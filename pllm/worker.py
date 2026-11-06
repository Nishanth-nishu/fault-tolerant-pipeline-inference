"""Pipeline stage worker: a gRPC (asyncio) server that owns a layer range and per-request KV caches.

Stage 0 receives token ids from the coordinator, runs embeddings + its layers, encodes the boundary activation
and forwards it to stage 1 over gRPC. Stage 1 runs the remaining layers + LM head and returns next-token scores.
Because the gRPC handler awaits the downstream call *outside* the compute thread, stage 0 can compute
micro-batch k+1 while stage 1 computes micro-batch k (pipeline parallelism).

Run:  python -m pllm.worker --stage 1 --port 50052 --model gpt2
      python -m pllm.worker --stage 0 --port 50051 --model gpt2 --downstream 127.0.0.1:50052
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
import torch

from . import codec as C
from .model import CacheMiss, SeqSlice, build_stage
from .proto import pipeline_pb2 as pb
from .proto import pipeline_pb2_grpc as rpc

GRPC_OPTS = [("grpc.max_send_message_length", 256 << 20), ("grpc.max_receive_message_length", 256 << 20),
             ("grpc.so_reuseport", 0),
             # Fast reconnect after a peer restart (default backoff grows to 120 s and would stall recovery).
             ("grpc.initial_reconnect_backoff_ms", 100), ("grpc.min_reconnect_backoff_ms", 100),
             ("grpc.max_reconnect_backoff_ms", 1000), ("grpc.enable_retries", 0)]


class ActivationLog:
    """Stage-0 log of boundary activations per request (enables stage-1-only recovery, Petals-style)."""

    def __init__(self):
        self.buf: dict[str, torch.Tensor] = {}
        self.len: dict[str, int] = {}

    def write(self, rid: str, pos: int, x: torch.Tensor):
        b = self.buf.get(rid)
        end = pos + x.shape[0]
        if b is None or b.shape[0] < end:
            nb = torch.zeros(max(end, 2 * (b.shape[0] if b is not None else 64)), x.shape[1])
            if b is not None:
                nb[: b.shape[0]] = b
            self.buf[rid] = b = nb
        b[pos:end] = x
        self.len[rid] = end

    def read(self, rid: str, upto: int):
        if self.len.get(rid, 0) < upto:
            return None
        return self.buf[rid][:upto]

    def drop(self, rid):
        self.buf.pop(rid, None)
        self.len.pop(rid, None)


class StageServicer(rpc.StageServiceServicer):
    def __init__(self, args):
        self.args = args
        self.name = f"s{args.stage}"
        self.epoch = args.epoch
        torch.set_num_threads(args.threads)
        self.stage, _ = build_stage(args.model, args.stage, args.split, lean=True,
                                    head_on_first=args.head_on_first)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.name}-compute")
        self.queue_depth = 0
        self.alog = ActivationLog() if (args.stage == 0 and args.log_activations) else None
        self.downstream = None
        if args.stage == 0:
            ch = grpc.aio.insecure_channel(args.downstream, options=GRPC_OPTS)
            self.downstream = rpc.StageServiceStub(ch)
        self.rng = random.Random(args.seed)

    # ---------------------------------------------------------------------------------------------------
    async def _netem(self, nbytes: int):
        a = self.args
        d = a.net_delay_ms / 1e3 + (nbytes * 8 / (a.net_bw_mbps * 1e6) if a.net_bw_mbps > 0 else 0.0)
        if d > 0:
            await asyncio.sleep(d)

    def _compute(self, req: pb.ForwardRequest, t_enq: float):
        t0 = time.perf_counter()
        queue_ms = (t0 - t_enq) * 1e3
        seqs = [SeqSlice(i.request_id, i.cache_position, i.num_tokens, i.want_logits) for i in req.items]
        p = req.payload
        if self.stage.is_first:
            x = torch.from_numpy(np.frombuffer(p.data, dtype=np.int64).copy())
            codec_ms = 0.0
        else:
            tc = time.perf_counter()
            x = C.decode(p.data, p.scales, p.shape, p.codec)
            codec_ms = (time.perf_counter() - tc) * 1e3
        out = self.stage.forward(x, seqs)
        if self.alog is not None:
            off = 0
            for s in seqs:
                self.alog.write(s.request_id, s.cache_position, out[off:off + s.num_tokens])
                off += s.num_tokens
        enc = None
        if not self.stage.is_last and req.forward_downstream:
            tc = time.perf_counter()
            enc = C.encode(out, req.act_codec or "fp32")
            codec_ms += (time.perf_counter() - tc) * 1e3
        compute_ms = (time.perf_counter() - t0) * 1e3 - codec_ms
        return out, enc, queue_ms, compute_ms, codec_ms

    def _results(self, req, logits: torch.Tensor):
        res, row = [], 0
        k = req.logits_topk
        for it in req.items:
            if not it.want_logits:
                continue
            r = logits[row]
            row += 1
            if k > 0:
                v, ix = torch.topk(r, k)
                res.append(pb.SeqResult(request_id=it.request_id, op_seq=it.op_seq,
                                        scores=pb.Tensor(shape=[k], codec="fp32", data=v.float().numpy().tobytes()),
                                        topk_ids=ix.tolist()))
            else:
                res.append(pb.SeqResult(request_id=it.request_id, op_seq=it.op_seq,
                                        scores=pb.Tensor(shape=[r.numel()], codec="fp32",
                                                         data=r.float().numpy().tobytes())))
        return res

    # ---------------------------------------------------------------------------------------------------
    async def Forward(self, req: pb.ForwardRequest, ctx):
        a = self.args
        exp = req.expected_epochs.get(self.name, -1)
        if exp != -1 and exp != self.epoch:
            return pb.ForwardReply(status=pb.STALE_EPOCH, failed_stage=self.name, epochs={self.name: self.epoch},
                                   message=f"{self.name} epoch {self.epoch} != expected {exp}")
        if self.queue_depth >= a.max_queue:
            return pb.ForwardReply(status=pb.OVERLOADED, failed_stage=self.name, epochs={self.name: self.epoch},
                                   message=f"{self.name} queue full ({self.queue_depth})")
        t_in = time.perf_counter()
        self.queue_depth += 1
        try:
            if a.fault_delay_ms > 0:
                await asyncio.sleep(a.fault_delay_ms / 1e3)
            loop = asyncio.get_running_loop()
            try:
                out, enc, q_ms, c_ms, k_ms = await loop.run_in_executor(self.pool, self._compute, req, time.perf_counter())
            except CacheMiss as e:
                return pb.ForwardReply(status=pb.CACHE_MISS, failed_stage=self.name, message=str(e),
                                       epochs={self.name: self.epoch})
        finally:
            self.queue_depth -= 1
        stat = pb.StageStat(stage=self.name, epoch=self.epoch, queue_ms=q_ms, compute_ms=c_ms, codec_ms=k_ms)
        if self.stage.is_last and not self.stage.has_head:
            res = [pb.SeqResult(request_id=it.request_id, op_seq=it.op_seq) for it in req.items if it.want_logits]
            reply = pb.ForwardReply(status=pb.OK, results=res, stats=[stat], epochs={self.name: self.epoch},
                                    hidden_out=pb.Tensor(shape=list(out.shape), codec="fp32",
                                                         data=out.float().contiguous().numpy().tobytes()))
        elif self.stage.is_last:
            reply = pb.ForwardReply(status=pb.OK, results=self._results(req, out), stats=[stat],
                                    epochs={self.name: self.epoch})
        elif not req.forward_downstream:
            reply = pb.ForwardReply(status=pb.OK, stats=[stat], epochs={self.name: self.epoch})
        else:
            data, scales, shape = enc
            down = pb.ForwardRequest(batch_id=req.batch_id, items=req.items,
                                     payload=pb.Tensor(shape=shape, codec=req.act_codec or "fp32", data=data, scales=scales),
                                     expected_epochs=req.expected_epochs, act_codec=req.act_codec,
                                     logits_topk=req.logits_topk)
            stat.act_bytes_out = len(data) + len(scales)
            stat.msg_bytes_out = down.ByteSize()
            reply = await self._call_downstream(down, req, stat)
            if reply.status == pb.OK and self.stage.has_head and reply.HasField("hidden_out"):
                # Ring placement: s1 returned final hidden rows; the tied LM head runs here, next to the embedding.
                t_h = time.perf_counter()
                h = C.decode(reply.hidden_out.data, b"", reply.hidden_out.shape, "fp32")
                logits = await asyncio.get_running_loop().run_in_executor(self.pool, self.stage.head, h)
                res = self._results(req, logits)
                del reply.results[:]
                reply.results.extend(res)
                reply.ClearField("hidden_out")
                for st in reply.stats:
                    if st.stage == self.name:
                        st.compute_ms += (time.perf_counter() - t_h) * 1e3
        for st in reply.stats:  # protobuf copies on insert: stamp the copy that is actually returned
            if st.stage == self.name:
                st.handler_ms = (time.perf_counter() - t_in) * 1e3
        if a.fault_drop_reply_p > 0 and self.rng.random() < a.fault_drop_reply_p:
            # Work was done (KV written) but the reply is "lost": exercises idempotent retries.
            await ctx.abort(grpc.StatusCode.UNAVAILABLE, "injected reply loss")
        return reply

    async def _call_downstream(self, down, req, stat):
        await self._netem(stat.msg_bytes_out)
        timeout = (req.deadline_ms or self.args.downstream_timeout_ms) / 1e3
        t_d = time.perf_counter()
        try:
            r = await self.downstream.Forward(down, timeout=timeout)
            stat.downstream_ms = (time.perf_counter() - t_d) * 1e3
        except grpc.aio.AioRpcError as e:
            return pb.ForwardReply(status=pb.DOWNSTREAM_FAILED, failed_stage="s1",
                                   message=f"downstream {e.code().name}: {e.details()}",
                                   stats=[stat], epochs={self.name: self.epoch})
        await self._netem(r.ByteSize())
        r.stats.insert(0, stat)
        r.epochs[self.name] = self.epoch
        return r

    async def ReplayBoundary(self, req: pb.ReplayRequest, ctx):
        """Targeted recovery of stage 1: resend the logged boundary activations [0, upto) without recomputing."""
        if self.alog is None:
            return pb.Ack(status=pb.BAD_REQUEST, message="activation log disabled")
        x = self.alog.read(req.request_id, req.upto_position)
        if x is None:
            return pb.Ack(status=pb.CACHE_MISS, message="activation log too short")
        data, scales, shape = C.encode(x, req.act_codec or "fp32")
        item = pb.SeqItem(request_id=req.request_id, op_seq=-1, cache_position=0,
                          num_tokens=req.upto_position, want_logits=False, replay=True)
        down = pb.ForwardRequest(batch_id=f"replay-{req.request_id}", items=[item],
                                 payload=pb.Tensor(shape=shape, codec=req.act_codec or "fp32", data=data, scales=scales),
                                 expected_epochs=req.expected_epochs, act_codec=req.act_codec)
        stat = pb.StageStat(stage=self.name, epoch=self.epoch, act_bytes_out=len(data) + len(scales),
                            msg_bytes_out=down.ByteSize())
        r = await self._call_downstream(down, pb.ForwardRequest(), stat)
        return pb.Ack(status=r.status, message=r.message, value=stat.act_bytes_out)

    async def Release(self, req: pb.ReleaseRequest, ctx):
        n = 0
        if req.all:
            n = len(self.stage.caches)
            self.stage.reset()
            if self.alog:
                self.alog = ActivationLog()
        for rid in req.request_ids:
            n += self.stage.drop(rid)
            if self.alog:
                self.alog.drop(rid)
        return pb.Ack(status=pb.OK, value=n)

    async def Health(self, req, ctx):
        return pb.HealthReply(stage=self.name, epoch=self.epoch, queue_depth=self.queue_depth,
                              active_sequences=len(self.stage.caches), cache_bytes=self.stage.cache_bytes(),
                              layer_start=self.stage.layer_start, layer_end=self.stage.layer_end,
                              model=self.args.model, pid=os.getpid())


def parse(argv=None):
    p = argparse.ArgumentParser(description="pipeline stage worker")
    p.add_argument("--stage", type=int, required=True, choices=[0, 1])
    p.add_argument("--model", default="gpt2")
    p.add_argument("--split", type=int, default=None, help="first layer owned by stage 1 (default n_layers//2)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--downstream", default="127.0.0.1:50052")
    p.add_argument("--epoch", type=int, default=None, help="worker epoch (default: wall-clock ms, unique per start)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    p.add_argument("--max-queue", type=int, default=8, help="bounded inbound queue (backpressure)")
    p.add_argument("--downstream-timeout-ms", type=float, default=10000)
    p.add_argument("--log-activations", action="store_true", help="stage 0: keep boundary activations for targeted recovery")
    p.add_argument("--net-delay-ms", type=float, default=0.0, help="emulated one-way latency on the s0<->s1 link")
    p.add_argument("--net-bw-mbps", type=float, default=0.0, help="emulated bandwidth on the s0<->s1 link (0=unlimited)")
    p.add_argument("--fault-drop-reply-p", type=float, default=0.0)
    p.add_argument("--fault-delay-ms", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-on-first", action="store_true", help="tied-head co-location: LM head on stage 0")
    p.add_argument("--cpus", default="", help="comma-separated logical CPU ids to pin this worker to")
    p.add_argument("--parent-pid", type=int, default=0, help="exit when this process dies (no orphaned workers)")
    p.add_argument("--standby", action="store_true", help="hot spare: load weights, wait for 'GO <epoch>' on stdin")
    return p.parse_args(argv)


async def _parent_watchdog(pid: int):
    import psutil
    while True:
        await asyncio.sleep(1.0)
        if not psutil.pid_exists(pid):
            os._exit(0)


async def serve(args):
    if args.parent_pid:
        asyncio.get_running_loop().create_task(_parent_watchdog(args.parent_pid))
    servicer = StageServicer(args)  # loads weights (the slow part of a restart)
    if args.standby:
        # Hot spare: weights are resident; block until the supervisor promotes us with "GO <epoch>".
        print(f"STANDBY {servicer.name} pid={os.getpid()}", flush=True)
        line = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
        parts = line.split()
        if not parts or parts[0] != "GO":
            return
        servicer.epoch = args.epoch = int(parts[1])
    server = grpc.aio.server(options=GRPC_OPTS)
    rpc.add_StageServiceServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{args.host}:{args.port}")
    if bound == 0:
        raise SystemExit(f"cannot bind {args.host}:{args.port}")
    await server.start()
    print(f"READY {servicer.name} port={args.port} epoch={args.epoch} layers=[{servicer.stage.layer_start},"
          f"{servicer.stage.layer_end}) pid={os.getpid()}", flush=True)
    await server.wait_for_termination()


def set_affinity(cpus: str | list[int]):
    if not cpus:
        return
    import psutil
    ids = [int(c) for c in cpus.split(",")] if isinstance(cpus, str) else list(cpus)
    psutil.Process().cpu_affinity(ids)


def main(argv=None):
    args = parse(argv)
    set_affinity(args.cpus)
    if args.epoch is None:
        args.epoch = int(time.time() * 1000) % (1 << 40)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
