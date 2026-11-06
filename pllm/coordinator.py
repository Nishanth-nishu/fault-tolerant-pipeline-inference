"""Coordinator: admission control, iteration-level scheduling, token commits, failure detection and recovery.

Request lifecycle
-----------------
``history = prompt + committed tokens``; ``pos`` = number of history tokens whose KV state is (believed to be)
present on *both* stages. Every operation feeds ``history[pos : pos+n]`` at ``cache_position=pos``. When an op
consumes the last history token it asks for next-token scores and the coordinator commits exactly one token.
Prefill (possibly chunked) and decode are therefore the same operation, and recovery is "set pos back and
re-feed history" with outputs suppressed.

Correctness invariants (checked in tests/):
* a request has at most one op in flight -> per-request order is preserved;
* a token is committed only if the reply's (request_id, op_seq) equals the request's outstanding op and the
  reply's worker epochs equal the coordinator's current view -> stale replies are rejected;
* KV writes are position-indexed -> re-executing an op after a lost reply is idempotent;
* after ``max_retries`` failed attempts the request fails explicitly (never a silent partial answer).
"""
from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import grpc
import numpy as np

from .proto import pipeline_pb2 as pb
from .proto import pipeline_pb2_grpc as rpc
from .worker import GRPC_OPTS


class Overloaded(Exception):
    """Admission queue is full (explicit load shedding)."""


class RequestFailed(Exception):
    pass


class St(Enum):
    WAITING = 0
    ACTIVE = 1
    DONE = 2
    FAILED = 3


@dataclass
class EngineConfig:
    max_active: int = 16            # sequences holding KV state (memory bound)
    max_waiting: int = 64           # bounded admission queue -> Overloaded beyond this
    max_batch_seqs: int = 8         # micro-batch size limit (sequences)
    max_batch_tokens: int = 256     # micro-batch token budget (chunked prefill)
    prefill_chunk: int = 128
    max_inflight_batches: int = 2   # pipeline depth: >=2 lets both stages work concurrently
    rpc_timeout_s: float = 10.0
    max_retries: int = 3
    overload_backoff_s: float = 0.005
    act_codec: str = "fp32"
    logits_topk: int = 8
    recovery: str = "full"          # full | targeted
    batching: bool = True           # False -> one sequence per micro-batch
    pipeline_microbatch: bool = True  # cap micro-batch size at ceil(runnable / max_inflight_batches)
    depth_policy: str = "fixed"     # fixed | auto (online A/B between depth 1 and max_inflight_batches)
    auto_window_s: float = 1.0      # measurement window per arm
    auto_exploit_windows: int = 8   # exploit the winner this many windows before re-measuring
    auto_samples: int = 2           # clean windows per arm before deciding
    auto_min_ops: int = 32          # ...and each window must contain at least this many sequence-steps
    health_timeout_s: float = 1.0
    restart_timeout_s: float = 120.0
    recovery_backoff_s: float = 0.2


@dataclass
class Request:
    rid: str
    prompt: list[int]
    max_new_tokens: int
    eos_id: int | None = None
    generated: list[int] = field(default_factory=list)
    state: St = St.WAITING
    pos: int = 0
    op_seq: int = 0
    busy: bool = False
    needs_recovery: set = field(default_factory=set)   # stages whose KV for this request was lost
    attempts: int = 0
    error: str | None = None
    t_arrival: float = 0.0
    t_admit: float = 0.0
    t_first: float = 0.0
    t_done: float = 0.0
    token_times: list[float] = field(default_factory=list)
    recoveries: list[dict] = field(default_factory=list)
    future: asyncio.Future | None = None

    @property
    def history(self):
        return self.prompt + self.generated

    def finished(self):
        return len(self.generated) >= self.max_new_tokens or (
            self.eos_id is not None and self.generated and self.generated[-1] == self.eos_id)


@dataclass
class Counters:
    batches: int = 0
    batch_seqs: int = 0
    batch_tokens: int = 0
    tokens_committed: int = 0
    overload_rejects: int = 0         # admission rejections
    worker_overloads: int = 0         # OVERLOADED replies from workers (retried with backoff)
    rpc_errors: int = 0
    stale_rejected: int = 0           # replies discarded because of epoch/op mismatch
    retries: int = 0
    recoveries: int = 0
    failures: int = 0
    act_bytes: int = 0                # serialized activation payload bytes on the s0->s1 link
    link_msg_bytes: int = 0           # full s0->s1 request messages (payload + control)
    coord_req_bytes: int = 0          # coordinator->s0 requests
    coord_reply_bytes: int = 0        # s0->coordinator replies (includes scores)
    replay_tokens: int = 0
    replay_act_bytes: int = 0


class GrpcTransport:
    """Talks to two remote stage workers."""

    def __init__(self, addr0: str, addr1: str):
        self.addrs = {0: addr0, 1: addr1}
        self.ch = {s: grpc.aio.insecure_channel(a, options=GRPC_OPTS) for s, a in self.addrs.items()}
        self.stub = {s: rpc.StageServiceStub(c) for s, c in self.ch.items()}

    async def forward(self, req: pb.ForwardRequest, timeout: float) -> pb.ForwardReply:
        return await self.stub[0].Forward(req, timeout=timeout)

    async def forward_stage(self, stage: int, req: pb.ForwardRequest, timeout: float) -> pb.ForwardReply:
        return await self.stub[stage].Forward(req, timeout=timeout)

    async def health(self, stage: int, timeout: float) -> pb.HealthReply:
        return await self.stub[stage].Health(pb.HealthRequest(), timeout=timeout)

    async def release(self, stage: int, rids=(), all_=False, timeout=2.0):
        return await self.stub[stage].Release(pb.ReleaseRequest(request_ids=list(rids), all=all_), timeout=timeout)

    async def replay_boundary(self, rid, upto, epochs, codec, timeout):
        return await self.stub[0].ReplayBoundary(
            pb.ReplayRequest(request_id=rid, upto_position=upto, expected_epochs=epochs, act_codec=codec),
            timeout=timeout)

    async def close(self):
        for c in self.ch.values():
            await c.close()


class Engine:
    def __init__(self, transport, cfg: EngineConfig, supervisor=None, stage_names=("s0", "s1")):
        self.t = transport
        self.cfg = cfg
        self.sup = supervisor
        self.names = stage_names
        self.epochs: dict[str, int] = {}
        self.waiting: list[Request] = []
        self.active: list[Request] = []
        self.inflight = 0
        self.c = Counters()
        self.batch_log: list[dict] = []
        self.events: list[dict] = []
        self._wake = asyncio.Event()
        self._fail_lock = asyncio.Lock()
        self._task = None
        self._ids = itertools.count()
        self._recovering: set[str] = set()
        self.depth = cfg.max_inflight_batches
        self._auto = {"t0": None, "scores": {}, "exploit_left": 0, "history": []}
        self._admitted = 0

    # ------------------------------------------------------------------------------------------------------
    async def start(self):
        for i, n in enumerate(self.names):
            h = await self.t.health(i, self.cfg.health_timeout_s * 10)
            self.epochs[n] = h.epoch
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def submit(self, prompt: list[int], max_new_tokens: int, eos_id=None, rid=None) -> Request:
        if len(self.waiting) >= self.cfg.max_waiting:
            self.c.overload_rejects += 1
            raise Overloaded(f"admission queue full ({len(self.waiting)})")
        r = Request(rid or f"r{next(self._ids)}-{uuid.uuid4().hex[:6]}", list(prompt), max_new_tokens, eos_id)
        r.t_arrival = time.perf_counter()
        r.future = asyncio.get_running_loop().create_future()
        self.waiting.append(r)
        self._wake.set()
        return r

    async def generate(self, prompt, max_new_tokens, eos_id=None) -> Request:
        r = self.submit(prompt, max_new_tokens, eos_id)
        await r.future
        return r

    # ------------------------------------------------------------------------------------------------------
    async def _loop(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self.waiting and len(self.active) < self.cfg.max_active:
                r = self.waiting.pop(0)
                r.state, r.t_admit = St.ACTIVE, time.perf_counter()
                self._admitted += 1
                self.active.append(r)
            for r in self.active:
                if r.needs_recovery and not r.busy and r.rid not in self._recovering:
                    r.busy = True
                    self._recovering.add(r.rid)
                    asyncio.create_task(self._recover(r))
            self._auto_tick()
            while self.inflight < self.depth:
                batch = self._form_batch()
                if not batch:
                    break
                self.inflight += 1
                asyncio.create_task(self._run_batch(batch))

    def _auto_tick(self):
        """Online A/B of pipeline depth. On one host the stages share DRAM bandwidth, so smaller overlapping
        micro-batches can lose to one large batch; with a slow link, overlap hides latency. Measure, don't guess.

        Metric: sequence-steps per second (one decode step of one sequence = 1; robust to prefill/decode mix).
        Arms are sampled alternately (A,B,A,B) and windows in which new requests were admitted are discarded
        (their prefill bursts would bias whichever arm happened to be active)."""
        cfg, st = self.cfg, self._auto
        if cfg.depth_policy != "auto" or cfg.max_inflight_batches < 2:
            return
        now = time.perf_counter()
        runnable = sum(1 for r in self.active if r.state == St.ACTIVE)
        if runnable < 2:
            st["t0"] = None
            return
        if st["t0"] is None:
            st.update(t0=now, ops0=self.c.batch_seqs, adm0=self._admitted, dirty=0)
            return
        if now - st["t0"] < cfg.auto_window_s or self.c.batch_seqs - st["ops0"] < cfg.auto_min_ops:
            return  # window closes on time AND on work, so slow models still get enough samples per window
        rate = (self.c.batch_seqs - st["ops0"]) / (now - st["t0"])
        contaminated = self._admitted != st["adm0"]
        arms = [cfg.max_inflight_batches, 1]
        if st["exploit_left"] > 0:
            st["exploit_left"] -= 1
            if st["exploit_left"] == 0:
                st["scores"] = {}
                self.depth = arms[0]
        elif contaminated and st.get("dirty", 0) < 2:
            st["dirty"] = st.get("dirty", 0) + 1     # retry the same arm on a clean window
        else:
            st["dirty"] = 0
            st["scores"].setdefault(self.depth, []).append(rate)
            n = [len(st["scores"].get(d, [])) for d in arms]
            if min(n) >= cfg.auto_samples:
                mean = {d: sum(v) / len(v) for d, v in st["scores"].items()}
                self.depth = max(mean, key=mean.get)
                st["exploit_left"] = cfg.auto_exploit_windows
                st["history"].append({"t": now, **{f"d{k}": round(v, 1) for k, v in mean.items()}, "chosen": self.depth})
            else:
                self.depth = arms[1] if self.depth == arms[0] else arms[0]   # alternate A,B,A,B
        st.update(t0=now, ops0=self.c.batch_seqs, adm0=self._admitted)

    def _form_batch(self):
        """Stall-free selection (Sarathi-Serve): decodes first, then prefill chunks fill the token budget."""
        cfg = self.cfg
        ready = [r for r in self.active if not r.busy and not r.needs_recovery and r.state == St.ACTIVE]
        if not ready:
            return []
        max_seqs = cfg.max_batch_seqs if cfg.batching else 1
        if cfg.batching and cfg.pipeline_microbatch and self.depth > 1:
            # Pipeline micro-batching (GPipe/Orca): split the runnable set into >= depth micro-batches so that
            # stage 0 can work on one while stage 1 works on another, instead of one big batch idling a stage.
            runnable = sum(1 for r in self.active if not r.needs_recovery and r.state == St.ACTIVE)
            max_seqs = min(max_seqs, max(1, -(-runnable // self.depth)))
        decodes = [r for r in ready if len(r.history) - r.pos == 1]
        prefills = [r for r in ready if len(r.history) - r.pos > 1]
        batch, budget = [], cfg.max_batch_tokens
        for r in decodes + prefills:
            if len(batch) >= max_seqs or budget <= 0:
                break
            n = min(len(r.history) - r.pos, cfg.prefill_chunk, budget)
            batch.append((r, r.pos, n))
            budget -= n
        for r, _, _ in batch:
            r.busy = True
            r.op_seq += 1
        return batch

    def _build(self, batch, forward_downstream=True, want=True, replay=False) -> pb.ForwardRequest:
        items, toks = [], []
        for r, pos, n in batch:
            want_logits = want and (pos + n == len(r.history))
            items.append(pb.SeqItem(request_id=r.rid, op_seq=r.op_seq, cache_position=pos, num_tokens=n,
                                    want_logits=want_logits, replay=replay))
            toks.extend(r.history[pos:pos + n])
        data = np.asarray(toks, dtype=np.int64).tobytes()
        return pb.ForwardRequest(batch_id=uuid.uuid4().hex[:12], items=items,
                                 payload=pb.Tensor(shape=[len(toks)], codec="i64", data=data),
                                 expected_epochs=dict(self.epochs), forward_downstream=forward_downstream,
                                 act_codec=self.cfg.act_codec, logits_topk=self.cfg.logits_topk,
                                 deadline_ms=self.cfg.rpc_timeout_s * 1e3 * 0.9)

    def _account(self, req: pb.ForwardRequest, reply: pb.ForwardReply | None, replay=False):
        self.c.coord_req_bytes += req.ByteSize()
        if reply is not None:
            self.c.coord_reply_bytes += reply.ByteSize()
            for s in reply.stats:
                self.c.act_bytes += s.act_bytes_out
                self.c.link_msg_bytes += s.msg_bytes_out
                if replay:
                    self.c.replay_act_bytes += s.act_bytes_out

    async def _run_batch(self, batch):
        t0 = time.perf_counter()
        req = self._build(batch)
        epochs_sent = dict(req.expected_epochs)
        reply, err = None, None
        try:
            reply = await self.t.forward(req, timeout=self.cfg.rpc_timeout_s)
        except grpc.aio.AioRpcError as e:
            err = f"rpc {e.code().name}: {e.details()}"
            self.c.rpc_errors += 1
        except Exception as e:  # transport-level failure
            err = f"{type(e).__name__}: {e}"
            self.c.rpc_errors += 1
        self._account(req, reply)
        t1 = time.perf_counter()
        try:
            if reply is not None and reply.status == pb.OK:
                self._commit(batch, reply, epochs_sent, t0, t1)
            elif reply is not None and reply.status == pb.OVERLOADED:
                self.c.worker_overloads += 1
                await asyncio.sleep(self.cfg.overload_backoff_s)
                for r, _, _ in batch:
                    r.busy = False
            else:
                why = err or f"{pb.Status.Name(reply.status)} {reply.failed_stage}: {reply.message}"
                await self._on_failure(batch, why, reply)
        finally:
            self.inflight -= 1
            self._wake.set()

    def _commit(self, batch, reply, epochs_sent, t0, t1):
        cur = dict(self.epochs)
        live = dict(reply.epochs)
        if epochs_sent != cur or any(live.get(k, v) != v for k, v in cur.items()):
            # The cluster changed while this batch was in flight: the reply may stem from a stale worker.
            self.c.stale_rejected += len(batch)
            for r, _, _ in batch:
                r.busy = False
            return
        res = {x.request_id: x for x in reply.results}
        st = {s.stage: s for s in reply.stats}
        self.c.batches += 1
        self.c.batch_seqs += len(batch)
        self.c.batch_tokens += sum(n for _, _, n in batch)
        self.batch_log.append({"t": t1, "seqs": len(batch), "tokens": sum(n for _, _, n in batch), "rtt_ms": (t1 - t0) * 1e3,
                               **{f"{k}_{f}": getattr(v, f) for k, v in st.items()
                                  for f in ("queue_ms", "compute_ms", "codec_ms", "handler_ms", "downstream_ms")}})
        for r, pos, n in batch:
            r.busy = False
            r.attempts = 0
            want = pos + n == len(r.history)
            if want:
                x = res.get(r.rid)
                if x is None or x.op_seq != r.op_seq:
                    self.c.stale_rejected += 1
                    continue
                tok = self._select(x)
                r.pos = pos + n
                r.generated.append(tok)
                now = time.perf_counter()
                if not r.token_times:
                    r.t_first = now
                r.token_times.append(now)
                self.c.tokens_committed += 1
                if r.finished():
                    self._finish(r, St.DONE)
            else:
                r.pos = pos + n

    def _select(self, x: pb.SeqResult) -> int:
        v = np.frombuffer(x.scores.data, dtype=np.float32)
        i = int(np.argmax(v))  # greedy; first max on ties (matches torch.argmax)
        return int(x.topk_ids[i]) if len(x.topk_ids) else i

    def _finish(self, r: Request, state: St, error: str | None = None):
        r.state, r.t_done, r.error = state, time.perf_counter(), error
        if r in self.active:
            self.active.remove(r)
        if state == St.FAILED:
            self.c.failures += 1
        asyncio.create_task(self._release(r.rid))
        if r.future and not r.future.done():
            r.future.set_result(r)

    async def _release(self, rid):
        for s in (0, 1):
            try:
                await self.t.release(s, [rid])
            except Exception:
                pass

    # ------------------------------------------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------------------------------------------
    async def _probe(self) -> dict[str, int | None]:
        out = {}
        for i, n in enumerate(self.names):
            try:
                h = await self.t.health(i, self.cfg.health_timeout_s)
                out[n] = h.epoch
            except Exception:
                out[n] = None
        return out

    async def _ensure_cluster(self, reason: str):
        """Detect dead/restarted workers; restart dead ones via the supervisor; advance epochs.
        Every active request whose KV lived on a changed stage is marked for recovery."""
        async with self._fail_lock:
            t0 = time.perf_counter()
            live = await self._probe()
            changed = []
            for i, n in enumerate(self.names):
                if live[n] is None:
                    if self.sup is None:
                        deadline = time.monotonic() + self.cfg.restart_timeout_s
                        while live[n] is None and time.monotonic() < deadline:
                            await asyncio.sleep(0.2)
                            live = await self._probe()
                        if live[n] is None:
                            raise RuntimeError(f"stage {n} unreachable")
                    else:
                        live[n] = await self.sup.restart(i)
                if live[n] != self.epochs[n]:
                    changed.append(n)
                    self.epochs[n] = live[n]
            if changed:
                for r in self.active:
                    r.needs_recovery.update(changed)
                self.events.append({"t": t0, "kind": "epoch_change", "stages": changed, "reason": reason,
                                    "detect_restart_ms": (time.perf_counter() - t0) * 1e3})
            return changed

    async def _on_failure(self, batch, why, reply):
        for r, _, _ in batch:
            r.attempts += 1
            self.c.retries += 1
        try:
            changed = await self._ensure_cluster(why)
        except Exception as e:
            changed = []
            why = f"{why}; cluster unrecoverable: {e}"
            for r, _, _ in batch:
                r.attempts = self.cfg.max_retries + 1
        if reply is not None and reply.status == pb.CACHE_MISS and not changed:
            for r, _, _ in batch:
                r.needs_recovery.update(self.names)
        for r, _, _ in batch:
            r.busy = False
            if r.attempts > self.cfg.max_retries and r.state == St.ACTIVE:
                self._finish(r, St.FAILED, f"retry limit exceeded: {why}")
        self.events.append({"t": time.perf_counter(), "kind": "batch_failure", "why": why,
                            "rids": [r.rid for r, _, _ in batch]})

    async def _recover(self, r: Request):
        """Rebuild lost KV state for one request, then resume from the next uncommitted step.

        full     : drop the request on both stages and replay history[0:pos] through the whole pipeline.
        targeted : only s1 lost -> s0 resends its logged boundary activations (no recompute on s0);
                   only s0 lost -> replay history through s0 alone (forward_downstream=False); s1 keeps its KV.
        Replay outputs are suppressed (want_logits=False); the uncommitted token is produced by the regular step.
        """
        t0 = time.perf_counter()
        lost = set(r.needs_recovery)
        mode = self.cfg.recovery
        upto = r.pos
        ok, why = True, ""
        try:
            if mode == "targeted" and lost == {"s1"} and upto > 0:
                await self.t.release(1, [r.rid])
                ack = await self.t.replay_boundary(r.rid, upto, dict(self.epochs), self.cfg.act_codec,
                                                   self.cfg.rpc_timeout_s)
                if ack.status != pb.OK:
                    raise RuntimeError(f"boundary replay: {pb.Status.Name(ack.status)} {ack.message}")
                self.c.replay_act_bytes += ack.value
                path = "s1-from-log"
            elif mode == "targeted" and lost == {"s0"} and upto > 0:
                await self.t.release(0, [r.rid])
                await self._replay(r, upto, forward_downstream=False)
                path = "s0-local"
            else:
                for s in (0, 1):
                    await self.t.release(s, [r.rid])
                await self._replay(r, upto, forward_downstream=True)
                path = "full"
            self.c.replay_tokens += upto
        except Exception as e:
            ok, why = False, f"{type(e).__name__}: {e}"
        dt = (time.perf_counter() - t0) * 1e3
        r.recoveries.append({"lost": sorted(lost), "path": path if ok else "error", "replayed": upto, "ms": dt, "ok": ok,
                             **({} if ok else {"error": why})})
        self.c.recoveries += 1
        self._recovering.discard(r.rid)
        r.busy = False
        if ok:
            r.needs_recovery.clear()
        else:
            r.attempts += 1
            if r.attempts > self.cfg.max_retries:
                self._finish(r, St.FAILED, f"recovery failed: {why}")
            else:
                await asyncio.sleep(self.cfg.recovery_backoff_s * (2 ** (r.attempts - 1)))
                try:
                    await self._ensure_cluster(why)
                except Exception:
                    pass
        self._wake.set()

    async def _replay(self, r: Request, upto: int, forward_downstream: bool):
        pos = 0
        while pos < upto:
            n = min(self.cfg.prefill_chunk, upto - pos)
            r.op_seq += 1
            req = self._build([(r, pos, n)], forward_downstream=forward_downstream, want=False, replay=True)
            if forward_downstream:
                rep = await self.t.forward(req, timeout=self.cfg.rpc_timeout_s)
            else:
                rep = await self.t.forward_stage(0, req, timeout=self.cfg.rpc_timeout_s)
            self._account(req, rep, replay=True)
            if rep.status != pb.OK:
                raise RuntimeError(f"replay {pb.Status.Name(rep.status)}: {rep.message}")
            pos += n
