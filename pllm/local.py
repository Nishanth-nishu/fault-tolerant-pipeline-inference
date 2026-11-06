"""Single-process baseline transport: the *unpartitioned* model executed in-process behind the same Engine
(same scheduler, batching, token selection), so baseline vs. distributed differ only in partitioning + RPC."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .model import SeqSlice, build_stage
from .proto import pipeline_pb2 as pb


class LocalTransport:
    def __init__(self, model: str, threads: int | None = None, cpus: str = ""):
        if cpus:
            from .worker import set_affinity
            set_affinity(cpus)
        if threads:
            torch.set_num_threads(threads)
        self.stage, self.tok = build_stage(model, -1, None)
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _run(self, req: pb.ForwardRequest):
        t0 = time.perf_counter()
        seqs = [SeqSlice(i.request_id, i.cache_position, i.num_tokens, i.want_logits) for i in req.items]
        x = torch.from_numpy(np.frombuffer(req.payload.data, dtype=np.int64).copy())
        out = self.stage.forward(x, seqs)
        res, row, k = [], 0, req.logits_topk
        for it in req.items:
            if not it.want_logits:
                continue
            r = out[row]
            row += 1
            if k > 0:
                v, ix = torch.topk(r, k)
                res.append(pb.SeqResult(request_id=it.request_id, op_seq=it.op_seq,
                                        scores=pb.Tensor(shape=[k], codec="fp32", data=v.numpy().tobytes()),
                                        topk_ids=ix.tolist()))
            else:
                res.append(pb.SeqResult(request_id=it.request_id, op_seq=it.op_seq,
                                        scores=pb.Tensor(shape=[r.numel()], codec="fp32", data=r.numpy().tobytes())))
        st = pb.StageStat(stage="s0", epoch=1, compute_ms=(time.perf_counter() - t0) * 1e3)
        return pb.ForwardReply(status=pb.OK, results=res, stats=[st], epochs={"s0": 1})

    async def forward(self, req, timeout):
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._run, req)

    async def forward_stage(self, stage, req, timeout):
        return await self.forward(req, timeout)

    async def health(self, stage, timeout):
        return pb.HealthReply(stage="s0", epoch=1)

    async def release(self, stage, rids=(), all_=False, timeout=2.0):
        if all_:
            self.stage.reset()
        for r in rids:
            self.stage.drop(r)
        return pb.Ack(status=pb.OK)

    async def close(self):
        self.pool.shutdown(wait=False)
