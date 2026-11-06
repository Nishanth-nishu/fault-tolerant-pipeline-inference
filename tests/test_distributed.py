"""End-to-end tests over real gRPC worker processes on loopback TCP.

Primary success criterion: every request either matches the single-process greedy reference exactly or fails
explicitly; tokens are never stale, duplicated or out of order — including under worker kills, lost replies,
overload and timeouts.
"""
import asyncio
import os
import contextlib

import pytest

from pllm.coordinator import Engine, EngineConfig, GrpcTransport, Overloaded, St
from pllm.proto import pipeline_pb2 as pb
from pllm.supervisor import Supervisor

from .conftest import PROMPTS, free_ports, run

N_NEW = 24


@contextlib.contextmanager
def cluster(**kw):
    kw.setdefault("threads", 2)
    kw.setdefault("log_dir", os.path.join(os.path.dirname(__file__), "..", "scratch_logs", "tests"))
    sup = Supervisor("gpt2", None, ports=tuple(free_ports(2)), **kw)
    run(sup.start())
    try:
        yield sup
    finally:
        sup.stop()


@pytest.fixture(scope="module")
def basic():
    with cluster(s0_args=["--log-activations"], hot_spare=True) as sup:
        yield sup


async def _run(sup, cfg=None, prompts=PROMPTS, tok=None, during=None):
    tr = GrpcTransport(sup.addr(0), sup.addr(1))
    eng = Engine(tr, cfg or EngineConfig(), supervisor=sup)
    await eng.start()
    try:
        tasks = [asyncio.create_task(eng.generate(tok(p).input_ids, N_NEW)) for p in prompts]
        if during:
            await during(eng, sup)
        rs = await asyncio.gather(*tasks)
    finally:
        await eng.stop()
        await tr.close()
    return eng, rs


def check_exact(rs, prompts, refs):
    for r, p in zip(rs, prompts):
        assert r.state == St.DONE, (r.rid, r.error)
        assert r.generated == refs[p][:N_NEW], f"mismatch for {p!r}"
        assert len(r.token_times) == N_NEW, "duplicate or missing token emissions"
        assert all(a <= b for a, b in zip(r.token_times, r.token_times[1:])), "out-of-order emission"


def test_matches_reference_concurrent(basic, gpt2, references):
    eng, rs = run(_run(basic, tok=gpt2[0]))
    check_exact(rs, PROMPTS, references)
    assert eng.c.batch_seqs > eng.c.batches, "micro-batching should have grouped sequences"
    assert eng.c.act_bytes > 0 and eng.c.stale_rejected == 0


def test_no_batching_chunked_prefill(basic, gpt2, references):
    cfg = EngineConfig(batching=False, prefill_chunk=3, max_inflight_batches=2)
    eng, rs = run(_run(basic, cfg, tok=gpt2[0]))
    check_exact(rs, PROMPTS, references)
    assert eng.c.batch_seqs == eng.c.batches


@pytest.mark.parametrize("codec", ["bf16", "int8h"])
def test_compressed_link_runs(basic, gpt2, references, codec):
    eng, rs = run(_run(basic, EngineConfig(act_codec=codec), tok=gpt2[0]))
    assert all(r.state == St.DONE and len(r.generated) == N_NEW for r in rs)
    agree = sum(a == b for r, p in zip(rs, PROMPTS) for a, b in zip(r.generated, references[p])) / (N_NEW * len(rs))
    assert agree > 0.6, agree  # lossy link: high but not necessarily exact agreement (see bench/accuracy.py)


def test_stale_epoch_rejected_by_worker(basic):
    async def go():
        tr = GrpcTransport(basic.addr(0), basic.addr(1))
        h0 = await tr.health(0, 2)
        import numpy as np
        req = pb.ForwardRequest(items=[pb.SeqItem(request_id="stale", op_seq=1, cache_position=0, num_tokens=2,
                                                  want_logits=True)],
                                payload=pb.Tensor(shape=[2], codec="i64", data=np.array([1, 2], dtype=np.int64).tobytes()),
                                expected_epochs={"s0": h0.epoch - 1}, forward_downstream=True)
        r = await tr.forward(req, 5)
        h0b = await tr.health(0, 2)
        await tr.close()
        return r, h0, h0b
    r, h0, h0b = run(go())
    assert r.status == pb.STALE_EPOCH and r.failed_stage == "s0"
    assert h0b.active_sequences == h0.active_sequences, "a stale request must not create state"


def test_stale_reply_rejected_by_coordinator():
    """A reply produced under an old epoch view must never commit a token."""
    class Fake:
        async def health(self, s, t):
            return pb.HealthReply(epoch=7)

        async def forward(self, req, timeout):
            it = req.items[0]
            return pb.ForwardReply(status=pb.OK, epochs={"s0": 7, "s1": 6},  # s1 restarted meanwhile
                                   results=[pb.SeqResult(request_id=it.request_id, op_seq=it.op_seq,
                                                         scores=pb.Tensor(data=b"\0\0\x80?"), topk_ids=[42])])

        async def release(self, *a, **k):
            return pb.Ack()

    async def go():
        eng = Engine(Fake(), EngineConfig())
        await eng.start()
        r = eng.submit([1, 2, 3], 2)
        await asyncio.sleep(0.2)
        await eng.stop()
        return eng, r
    eng, r = run(go())
    assert r.generated == [] and eng.c.stale_rejected > 0


def test_admission_overload_is_explicit(basic, gpt2, references):
    async def go():
        tr = GrpcTransport(basic.addr(0), basic.addr(1))
        eng = Engine(tr, EngineConfig(max_active=1, max_waiting=2), supervisor=basic)
        await eng.start()
        tok = gpt2[0]
        accepted, rejected = [], 0
        for p in PROMPTS:
            try:
                accepted.append((p, eng.submit(tok(p).input_ids, N_NEW)))
            except Overloaded:
                rejected += 1
        await asyncio.gather(*[r.future for _, r in accepted])
        await eng.stop()
        await tr.close()
        return accepted, rejected, eng
    accepted, rejected, eng = run(go())
    assert rejected == len(PROMPTS) - 2 == eng.c.overload_rejects
    check_exact([r for _, r in accepted], [p for p, _ in accepted], references)


def test_worker_backpressure_and_lost_replies(gpt2, references):
    """Tiny worker queues (OVERLOADED replies) + 15% injected reply loss after KV writes: output still exact."""
    with cluster(s0_args=["--max-queue", "1", "--fault-drop-reply-p", "0.15", "--seed", "3"],
                 s1_args=["--max-queue", "1"]) as sup:
        cfg = EngineConfig(batching=False, max_inflight_batches=4, max_retries=50)
        eng, rs = run(_run(sup, cfg, tok=gpt2[0]))
    check_exact(rs, PROMPTS, references)
    assert eng.c.rpc_errors > 0, "fault injection should have dropped some replies"
    assert eng.c.worker_overloads > 0, "bounded worker queues should have pushed back"


async def _kill_when(eng, sup, stages, after_tokens=30):
    while eng.c.tokens_committed < after_tokens:
        await asyncio.sleep(0.002)
    for s in stages:
        sup.kill(s)


@pytest.mark.parametrize("mode,stages", [("full", [1]), ("full", [0]), ("full", [0, 1]),
                                         ("targeted", [1]), ("targeted", [0]), ("targeted", [0, 1])])
def test_worker_kill_recovery(basic, gpt2, references, mode, stages):
    run(basic.wait_spares())
    cfg = EngineConfig(recovery=mode, rpc_timeout_s=5)
    n0 = len(basic.restart_ms)
    eng, rs = run(_run(basic, cfg, tok=gpt2[0], during=lambda e, s: _kill_when(e, s, stages)))
    check_exact(rs, PROMPTS, references)
    assert eng.c.recoveries >= 1 and eng.c.failures == 0
    for x in basic.restart_ms[n0:]:
        assert x["hot_spare"] and x["spare_loaded"] and x["ms"] < 5000, x  # warm promotion, no model load
    paths = {x["path"] for r in rs for x in r.recoveries}
    if mode == "targeted" and stages == [1]:
        assert paths <= {"s1-from-log", "full"} and "s1-from-log" in paths
    if mode == "targeted" and stages == [0]:
        assert "s0-local" in paths


def test_timeout_fails_explicitly(gpt2):
    """A stage that is alive but too slow: deadlines fire, bounded retries are exhausted, request FAILS."""
    with cluster(s1_args=["--fault-delay-ms", "1500"]) as sup:
        cfg = EngineConfig(rpc_timeout_s=0.4, max_retries=2, recovery_backoff_s=0.01)
        eng, rs = run(_run(sup, cfg, prompts=PROMPTS[:2], tok=gpt2[0]))
    for r in rs:
        assert r.state == St.FAILED and "retry limit" in r.error
        assert r.generated == [], "no token may be emitted from timed-out steps"


def test_head_on_first_ring_exact_with_kill(gpt2, references):
    """Ring placement (LM head next to the tied embedding on s0): exact outputs, also across an s1 kill."""
    with cluster(head_on_first=True) as sup:
        eng, rs = run(_run(sup, EngineConfig(recovery="full"), tok=gpt2[0],
                           during=lambda e, s: _kill_when(e, s, [1])))
    check_exact(rs, PROMPTS, references)
    assert eng.c.recoveries >= 1
