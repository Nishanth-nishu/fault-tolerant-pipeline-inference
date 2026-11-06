"""Command-line entry points.

  python -m pllm.cli demo   --model gpt2 [--kill s1 --kill-after 20] [--recovery targeted] [--hot-spare]
      Launch both workers locally, generate for a few prompts concurrently, optionally SIGKILL a worker mid-run,
      and verify every output against the single-process Hugging Face greedy reference.

  python -m pllm.cli generate --s0 HOST:PORT --s1 HOST:PORT --prompt "..." [--max-new 32]
      Use already-running workers (e.g. on two RCE nodes, started with `python -m pllm.worker ...`).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

DEMO_PROMPTS = ["The capital of France is", "In 1492, Columbus", "def fibonacci(n):",
                "Machine learning is a field of", "The quick brown fox", "Once upon a time,"]


async def demo(a):
    from .coordinator import Engine, EngineConfig, GrpcTransport, St
    from .model import load_hf
    from .supervisor import Supervisor
    tok, hf = load_hf(a.model)
    print(f"[demo] computing single-process reference for {len(DEMO_PROMPTS)} prompts ...")
    refs = []
    for p in DEMO_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        refs.append(hf.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist())
    sup = Supervisor(a.model, a.split, ports=(a.port, a.port + 1), threads=a.threads, log_dir=a.log_dir,
                     s0_args=["--log-activations"], hot_spare=a.hot_spare)
    t = time.time()
    await sup.start()
    if a.hot_spare:
        await sup.wait_spares()
    print(f"[demo] workers up in {time.time() - t:.1f}s: s0={sup.addr(0)} s1={sup.addr(1)}")
    tr = GrpcTransport(sup.addr(0), sup.addr(1))
    eng = Engine(tr, EngineConfig(act_codec=a.codec, recovery=a.recovery), supervisor=sup)
    await eng.start()
    try:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(eng.generate(tok(p).input_ids, a.max_new)) for p in DEMO_PROMPTS]
        if a.kill:
            while eng.c.tokens_committed < a.kill_after:
                await asyncio.sleep(0.002)
            for s in a.kill.split("+"):
                print(f"[demo] !!! SIGKILL {s} after {eng.c.tokens_committed} committed tokens")
                sup.kill(int(s[1]))
        rs = await asyncio.gather(*tasks)
        dt = time.perf_counter() - t0
        ok = 0
        for p, r, ref in zip(DEMO_PROMPTS, rs, refs):
            match = r.state == St.DONE and r.generated == ref
            ok += match
            rec = f" recovered via {[x['path'] for x in r.recoveries]}" if r.recoveries else ""
            print(f"  [{'MATCH' if match else r.state.name}] {p!r} -> {tok.decode(r.generated)[:60]!r}{rec}")
        c = eng.c
        print(f"[demo] {ok}/{len(rs)} outputs identical to single-process reference | {c.tokens_committed} tokens in "
              f"{dt:.2f}s | avg micro-batch {c.batch_seqs / max(1, c.batches):.2f} seqs | "
              f"activation bytes {c.act_bytes:,} | recoveries {c.recoveries} | stale replies rejected {c.stale_rejected}")
        if sup.restart_ms:
            print(f"[demo] restarts: {sup.restart_ms}")
        return 0 if ok == len(rs) else 1
    finally:
        await eng.stop()
        await tr.close()
        sup.stop()


async def generate(a):
    from .coordinator import Engine, EngineConfig, GrpcTransport
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr = GrpcTransport(a.s0, a.s1)
    eng = Engine(tr, EngineConfig(act_codec=a.codec, recovery=a.recovery))
    await eng.start()
    try:
        r = await eng.generate(tok(a.prompt).input_ids, a.max_new, eos_id=tok.eos_token_id)
        print(tok.decode(r.generated) if r.error is None else f"FAILED: {r.error}")
        print(f"ttft={(r.t_first - r.t_arrival) * 1e3:.1f}ms tokens={len(r.generated)} "
              f"act_bytes={eng.c.act_bytes} epochs={eng.epochs}", file=sys.stderr)
    finally:
        await eng.stop()
        await tr.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pllm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--model", default="gpt2")
    d.add_argument("--split", type=int, default=None)
    d.add_argument("--port", type=int, default=50051)
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--max-new", type=int, default=32)
    d.add_argument("--codec", default="fp32")
    d.add_argument("--recovery", default="full", choices=["full", "targeted"])
    d.add_argument("--hot-spare", action="store_true")
    d.add_argument("--kill", default="", help="s0 | s1 | s0+s1")
    d.add_argument("--kill-after", type=int, default=40)
    d.add_argument("--log-dir", default=None)
    g = sub.add_parser("generate")
    g.add_argument("--s0", required=True)
    g.add_argument("--s1", required=True)
    g.add_argument("--model", default="gpt2")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new", type=int, default=32)
    g.add_argument("--codec", default="fp32")
    g.add_argument("--recovery", default="full", choices=["full", "targeted"])
    a = ap.parse_args(argv)
    return asyncio.run(demo(a) if a.cmd == "demo" else generate(a))


if __name__ == "__main__":
    sys.exit(main())
