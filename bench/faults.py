"""Failure-injection benchmark.

kill     : SIGKILL a stage worker at a random point of a concurrent workload.
           recovery in {full, targeted} x hot_spare in {off, on} x killed in {s0, s1, both}, N trials each.
           Reports output agreement with the HF greedy reference, duplicates / out-of-order emissions, failure
           detection time, worker restart time, per-request replay time, time-to-next-token after the kill
           (user-visible stall) and replay traffic.
lossy    : drop a fraction p of stage-0 replies *after* KV was written (idempotent retry path).

Usage: python -m bench.faults --model gpt2 --trials 5 --out results/faults
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time

import numpy as np
import pandas as pd

from pllm.coordinator import Engine, EngineConfig, GrpcTransport, St
from pllm.model import load_hf
from pllm.supervisor import Supervisor

from .common import dump, env_info, wikitext_prompts


def references(hf, tok, prompts, gen):
    return [hf.generate(__import__("torch").tensor([p]), max_new_tokens=gen, do_sample=False,
                        pad_token_id=tok.eos_token_id)[0, len(p):].tolist() for p in prompts]


def integrity(rs, refs, gen):
    exact = sum(r.state == St.DONE and r.generated == ref for r, ref in zip(rs, refs))
    failed = sum(r.state == St.FAILED for r in rs)
    wrong = sum(r.state == St.DONE and r.generated != ref for r, ref in zip(rs, refs))
    dup = sum(r.state == St.DONE and len(r.token_times) != gen for r in rs)
    ooo = sum(any(b < a for a, b in zip(r.token_times, r.token_times[1:])) for r in rs)
    return exact, failed, wrong, dup, ooo


async def kill_trial(sup, cfg, prompts, refs, gen, stages, rnd):
    tr = GrpcTransport(sup.addr(0), sup.addr(1))
    eng = Engine(tr, cfg, supervisor=sup)
    await eng.start()
    total = len(prompts) * gen
    kill_at = rnd.randint(int(0.2 * total), int(0.7 * total))
    n_restart_before = len(sup.restart_ms)
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(eng.generate(p, gen)) for p in prompts]
    while eng.c.tokens_committed < kill_at:
        await asyncio.sleep(0.001)
    t_kill = time.perf_counter()
    for s in stages:
        sup.kill(s)
    rs = await asyncio.gather(*tasks)
    mk = time.perf_counter() - t0
    await eng.stop()
    await tr.close()
    exact, failed, wrong, dup, ooo = integrity(rs, refs, gen)
    det = [e["t"] - t_kill for e in eng.events if e["kind"] == "epoch_change"]
    restarts = sup.restart_ms[n_restart_before:]
    rec = [x for r in rs for x in r.recoveries]
    nxt = [min([t for t in r.token_times if t > t_kill], default=np.nan) - t_kill for r in rs
           if any(t > t_kill for t in r.token_times)]
    return {"kill_at_tokens": kill_at, "requests": len(rs), "exact": exact, "failed": failed, "wrong": wrong,
            "duplicates": dup, "out_of_order": ooo,
            "detect_ms": det[0] * 1e3 if det else np.nan,
            "restart_ms": sum(x["ms"] for x in restarts),
            "hot_spare_used": all(x["hot_spare"] for x in restarts) if restarts else False,
            "replay_ms_mean": float(np.mean([x["ms"] for x in rec])) if rec else np.nan,
            "replay_ms_max": float(np.max([x["ms"] for x in rec])) if rec else np.nan,
            "recovery_paths": ",".join(sorted({x["path"] for x in rec})),
            "ttnt_ms_median": float(np.nanmedian(nxt)) * 1e3 if nxt else np.nan,
            "ttnt_ms_max": float(np.nanmax(nxt)) * 1e3 if nxt else np.nan,
            "replay_tokens": eng.c.replay_tokens, "replay_act_bytes": eng.c.replay_act_bytes,
            "stale_rejected": eng.c.stale_rejected, "rpc_errors": eng.c.rpc_errors, "makespan_s": mk}


async def exp_kill(a, tok, hf, rows):
    rnd = random.Random(a.seed)
    prompts = wikitext_prompts(tok, a.concurrency, a.prompt_len, seed=1)
    refs = references(hf, tok, prompts, a.gen)
    for spare in (False, True):
        sup = Supervisor(a.model, None, threads=a.threads // 2, ports=(a.port, a.port + 1),
                         s0_args=["--log-activations"], hot_spare=spare)
        await sup.start()
        try:
            for mode in ("full", "targeted"):
                for stages in ([0], [1], [0, 1]):
                    for t in range(a.trials):
                        if spare:  # make sure a fully loaded spare exists for every trial
                            await sup.wait_spares()
                        cfg = EngineConfig(recovery=mode, rpc_timeout_s=a.rpc_timeout)
                        r = await kill_trial(sup, cfg, prompts, refs, a.gen, stages, rnd)
                        r.update(model=a.model, recovery=mode, hot_spare=spare,
                                 killed="+".join(f"s{s}" for s in stages), trial=t)
                        rows.append(r)
                        print(f"[kill] spare={spare!s:5} {mode:8s} kill={r['killed']:5s} t={t} exact={r['exact']}/{r['requests']} "
                              f"failed={r['failed']} dup={r['duplicates']} detect={r['detect_ms']:.0f}ms "
                              f"restart={r['restart_ms']:.0f}ms replay={r['replay_ms_mean']:.1f}ms "
                              f"ttnt_max={r['ttnt_ms_max']:.0f}ms paths={r['recovery_paths']}", flush=True)
        finally:
            sup.stop()


async def exp_lossy(a, tok, hf, rows):
    prompts = wikitext_prompts(tok, a.concurrency, a.prompt_len, seed=2)
    refs = references(hf, tok, prompts, a.gen)
    for p in a.drop_probs:
        sup = Supervisor(a.model, None, threads=a.threads // 2, ports=(a.port, a.port + 1),
                         s0_args=["--fault-drop-reply-p", str(p), "--seed", str(a.seed)])
        await sup.start()
        try:
            for t in range(a.trials):
                tr = GrpcTransport(sup.addr(0), sup.addr(1))
                eng = Engine(tr, EngineConfig(max_retries=20), supervisor=sup)
                await eng.start()
                t0 = time.perf_counter()
                rs = await asyncio.gather(*[eng.generate(pp, a.gen) for pp in prompts])
                mk = time.perf_counter() - t0
                await eng.stop()
                await tr.close()
                exact, failed, wrong, dup, ooo = integrity(rs, refs, a.gen)
                rows.append({"model": a.model, "drop_p": p, "trial": t, "requests": len(rs), "exact": exact,
                             "failed": failed, "wrong": wrong, "duplicates": dup, "out_of_order": ooo,
                             "rpc_errors": eng.c.rpc_errors, "retries": eng.c.retries,
                             "tok_per_s": sum(len(r.generated) for r in rs) / mk})
                print(f"[lossy] p={p} t={t} exact={exact}/{len(rs)} failed={failed} retries={eng.c.retries} "
                      f"tok/s={rows[-1]['tok_per_s']:.1f}", flush=True)
        finally:
            sup.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--exp", nargs="+", default=["kill", "lossy"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rpc-timeout", type=float, default=5.0)
    ap.add_argument("--drop-probs", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=50251)
    ap.add_argument("--out", default="results/faults")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok, hf = load_hf(a.model)
    tag = a.model.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    dump(f"{a.out}/env_{tag}.json", env_info({"model": a.model, "args": vars(a)}))
    async def all_exps():
        for e in a.exp:
            rows = []
            await {"kill": exp_kill, "lossy": exp_lossy}[e](a, tok, hf, rows)
            pd.DataFrame(rows).to_csv(f"{a.out}/{e}_{tag}.csv", index=False)
    asyncio.run(all_exps())


if __name__ == "__main__":
    main()
