"""Memory footprint: resident memory of each pipeline worker vs. the single-process baseline, plus the
parameter bytes each stage owns. This is the capacity argument for pipeline parallelism: a model that does not fit
one node's memory fits two.

Usage: python -m bench.memory --models gpt2 Qwen/Qwen2.5-0.5B --out results/memory.csv
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

import pandas as pd
import psutil

BASELINE_SNIPPET = r"""
import os, sys, psutil, torch
sys.path.insert(0, os.getcwd())
from pllm.model import build_stage, SeqSlice
st, _ = build_stage(sys.argv[1], -1, None)
st.forward(torch.arange(64), [SeqSlice("w", 0, 64)])
print(psutil.Process().memory_info().rss)
"""


def stage_param_bytes(model_name, split, head_on_first=False):
    from pllm.model import Stage, load_hf
    _, hf = load_hf(model_name)
    n = hf.config.num_hidden_layers

    def nbytes(stage):
        seen, tot = set(), 0
        mods = list(stage.blocks)
        tensors = [p for m in mods for p in m.parameters()]
        for t in (getattr(stage, "wte", None), getattr(stage, "wpe", None), stage.lm_head):
            if t is not None:
                tensors.append(t)
        if stage.ln_f is not None:
            tensors += list(stage.ln_f.parameters())
        for t in tensors:
            if id(t) not in seen:
                seen.add(id(t))
                tot += t.numel() * t.element_size()
        return tot
    return (nbytes(Stage(hf, 0, split, head_on_first=head_on_first)),
            nbytes(Stage(hf, split, n, head_on_first=head_on_first)), nbytes(Stage(hf, 0, n)))


async def worker_rss(model_name, split, head_on_first=False):
    from pllm.coordinator import Engine, EngineConfig, GrpcTransport
    from pllm.supervisor import Supervisor
    sup = Supervisor(model_name, split, ports=(50351, 50352), threads=2, head_on_first=head_on_first)
    await sup.start()
    tr = GrpcTransport(sup.addr(0), sup.addr(1))
    eng = Engine(tr, EngineConfig())
    await eng.start()
    await eng.generate(list(range(100, 164)), 4)  # touch caches / allocator
    rss = [psutil.Process(sup.procs[s].pid).memory_info().rss for s in (0, 1)]
    await eng.stop()
    await tr.close()
    sup.stop()
    return rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--out", default="results/memory.csv")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = []
    loop = asyncio.new_event_loop()  # one loop for every cluster (grpc.aio is loop-bound)
    for m in a.models:
        from transformers import AutoConfig
        n = AutoConfig.from_pretrained(m).num_hidden_layers
        split = n // 2
        base = int(subprocess.check_output([sys.executable, "-c", BASELINE_SNIPPET, m], cwd=os.getcwd(),
                                           stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1])
        for hof in (False, True):
            p0, p1, pall = stage_param_bytes(m, split, hof)
            r0, r1 = loop.run_until_complete(worker_rss(m, split, hof))
            rows.append({"model": m, "split": split, "head_placement": "first (ring)" if hof else "last",
                         "params_MB_s0": p0 / 1e6, "params_MB_s1": p1 / 1e6, "params_MB_full": pall / 1e6,
                         "rss_MB_s0": r0 / 1e6, "rss_MB_s1": r1 / 1e6, "rss_MB_baseline": base / 1e6,
                         "max_worker_over_baseline": max(r0, r1) / base,
                         "sum_workers_over_baseline": (r0 + r1) / base})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
