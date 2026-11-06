"""Shared benchmark utilities: workload construction, load generators, metric aggregation, environment capture."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import time

import numpy as np

from pllm.coordinator import Engine, EngineConfig, GrpcTransport, Overloaded, St


def env_info(extra=None) -> dict:
    import torch
    import transformers
    import grpc
    info = {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "torch": torch.__version__, "transformers": transformers.__version__,
            "grpcio": grpc.__version__, "device": "cpu", "precision": "fp32 weights / fp32 compute",
            "placement": "both workers + coordinator on one host, loopback TCP",
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    if extra:
        info.update(extra)
    return info


def wikitext_prompts(tok, n, plen, seed=0):
    """Real-text prompts of exactly `plen` tokens drawn from WikiText-2 test paragraphs."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paras = [t.strip() for t in ds["text"] if len(t.strip()) > 400 and not t.strip().startswith("=")]
    rnd = random.Random(seed)
    rnd.shuffle(paras)
    out = []
    for p in paras:
        ids = tok(p).input_ids
        if len(ids) >= plen:
            out.append(ids[:plen])
        if len(out) == n:
            break
    return out


def req_metrics(r) -> dict:
    d = {"rid": r.rid, "state": r.state.name, "prompt_len": len(r.prompt), "gen": len(r.generated),
         "queue_ms": (r.t_admit - r.t_arrival) * 1e3 if r.t_admit else float("nan"),
         "ttft_ms": (r.t_first - r.t_arrival) * 1e3 if r.t_first else float("nan"),
         "e2e_ms": (r.t_done - r.t_arrival) * 1e3 if r.t_done else float("nan"),
         "recoveries": len(r.recoveries), "error": r.error or ""}
    itl = np.diff(r.token_times) * 1e3 if len(r.token_times) > 1 else np.array([])
    d["tpot_ms"] = float(itl.mean()) if len(itl) else float("nan")
    d["max_itl_ms"] = float(itl.max()) if len(itl) else float("nan")
    d["_itl"] = itl
    return d


def summarize(reqs: list[dict], eng: Engine, makespan_s: float) -> dict:
    ok = [r for r in reqs if r["state"] == "DONE"]
    itl = np.concatenate([r["_itl"] for r in ok]) if ok else np.array([])
    toks = sum(r["gen"] for r in ok)
    ttft = np.array([r["ttft_ms"] for r in ok])
    q = np.array([r["queue_ms"] for r in ok])
    c = eng.c
    bl = eng.batch_log
    s = {"requests": len(reqs), "completed": len(ok), "failed": len(reqs) - len(ok),
         "makespan_s": makespan_s, "tokens": toks, "tok_per_s": toks / makespan_s if makespan_s else 0,
         "req_per_s": len(ok) / makespan_s if makespan_s else 0,
         "ttft_p50_ms": np.percentile(ttft, 50) if len(ttft) else np.nan,
         "ttft_p95_ms": np.percentile(ttft, 95) if len(ttft) else np.nan,
         "tpot_p50_ms": np.percentile(itl, 50) if len(itl) else np.nan,
         "tpot_p95_ms": np.percentile(itl, 95) if len(itl) else np.nan,
         "queue_p50_ms": np.percentile(q, 50) if len(q) else np.nan,
         "queue_p95_ms": np.percentile(q, 95) if len(q) else np.nan,
         "avg_batch_seqs": c.batch_seqs / max(1, c.batches),
         "act_bytes_per_tok": c.act_bytes / max(1, c.tokens_committed),
         "link_msg_bytes_per_tok": c.link_msg_bytes / max(1, c.tokens_committed),
         "ctrl_bytes_per_tok": (c.link_msg_bytes - c.act_bytes) / max(1, c.tokens_committed),
         "coord_bytes_per_tok": (c.coord_req_bytes + c.coord_reply_bytes) / max(1, c.tokens_committed),
         "act_bytes_total": c.act_bytes, "link_msg_bytes_total": c.link_msg_bytes,
         "worker_overloads": c.worker_overloads, "admission_rejects": c.overload_rejects,
         "rpc_errors": c.rpc_errors, "stale_rejected": c.stale_rejected, "recoveries": c.recoveries}
    for k in ("s0", "s1"):
        for f in ("compute_ms", "queue_ms", "codec_ms"):
            v = [b.get(f"{k}_{f}") for b in bl if b.get(f"{k}_{f}") is not None]
            s[f"{k}_{f}_mean"] = float(np.mean(v)) if v else np.nan
    if bl:
        s["batch_rtt_ms_mean"] = float(np.mean([b["rtt_ms"] for b in bl]))
    return s


async def closed_loop(eng: Engine, prompts, concurrency: int, per_client: int, max_new: int):
    """`concurrency` clients, each issuing `per_client` requests back-to-back."""
    out = []
    it = iter(range(10 ** 9))

    async def client(ci):
        for j in range(per_client):
            p = prompts[(ci * per_client + j) % len(prompts)]
            r = await eng.generate(p, max_new)
            out.append(req_metrics(r))
    t0 = time.perf_counter()
    await asyncio.gather(*[client(i) for i in range(concurrency)])
    return out, time.perf_counter() - t0


async def open_loop(eng: Engine, prompts, rate: float, duration_s: float, max_new: int, seed=0):
    """Poisson arrivals at `rate` req/s for `duration_s`; rejected submissions are counted, not retried."""
    rnd = random.Random(seed)
    futs, rejected, i = [], 0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        try:
            futs.append(eng.submit(prompts[i % len(prompts)], max_new))
        except Overloaded:
            rejected += 1
        i += 1
        await asyncio.sleep(rnd.expovariate(rate))
    rs = [await r.future for r in futs]
    return [req_metrics(r) for r in rs], time.perf_counter() - t0, rejected, i


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
