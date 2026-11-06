"""Performance benchmarks: single-process baseline vs. two-stage gRPC pipeline.

Experiments
  concurrency : closed-loop clients C in {1,2,4,8} x prompt length {16,64} x systems, gen=32
  network     : emulated s0<->s1 link (latency, bandwidth) x activation codec x depth policy, C=4
  load        : open-loop Poisson arrivals, bounded admission queue (latency / goodput / rejections vs. rate)

Methodology
  * Equal hardware: the baseline process is pinned to 6 physical P-cores (6 threads); the pipeline splits the
    same 6 cores 3+3 between s0 and s1; the coordinator runs on E-cores in both cases.
  * Paired, interleaved rounds: all systems stay live and, for every (config, repeat), run in a freshly
    shuffled order, so background load on a shared host affects every system alike. Host CPU% is logged.
Usage: python -m bench.perf --model gpt2 --exp concurrency network load --out results/perf
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random

import pandas as pd

from pllm.coordinator import Engine, EngineConfig, GrpcTransport
from pllm.local import LocalTransport
from pllm.model import load_hf
from pllm.supervisor import Supervisor

from .common import closed_loop, dump, env_info, open_loop, summarize, wikitext_prompts

SYSTEMS = {
    # name: (distributed?, EngineConfig overrides)
    "baseline": (False, {"max_inflight_batches": 1}),
    "baseline-nobatch": (False, {"batching": False, "max_inflight_batches": 1}),
    "pipe-d1": (True, {"max_inflight_batches": 1}),
    "pipe-d2": (True, {"max_inflight_batches": 2}),
    "pipe-d2-nobatch": (True, {"max_inflight_batches": 2, "batching": False}),
    "pipe-auto": (True, {"max_inflight_batches": 2, "depth_policy": "auto", "auto_window_s": 0.5}),
}

NET_SYSTEMS = {f"pipe-d2-{c}": {"act_codec": c, "max_inflight_batches": 2} for c in ("fp32", "bf16", "int8h", "int4h")}
NET_SYSTEMS.update({"pipe-d1-fp32": {"act_codec": "fp32", "max_inflight_batches": 1},
                    "pipe-auto-fp32": {"act_codec": "fp32", "max_inflight_batches": 2, "depth_policy": "auto",
                                       "auto_window_s": 0.5}})


def pin(cpus):
    if cpus:
        import psutil
        psutil.Process().cpu_affinity([int(c) for c in cpus.split(",")])


async def run_closed(eng, prompts, C, per_client, gen, warm=True):
    if warm:
        await closed_loop(eng, prompts, min(C, 2), 1, 4)
    eng.c.__init__()
    eng.batch_log.clear()
    reqs, mk = await closed_loop(eng, prompts, C, per_client, gen)
    return reqs, summarize(reqs, eng, mk)


class Systems:
    """One live engine per system so that rounds can be interleaved (paired design)."""

    def __init__(self, a, sup, configs: dict[str, tuple[bool, dict]]):
        self.a, self.sup, self.configs = a, sup, configs
        self.eng, self.tr = {}, {}
        self.local = None

    async def start(self):
        for n, (dist, ov) in self.configs.items():
            if dist:
                tr = GrpcTransport(self.sup.addr(0), self.sup.addr(1))
                eng = Engine(tr, EngineConfig(**ov), supervisor=self.sup)
            else:
                if self.local is None:
                    pin(self.a.base_cpus)
                    self.local = LocalTransport(self.a.model, threads=self.a.base_threads)
                tr = self.local
                eng = Engine(tr, EngineConfig(**ov), stage_names=("s0",))
            await eng.start()
            self.eng[n], self.tr[n] = eng, tr

    async def round(self, n, prompts, C, per_client, gen):
        import psutil
        pin(self.a.coord_cpus if self.configs[n][0] else self.a.base_cpus)
        eng = self.eng[n]
        psutil.cpu_percent(None)
        reqs, s = await run_closed(eng, prompts, C, per_client, gen, warm=True)
        s["host_cpu_pct"] = psutil.cpu_percent(None)
        s["depth_history"] = str(eng._auto["history"][-3:]) if eng.cfg.depth_policy == "auto" else ""
        return reqs, s

    async def stop(self):
        for n, e in self.eng.items():
            await e.stop()
            if self.tr[n] is not self.local:
                await self.tr[n].close()


def make_sup(a, s0_extra=(), s1_extra=()):
    s0 = ["--cpus", a.s0_cpus] if a.s0_cpus else []
    s1 = ["--cpus", a.s1_cpus] if a.s1_cpus else []
    return Supervisor(a.model, a.split, threads=a.worker_threads, ports=(a.port, a.port + 1),
                      s0_args=s0 + list(s0_extra), s1_args=s1 + list(s1_extra))


async def exp_concurrency(a, tok, rows, raw):
    sup = make_sup(a)
    await sup.start()
    rnd = random.Random(0)
    S = Systems(a, sup, {n: SYSTEMS[n] for n in a.systems})
    await S.start()
    try:
        for plen in a.prompt_lens:
            prompts = wikitext_prompts(tok, 64, plen)
            for C in a.concurrency:
                for rep in range(a.repeats):
                    order = list(a.systems)
                    rnd.shuffle(order)
                    for n in order:
                        reqs, s = await S.round(n, prompts, C, a.per_client, a.gen)
                        s.update(exp="concurrency", system=n, concurrency=C, prompt_len=plen, gen=a.gen, rep=rep)
                        rows.append(s)
                        for r in reqs:
                            raw.append({**{k: v for k, v in r.items() if k != "_itl"}, "system": n,
                                        "concurrency": C, "prompt_len": plen, "rep": rep})
                        print(f"[conc] {n:16s} P={plen:3d} C={C} rep={rep} tok/s={s['tok_per_s']:7.1f} "
                              f"ttft50={s['ttft_p50_ms']:7.1f} tpot50={s['tpot_p50_ms']:6.1f} "
                              f"tpot95={s['tpot_p95_ms']:6.1f} batch={s['avg_batch_seqs']:.2f} "
                              f"cpu={s['host_cpu_pct']:.0f}%", flush=True)
    finally:
        await S.stop()
        sup.stop()


async def exp_network(a, tok, rows, raw):
    prompts = wikitext_prompts(tok, 64, 64)
    rnd = random.Random(1)
    for delay, bw in a.links:
        sup = make_sup(a, s0_extra=["--net-delay-ms", str(delay), "--net-bw-mbps", str(bw)])
        await sup.start()
        S = Systems(a, sup, {n: (True, NET_SYSTEMS[n]) for n in a.net_systems})
        await S.start()
        try:
            for rep in range(a.repeats):
                order = list(a.net_systems)
                rnd.shuffle(order)
                for n in order:
                    ov = NET_SYSTEMS[n]
                    reqs, s = await S.round(n, prompts, a.net_concurrency, a.per_client, a.gen)
                    s.update(exp="network", system=n, codec=ov["act_codec"],
                             depth=ov.get("depth_policy", ov["max_inflight_batches"]) if ov.get("depth_policy") == "auto"
                             else ov["max_inflight_batches"],
                             delay_ms=delay, bw_mbps=bw, concurrency=a.net_concurrency, prompt_len=64, gen=a.gen, rep=rep)
                    rows.append(s)
                    print(f"[net] delay={delay}ms bw={bw}Mbps {n:15s} rep={rep} tok/s={s['tok_per_s']:7.1f} "
                          f"ttft50={s['ttft_p50_ms']:7.1f} tpot50={s['tpot_p50_ms']:6.1f} "
                          f"act_B/tok={s['act_bytes_per_tok']:.0f} cpu={s['host_cpu_pct']:.0f}%", flush=True)
        finally:
            await S.stop()
            sup.stop()


async def exp_load(a, tok, rows, raw):
    prompts = wikitext_prompts(tok, 64, 64)
    sup = make_sup(a)
    await sup.start()
    try:
        for n in ("baseline", "pipe-auto"):
            dist, ov = SYSTEMS[n]
            for rate in a.rates:
                S = Systems(a, sup, {n: (dist, dict(ov, max_active=8, max_waiting=a.max_waiting))})
                await S.start()
                eng = S.eng[n]
                pin(a.coord_cpus if dist else a.base_cpus)
                await closed_loop(eng, prompts, 2, 1, 4)
                eng.c.__init__()
                eng.batch_log.clear()
                reqs, mk, rejected, offered = await open_loop(eng, prompts, rate, a.load_duration, a.gen)
                s = summarize(reqs, eng, mk)
                s.update(exp="load", system=n, rate=rate, offered=offered, rejected=rejected,
                         reject_rate=rejected / max(1, offered), prompt_len=64, gen=a.gen)
                rows.append(s)
                print(f"[load] {n:10s} rate={rate:5.1f} goodput={s['req_per_s']:.2f} req/s "
                      f"rej={s['reject_rate']:.2f} ttft95={s['ttft_p95_ms']:.0f} q95={s['queue_p95_ms']:.0f}", flush=True)
                await S.stop()
    finally:
        sup.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--exp", nargs="+", default=["concurrency", "network", "load"])
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--prompt-lens", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--per-client", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--split", type=int, default=None, help="first layer of stage 1 (default: n_layers//2)")
    ap.add_argument("--base-threads", type=int, default=6)
    ap.add_argument("--worker-threads", type=int, default=3)
    ap.add_argument("--base-cpus", default="0,2,4,6,8,10", help="baseline: the same physical cores as s0+s1")
    ap.add_argument("--s0-cpus", default="0,2,4")
    ap.add_argument("--s1-cpus", default="6,8,10")
    ap.add_argument("--coord-cpus", default="12,13,14,15,16,17,18,19")
    ap.add_argument("--links", nargs="+", default=["0,0", "2,0", "2,1000", "2,100", "5,20"],
                    help="delay_ms,bw_mbps pairs (bw 0 = unlimited)")
    ap.add_argument("--net-concurrency", type=int, default=4)
    ap.add_argument("--net-systems", nargs="+", default=list(NET_SYSTEMS))
    ap.add_argument("--rates", nargs="+", type=float, default=[1, 2, 4, 8, 16])
    ap.add_argument("--load-duration", type=float, default=20)
    ap.add_argument("--max-waiting", type=int, default=16)
    ap.add_argument("--port", type=int, default=50151)
    ap.add_argument("--out", default="results/perf")
    a = ap.parse_args()
    a.links = [tuple(float(x) for x in l.split(",")) for l in a.links]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_num_threads(a.base_threads)
    tok, _ = load_hf(a.model)
    tag = a.model.split("/")[-1] + (f"-split{a.split}" if a.split is not None else "")
    os.makedirs(a.out, exist_ok=True)
    dump(f"{a.out}/env_{tag}.json", env_info({"model": a.model, "args": vars(a),
                                              "placement": "pinned: baseline on base-cpus; s0/s1 on disjoint halves "
                                                           "of the same physical P-cores; coordinator on E-cores"}))

    async def all_exps():  # a single event loop for every experiment (grpc.aio is loop-bound)
        for e in a.exp:
            rows, raw = [], []
            await {"concurrency": exp_concurrency, "network": exp_network, "load": exp_load}[e](a, tok, rows, raw)
            pd.DataFrame(rows).to_csv(f"{a.out}/{e}_{tag}.csv", index=False)
            if raw:
                pd.DataFrame(raw).to_csv(f"{a.out}/{e}_{tag}_requests.csv", index=False)
    asyncio.run(all_exps())


if __name__ == "__main__":
    main()
