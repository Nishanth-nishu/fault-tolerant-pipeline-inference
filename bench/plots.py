"""Render figures + a markdown results digest from the CSVs written by bench.accuracy / bench.perf / bench.faults.

Usage: python -m bench.plots --results results --out docs/figures
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAL = ["#2a6fdb", "#e5732b", "#2fa36b", "#c63f5b", "#7d5bd1", "#8a8a8a"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "legend.frameon": False})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    return name


def agg(df, keys, cols):
    g = df.groupby(keys)[cols]
    m, s = g.mean(), g.std().fillna(0)
    return m.join(s, rsuffix="_std").reset_index()


def fig_concurrency(df, out, tag):
    figs = []
    cols = ["tok_per_s", "ttft_p50_ms", "tpot_p50_ms", "tpot_p95_ms"]
    a = agg(df, ["system", "prompt_len", "concurrency"], cols)
    for plen in sorted(a.prompt_len.unique()):
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for i, sysname in enumerate(sorted(a.system.unique())):
            d = a[(a.system == sysname) & (a.prompt_len == plen)].sort_values("concurrency")
            kw = dict(marker="o", color=PAL[i % len(PAL)], label=sysname, ms=3, capsize=2)
            ax[0].errorbar(d.concurrency, d.tok_per_s, d.tok_per_s_std, **kw)
            ax[1].errorbar(d.concurrency, d.ttft_p50_ms, d.ttft_p50_ms_std, **kw)
            ax[2].errorbar(d.concurrency, d.tpot_p50_ms, d.tpot_p50_ms_std, **kw)
        for x, t in zip(ax, ["generated tokens / s", "TTFT p50 (ms)", "inter-token latency p50 (ms)"]):
            x.set_xlabel("concurrent clients")
            x.set_title(t)
            x.set_xscale("log", base=2)
        ax[0].legend(fontsize=7)
        fig.suptitle(f"{tag}: baseline vs. 2-stage pipeline, prompt={plen} tokens, gen=32")
        figs.append(_save(fig, out, f"concurrency_{tag}_p{plen}.png"))
    return figs


def fig_network(df, out, tag):
    a = agg(df, ["system", "delay_ms", "bw_mbps"], ["tok_per_s", "tpot_p50_ms", "ttft_p50_ms", "act_bytes_per_tok"])
    a["link"] = a.apply(lambda r: f"{r.delay_ms:g}ms/" + ("inf" if r.bw_mbps == 0 else f"{r.bw_mbps:g}Mb"), axis=1)
    order = df.drop_duplicates(["delay_ms", "bw_mbps"])[["delay_ms", "bw_mbps"]].values.tolist()
    links = [f"{d:g}ms/" + ("inf" if b == 0 else f"{b:g}Mb") for d, b in order]
    systems = [s for s in ["pipe-d2-fp32", "pipe-d2-bf16", "pipe-d2-int8h", "pipe-d2-int4h", "pipe-d1-fp32",
                           "pipe-auto-fp32"] if s in set(a.system)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    w = 0.8 / len(systems)
    for i, n in enumerate(systems):
        d = a[a.system == n].set_index("link").reindex(links)
        x = np.arange(len(links)) + i * w
        ax[0].bar(x, d.tok_per_s, w, yerr=d.tok_per_s_std, color=PAL[i], label=n)
        ax[1].bar(x, d.ttft_p50_ms, w, yerr=d.ttft_p50_ms_std, color=PAL[i])
        ax[2].bar(x, d.tpot_p50_ms, w, yerr=d.tpot_p50_ms_std, color=PAL[i])
    for x, t in zip(ax, ["throughput (tokens/s)", "TTFT p50 (ms)", "inter-token latency p50 (ms)"]):
        x.set_xticks(np.arange(len(links)) + 0.4 - w / 2)
        x.set_xticklabels(links, rotation=20)
        x.set_title(t)
        x.set_xlabel("emulated s0<->s1 link (one-way delay / bandwidth)")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=len(systems), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle(f"{tag}: activation codec and pipeline depth vs. link quality (C=4, prompt=64, gen=32)")
    return [_save(fig, out, f"network_{tag}.png")]


def paired_ratio(df, num, den, metric="tok_per_s", keys=("prompt_len", "concurrency")):
    """Per-repeat ratio num/den (interleaved rounds share background load), mean and 95% CI."""
    from scipy import stats
    out = []
    for k, g in df.groupby(list(keys)):
        a = g[g.system == num].set_index("rep")[metric]
        b = g[g.system == den].set_index("rep")[metric]
        r = (a / b).dropna()
        if len(r) == 0:
            continue
        ci = stats.t.ppf(0.975, len(r) - 1) * r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0
        out.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "ratio": r.mean(), "ci95": ci, "n": len(r)})
    return pd.DataFrame(out)


def fig_ratio(df, out, tag):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, plen in enumerate(sorted(df.prompt_len.unique())):
        d = df[df.prompt_len == plen]
        for i, n in enumerate([s for s in ["pipe-d1", "pipe-d2", "pipe-auto", "pipe-d2-nobatch"] if s in set(d.system)]):
            r = paired_ratio(d, n, "baseline")
            ax[j].errorbar(r.concurrency, r.ratio, r.ci95, marker="o", ms=3, capsize=2, color=PAL[i], label=n)
        ax[j].axhline(1, color="#999", lw=1, ls=":")
        ax[j].set_xscale("log", base=2)
        ax[j].set_xlabel("concurrent clients")
        ax[j].set_title(f"throughput / baseline (paired, 95% CI), prompt={plen}")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"{tag}: pipeline throughput relative to single-process baseline on the same cores")
    return [_save(fig, out, f"ratio_{tag}.png")]


def fig_load(df, out, tag):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, s in enumerate(sorted(df.system.unique())):
        d = df[df.system == s].sort_values("rate")
        ax[0].plot(d.rate, d.req_per_s, "o-", color=PAL[i], label=s, ms=3)
        ax[1].plot(d.rate, d.ttft_p95_ms, "o-", color=PAL[i], ms=3)
        ax[2].plot(d.rate, d.reject_rate, "o-", color=PAL[i], ms=3)
    ax[0].plot(df.rate, df.rate, ":", color="#999", lw=1, label="offered")
    for x, t in zip(ax, ["goodput (req/s)", "TTFT p95 (ms)", "admission reject rate"]):
        x.set_xlabel("offered load (req/s, Poisson)")
        x.set_title(t)
    ax[1].set_yscale("log")
    ax[0].legend()
    fig.suptitle(f"{tag}: open-loop load with bounded admission queue")
    return [_save(fig, out, f"load_{tag}.png")]


def fig_sweep(df, out, tag):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, c in enumerate(sorted(df.codec.unique())):
        d = df[df.codec == c].sort_values("split")
        ax[1].plot(d.split, d.act_rel_err, "o-", color=PAL[i], label=c, ms=3)
        ax[2].plot(d.split, d.rel_delta_pct.abs() + 1e-4, "o-", color=PAL[i], label=c, ms=3)
    d = df.drop_duplicates("split").sort_values("split")
    ax[0].plot(d.split, d.outlier_ratio, "o-", color="#444", ms=3)
    ax[0].set_title("boundary activation max|x| / rms")
    ax[1].set_title("relative activation error after codec")
    ax[2].set_title("|WikiText-2 PPL change| (%)")
    for x in ax:
        x.set_xlabel("split layer (first layer on stage 1)")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[1].legend()
    fig.suptitle(f"{tag}: where to cut? outliers and codec error vs. split point")
    return [_save(fig, out, f"sweep_{tag}.png")]


def fig_faults(df, out, tag):
    d = df.copy()
    d["cfg"] = d.recovery + d.hot_spare.map({True: "+spare", False: ""})
    a = agg(d, ["cfg", "killed"], ["ttnt_ms_max", "restart_ms", "replay_ms_mean", "detect_ms"])
    cfgs = list(dict.fromkeys(["full", "targeted", "full+spare", "targeted+spare"]))
    cfgs = [c for c in cfgs if c in set(a.cfg)]
    killed = ["s0", "s1", "s0+s1"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        s = a[a.cfg == c].set_index("killed").reindex(killed)
        x = np.arange(len(killed)) + i * w
        ax[0].bar(x, s.ttnt_ms_max, w, yerr=s.ttnt_ms_max_std, color=PAL[i], label=c)
        ax[1].bar(x, s.replay_ms_mean, w, yerr=s.replay_ms_mean_std, color=PAL[i], label=c)
    for x, t in zip(ax, ["user-visible stall: kill -> next token, max over requests (ms)",
                         "KV rebuild (replay) per request (ms)"]):
        x.set_xticks(np.arange(len(killed)) + 0.4 - w / 2)
        x.set_xticklabels([f"kill {k}" for k in killed])
        x.set_title(t, fontsize=8.5)
        x.set_yscale("log")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=len(cfgs), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22)
    fig.suptitle(f"{tag}: recovery cost by strategy (C=4, prompt=64, gen=32)")
    return [_save(fig, out, f"faults_{tag}.png")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    made = []
    for f in glob.glob(f"{a.results}/perf/concurrency_*.csv"):
        if f.endswith("_requests.csv"):
            continue
        df = pd.read_csv(f)
        made += fig_concurrency(df, a.out, os.path.basename(f)[12:-4])
        if "baseline" in set(df.system):
            made += fig_ratio(df, a.out, os.path.basename(f)[12:-4])
    for f in glob.glob(f"{a.results}/perf/network_*.csv"):
        made += fig_network(pd.read_csv(f), a.out, os.path.basename(f)[8:-4])
    for f in glob.glob(f"{a.results}/perf/load_*.csv"):
        made += fig_load(pd.read_csv(f), a.out, os.path.basename(f)[5:-4])
    for f in glob.glob(f"{a.results}/accuracy/sweep_*.csv"):
        made += fig_sweep(pd.read_csv(f), a.out, os.path.basename(f)[6:-4])
    for f in glob.glob(f"{a.results}/faults/kill_*.csv"):
        made += fig_faults(pd.read_csv(f), a.out, os.path.basename(f)[5:-4])
    print("\n".join(made))


if __name__ == "__main__":
    main()
