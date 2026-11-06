"""Model-quality benchmarks for the partitioned pipeline and its inter-stage activation codecs.

Numerics are identical to the distributed system (same Stage code, same codec encode/decode at the cut);
running in-process removes timing noise so thousands of evaluation windows are affordable.

  * WikiText-2 (raw, test) perplexity, non-overlapping windows       -> ppl_*.csv
  * LAMBADA (OpenAI) last-word accuracy, teacher-forced greedy         -> lambada_*.csv
  * Greedy generation agreement vs. the unpartitioned HF model         -> greedy_*.csv
  * Split-point x codec sweep (where to cut?): boundary outlier ratio, activation error, PPL delta -> sweep_*.csv

Usage: python -m bench.accuracy --models gpt2 Qwen/Qwen2.5-0.5B --out results/accuracy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid

import pandas as pd
import torch
import torch.nn.functional as F

from pllm import codec as C
from pllm.model import SeqSlice, Stage, load_hf

torch.set_grad_enabled(False)


def wikitext_ids(tok, n_tokens):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids[:n_tokens]


class Pipe:
    """Two stages + codec at the cut (or the unpartitioned model when split is None)."""

    def __init__(self, hf, split, codec="fp32"):
        n = hf.config.num_hidden_layers
        self.hf = hf
        self.codec = codec
        self.stages = [Stage(hf, 0, n)] if split is None else [Stage(hf, 0, split), Stage(hf, split, n)]

    def logits(self, ids, all_logits=True, capture=None):
        rid = uuid.uuid4().hex
        sl = [SeqSlice(rid, 0, len(ids))]
        x = ids
        for i, s in enumerate(self.stages):
            x = s.forward(x, sl, all_logits=all_logits)
            if i == 0 and len(self.stages) == 2:
                if capture is not None:
                    capture.append(x.clone())
                x = C.roundtrip(x, self.codec)
        for s in self.stages:
            s.drop(rid)
        return x

    def generate(self, ids, n_new):
        rid = uuid.uuid4().hex
        out, x, pos = [], ids, 0
        for _ in range(n_new):
            sl = [SeqSlice(rid, pos, len(x))]
            h = x
            for i, s in enumerate(self.stages):
                h = s.forward(h, sl)
                if i == 0 and len(self.stages) == 2:
                    h = C.roundtrip(h, self.codec)
            pos += len(x)
            t = int(h[-1].argmax())
            out.append(t)
            x = torch.tensor([t])
        for s in self.stages:
            s.drop(rid)
        return out


def ppl(fn, ids, window):
    nll, cnt = 0.0, 0
    for i in range(0, len(ids) - 1, window):
        chunk = ids[i:i + window + 1]
        if len(chunk) < 2:
            break
        lg = fn(chunk[:-1])
        nll += F.cross_entropy(lg.float(), chunk[1:], reduction="sum").item()
        cnt += len(chunk) - 1
    return math.exp(nll / cnt), cnt


def lambada(fn, tok, n):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test").select(range(n))
    hits = 0
    for ex in ds:
        ctx, last = ex["text"].rsplit(" ", 1)
        c = tok(ctx, return_tensors="pt").input_ids[0]
        t = tok(" " + last, return_tensors="pt").input_ids[0]
        full = torch.cat([c, t])
        lg = fn(full[:-1])
        pred = lg[len(c) - 1:].argmax(-1)
        hits += int(torch.equal(pred, t))
    return hits / n


def gen_prompts(tok, n, plen):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for line in ds["text"]:
        line = line.strip()
        if len(line) < 200 or line.startswith("="):
            continue
        ids = tok(line, return_tensors="pt").input_ids[0]
        if len(ids) >= plen:
            out.append(ids[:plen])
        if len(out) == n:
            break
    return out


def hf_logits(hf):
    return lambda ids: hf(ids[None]).logits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--codecs", nargs="+", default=list(C.CODECS))
    ap.add_argument("--wikitext-tokens", type=int, default=65536)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--lambada-n", type=int, default=1000)
    ap.add_argument("--gen-n", type=int, default=64)
    ap.add_argument("--gen-plen", type=int, default=32)
    ap.add_argument("--gen-new", type=int, default=32)
    ap.add_argument("--sweep-tokens", type=int, default=16384)
    ap.add_argument("--sweep-codecs", nargs="+", default=["bf16", "int8", "int8h", "int4h"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="results/accuracy")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for name in a.models:
        tag = name.split("/")[-1]
        tok, hf = load_hf(name)
        n = hf.config.num_hidden_layers
        split = n // 2
        ids = wikitext_ids(tok, a.wikitext_tokens)
        rows = []
        t0 = time.time()
        if "ppl" not in a.skip:
            p_hf, cnt = ppl(hf_logits(hf), ids, a.window)
            rows.append({"model": name, "config": "hf-reference", "codec": "-", "split": "-", "ppl": p_hf, "tokens": cnt})
            for codec in a.codecs:
                try:
                    p, _ = ppl(Pipe(hf, split, codec).logits, ids, a.window)
                except OverflowError as e:
                    p = float("nan")
                    print(f"  {codec}: {e}")
                rows.append({"model": name, "config": "split", "codec": codec, "split": split, "ppl": p, "tokens": cnt,
                             "rel_delta_pct": 100 * (p - p_hf) / p_hf})
                print(f"[{tag}] wikitext2 ppl codec={codec}: {p:.4f} (hf {p_hf:.4f})  {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f"{a.out}/ppl_{tag}.csv", index=False)
        if "lambada" not in a.skip:
            lrows = [{"model": name, "config": "hf-reference", "codec": "-", "acc": lambada(hf_logits(hf), tok, a.lambada_n)}]
            for codec in a.codecs:
                try:
                    acc = lambada(Pipe(hf, split, codec).logits, tok, a.lambada_n)
                except OverflowError:
                    acc = float("nan")
                lrows.append({"model": name, "config": "split", "codec": codec, "acc": acc})
                print(f"[{tag}] lambada acc codec={codec}: {acc:.4f} (hf {lrows[0]['acc']:.4f})", flush=True)
            for r in lrows:
                r["n"] = a.lambada_n
            pd.DataFrame(lrows).to_csv(f"{a.out}/lambada_{tag}.csv", index=False)
        if "greedy" not in a.skip:
            prompts = gen_prompts(tok, a.gen_n, a.gen_plen)
            refs = [hf.generate(p[None], max_new_tokens=a.gen_new, do_sample=False,
                                pad_token_id=tok.eos_token_id)[0, len(p):].tolist() for p in prompts]
            grows = []
            for codec in a.codecs:
                pipe = Pipe(hf, split, codec)
                exact, prefix, tokagree = 0, [], []
                for p, ref in zip(prompts, refs):
                    try:
                        g = pipe.generate(p, a.gen_new)
                    except OverflowError:
                        g = [-1] * a.gen_new
                    exact += int(g == ref)
                    k = next((i for i, (x, y) in enumerate(zip(g, ref)) if x != y), len(ref))
                    prefix.append(k)
                    tokagree.append(sum(x == y for x, y in zip(g, ref)) / len(ref))
                grows.append({"model": name, "codec": codec, "n_prompts": len(prompts), "new_tokens": a.gen_new,
                              "exact_match_rate": exact / len(prompts),
                              "mean_prefix_agreement": sum(prefix) / len(prefix) / a.gen_new,
                              "token_agreement": sum(tokagree) / len(tokagree)})
                print(f"[{tag}] greedy codec={codec}: exact={exact}/{len(prompts)} prefix={grows[-1]['mean_prefix_agreement']:.3f}", flush=True)
            pd.DataFrame(grows).to_csv(f"{a.out}/greedy_{tag}.csv", index=False)
        if "sweep" not in a.skip:
            sids = ids[: a.sweep_tokens]
            p_hf, _ = ppl(hf_logits(hf), sids, a.window)
            srows = []
            splits = list(range(1, n)) if n <= 12 else sorted(set([1, 2, 3] + list(range(4, n, 4)) + [n - 2, n - 1]))
            for k in splits:
                cap = []
                Pipe(hf, k, "fp32").logits(sids[: a.window], capture=cap)
                x = cap[0]
                rms = x.pow(2).mean().sqrt().item()
                outlier = (x.abs().max() / x.pow(2).mean().sqrt()).item()
                for codec in a.sweep_codecs:
                    rel = ((C.roundtrip(x, codec) - x).norm() / x.norm()).item()
                    p, _ = ppl(Pipe(hf, k, codec).logits, sids, a.window)
                    srows.append({"model": name, "split": k, "codec": codec, "boundary_rms": rms,
                                  "outlier_ratio": outlier, "act_rel_err": rel, "ppl": p, "ppl_hf": p_hf,
                                  "rel_delta_pct": 100 * (p - p_hf) / p_hf,
                                  "bytes_per_token": C.wire_bytes(1, x.shape[1], codec)})
                print(f"[{tag}] sweep split={k} outlier={outlier:.1f} " +
                      " ".join(f"{r['codec']}:{r['rel_delta_pct']:+.2f}%" for r in srows[-len(a.sweep_codecs):]), flush=True)
            pd.DataFrame(srows).to_csv(f"{a.out}/sweep_{tag}.csv", index=False)
        summary[name] = {"seconds": time.time() - t0}
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
