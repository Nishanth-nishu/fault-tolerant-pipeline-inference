"""Cost-balanced partitioning: choose the split layer that minimises the slower stage's per-step time.

The LM head (H x V GEMM, V=50k for GPT-2) costs as much as several Transformer blocks during decode, so the
naive n/2 split leaves stage 1 as the bottleneck. We time each block and the head in isolation for a decode
micro-batch and pick k = argmin_k max(T_embed + sum_{i<k} T_i, sum_{i>=k} T_i + T_head)."""
from __future__ import annotations

import time

import torch

from .model import SeqSlice, Stage, load_hf


@torch.inference_mode()
def layer_costs(model_name: str, batch: int = 2, ctx: int = 64, threads: int = 4, iters: int = 30):
    torch.set_num_threads(threads)
    tok, hf = load_hf(model_name)
    n = hf.config.num_hidden_layers
    costs = []
    seqs_pre = [SeqSlice(f"b{i}", 0, ctx) for i in range(batch)]
    seqs = [SeqSlice(f"b{i}", ctx, 1) for i in range(batch)]
    H = hf.config.hidden_size
    for li in range(n):
        st = Stage(hf, li, li + 1)
        if st.is_first:
            st.forward(torch.randint(0, 1000, (batch * ctx,)), seqs_pre)
            x = torch.randint(0, 1000, (batch,))
        else:
            st.forward(torch.randn(batch * ctx, H), seqs_pre)
            x = torch.randn(batch, H)
        for _ in range(5):
            st.forward(x, seqs)
        t = time.perf_counter()
        for _ in range(iters):
            st.forward(x, seqs)
        costs.append((time.perf_counter() - t) / iters * 1e3)  # last layer includes the LM head
    return costs


def balanced_split(costs: list[float]) -> tuple[int, float, float]:
    n = len(costs)
    best = min(range(1, n), key=lambda k: max(sum(costs[:k]), sum(costs[k:])))
    return best, sum(costs[:best]), sum(costs[best:])


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    c = layer_costs(m)
    k, a, b = balanced_split(c)
    print("per-layer ms:", [round(x, 2) for x in c])
    print(f"naive split {len(c)//2}: s0={sum(c[:len(c)//2]):.2f} s1={sum(c[len(c)//2:]):.2f} | balanced split {k}: s0={a:.2f} s1={b:.2f}")
