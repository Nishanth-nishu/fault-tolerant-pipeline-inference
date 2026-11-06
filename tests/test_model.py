"""Partition adapter correctness: split execution must reproduce the unpartitioned HF model."""
import pytest
import torch

from pllm import codec as C
from pllm.model import CacheMiss, SeqSlice, Stage, load_hf

TOL = 5e-4  # fp32 re-association tolerance on logits


@pytest.mark.parametrize("name", ["gpt2", "Qwen/Qwen2.5-0.5B"])
def test_split_matches_hf_all_splits(name):
    tok, hf = load_hf(name)
    ids = tok("Distributed systems make inference across machines possible because", return_tensors="pt").input_ids[0]
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    n = hf.config.num_hidden_layers
    splits = range(1, n) if n <= 12 else [1, n // 4, n // 2, n - 1]
    for k in splits:
        s0, s1 = Stage(hf, 0, k), Stage(hf, k, n)
        sl = [SeqSlice("a", 0, len(ids))]
        out = s1.forward(s0.forward(ids, sl), sl, all_logits=True)
        assert (out - ref).abs().max().item() < TOL, f"split {k}"


def test_chunked_prefill_decode_and_mixed_batch(gpt2):
    tok, hf = gpt2
    ids = tok("A B C D E F G H I J K L M N O P", return_tensors="pt").input_ids[0]
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    s0, s1 = Stage(hf, 0, 6), Stage(hf, 6, 12)
    L = len(ids)
    # chunk 1: 7 tokens for seq a, mixed in a micro-batch with a full prefill of seq b
    x = torch.cat([ids[:7], ids])
    seqs = [SeqSlice("a", 0, 7), SeqSlice("b", 0, L)]
    out = s1.forward(s0.forward(x, seqs), seqs)
    assert (out[0] - ref[6]).abs().max() < TOL and (out[1] - ref[-1]).abs().max() < TOL
    # remaining tokens of a one at a time, while b decodes a different token alongside
    for p in range(7, L):
        seqs = [SeqSlice("a", p, 1), SeqSlice("b", L, 1)]
        x = torch.tensor([ids[p].item(), 13])
        out = s1.forward(s0.forward(x, seqs), seqs)
        assert (out[0] - ref[p]).abs().max() < TOL
        s0.caches["b"].length = L  # roll b back so each iteration is independent
        s1.caches["b"].length = L


def test_retry_is_idempotent(gpt2):
    tok, hf = gpt2
    ids = tok("one two three four five", return_tensors="pt").input_ids[0]
    s = Stage(hf, 0, 12)
    s.forward(ids[:-1], [SeqSlice("a", 0, len(ids) - 1)])
    a = s.forward(ids[-1:], [SeqSlice("a", len(ids) - 1, 1)])
    for _ in range(3):  # same op re-executed (lost reply) must give the same result and cache length
        b = s.forward(ids[-1:], [SeqSlice("a", len(ids) - 1, 1)])
        assert torch.equal(a, b)
        assert s.cache_len("a") == len(ids)


def test_cache_miss_detected(gpt2):
    _, hf = gpt2
    s = Stage(hf, 6, 12)
    with pytest.raises(CacheMiss):
        s.forward(torch.zeros(1, 768), [SeqSlice("ghost", 5, 1)])
    s.forward(torch.zeros(2, 768), [SeqSlice("a", 0, 2)])
    with pytest.raises(CacheMiss):  # gap: position 4 while cache holds 2
        s.forward(torch.zeros(1, 768), [SeqSlice("a", 4, 1)])


@pytest.mark.parametrize("codec,rel_tol", [("fp32", 0), ("fp16", 1e-3), ("bf16", 1e-2), ("int8", 0.5),
                                           ("int8h", 0.02), ("int4h", 0.2)])
def test_codec_roundtrip(codec, rel_tol):
    torch.manual_seed(0)
    x = torch.randn(13, 768)
    x[:, 3] *= 200  # residual-stream style outlier channel
    y = C.roundtrip(x, codec)
    rel = ((y - x).norm() / x.norm()).item()
    assert rel <= rel_tol + 1e-7, rel
    data, scales, shape = C.encode(x, codec)
    assert len(data) + len(scales) == C.wire_bytes(13, 768, codec)


def test_hadamard_rotation_tames_outliers():
    """Residual-stream outlier channels wreck per-token absmax int8; block-Hadamard rotation fixes it."""
    torch.manual_seed(0)
    x = torch.randn(64, 768)
    x[:, [3, 400]] *= 300
    err = {c: ((C.roundtrip(x, c) - x).norm() / x.norm()).item() for c in ("int8", "int8h")}
    assert err["int8h"] < err["int8"] / 3, err


def test_fp16_overflow_guard():
    with pytest.raises(OverflowError):
        C.encode(torch.full((1, 64), 1e6), "fp16")


@pytest.mark.parametrize("name", ["gpt2", "Qwen/Qwen2.5-0.5B"])
def test_head_on_first_matches_hf(name):
    """Tied-head co-location: s1 returns hidden rows, s0 applies the LM head; logits must equal HF."""
    tok, hf = load_hf(name)
    ids = tok("Pipeline placement of the language-model head", return_tensors="pt").input_ids[0]
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    n = hf.config.num_hidden_layers
    s0, s1 = Stage(hf, 0, n // 2, head_on_first=True), Stage(hf, n // 2, n, head_on_first=True)
    assert s0.has_head and not s1.has_head and s1.lm_head is None
    sl = [SeqSlice("a", 0, len(ids))]
    hidden = s1.forward(s0.forward(ids, sl), sl, all_logits=True)
    assert hidden.shape[-1] == hf.config.hidden_size
    assert (s0.head(hidden) - ref).abs().max().item() < TOL
