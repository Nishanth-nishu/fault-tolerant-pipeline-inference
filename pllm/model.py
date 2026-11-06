"""Layer-partitioned decoder-only Transformer stages with explicit, position-indexed KV caches.

A *stage* owns a contiguous range of Transformer blocks ``[layer_start, layer_end)``. Stage 0 additionally
owns the token/position embeddings; the last stage owns the final norm and the LM head. The same class is
used for the single-process baseline (one stage with every layer) and for each pipeline worker.

Design choices (see docs/DESIGN.md):

* **Flattened token batches with selective batching** (Orca, OSDI'22). A micro-batch is a list of
  ``SeqSlice`` (request, cache_position, n_tokens). Tokens of all sequences are concatenated into one
  ``[T, H]`` matrix so every dense op (QKV, output projection, MLP) runs once per micro-batch; only the
  attention is evaluated per sequence against that sequence's own KV cache. Prefill chunks and decode
  tokens can therefore be mixed in a micro-batch (Sarathi-Serve style chunked prefill).
* **Position-indexed KV writes make steps idempotent.** A step writes K/V into slots
  ``[pos, pos+n)`` and attends to ``[0, pos+n)``. Re-executing a step (RPC retry after a lost reply) just
  overwrites the same slots, so retries cannot corrupt the cache or duplicate state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------
# KV cache
# ----------------------------------------------------------------------------------------------------------


class SeqCache:
    """Per-request KV cache for the layers a stage owns. Grows geometrically; slots are position-indexed."""

    __slots__ = ("k", "v", "length", "capacity")

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, capacity: int, dtype: torch.dtype):
        self.capacity = max(16, capacity)
        self.k = torch.zeros(n_layers, n_kv_heads, self.capacity, head_dim, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0  # number of valid positions

    def ensure(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        new_cap = max(needed, self.capacity * 2)
        k = torch.zeros(*self.k.shape[:2], new_cap, self.k.shape[-1], dtype=self.k.dtype)
        v = torch.zeros_like(k)
        k[:, :, : self.capacity] = self.k
        v[:, :, : self.capacity] = self.v
        self.k, self.v, self.capacity = k, v, new_cap

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


@dataclass
class SeqSlice:
    """One sequence's share of a flattened micro-batch."""

    request_id: str
    cache_position: int  # absolute position of the first token in this slice
    num_tokens: int
    want_logits: bool = True  # only the last position's logits are returned


# ----------------------------------------------------------------------------------------------------------
# Architecture specs
# ----------------------------------------------------------------------------------------------------------


@dataclass
class ArchSpec:
    family: str  # "gpt2" | "qwen2" (llama-like: RMSNorm, RoPE, GQA, SwiGLU)
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    eps: float
    rope_theta: float = 10000.0
    max_positions: int = 1 << 30
    extra: dict = field(default_factory=dict)


def arch_from_hf(model) -> ArchSpec:
    cfg = model.config
    mt = cfg.model_type
    if mt == "gpt2":
        return ArchSpec("gpt2", cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head, cfg.n_embd // cfg.n_head,
                        cfg.vocab_size, cfg.layer_norm_epsilon, max_positions=cfg.n_positions)
    if mt in ("qwen2", "llama", "mistral", "qwen3"):
        rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        theta = rp.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return ArchSpec("qwen2", cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
                        cfg.num_key_value_heads, hd, cfg.vocab_size, cfg.rms_norm_eps, rope_theta=theta,
                        extra={"qk_norm": mt == "qwen3"})
    raise ValueError(f"unsupported model_type {mt}")


def _gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def _rms(x, w, eps):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


# ----------------------------------------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------------------------------------


class Stage:
    """Executes a contiguous slice of a pretrained HF causal LM. Holds references only to owned weights."""

    def __init__(self, hf_model, layer_start: int, layer_end: int, dtype: torch.dtype = torch.float32,
                 head_on_first: bool = False):
        self.spec = arch_from_hf(hf_model)
        s = self.spec
        assert 0 <= layer_start < layer_end <= s.n_layers, (layer_start, layer_end, s.n_layers)
        self.layer_start, self.layer_end = layer_start, layer_end
        self.is_first = layer_start == 0
        self.is_last = layer_end == s.n_layers
        # Tied-head co-location: with tied input/output embeddings, putting the LM head on stage 0 (next to the
        # embedding) keeps the vocab matrix in ONE process; the last stage then returns final hidden rows.
        self.has_head = self.is_first if head_on_first else self.is_last
        self.dtype = dtype
        self.caches: dict[str, SeqCache] = {}
        m = hf_model
        if s.family == "gpt2":
            tr = m.transformer
            self.wte = tr.wte.weight if self.is_first else None
            self.wpe = tr.wpe.weight if self.is_first else None
            self.blocks = [tr.h[i] for i in range(layer_start, layer_end)]
            self.ln_f = tr.ln_f if self.has_head else None
            self.lm_head = m.lm_head.weight if self.has_head else None
        else:
            core = m.model
            self.wte = core.embed_tokens.weight if self.is_first else None
            self.blocks = [core.layers[i] for i in range(layer_start, layer_end)]
            self.ln_f = core.norm if self.has_head else None
            self.lm_head = m.lm_head.weight if self.has_head else None
            self.inv_freq = 1.0 / (s.rope_theta ** (torch.arange(0, s.head_dim, 2, dtype=torch.float32) / s.head_dim))
        self.n_local = layer_end - layer_start

    # -- cache management -------------------------------------------------------------------------------
    def drop(self, request_id: str) -> bool:
        return self.caches.pop(request_id, None) is not None

    def reset(self) -> None:
        self.caches.clear()

    def cache_len(self, request_id: str) -> int:
        c = self.caches.get(request_id)
        return c.length if c else 0

    def cache_bytes(self) -> int:
        return sum(c.nbytes() for c in self.caches.values())

    def _cache_for(self, sl: SeqSlice) -> SeqCache:
        c = self.caches.get(sl.request_id)
        end = sl.cache_position + sl.num_tokens
        if c is None:
            if sl.cache_position != 0:
                raise CacheMiss(sl.request_id, sl.cache_position, 0)
            s = self.spec
            c = SeqCache(self.n_local, s.n_kv_heads, s.head_dim, end + 64, self.dtype)
            self.caches[sl.request_id] = c
        elif sl.cache_position > c.length:
            # A gap would mean attending to never-written slots: refuse instead of producing garbage.
            raise CacheMiss(sl.request_id, sl.cache_position, c.length)
        c.ensure(end)
        return c

    # -- forward ------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, x: torch.Tensor, seqs: list[SeqSlice], all_logits: bool = False) -> torch.Tensor:
        """x: LongTensor[T] of token ids (first stage) or FloatTensor[T, H] hidden states.

        Returns hidden states [T, H] (non-last stage) or logits [n_selected, V] (last stage) where the selected
        rows are the last position of every slice with ``want_logits`` (or every position if ``all_logits``).
        """
        assert sum(s.num_tokens for s in seqs) == x.shape[0], "token count mismatch"
        caches = [self._cache_for(sl) for sl in seqs]
        positions = torch.cat([torch.arange(sl.cache_position, sl.cache_position + sl.num_tokens) for sl in seqs])
        s = self.spec
        if self.is_first:
            h = F.embedding(x, self.wte)
            if s.family == "gpt2":
                h = h + F.embedding(positions, self.wpe)
        else:
            h = x.to(self.dtype)
        rope = None
        if s.family != "gpt2":
            freqs = positions[:, None].float() * self.inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            rope = (emb.cos().to(self.dtype)[:, None, :], emb.sin().to(self.dtype)[:, None, :])  # [T,1,hd]
        for li, blk in enumerate(self.blocks):
            h = self._block(blk, li, h, seqs, caches, rope)
        for sl, c in zip(seqs, caches):
            c.length = sl.cache_position + sl.num_tokens  # truncate: anything beyond is stale
        if not self.is_last:
            return h
        if all_logits:
            sel = h
        else:
            idx, off = [], 0
            for sl in seqs:
                off += sl.num_tokens
                if sl.want_logits:
                    idx.append(off - 1)
            sel = h[idx]
        return self.head(sel) if self.has_head else sel

    @torch.inference_mode()
    def head(self, sel: torch.Tensor) -> torch.Tensor:
        """Final norm + LM head on selected final hidden rows."""
        s = self.spec
        if s.family == "gpt2":
            sel = F.layer_norm(sel, (s.hidden,), self.ln_f.weight, self.ln_f.bias, s.eps)
        else:
            sel = _rms(sel, self.ln_f.weight, s.eps)
        return sel @ self.lm_head.T

    def _attend(self, q, k_all, v_all, sl: SeqSlice, cache: SeqCache, li: int):
        """q,k_all,v_all are this slice's new rows: q [n,Hq,hd], k/v [n,Hkv,hd]. Writes cache, returns [n,Hq*hd]."""
        p, n = sl.cache_position, sl.num_tokens
        cache.k[li, :, p:p + n] = k_all.transpose(0, 1)
        cache.v[li, :, p:p + n] = v_all.transpose(0, 1)
        K = cache.k[li, :, : p + n]
        V = cache.v[li, :, : p + n]
        Q = q.transpose(0, 1)  # [Hq, n, hd]
        if n == 1:
            mask = None
        else:
            mask = torch.arange(p + n)[None, :] <= (p + torch.arange(n))[:, None]
        o = F.scaled_dot_product_attention(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), attn_mask=mask,
                                           enable_gqa=self.spec.n_kv_heads != self.spec.n_heads)
        return o[0].transpose(0, 1).reshape(n, -1)

    def _block(self, blk, li, h, seqs, caches, rope):
        s = self.spec
        T = h.shape[0]
        if s.family == "gpt2":
            a = F.layer_norm(h, (s.hidden,), blk.ln_1.weight, blk.ln_1.bias, s.eps)
            qkv = torch.addmm(blk.attn.c_attn.bias, a, blk.attn.c_attn.weight)  # HF Conv1D: x @ W + b
            q, k, v = qkv.split(s.hidden, dim=1)
            q = q.view(T, s.n_heads, s.head_dim)
            k = k.view(T, s.n_kv_heads, s.head_dim)
            v = v.view(T, s.n_kv_heads, s.head_dim)
        else:
            at = blk.self_attn
            a = _rms(h, blk.input_layernorm.weight, s.eps)
            q = at.q_proj(a).view(T, s.n_heads, s.head_dim)
            k = at.k_proj(a).view(T, s.n_kv_heads, s.head_dim)
            v = at.v_proj(a).view(T, s.n_kv_heads, s.head_dim)
            if s.extra.get("qk_norm"):
                q = _rms(q, at.q_norm.weight, s.eps)
                k = _rms(k, at.k_norm.weight, s.eps)
            cos, sin = rope
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        outs, off = [], 0
        for sl, c in zip(seqs, caches):
            n = sl.num_tokens
            outs.append(self._attend(q[off:off + n], k[off:off + n], v[off:off + n], sl, c, li))
            off += n
        o = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
        if s.family == "gpt2":
            h = h + torch.addmm(blk.attn.c_proj.bias, o, blk.attn.c_proj.weight)
            m = F.layer_norm(h, (s.hidden,), blk.ln_2.weight, blk.ln_2.bias, s.eps)
            m = _gelu_new(torch.addmm(blk.mlp.c_fc.bias, m, blk.mlp.c_fc.weight))
            return h + torch.addmm(blk.mlp.c_proj.bias, m, blk.mlp.c_proj.weight)
        h = h + blk.self_attn.o_proj(o)
        m = _rms(h, blk.post_attention_layernorm.weight, s.eps)
        mlp = blk.mlp
        return h + mlp.down_proj(F.silu(mlp.gate_proj(m)) * mlp.up_proj(m))


class CacheMiss(RuntimeError):
    def __init__(self, request_id, position, have):
        super().__init__(f"cache miss for {request_id}: step at position {position} but cache holds {have}")
        self.request_id, self.position, self.have = request_id, position, have


# ----------------------------------------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def load_hf(model_name: str, dtype: torch.dtype = torch.float32):
    key = (model_name, dtype)
    if key not in _MODEL_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.eval()
        _MODEL_CACHE[key] = (tok, model)
    return _MODEL_CACHE[key]


def default_split(n_layers: int) -> int:
    return n_layers // 2


def _owned_tensor_ids(stage: "Stage") -> set[int]:
    ids = {id(p) for b in stage.blocks for p in b.parameters()}
    for t in (getattr(stage, "wte", None), getattr(stage, "wpe", None), stage.lm_head):
        if t is not None:
            ids.add(id(t))
    if stage.ln_f is not None:
        ids |= {id(p) for p in stage.ln_f.parameters()}
    return ids


def load_partial(model_name: str, layer_start: int, layer_end: int, dtype=torch.float32,
                 head_on_first: bool = False):
    """Materialise ONLY the tensors a stage owns. The model skeleton is built on the `meta` device (no memory),
    then the owned parameters are read individually from the safetensors checkpoint. A worker therefore never
    holds the other stage's weights, not even transiently during start-up."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import cached_file
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    probe = Stage(model, layer_start, layer_end, dtype, head_on_first)  # which tensors does this stage touch?
    owned = _owned_tensor_ids(probe)
    ckpt = safe_open(cached_file(model_name, "model.safetensors"), "pt")
    keys = set(ckpt.keys())

    def resolve(name):
        cands = [name]
        for pre in ("transformer.", "model."):
            if name.startswith(pre):
                cands.append(name[len(pre):])
        if name == "lm_head.weight":  # tied: stored once as the input embedding
            cands += ["wte.weight", "transformer.wte.weight", "model.embed_tokens.weight"]
        for c in cands:
            if c in keys:
                return c
        raise KeyError(f"{name} not in checkpoint")

    loaded: dict[int, torch.nn.Parameter] = {}
    for mod_name, mod in model.named_modules():
        for pname, param in list(mod._parameters.items()):
            if param is None or id(param) not in owned:
                continue
            if id(param) not in loaded:
                full = f"{mod_name}.{pname}" if mod_name else pname
                t = ckpt.get_tensor(resolve(full)).to(dtype)
                loaded[id(param)] = torch.nn.Parameter(t, requires_grad=False)
            mod._parameters[pname] = loaded[id(param)]  # keeps ties: same Parameter object everywhere
    return Stage(model, layer_start, layer_end, dtype, head_on_first), tok


def build_stage(model_name: str, stage_id: int, split: int | None, dtype=torch.float32,
                lean: bool = False, head_on_first: bool = False) -> tuple[Stage, object]:
    """stage_id 0/1 of a two-way split, or -1 for the unpartitioned model.

    lean=True (worker processes): only the tensors this stage owns are ever materialised (see load_partial)."""
    if lean and stage_id != -1:
        from transformers import AutoConfig
        n = AutoConfig.from_pretrained(model_name).num_hidden_layers
        split = default_split(n) if split is None else split
        lo, hi = (0, split) if stage_id == 0 else (split, n)
        return load_partial(model_name, lo, hi, dtype, head_on_first)
    tok, model = load_hf(model_name, dtype)
    n = model.config.num_hidden_layers
    split = default_split(n) if split is None else split
    if stage_id == -1:
        return Stage(model, 0, n, dtype), tok
    st = (Stage(model, 0, split, dtype, head_on_first) if stage_id == 0
          else Stage(model, split, n, dtype, head_on_first))
    return st, tok
