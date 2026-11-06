"""Activation codecs for the inter-stage link.

* ``fp32``  – lossless reference.
* ``fp16`` / ``bf16`` – 2x smaller; fp16 is unsafe for models with large residual outliers (overflow guard
  falls back to bf16 semantics is *not* done silently: values beyond fp16 range raise).
* ``int8``  – symmetric per-token absmax quantization (LLM.int8()-style vector-wise scaling, no outlier split).
* ``int8h`` / ``int4h`` – block-Hadamard rotation (block 64) followed by per-group (64) symmetric quantization.
  Rotating spreads massive residual-stream outliers across a block before quantization (QuaRot / TAH-Quant
  idea); the receiver dequantizes and applies the inverse (same) orthonormal Hadamard.

All codecs operate on ``[T, H]`` float tensors and round-trip through ``(bytes, meta)``.
"""
from __future__ import annotations

import numpy as np
import torch

CODECS = ("fp32", "fp16", "bf16", "int8", "int8h", "int4h")
_BLOCK = 64
_H_CACHE: dict[int, torch.Tensor] = {}


def _hadamard(n: int) -> torch.Tensor:
    if n not in _H_CACHE:
        assert n & (n - 1) == 0
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H_CACHE[n] = h / np.sqrt(n)
    return _H_CACHE[n]


def _rotate(x: torch.Tensor) -> torch.Tensor:
    T, H = x.shape
    assert H % _BLOCK == 0, f"hidden {H} not divisible by Hadamard block {_BLOCK}"
    return (x.view(T, H // _BLOCK, _BLOCK) @ _hadamard(_BLOCK)).view(T, H)


def _quant(x: torch.Tensor, bits: int, group: int | None):
    T, H = x.shape
    g = x.view(T, H // group, group) if group else x.view(T, 1, H)
    qmax = (1 << (bits - 1)) - 1
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(g / scale), -qmax, qmax).to(torch.int8)
    return q.view(T, H), scale.squeeze(-1).to(torch.float16)


def _dequant(q: torch.Tensor, scale: torch.Tensor, group: int | None):
    T, H = q.shape
    g = q.view(T, H // group, group) if group else q.view(T, 1, H)
    return (g.float() * scale.float().unsqueeze(-1)).view(T, H)


def _pack4(q: torch.Tensor) -> bytes:
    u = (q.to(torch.int16) + 8).to(torch.uint8).view(-1)
    return ((u[0::2] << 4) | u[1::2]).numpy().tobytes()


def _unpack4(b: bytes, shape) -> torch.Tensor:
    u = torch.from_numpy(np.frombuffer(b, dtype=np.uint8).copy())
    out = torch.empty(u.numel() * 2, dtype=torch.int16)
    out[0::2] = (u >> 4).to(torch.int16)
    out[1::2] = (u & 0xF).to(torch.int16)
    return (out - 8).to(torch.int8).view(*shape)


def encode(x: torch.Tensor, codec: str) -> tuple[bytes, bytes, list[int]]:
    """Returns (data, scales, shape)."""
    x = x.detach().float().contiguous()
    shape = list(x.shape)
    if codec == "fp32":
        return x.numpy().tobytes(), b"", shape
    if codec == "fp16":
        if x.abs().max() > 65504:
            raise OverflowError("activation exceeds fp16 range; use bf16 or int8h")
        return x.half().numpy().tobytes(), b"", shape
    if codec == "bf16":
        return x.bfloat16().view(torch.int16).numpy().tobytes(), b"", shape
    if codec == "int8":
        q, s = _quant(x, 8, None)
        return q.numpy().tobytes(), s.numpy().tobytes(), shape
    if codec in ("int8h", "int4h"):
        bits = 8 if codec == "int8h" else 4
        q, s = _quant(_rotate(x), bits, _BLOCK)
        data = q.numpy().tobytes() if bits == 8 else _pack4(q)
        return data, s.numpy().tobytes(), shape
    raise ValueError(codec)


def decode(data: bytes, scales: bytes, shape, codec: str) -> torch.Tensor:
    shape = tuple(int(s) for s in shape)
    if codec == "fp32":
        return torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy()).view(*shape)
    if codec == "fp16":
        return torch.from_numpy(np.frombuffer(data, dtype=np.float16).copy()).float().view(*shape)
    if codec == "bf16":
        return torch.from_numpy(np.frombuffer(data, dtype=np.int16).copy()).view(torch.bfloat16).float().view(*shape)
    T, H = shape
    s = torch.from_numpy(np.frombuffer(scales, dtype=np.float16).copy())
    if codec == "int8":
        q = torch.from_numpy(np.frombuffer(data, dtype=np.int8).copy()).view(T, H)
        return _dequant(q, s.view(T, 1), None)
    if codec in ("int8h", "int4h"):
        q = (torch.from_numpy(np.frombuffer(data, dtype=np.int8).copy()).view(T, H) if codec == "int8h"
             else _unpack4(data, (T, H)))
        return _rotate(_dequant(q, s.view(T, H // _BLOCK), _BLOCK))  # H is symmetric orthonormal: H^-1 = H
    raise ValueError(codec)


def roundtrip(x: torch.Tensor, codec: str) -> torch.Tensor:
    d, s, sh = encode(x, codec)
    return decode(d, s, sh, codec).to(x.dtype)


def wire_bytes(T: int, H: int, codec: str) -> int:
    per = {"fp32": 4 * H, "fp16": 2 * H, "bf16": 2 * H, "int8": H + 2, "int8h": H + 2 * H // _BLOCK,
           "int4h": H // 2 + 2 * H // _BLOCK}[codec]
    return T * per
