"""Public shape, layout, dtype, and device contract for op12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


HEAD_SIZE = 128
CHUNK_SIZE = 64
FP8_DTYPE = torch.float8_e4m3fn
Layout = Literal["BNSD", "TND"]


@dataclass(frozen=True)
class Problem:
    layout: Layout
    batch: int
    total_tokens: int
    sequence_length: int | None
    hq: int
    hk: int
    hv: int
    ho: int
    cu_seqlens: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_fp8_tensor(name: str, x: torch.Tensor, ndim: int) -> None:
    _require(isinstance(x, torch.Tensor), f"{name} must be a torch.Tensor")
    _require(x.is_cuda, f"{name} must be on CUDA")
    _require(x.ndim == ndim, f"{name} must have rank {ndim}, got {x.ndim}")
    _require(x.dtype == FP8_DTYPE, f"{name} must be E4M3 ({FP8_DTYPE}), got {x.dtype}")
    _require(x.shape[-1] == HEAD_SIZE, f"{name} D must be 128, got {x.shape[-1]}")
    _require(x.stride(-1) == 1, f"{name} D dimension must be contiguous")
    _require(x.data_ptr() % 16 == 0, f"{name} base address must be at least 16-byte aligned")


def _check_heads(hq: int, hk: int, hv: int) -> None:
    _require(min(hq, hk, hv) > 0, "head counts must be positive")
    mha = hq == hk == hv
    gqa = hq > hk and hk == hv and hq % hk == 0
    gva = hv > hq and hq == hk and hv % hq == 0
    _require(
        mha or gqa or gva,
        "head relation must be MHA, GQA (Hq > Hk = Hv, divisible), or "
        "GVA (Hv > Hq = Hk, divisible)",
    )


def _check_gate_beta(
    gate: torch.Tensor,
    beta: torch.Tensor,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    for name, x in (("gate", gate), ("beta", beta)):
        _require(x.is_cuda and x.device == device, f"{name} must be on the Q CUDA device")
        _require(x.dtype == torch.float32, f"{name} must be FP32, got {x.dtype}")
        _require(tuple(x.shape) == expected_shape, f"{name} must have shape {expected_shape}, got {tuple(x.shape)}")
        _require(x.is_contiguous(), f"{name} must be contiguous")
        _require(bool(torch.isfinite(x).all().item()), f"{name} contains NaN or Inf")
    _require(bool((gate >= 0).all().item()), "gate must be non-negative")


def validate_problem(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
) -> Problem:
    """Validate the frozen op12 contract and return normalized metadata."""

    _require(q.device == k.device == v.device, "Q/K/V must be on the same CUDA device")
    major, minor = torch.cuda.get_device_capability(q.device)
    _require((major, minor) == (10, 0), f"op12 requires B200 SM100, got compute capability {major}.{minor}")

    if q.ndim == 4:
        layout: Layout = "BNSD"
        for name, x in (("q", q), ("k", k), ("v", v)):
            _check_fp8_tensor(name, x, 4)
        bq, hq, sq, _ = q.shape
        bk, hk, sk, _ = k.shape
        bv, hv, sv, _ = v.shape
        _require(bq == bk == bv and sq == sk == sv, "BNSD Q/K/V batch and sequence dimensions must match")
        _require(cu_seqlens is None, "BNSD does not accept cu_seqlens")
        _check_heads(hq, hk, hv)
        ho = max(hq, hv)
        _check_gate_beta(gate, beta, (bq, ho, sq), q.device)
        normalized_cu = torch.arange(bq + 1, dtype=torch.int32, device=q.device) * sq
        batch, total_tokens, sequence_length = bq, bq * sq, sq
    elif q.ndim == 3:
        layout = "TND"
        for name, x in (("q", q), ("k", k), ("v", v)):
            _check_fp8_tensor(name, x, 3)
        tq, hq, _ = q.shape
        tk, hk, _ = k.shape
        tv, hv, _ = v.shape
        _require(tq == tk == tv, "TND Q/K/V total-token dimensions must match")
        _check_heads(hq, hk, hv)
        ho = max(hq, hv)
        _check_gate_beta(gate, beta, (tq, ho), q.device)
        _require(cu_seqlens is not None, "TND requires cu_seqlens")
        assert cu_seqlens is not None
        _require(cu_seqlens.is_cuda and cu_seqlens.device == q.device, "cu_seqlens must be on the Q CUDA device")
        _require(cu_seqlens.dtype == torch.int32, f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
        _require(cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, "cu_seqlens must have shape [B+1]")
        _require(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous")
        cu_host = cu_seqlens.detach().cpu()
        _require(int(cu_host[0]) == 0, "cu_seqlens[0] must be zero")
        _require(int(cu_host[-1]) == tq, f"cu_seqlens[-1] must equal T={tq}")
        _require(bool((cu_host[1:] >= cu_host[:-1]).all()), "cu_seqlens must be monotonically non-decreasing")
        normalized_cu = cu_seqlens
        batch, total_tokens, sequence_length = cu_seqlens.numel() - 1, tq, None
    else:
        raise ValueError(f"Q must be BNSD rank 4 or TND rank 3, got rank {q.ndim}")

    if initial_state is not None:
        expected = (batch, ho, HEAD_SIZE, HEAD_SIZE)
        _require(initial_state.is_cuda and initial_state.device == q.device, "initial_state must be on the Q CUDA device")
        _require(initial_state.dtype == torch.float16, f"initial_state must be FP16, got {initial_state.dtype}")
        _require(tuple(initial_state.shape) == expected, f"initial_state must have shape {expected}")
        _require(initial_state.stride(-1) == 1, "initial_state K dimension must be contiguous")

    return Problem(
        layout=layout,
        batch=batch,
        total_tokens=total_tokens,
        sequence_length=sequence_length,
        hq=hq,
        hk=hk,
        hv=hv,
        ho=ho,
        cu_seqlens=normalized_cu,
    )
