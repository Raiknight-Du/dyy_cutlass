"""Public op12 API; independent from FlashInfer's public API."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .backend_scaled_mm import run_scaled_mm
from .contracts import validate_problem
from .reference import gdn_recurrent_reference


@torch.inference_mode()
def gdn_prefill_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    use_cp: Literal["auto"] | bool = "auto",
    backend: Literal["scaled_mm", "unit_scale_fp8", "reference"] = "scaled_mm",
    return_stats: bool = False,
):
    """Run B200 E4M3 Gated DeltaNet prefill for BNSD or packed TND.

    Q/K/V are raw E4M3 values with implicit unit external scale. Gate, beta,
    cumulative gate math, inverse, epilogues, and state updates are FP32.
    Output and persistent state are FP16. Public state layout is [B,H,V,K].

    ``scaled_mm`` is the executable seven-FP8-GEMM bring-up backend with
    per-tile amax scaling. ``unit_scale_fp8`` executes the same seven logical
    GEMMs with direct E4M3 casts and literal-one MMA operand scales: it has no
    data-dependent quantization/dequantization scale. Both are unfused;
    ``reference`` is the FP32 recurrent gold path.
    """

    if use_cp not in (True, False, "auto"):
        raise ValueError("use_cp must be True, False, or 'auto'")
    p = validate_problem(q, k, v, gate, beta, cu_seqlens, initial_state)
    score_scale = 1.0 / math.sqrt(128) if scale is None else float(scale)
    if not math.isfinite(score_scale):
        raise ValueError("scale must be finite")

    if backend in ("scaled_mm", "unit_scale_fp8"):
        out, final_state, stats = run_scaled_mm(
            q,
            k,
            v,
            gate,
            beta,
            p,
            initial_state=initial_state,
            score_scale=score_scale,
            use_cp=use_cp,
            quantization_mode=(
                "unit_scale" if backend == "unit_scale_fp8" else "tile_scale"
            ),
        )
        stats_value = stats.as_dict()
    elif backend == "reference":
        out, final_state = gdn_recurrent_reference(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            scale=score_scale,
        )
        stats_value = {"backend": "fp32_recurrent_reference"}
    else:
        raise ValueError(f"unknown backend {backend!r}")

    primary = (out, final_state) if output_final_state else out
    return (primary, stats_value) if return_stats else primary
