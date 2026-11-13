"""On-chip tile quantization model used by the executable prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .contracts import FP8_DTYPE


E4M3_MAX = 448.0


@dataclass
class QuantizationStats:
    values: dict[str, list[float]] = field(default_factory=dict)
    zeroed_nonzero_values: dict[str, list[float]] = field(default_factory=dict)
    amax_values: dict[str, list[float]] = field(default_factory=dict)

    def add(
        self,
        name: str,
        saturation_rate: float,
        zeroed_nonzero_rate: float = 0.0,
        amax: float = 0.0,
    ) -> None:
        self.values.setdefault(name, []).append(saturation_rate)
        self.zeroed_nonzero_values.setdefault(name, []).append(zeroed_nonzero_rate)
        self.amax_values.setdefault(name, []).append(amax)

    def summarize(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "tiles": len(rates),
                "mean_saturation_rate": sum(rates) / len(rates),
                "max_saturation_rate": max(rates),
                "mean_zeroed_nonzero_rate": sum(self.zeroed_nonzero_values[name])
                / len(self.zeroed_nonzero_values[name]),
                "max_zeroed_nonzero_rate": max(self.zeroed_nonzero_values[name]),
                "max_input_amax": max(self.amax_values[name]),
            }
            for name, rates in self.values.items()
            if rates
        }


def quantize_e4m3_tile(
    x: torch.Tensor,
    name: str,
    stats: QuantizationStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one logical tile with s=maxabs/448 and return (FP8, dequant s).

    ``Tensor.to(float8_e4m3fn)`` uses CUDA's round-to-nearest conversion.  The
    explicit clamp gives saturating behavior instead of E4M3FN NaNs.
    """

    xf = x.to(torch.float32)
    amax = xf.abs().amax()
    one = torch.ones((), dtype=torch.float32, device=x.device)
    scale = torch.where(amax > 0, amax / E4M3_MAX, one)
    normalized = xf / scale
    saturation = (normalized.abs() >= E4M3_MAX).to(torch.float32).mean()
    q = normalized.clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE)
    zeroed_nonzero = ((xf != 0) & (q.float() == 0)).to(torch.float32).mean()
    stats.add(
        name,
        float(saturation.item()),
        float(zeroed_nonzero.item()),
        float(amax.item()),
    )
    return q, scale


def quantize_e4m3_no_scale(
    x: torch.Tensor,
    name: str,
    stats: QuantizationStats,
) -> torch.Tensor:
    """Directly cast one tile to E4M3 with no data-dependent scale.

    The clamp implements saturating E4M3FN conversion; unlike
    :func:`quantize_e4m3_tile`, the original numeric value is not divided by
    ``amax/448`` and there is no compensating dequantization multiply after
    MMA.  This is the exact unit-scale experiment requested for op12.
    """

    xf = x.to(torch.float32)
    amax = xf.abs().amax()
    saturation = (xf.abs() >= E4M3_MAX).to(torch.float32).mean()
    q = xf.clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE)
    zeroed_nonzero = ((xf != 0) & (q.float() == 0)).to(torch.float32).mean()
    stats.add(
        name,
        float(saturation.item()),
        float(zeroed_nonzero.item()),
        float(amax.item()),
    )
    return q


def fp8_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor | float,
    scale_b: torch.Tensor | float,
) -> torch.Tensor:
    """E4M3 Tensor-Core GEMM with FP32 accumulator/output on SM100.

    Each dimension presented to cuBLASLt is a multiple of 16 in this
    implementation.  ``use_fast_accum=False`` requests the accurate FP32
    accumulation mode.
    """

    if a.dtype != FP8_DTYPE or b.dtype != FP8_DTYPE:
        raise TypeError("fp8_scaled_mm operands must both be E4M3")
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=a.device)
    if not isinstance(scale_b, torch.Tensor):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=a.device)
    return torch._scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )


def fp8_unit_scale_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """E4M3 Tensor-Core GEMM without quantization/dequantization scaling.

    ``torch._scaled_mm`` requires scale tensors in its ABI.  Both operands are
    therefore passed literal unit scalars.  No amax-derived or caller-provided
    scale participates in the mathematical result.
    """

    if a.dtype != FP8_DTYPE or b.dtype != FP8_DTYPE:
        raise TypeError("fp8_unit_scale_mm operands must both be E4M3")
    one = torch.ones((), dtype=torch.float32, device=a.device)
    return torch._scaled_mm(
        a,
        b,
        scale_a=one,
        scale_b=one,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )
