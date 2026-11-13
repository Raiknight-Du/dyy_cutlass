from __future__ import annotations

import torch


def fp8_randn(shape, scale: float = 0.05):
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * scale).to(torch.float8_e4m3fn)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a, e = actual.float().flatten(), expected.float().flatten()
    rel = float((a - e).norm() / e.norm().clamp_min(1.0e-12))
    if a.numel() == 0:
        return rel, 1.0
    cosine = float(torch.nn.functional.cosine_similarity(a, e, dim=0))
    return rel, cosine
