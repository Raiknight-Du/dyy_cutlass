#!/usr/bin/env python3
"""Small BNSD+TND memory-safety probe for compute-sanitizer."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from op12_gdn import gdn_prefill_fp8


def make(shape):
    return (torch.randn(shape, device="cuda") * 0.02).to(torch.float8_e4m3fn)


def main():
    for layout in ("BNSD", "TND"):
        if layout == "BNSD":
            shape, scalar, cu = (1, 1, 17, 128), (1, 1, 17), None
        else:
            shape, scalar = (17, 1, 128), (17, 1)
            cu = torch.tensor([0, 17], dtype=torch.int32, device="cuda")
        q, k, v = make(shape), make(shape), make(shape)
        gate = torch.full(scalar, 0.99, dtype=torch.float32, device="cuda")
        beta = torch.full(scalar, 0.1, dtype=torch.float32, device="cuda")
        o, state = gdn_prefill_fp8(q, k, v, gate, beta, cu_seqlens=cu, output_final_state=True, use_cp=False)
        assert torch.isfinite(o).all() and torch.isfinite(state).all()
        torch.cuda.synchronize()
        print(layout, "PASS", tuple(o.shape), tuple(state.shape))


if __name__ == "__main__":
    main()
