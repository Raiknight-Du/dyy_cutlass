#!/usr/bin/env python3
"""One full 64-token/head op12 launch for NCU FP8 Tensor-Core evidence."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from op12_gdn import gdn_prefill_fp8


def main():
    shape = (1, 1, 64, 128)
    q = torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    gate = torch.ones((1, 1, 64), dtype=torch.float32, device="cuda")
    beta = torch.full_like(gate, 0.1)
    gdn_prefill_fp8(
        q,
        k,
        v,
        gate,
        beta,
        output_final_state=True,
        use_cp=False,
        backend="unit_scale_fp8",
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
