#!/usr/bin/env python3
"""One 64-token launch for NCU evidence of the MMA-only FP8 backend."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from op12_gdn import chunk_gated_delta_rule_mma_only_fp8


def main() -> None:
    shape = (64, 1, 128)
    q = torch.zeros(shape, dtype=torch.float16, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    gate = torch.ones((64, 1), dtype=torch.float32, device="cuda")
    beta = torch.full_like(gate, 0.1)
    cu = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    chunk_gated_delta_rule_mma_only_fp8(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens=cu,
        output_final_state=True,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
