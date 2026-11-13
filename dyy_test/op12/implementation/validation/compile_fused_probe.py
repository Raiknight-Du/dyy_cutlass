#!/usr/bin/env python3
"""Compile/launch probe for the experimental single-CTA E4M3 CuTe port."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "upstreams" / "flashinfer"))

import torch

from op12_gdn.cute_fused import chunk_gated_delta_rule_sm100_fp8_experimental


def main():
    t, h, d = 64, 1, 128
    q = torch.zeros((t, h, d), dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    # The first compile milestone keeps the upstream single-io-dtype O staging.
    # It is not wired into the public FP16-output API.
    output = torch.empty_like(q)
    gate = torch.ones((t, h), dtype=torch.float32, device="cuda")
    beta = torch.full_like(gate, 0.1)
    cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    state = torch.empty((1, h, d, d), dtype=torch.float16, device="cuda")
    chunk_gated_delta_rule_sm100_fp8_experimental(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu,
        None,
        state,
        1.0 / (d**0.5),
    )
    torch.cuda.synchronize()
    print("EXPERIMENTAL_FUSED_COMPILE_AND_LAUNCH_PASS")


if __name__ == "__main__":
    main()
