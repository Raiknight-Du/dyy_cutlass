#!/usr/bin/env python3
"""Single-call NCU driver for pinned FlashInfer GDN prefill."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PINNED_COMMIT = "e77a4a0d276367895c3b50a642fd8f326c03fb72"
REPO = Path(os.environ["FLASHINFER_REPO"]).resolve()
sys.path.insert(0, str(REPO))

import flashinfer  # noqa: E402
from flashinfer.gdn_prefill import chunk_gated_delta_rule  # noqa: E402


CASES = {
    "noncp_underfilled_fp16": dict(batch=1, seq=32768, dtype=torch.float16, use_cp=False),
    "noncp_nearfull_fp16": dict(batch=8, seq=4096, dtype=torch.float16, use_cp=False),
    "cp_underfilled_fp16": dict(batch=1, seq=32768, dtype=torch.float16, use_cp=True),
    "noncp_underfilled_bf16": dict(batch=1, seq=32768, dtype=torch.bfloat16, use_cp=False),
    "cp_s8192_fp16": dict(batch=1, seq=8192, dtype=torch.float16, use_cp=True),
    "cp_s16384_fp16": dict(batch=1, seq=16384, dtype=torch.float16, use_cp=True),
    "cp_s32768_fp16": dict(batch=1, seq=32768, dtype=torch.float16, use_cp=True),
    "cp_s8192_bf16": dict(batch=1, seq=8192, dtype=torch.bfloat16, use_cp=True),
    "cp_s16384_bf16": dict(batch=1, seq=16384, dtype=torch.bfloat16, use_cp=True),
    "cp_s32768_bf16": dict(batch=1, seq=32768, dtype=torch.bfloat16, use_cp=True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    cfg = CASES[args.case]

    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(f"unexpected FlashInfer commit {head}")
    if REPO not in Path(flashinfer.__file__).resolve().parents:
        raise RuntimeError(f"unexpected FlashInfer import {flashinfer.__file__}")

    torch.manual_seed(20260825)
    torch.cuda.manual_seed_all(20260825)
    batch = cfg["batch"]
    seq = cfg["seq"]
    total = batch * seq
    h = 16
    d = 128
    dtype = cfg["dtype"]
    q = torch.randn(total, h, d, dtype=dtype, device="cuda").contiguous()
    k = F.normalize(
        torch.randn(total, h, d, dtype=torch.float32, device="cuda"),
        p=2.0,
        dim=-1,
    ).to(dtype).contiguous()
    v = torch.randn(total, h, d, dtype=dtype, device="cuda").contiguous()
    gate = torch.rand(total, h, dtype=torch.float32, device="cuda").contiguous()
    beta = torch.rand(total, h, dtype=torch.float32, device="cuda").sigmoid().contiguous()
    cu = torch.arange(0, total + 1, seq, dtype=torch.int64, device="cuda")
    initial = torch.randn(batch, h, d, d, dtype=torch.float32, device="cuda").mul_(0.01)
    output = torch.empty(total, h, d, dtype=dtype, device="cuda")
    output_state = torch.empty(batch, h, d, d, dtype=torch.float32, device="cuda")

    def invoke():
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g=gate,
            beta=beta,
            scale=1.0 / math.sqrt(d),
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False,
            output=output,
            output_state=output_state,
            use_cp=cfg["use_cp"],
        )

    for _ in range(4):
        invoke()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"OP12_GDN_{args.case}")
    result = invoke()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    out = result[0] if isinstance(result, tuple) else result
    if not torch.isfinite(out.float()).all() or not torch.isfinite(output_state).all():
        raise RuntimeError("profile result contains NaN/Inf")
    print(
        f"PROFILE_OK case={args.case} flashinfer={Path(flashinfer.__file__).resolve()} "
        f"shape=({batch},{h},{seq},{d}) dtype={dtype} use_cp={cfg['use_cp']}"
    )


if __name__ == "__main__":
    main()
