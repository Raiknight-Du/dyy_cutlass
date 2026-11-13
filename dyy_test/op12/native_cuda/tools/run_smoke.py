from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from op12_native_gdn import chunk_gated_delta_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cp", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260829)
    dtype = torch.float16
    seq_lens = [63, 65]
    total = sum(seq_lens)
    q = torch.randn(total, 8, 128, device="cuda", dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn(total, 2, 128, device="cuda", dtype=torch.float32), dim=-1
    ).to(dtype)
    v = torch.randn(total, 2, 128, device="cuda", dtype=dtype)
    g = torch.rand(total, 8, device="cuda")
    beta = torch.rand(total, 8, device="cuda")
    cu = torch.tensor([0, 63, 128], device="cuda", dtype=torch.int32)
    out, state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=1 / math.sqrt(128),
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=args.use_cp,
    )
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "PASS",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "output_shape": list(out.shape),
        "state_shape": list(state.shape),
        "output_finite": bool(torch.isfinite(out).all()),
        "state_finite": bool(torch.isfinite(state).all()),
        "use_cp": args.use_cp,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
