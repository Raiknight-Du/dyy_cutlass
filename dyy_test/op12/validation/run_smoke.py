#!/usr/bin/env python3
"""B200 smoke validation for the pinned FlashInfer GDN prefill source.

The script intentionally exercises only public APIs and the upstream reference
implementation.  It emits JSONL so the raw result can be archived verbatim.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cutlass
import torch
import torch.nn.functional as F


REPO = Path(os.environ["FLASHINFER_REPO"]).resolve()
EXPECTED_COMMIT = "e77a4a0d276367895c3b50a642fd8f326c03fb72"


def emit(kind: str, **payload) -> None:
    print(json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)


def load_reference_module():
    path = REPO / "tests" / "gdn" / "reference_delta_rule.py"
    spec = importlib.util.spec_from_file_location("op12_reference_delta_rule", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = load_reference_module()
from flashinfer.gdn_prefill import chunk_gated_delta_rule  # noqa: E402


def make_inputs(seq_lens, hq, hk, hv, dtype, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total = sum(seq_lens)
    q = (torch.rand(total, hq, 128, device="cuda") - 0.5).to(dtype).contiguous()
    k = (torch.rand(total, hk, 128, device="cuda") - 0.5).to(dtype)
    k = F.normalize(k.float(), p=2.0, dim=-1).to(dtype).contiguous()
    v = (torch.rand(total, hv, 128, device="cuda") - 0.5).to(dtype).contiguous()
    ho = max(hq, hv)
    alpha = (0.90 + 0.09 * torch.rand(total, ho, device="cuda")).float().contiguous()
    beta = (0.10 + 0.80 * torch.rand(total, ho, device="cuda")).float().contiguous()
    cu = torch.tensor(
        REFERENCE.exclusive_cumsum(seq_lens), dtype=torch.int64, device="cuda"
    )
    return q, k, v, alpha, beta, cu


def error_stats(actual, expected):
    a = actual.float()
    e = expected.float()
    diff = (a - e).abs()
    rel = diff / (e.abs() + 1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def run_non_cp_reference(name, dtype, seq_lens, heads, seed):
    hq, hk, hv = heads
    q, k, v, alpha, beta, cu = make_inputs(
        seq_lens, hq, hk, hv, dtype, seed
    )
    ho = max(hq, hv)
    out = torch.empty(sum(seq_lens), ho, 128, dtype=dtype, device="cuda")
    state = torch.empty(len(seq_lens), ho, 128, 128, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(128)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        None,
        True,
        cu,
        False,
        output=out,
        output_state=state,
        use_cp=False,
    )
    torch.cuda.synchronize()
    ref_out, ref_state = REFERENCE.blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        scale_factor=scale,
        state_dtype=torch.float32,
    )
    ref_out = ref_out.to(dtype)
    state_for_reference = state.transpose(-1, -2)
    if dtype == torch.bfloat16:
        tolerances = {"out_atol": 1e-2, "out_rtol": 1e-2, "state_atol": 5e-3, "state_rtol": 1e-3}
    else:
        tolerances = {"out_atol": 2e-3, "out_rtol": 1e-3, "state_atol": 1e-3, "state_rtol": 1e-4}
    torch.testing.assert_close(
        out, ref_out, atol=tolerances["out_atol"], rtol=tolerances["out_rtol"]
    )
    torch.testing.assert_close(
        state_for_reference,
        ref_state,
        atol=tolerances["state_atol"],
        rtol=tolerances["state_rtol"],
    )
    return {
        "dtype": str(dtype),
        "seq_lens": seq_lens,
        "heads": {"q": hq, "k": hk, "v": hv},
        "output_error": error_stats(out, ref_out),
        "state_error": error_stats(state_for_reference, ref_state),
        "tolerances": tolerances,
    }


def run_cp_vs_non_cp():
    seq_lens = [2048]
    q, k, v, alpha, beta, cu = make_inputs(
        seq_lens, 1, 1, 1, torch.bfloat16, 30
    )
    scale = 1.0 / math.sqrt(128)

    def invoke(use_cp):
        out = torch.empty(2048, 1, 128, dtype=torch.bfloat16, device="cuda")
        state = torch.empty(1, 1, 128, 128, dtype=torch.float32, device="cuda")
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            scale,
            None,
            True,
            cu,
            False,
            output=out,
            output_state=state,
            use_cp=use_cp,
        )
        torch.cuda.synchronize()
        return out, state

    non_cp_out, non_cp_state = invoke(False)
    cp_out, cp_state = invoke(True)
    torch.testing.assert_close(cp_out, non_cp_out, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(cp_state, non_cp_state, atol=4e-2, rtol=4e-2)
    return {
        "dtype": "torch.bfloat16",
        "seq_lens": seq_lens,
        "heads": {"q": 1, "k": 1, "v": 1},
        "output_error": error_stats(cp_out, non_cp_out),
        "state_error": error_stats(cp_state, non_cp_state),
        "atol": 4e-2,
        "rtol": 4e-2,
    }


def run_fp8_state(state_dtype):
    seq_lens = [64]
    q, k, v, alpha, beta, cu = make_inputs(
        seq_lens, 1, 1, 1, torch.bfloat16, 40 if state_dtype == torch.float8_e4m3fn else 41
    )
    scale = 1.0 / math.sqrt(128)
    torch.manual_seed(42)
    initial_ref = (0.01 * torch.randn(1, 1, 128, 128, device="cuda")).to(state_dtype)
    initial_kernel = initial_ref.transpose(-1, -2).contiguous()
    out = torch.empty(64, 1, 128, dtype=torch.bfloat16, device="cuda")
    state = torch.empty_like(initial_kernel)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        alpha,
        beta,
        scale,
        initial_kernel,
        True,
        cu,
        False,
        output=out,
        output_state=state,
        use_cp=False,
    )
    torch.cuda.synchronize()
    ref_out, ref_state = REFERENCE.blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=alpha,
        beta=beta,
        scale_factor=scale,
        state_dtype=state_dtype,
        initial_state=initial_ref,
    )
    ref_out = ref_out.to(torch.bfloat16)
    state_for_reference = state.transpose(-1, -2).float()
    ref_state_float = ref_state.float()
    torch.testing.assert_close(out, ref_out, atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(state_for_reference, ref_state_float, atol=1e-1, rtol=5e-2)
    if not torch.isfinite(out.float()).all() or not torch.isfinite(state_for_reference).all():
        raise AssertionError("FP8-state result contains non-finite values")
    return {
        "io_dtype": "torch.bfloat16",
        "state_dtype": str(state_dtype),
        "output_error": error_stats(out, ref_out),
        "state_error": error_stats(state_for_reference, ref_state_float),
        "atol": 1e-1,
        "rtol": 5e-2,
    }


def expect_fp8_io_rejection():
    q, k, v, alpha, beta, cu = make_inputs([64], 1, 1, 1, torch.float16, 50)
    q = q.to(torch.float8_e4m3fn)
    k = k.to(torch.float8_e4m3fn)
    v = v.to(torch.float8_e4m3fn)
    out = torch.empty(64, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    try:
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            1.0 / math.sqrt(128),
            None,
            False,
            cu,
            False,
            output=out,
            use_cp=False,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if "expected bfloat16 or float16" not in message:
            raise AssertionError(f"unexpected FP8-I/O rejection: {message}") from exc
        return {"rejected": True, "exception": message}
    raise AssertionError("FP8 Q/K/V unexpectedly reached a successful public API return")


def expect_bnsd_rejection():
    q = torch.zeros(1, 2, 64, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    alpha = torch.ones(1, 2, dtype=torch.float32, device="cuda")
    beta = torch.ones_like(alpha)
    cu = torch.tensor([0, 64], dtype=torch.int64, device="cuda")
    try:
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens=cu,
            use_cp=False,
        )
    except Exception as exc:
        return {"rejected": True, "exception": f"{type(exc).__name__}: {exc}"}
    raise AssertionError("BNSD four-dimensional input unexpectedly returned successfully")


def main():
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected source commit {head}")
    emit(
        "environment",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cutlass=cutlass.__version__,
        flashinfer_repo=str(REPO),
        flashinfer_commit=head,
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
    )
    cases = [
        ("fp16_tnd_gqa_non_cp_reference", lambda: run_non_cp_reference("fp16", torch.float16, [64], (4, 1, 1), 10)),
        ("bf16_tnd_gva_varlen_non_cp_reference", lambda: run_non_cp_reference("bf16", torch.bfloat16, [64, 128], (2, 2, 8), 20)),
        ("bf16_tnd_cp_vs_non_cp", run_cp_vs_non_cp),
        ("bf16_io_fp8_e4m3_state", lambda: run_fp8_state(torch.float8_e4m3fn)),
        ("bf16_io_fp8_e5m2_state", lambda: run_fp8_state(torch.float8_e5m2)),
        ("fp8_qkv_negative", expect_fp8_io_rejection),
        ("bnsd_4d_negative", expect_bnsd_rejection),
    ]
    failed = 0
    for name, fn in cases:
        emit("case_start", name=name)
        try:
            details = fn()
            emit("case_result", name=name, status="PASS", details=details)
        except Exception as exc:
            failed += 1
            emit(
                "case_result",
                name=name,
                status="FAIL",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
    emit("summary", total=len(cases), passed=len(cases) - failed, failed=failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
