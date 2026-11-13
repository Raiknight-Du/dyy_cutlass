#!/usr/bin/env python3
"""Reproducible B200 timing matrix for the pinned FlashInfer GDN prefill.

The script emits one JSON object per line.  Q/V follow the upstream benchmark
distribution; K is L2-normalized before the timed region.  Only the public
``chunk_gated_delta_rule`` API is called.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PINNED_COMMIT = "e77a4a0d276367895c3b50a642fd8f326c03fb72"
REPO = Path(os.environ["FLASHINFER_REPO"]).resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import flashinfer  # noqa: E402
from flashinfer.gdn_prefill import chunk_gated_delta_rule  # noqa: E402


def emit(kind: str, **payload: Any) -> None:
    print(json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def telemetry() -> dict[str, Any]:
    query = (
        "name,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "utilization.gpu,memory.used"
    )
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [x.strip() for x in out.split(",")]
    return {
        "gpu": fields[0],
        "temperature_c": float(fields[1]),
        "power_w": float(fields[2]),
        "sm_clock_mhz": int(fields[3]),
        "mem_clock_mhz": int(fields[4]),
        "gpu_util_pct": int(fields[5]),
        "memory_used_mib": int(fields[6]),
    }


def compute_processes() -> list[str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def predicted_path(use_cp: bool | str, batch: int, output_heads: int, sm_count: int) -> str:
    if use_cp is True:
        return "cp"
    if use_cp is False:
        return "non_cp"
    return "cp" if batch * output_heads * 4 < sm_count else "non_cp"


def main_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for dtype in ("float16", "bfloat16"):
        for seq in (8192, 16384, 32768):
            for path in (False, True, "auto"):
                cases.append(
                    {
                        "category": "main",
                        "name": f"main_{dtype}_b1_h16_s{seq}_{path}",
                        "seq_lens": [seq],
                        "heads": (16, 16, 16),
                        "dtype": dtype,
                        "use_cp": path,
                        "initial_state": True,
                        "output_final_state": True,
                    }
                )
    return cases


def diagnostic_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for batch in (1, 2, 4, 8, 16):
        seq = 32768 // batch
        for path in (False, "auto"):
            cases.append(
                {
                    "category": "parallelism",
                    "name": f"parallel_fp16_b{batch}_h16_s{seq}_{path}",
                    "seq_lens": [seq] * batch,
                    "heads": (16, 16, 16),
                    "dtype": "float16",
                    "use_cp": path,
                    "initial_state": True,
                    "output_final_state": True,
                }
            )
    for label, heads in (
        ("mha", (16, 16, 16)),
        ("gqa", (64, 16, 16)),
        ("gva", (16, 16, 64)),
    ):
        for path in (False, "auto"):
            cases.append(
                {
                    "category": "head_mode",
                    "name": f"heads_{label}_fp16_b1_s8192_{path}",
                    "seq_lens": [8192],
                    "heads": heads,
                    "dtype": "float16",
                    "use_cp": path,
                    "initial_state": True,
                    "output_final_state": True,
                }
            )
    for initial_state in (False, True):
        for output_final_state in (False, True):
            cases.append(
                {
                    "category": "state_ablation",
                    "name": (
                        "state_fp16_b1_h16_s8192_noncp_"
                        f"init{int(initial_state)}_out{int(output_final_state)}"
                    ),
                    "seq_lens": [8192],
                    "heads": (16, 16, 16),
                    "dtype": "float16",
                    "use_cp": False,
                    "initial_state": initial_state,
                    "output_final_state": output_final_state,
                }
            )
    for path in (False, "auto"):
        cases.append(
            {
                "category": "varlen",
                "name": f"varlen_fp16_1024_3072_8192_20480_{path}",
                "seq_lens": [1024, 3072, 8192, 20480],
                "heads": (16, 16, 16),
                "dtype": "float16",
                "use_cp": path,
                "initial_state": True,
                "output_final_state": True,
            }
        )
    return cases


def select_cases(suite: str) -> list[dict[str, Any]]:
    if suite == "main":
        return main_cases()
    if suite == "diagnostic":
        return diagnostic_cases()
    return main_cases() + diagnostic_cases()


def run_case(case: dict[str, Any], warmup: int, iterations: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seq_lens = list(case["seq_lens"])
    batch = len(seq_lens)
    total_tokens = sum(seq_lens)
    hq, hk, hv = case["heads"]
    ho = max(hq, hv)
    d = 128
    dtype = getattr(torch, case["dtype"])
    device = torch.device("cuda")

    q0 = torch.randn(total_tokens, hq, d, dtype=dtype, device=device).contiguous()
    k0 = F.normalize(
        torch.randn(total_tokens, hk, d, dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    ).to(dtype).contiguous()
    v0 = torch.randn(total_tokens, hv, d, dtype=dtype, device=device).contiguous()
    gate = torch.rand(total_tokens, ho, dtype=torch.float32, device=device).contiguous()
    beta = torch.rand(total_tokens, ho, dtype=torch.float32, device=device).sigmoid().contiguous()
    cu = torch.tensor(
        [0] + list(__import__("itertools").accumulate(seq_lens)),
        dtype=torch.int64,
        device=device,
    )
    initial = (
        torch.randn(batch, ho, d, d, dtype=torch.float32, device=device).mul_(0.01)
        if case["initial_state"]
        else None
    )
    output = torch.empty(total_tokens, ho, d, dtype=dtype, device=device)
    output_state = (
        torch.empty(batch, ho, d, d, dtype=torch.float32, device=device)
        if case["output_final_state"]
        else None
    )

    qkv_bytes = q0.nbytes + k0.nbytes + v0.nbytes
    num_buffers = max(1, min(iterations, (4 * 1024**3) // max(1, qkv_bytes)))
    qs = [q0] + [q0.clone() for _ in range(num_buffers - 1)]
    ks = [k0] + [k0.clone() for _ in range(num_buffers - 1)]
    vs = [v0] + [v0.clone() for _ in range(num_buffers - 1)]
    invoke_count = 0

    def invoke():
        nonlocal invoke_count
        idx = invoke_count % num_buffers
        result = chunk_gated_delta_rule(
            qs[idx],
            ks[idx],
            vs[idx],
            g=gate,
            beta=beta,
            scale=1.0 / math.sqrt(d),
            initial_state=initial,
            output_final_state=bool(case["output_final_state"]),
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False,
            output=output,
            output_state=output_state,
            use_cp=case["use_cp"],
        )
        invoke_count += 1
        return result

    telemetry_before = telemetry()
    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()

    result = invoke()
    torch.cuda.synchronize()
    result_output = result[0] if isinstance(result, tuple) else result
    finite_output = bool(torch.isfinite(result_output.float()).all().item())
    finite_state = True
    if isinstance(result, tuple):
        finite_state = bool(torch.isfinite(result[1].float()).all().item())

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for i in range(iterations):
        starts[i].record()
        invoke()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [float(starts[i].elapsed_time(ends[i])) for i in range(iterations)]
    telemetry_after = telemetry()

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.fmean(times_ms)
    stdev_ms = statistics.pstdev(times_ms)
    cv_pct = stdev_ms / mean_ms * 100.0 if mean_ms else 0.0
    upstream_flops = 4 * total_tokens * hv * d * d
    logical_chunks = sum(math.ceil(s / 64) for s in seq_lens)
    seven_gemm_flops = logical_chunks * ho * 10_485_760
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    predicted = predicted_path(case["use_cp"], batch, ho, sm_count)

    result_row = {
        **case,
        "trial_seed": seed,
        "batch": batch,
        "total_tokens": total_tokens,
        "output_heads": ho,
        "head_size": d,
        "predicted_dispatch": predicted,
        "non_cp_grid": batch * ho,
        "waves_per_sm_non_cp": batch * ho / sm_count,
        "num_buffers": num_buffers,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms_median": median_ms,
        "latency_ms_mean": mean_ms,
        "latency_ms_p5": percentile(times_ms, 0.05),
        "latency_ms_p95": percentile(times_ms, 0.95),
        "latency_ms_min": min(times_ms),
        "latency_ms_max": max(times_ms),
        "latency_cv_pct": cv_pct,
        "tokens_per_second": total_tokens / (median_ms / 1000.0),
        "upstream_work_tflops": upstream_flops / (median_ms / 1000.0) / 1e12,
        "seven_gemm_tflops": seven_gemm_flops / (median_ms / 1000.0) / 1e12,
        "logical_chunks": logical_chunks,
        "finite_output": finite_output,
        "finite_state": finite_state,
        "telemetry_before": telemetry_before,
        "telemetry_after": telemetry_after,
        "times_ms": times_ms,
    }

    del qs, ks, vs, q0, k0, v0, gate, beta, cu, initial, output, output_state
    del starts, ends, result, result_output
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.1)
    return result_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("main", "diagnostic", "all"), default="all")
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if git("rev-parse", "HEAD") != PINNED_COMMIT:
        raise RuntimeError("FlashInfer HEAD does not match the pinned op12 commit")
    imported = Path(flashinfer.__file__).resolve()
    if REPO not in imported.parents:
        raise RuntimeError(f"Imported FlashInfer is not the pinned source: {imported}")
    if torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("This benchmark is fixed to B200 / compute capability 10.0")

    props = torch.cuda.get_device_properties(0)
    emit(
        "environment",
        timestamp_utc=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        hostname=platform.node(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        flashinfer_file=str(imported),
        flashinfer_version=getattr(flashinfer, "__version__", "unknown"),
        flashinfer_commit=git("rev-parse", "HEAD"),
        flashinfer_status=git("status", "--short"),
        gpu=props.name,
        capability=list(torch.cuda.get_device_capability(0)),
        sm_count=props.multi_processor_count,
        total_memory=props.total_memory,
        telemetry=telemetry(),
        compute_processes=compute_processes(),
        trial=args.trial,
        suite=args.suite,
    )

    failed = 0
    cases = select_cases(args.suite)
    for index, case in enumerate(cases):
        emit("case_start", index=index, total=len(cases), name=case["name"])
        try:
            result = run_case(
                case,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=20260825 + args.trial * 1000 + index,
            )
            emit("case_result", status="PASS", **result)
        except Exception as exc:
            failed += 1
            emit(
                "case_result",
                status="FAIL",
                name=case["name"],
                error=f"{type(exc).__name__}: {exc}",
            )
    emit("summary", total=len(cases), failed=failed, passed=len(cases) - failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
