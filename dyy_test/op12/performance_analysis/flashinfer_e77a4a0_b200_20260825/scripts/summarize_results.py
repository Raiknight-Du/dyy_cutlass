#!/usr/bin/env python3
"""Summarize benchmark JSONL and NCU CSV into compact analysis tables."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(os.environ["OP12_PERF_OUT"]).resolve()


def load_bench() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "raw" / "bench").glob("trial_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if obj.get("kind") == "case_result" and obj.get("status") == "PASS":
                rows.append(obj)
    return rows


def write_bench_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)
    fields = [
        "name", "category", "dtype", "seq_lens", "heads", "use_cp",
        "predicted_dispatch", "batch", "total_tokens", "output_heads",
        "non_cp_grid", "waves_per_sm_non_cp", "trial_count",
        "median_of_medians_ms", "min_trial_median_ms", "max_trial_median_ms",
        "cross_trial_cv_pct", "tokens_per_second", "upstream_work_tflops",
        "seven_gemm_tflops", "max_within_trial_cv_pct", "all_finite",
    ]
    out_rows = []
    for name, items in sorted(grouped.items()):
        meds = [float(x["latency_ms_median"]) for x in items]
        mean = sum(meds) / len(meds)
        variance = sum((x - mean) ** 2 for x in meds) / len(meds)
        exemplar = items[0]
        median_ms = sorted(meds)[len(meds) // 2]
        upstream_flops = 4 * exemplar["total_tokens"] * exemplar["heads"][2] * 128 * 128
        seven_flops = exemplar["logical_chunks"] * exemplar["output_heads"] * 10_485_760
        out_rows.append({
            "name": name,
            "category": exemplar["category"],
            "dtype": exemplar["dtype"],
            "seq_lens": ",".join(map(str, exemplar["seq_lens"])),
            "heads": "/".join(map(str, exemplar["heads"])),
            "use_cp": str(exemplar["use_cp"]),
            "predicted_dispatch": exemplar["predicted_dispatch"],
            "batch": exemplar["batch"],
            "total_tokens": exemplar["total_tokens"],
            "output_heads": exemplar["output_heads"],
            "non_cp_grid": exemplar["non_cp_grid"],
            "waves_per_sm_non_cp": exemplar["waves_per_sm_non_cp"],
            "trial_count": len(items),
            "median_of_medians_ms": median_ms,
            "min_trial_median_ms": min(meds),
            "max_trial_median_ms": max(meds),
            "cross_trial_cv_pct": math.sqrt(variance) / mean * 100 if mean else 0,
            "tokens_per_second": exemplar["total_tokens"] / (median_ms / 1000),
            "upstream_work_tflops": upstream_flops / (median_ms / 1000) / 1e12,
            "seven_gemm_tflops": seven_flops / (median_ms / 1000) / 1e12,
            "max_within_trial_cv_pct": max(float(x["latency_cv_pct"]) for x in items),
            "all_finite": all(x["finite_output"] and x["finite_state"] for x in items),
        })
    path = ROOT / "tables" / "benchmark_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    (ROOT / "tables" / "benchmark_summary.json").write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_number(value: str) -> float | None:
    text = value.replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def short_kernel_name(name: str) -> str:
    if "GatedDeltaNetChunkedKernel" in name:
        return "noncp_fused"
    if "CPDeltaRuleTPrecomputeSm100" in name:
        return "cp_1_t_precompute"
    if "CPDeltaRuleMNPrecompute" in name:
        return "cp_2_mn_precompute"
    if "CPDeltaRuleFixup" in name:
        return "cp_3_fixup"
    if "CPDeltaRulePrefill" in name:
        return "cp_4_prefill"
    if "unrolled_elementwise_kernel" in name:
        return "cu_seqlens_dtype_copy"
    return name[:96]


def write_ncu_summary() -> None:
    # NCU --page details uses human-readable names.  Preserve these because they
    # are also the most convenient source for launch, SOL, occupancy and
    # scheduler data in the report.
    wanted_display = {
        "Duration",
        "Compute (SM) Throughput",
        "Memory Throughput",
        "DRAM Throughput",
        "L1/TEX Cache Throughput",
        "L2 Cache Throughput",
        "L2 Cache Hit Rate",
        "Registers Per Thread",
        "Dynamic Shared Memory Per Block",
        "Shared Memory Configuration Size",
        "Theoretical Occupancy",
        "Achieved Occupancy",
        "Overall GPU Occupancy",
        "Waves Per SM",
        "Active Warps Per Scheduler",
        "Eligible Warps Per Scheduler",
        "Issued Warp Per Scheduler",
        "One or More Eligible",
        "No Eligible",
        "Local Memory Spilling Requests",
    }
    rows_out: list[dict[str, Any]] = []
    for path in sorted((ROOT / "raw" / "ncu").glob("*_details.csv")):
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Metric Name", "") not in wanted_display:
                    continue
                rows_out.append({
                    "profile": path.name.removesuffix("_details.csv"),
                    "id": row.get("ID", ""),
                    "kernel_short": short_kernel_name(row.get("Kernel Name", "")),
                    "kernel": row.get("Kernel Name", ""),
                    "block_size": row.get("Block Size", ""),
                    "grid_size": row.get("Grid Size", ""),
                    "section": row.get("Section Name", ""),
                    "metric": row.get("Metric Name", ""),
                    "unit": row.get("Metric Unit", ""),
                    "value": row.get("Metric Value", ""),
                    "numeric_value": parse_number(row.get("Metric Value", "")),
                })
    fields = [
        "profile", "id", "kernel_short", "kernel", "block_size", "grid_size",
        "section", "metric", "unit", "value", "numeric_value",
    ]
    path = ROOT / "tables" / "ncu_key_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    (ROOT / "tables" / "ncu_key_metrics.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # NCU --page raw is a wide CSV: its first data row contains units and the
    # following rows contain one record per launch.  Extract exact counters used
    # for byte traffic, Tensor Core activity and warp-stall attribution.
    wanted_raw = {
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_pipe_tensor.sum",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    }
    raw_rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "raw" / "ncu").glob("*_raw.csv")):
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            source_rows = list(reader)
        if not source_rows:
            continue
        units = source_rows[0]
        for row in source_rows[1:]:
            if not row.get("Kernel Name"):
                continue
            for metric in sorted(wanted_raw):
                if metric not in row or row.get(metric, "") == "":
                    continue
                raw_rows.append({
                        "profile": path.name.removesuffix("_raw.csv"),
                        "id": row.get("ID", ""),
                        "kernel_short": short_kernel_name(row.get("Kernel Name", "")),
                        "kernel": row.get("Kernel Name", ""),
                        "block_size": row.get("Block Size", ""),
                        "grid_size": row.get("Grid Size", ""),
                        "metric": metric,
                        "unit": units.get(metric, ""),
                        "value": row.get(metric, ""),
                        "numeric_value": parse_number(row.get(metric, "")),
                    })
    raw_fields = [
        "profile", "id", "kernel_short", "kernel", "block_size", "grid_size",
        "metric", "unit", "value", "numeric_value",
    ]
    path = ROOT / "tables" / "ncu_raw_counters.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    (ROOT / "tables" / "ncu_raw_counters.json").write_text(
        json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_sass_summary() -> None:
    """Aggregate executed SASS instructions for the architecture-specific families."""
    out: dict[tuple[str, str, str], float] = defaultdict(float)
    for path in sorted((ROOT / "raw" / "ncu").glob("*_sass.csv")):
        current_kernel = ""
        header: list[str] = []
        indices: dict[str, int] = {}
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if row[0] == "Kernel Name":
                    current_kernel = row[1]
                    header = []
                    indices = {}
                    continue
                if row[0] == "Address":
                    header = row
                    indices = {name: i for i, name in enumerate(row)}
                    continue
                if not current_kernel or not header or len(row) <= 1:
                    continue
                source = row[indices.get("Source", 1)].strip()
                source = re.sub(r"^@[!A-Z0-9.]+\\s+", "", source)
                opcode = source.split()[0] if source.split() else ""
                if not re.search(
                    r"UTC|UTMA|TMEM|MMA|CPASYNC|BAR|MEMBAR", opcode, re.IGNORECASE
                ):
                    continue
                try:
                    count = float(row[indices["Instructions Executed"]] or 0)
                except (KeyError, ValueError, IndexError):
                    count = 0.0
                out[(
                    path.name.removesuffix("_sass.csv"),
                    short_kernel_name(current_kernel),
                    opcode,
                )] += count
    rows = [
        {
            "profile": profile,
            "kernel_short": kernel,
            "opcode": opcode,
            "instructions_executed": int(count),
        }
        for (profile, kernel, opcode), count in sorted(out.items())
    ]
    fields = ["profile", "kernel_short", "opcode", "instructions_executed"]
    path = ROOT / "tables" / "sass_instruction_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "tables" / "sass_instruction_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_profile_configurations() -> None:
    rows = [
        {"id": "N1", "profile": "noncp_underfilled_fp16", "path": "non-CP", "batch": 1, "heads": "16/16/16", "seq": 32768, "d": 128, "dtype": "FP16", "use_cp": "False", "total_tokens": 32768, "expected_core_grid": 16},
        {"id": "N2", "profile": "noncp_nearfull_fp16", "path": "non-CP", "batch": 8, "heads": "16/16/16", "seq": 4096, "d": 128, "dtype": "FP16", "use_cp": "False", "total_tokens": 32768, "expected_core_grid": 128},
        {"id": "N3", "profile": "noncp_underfilled_bf16", "path": "non-CP", "batch": 1, "heads": "16/16/16", "seq": 32768, "d": 128, "dtype": "BF16", "use_cp": "False", "total_tokens": 32768, "expected_core_grid": 16},
    ]
    for dtype in ("FP16", "BF16"):
        suffix = dtype.lower()
        for seq, cp_chunk_len in ((8192, 1024), (16384, 2048), (32768, 4096)):
            rows.append({
                "id": f"C{1 + (0 if dtype == 'FP16' else 3) + (0 if seq == 8192 else 1 if seq == 16384 else 2)}",
                "profile": f"cp_s{seq}_{suffix}",
                "path": "CP",
                "batch": 1,
                "heads": "16/16/16",
                "seq": seq,
                "d": 128,
                "dtype": dtype,
                "use_cp": "True",
                "total_tokens": seq,
                "cp_chunk_len": cp_chunk_len,
                "partitions_per_head": 8,
                "expected_core_grid": 128,
            })
    fields = ["id", "profile", "path", "batch", "heads", "seq", "d", "dtype", "use_cp", "total_tokens", "cp_chunk_len", "partitions_per_head", "expected_core_grid"]
    with (ROOT / "tables" / "ncu_profile_configurations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cp_matrix_summary() -> None:
    key_rows = list(csv.DictReader((ROOT / "tables" / "ncu_key_metrics.csv").open(encoding="utf-8-sig")))
    raw_rows = list(csv.DictReader((ROOT / "tables" / "ncu_raw_counters.csv").open(encoding="utf-8-sig")))
    key = {(r["profile"], r["kernel_short"], r["metric"]): r for r in key_rows}
    raw = {(r["profile"], r["kernel_short"], r["metric"]): r for r in raw_rows}
    stages = ["cp_1_t_precompute", "cp_2_mn_precompute", "cp_3_fixup", "cp_4_prefill"]
    stage_label = {
        "cp_1_t_precompute": "T precompute",
        "cp_2_mn_precompute": "M/N precompute",
        "cp_3_fixup": "fixup",
        "cp_4_prefill": "final prefill",
    }
    fields = [
        "profile", "dtype", "seq", "cp_chunk_len", "partitions_per_head", "stage", "stage_label",
        "grid_size", "block_size", "duration_us", "duration_share_pct", "registers_per_thread",
        "dynamic_smem_kib", "waves_per_sm", "theoretical_occupancy_pct", "achieved_occupancy_pct",
        "sm_throughput_pct", "dram_throughput_pct", "l1tex_throughput_pct", "l2_throughput_pct",
        "tensor_active_pct", "eligible_warps_per_scheduler", "no_eligible_pct",
        "dram_read_mb", "dram_write_mb", "stall_barrier", "stall_long_scoreboard",
        "stall_short_scoreboard", "stall_wait", "stall_not_selected", "stall_math_pipe_throttle",
    ]
    out_rows = []

    def key_num(profile: str, stage: str, metric: str) -> float:
        row = key.get((profile, stage, metric))
        return float(row["numeric_value"]) if row and row.get("numeric_value") else float("nan")

    def raw_num(profile: str, stage: str, metric: str) -> float:
        row = raw.get((profile, stage, metric))
        return float(row["numeric_value"]) if row and row.get("numeric_value") else float("nan")

    def raw_mb(profile: str, stage: str, metric: str) -> float:
        row = raw.get((profile, stage, metric))
        if not row or not row.get("numeric_value"):
            return float("nan")
        value = float(row["numeric_value"])
        unit = row.get("unit", "").lower()
        if "mbyte" in unit:
            return value
        if "kbyte" in unit:
            return value / 1e3
        return value / 1e6

    for dtype in ("fp16", "bf16"):
        for seq, cp_chunk_len in ((8192, 1024), (16384, 2048), (32768, 4096)):
            profile = f"cp_s{seq}_{dtype}"
            durations = {stage: key_num(profile, stage, "Duration") for stage in stages}
            total = sum(x for x in durations.values() if math.isfinite(x))
            for stage in stages:
                launch = key.get((profile, stage, "Duration"), {})
                duration = durations[stage]
                out_rows.append({
                    "profile": profile,
                    "dtype": dtype.upper(),
                    "seq": seq,
                    "cp_chunk_len": cp_chunk_len,
                    "partitions_per_head": 8,
                    "stage": stage,
                    "stage_label": stage_label[stage],
                    "grid_size": launch.get("grid_size", ""),
                    "block_size": launch.get("block_size", ""),
                    "duration_us": duration,
                    "duration_share_pct": duration / total * 100 if total else float("nan"),
                    "registers_per_thread": key_num(profile, stage, "Registers Per Thread"),
                    "dynamic_smem_kib": key_num(profile, stage, "Dynamic Shared Memory Per Block") / 1024,
                    "waves_per_sm": key_num(profile, stage, "Waves Per SM"),
                    "theoretical_occupancy_pct": key_num(profile, stage, "Theoretical Occupancy"),
                    "achieved_occupancy_pct": key_num(profile, stage, "Achieved Occupancy"),
                    "sm_throughput_pct": raw_num(profile, stage, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                    "dram_throughput_pct": key_num(profile, stage, "DRAM Throughput"),
                    "l1tex_throughput_pct": key_num(profile, stage, "L1/TEX Cache Throughput"),
                    "l2_throughput_pct": key_num(profile, stage, "L2 Cache Throughput"),
                    "tensor_active_pct": raw_num(profile, stage, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                    "eligible_warps_per_scheduler": key_num(profile, stage, "Eligible Warps Per Scheduler"),
                    "no_eligible_pct": key_num(profile, stage, "No Eligible"),
                    "dram_read_mb": raw_mb(profile, stage, "dram__bytes_read.sum"),
                    "dram_write_mb": raw_mb(profile, stage, "dram__bytes_write.sum"),
                    "stall_barrier": raw_num(profile, stage, "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
                    "stall_long_scoreboard": raw_num(profile, stage, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
                    "stall_short_scoreboard": raw_num(profile, stage, "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
                    "stall_wait": raw_num(profile, stage, "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
                    "stall_not_selected": raw_num(profile, stage, "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"),
                    "stall_math_pipe_throttle": raw_num(profile, stage, "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"),
                })
    with (ROOT / "tables" / "cp_ncu_matrix_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    rows = load_bench()
    write_bench_summary(rows)
    write_ncu_summary()
    write_sass_summary()
    write_profile_configurations()
    write_cp_matrix_summary()
    print(f"bench_rows={len(rows)}")


if __name__ == "__main__":
    main()
