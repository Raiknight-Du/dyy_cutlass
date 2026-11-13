#!/usr/bin/env python3
"""Extract compact native GDN NCU and SASS evidence tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DETAIL_METRICS = {
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

RAW_METRICS = {
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


def short_kernel(name: str) -> str:
    if "GatedDeltaNetChunkedKernel" in name:
        return "recurrent_tcgen05"
    if "CPDeltaRuleTPrecomputeSm100" in name:
        return "cp_0_t_precompute"
    if "CPDeltaRuleMNPrecompute" in name:
        return "cp_1_mn_precompute"
    if "CPDeltaRuleFixup" in name:
        return "cp_2_fixup"
    if "CPDeltaRulePrefill" in name:
        return "cp_3_final"
    if "gdn_recurrent_kernel" in name:
        return "recurrent"
    if "gdn_cp_affine_kernel" in name:
        return "cp_1_affine_mn"
    if "gdn_cp_fixup_kernel" in name:
        return "cp_2_fixup"
    if "gdn_cp_final_kernel" in name:
        return "cp_3_final"
    return name[:96]


def parse_number(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def details(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_details.csv")):
        profile = path.name.removesuffix("_details.csv")
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            for row in csv.DictReader(stream):
                if row.get("Metric Name") not in DETAIL_METRICS:
                    continue
                rows.append(
                    {
                        "profile": profile,
                        "id": row.get("ID", ""),
                        "kernel_short": short_kernel(row.get("Kernel Name", "")),
                        "kernel": row.get("Kernel Name", ""),
                        "block_size": row.get("Block Size", ""),
                        "grid_size": row.get("Grid Size", ""),
                        "section": row.get("Section Name", ""),
                        "metric": row.get("Metric Name", ""),
                        "unit": row.get("Metric Unit", ""),
                        "value": row.get("Metric Value", ""),
                        "numeric_value": parse_number(row.get("Metric Value", "")),
                    }
                )
    return rows


def raw(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_raw.csv")):
        profile = path.name.removesuffix("_raw.csv")
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            source = list(csv.DictReader(stream))
        if len(source) < 2:
            continue
        units = source[0]
        for row in source[1:]:
            if not row.get("Kernel Name"):
                continue
            for metric in sorted(RAW_METRICS):
                value = row.get(metric, "")
                if value == "":
                    continue
                rows.append(
                    {
                        "profile": profile,
                        "id": row.get("ID", ""),
                        "kernel_short": short_kernel(row.get("Kernel Name", "")),
                        "kernel": row.get("Kernel Name", ""),
                        "block_size": row.get("Block Size", ""),
                        "grid_size": row.get("Grid Size", ""),
                        "metric": metric,
                        "unit": units.get(metric, ""),
                        "value": value,
                        "numeric_value": parse_number(value),
                    }
                )
    return rows


def sass(root: Path) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for path in sorted(root.glob("*_sass.csv")):
        profile = path.name.removesuffix("_sass.csv")
        kernel = ""
        header: list[str] = []
        index: dict[str, int] = {}
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                if row[0] == "Kernel Name":
                    kernel = short_kernel(row[1])
                    header, index = [], {}
                    continue
                if row[0] == "Address":
                    header = row
                    index = {name: i for i, name in enumerate(row)}
                    continue
                if not kernel or not header:
                    continue
                source = row[index.get("Source", 1)].strip()
                source = re.sub(r"^@[!A-Z0-9.]+\s+", "", source)
                opcode = source.split()[0] if source.split() else ""
                if not re.search(
                    r"^(FFMA|FADD|FMUL|HMMA|UTC|UTMA|TMEM|LDG|STG|LDS|STS|BAR|MEMBAR)",
                    opcode,
                    re.IGNORECASE,
                ):
                    continue
                try:
                    count = float(row[index["Instructions Executed"]] or 0)
                except (KeyError, ValueError, IndexError):
                    count = 0.0
                totals[(profile, kernel, opcode)] += count
    return [
        {
            "profile": profile,
            "kernel_short": kernel,
            "opcode": opcode,
            "instructions_executed": int(count),
        }
        for (profile, kernel, opcode), count in sorted(totals.items())
    ]


def cp_summary(
    detail_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    dkey = {
        (x["profile"], x["kernel_short"], x["metric"]): x for x in detail_rows
    }
    rkey = {(x["profile"], x["kernel_short"], x["metric"]): x for x in raw_rows}
    profiles = sorted({x["profile"] for x in detail_rows if "_cp_" in x["profile"]})

    def value(table: dict, profile: str, stage: str, metric: str) -> float:
        row = table.get((profile, stage, metric))
        value = row.get("numeric_value") if row else None
        return float(value) if value is not None else float("nan")

    def detail_kib(profile: str, stage: str, metric: str) -> float:
        row = dkey.get((profile, stage, metric))
        if not row or row.get("numeric_value") is None:
            return float("nan")
        result = float(row["numeric_value"])
        unit = str(row.get("unit", "")).lower()
        if "mbyte" in unit:
            return result * 1024.0
        if "kbyte" in unit:
            return result
        if "byte" in unit:
            return result / 1024.0
        return result

    def detail_duration_ms(profile: str, stage: str) -> float:
        row = dkey.get((profile, stage, "Duration"))
        if not row or row.get("numeric_value") is None:
            return float("nan")
        result = float(row["numeric_value"])
        unit = str(row.get("unit", "")).lower()
        if unit in ("ns", "nsecond", "nseconds"):
            return result / 1.0e6
        if unit in ("us", "usecond", "useconds"):
            return result / 1.0e3
        if unit in ("s", "second", "seconds"):
            return result * 1.0e3
        return result

    result: list[dict[str, Any]] = []
    for profile in profiles:
        stages = (
            (
                "cp_0_t_precompute",
                "cp_1_mn_precompute",
                "cp_2_fixup",
                "cp_3_final",
            )
            if profile.startswith("tcgen05_")
            else ("cp_1_affine_mn", "cp_2_fixup", "cp_3_final")
        )
        durations = {stage: detail_duration_ms(profile, stage) for stage in stages}
        total = sum(v for v in durations.values() if math.isfinite(v))
        for stage in stages:
            launch = dkey.get((profile, stage, "Duration"), {})
            duration = durations[stage]
            result.append(
                {
                    "profile": profile,
                    "kernel_short": stage,
                    "grid_size": launch.get("grid_size", ""),
                    "block_size": launch.get("block_size", ""),
                    "duration_ms": duration,
                    "duration_share_pct": duration / total * 100 if total else float("nan"),
                    "registers_per_thread": value(dkey, profile, stage, "Registers Per Thread"),
                    "dynamic_smem_kib": detail_kib(profile, stage, "Dynamic Shared Memory Per Block"),
                    "waves_per_sm": value(dkey, profile, stage, "Waves Per SM"),
                    "theoretical_occupancy_pct": value(dkey, profile, stage, "Theoretical Occupancy"),
                    "achieved_occupancy_pct": value(dkey, profile, stage, "Achieved Occupancy"),
                    "sm_throughput_pct": value(rkey, profile, stage, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                    "dram_throughput_pct": value(dkey, profile, stage, "DRAM Throughput"),
                    "l1tex_throughput_pct": value(dkey, profile, stage, "L1/TEX Cache Throughput"),
                    "tensor_active_pct": value(rkey, profile, stage, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                    "eligible_warps_per_scheduler": value(dkey, profile, stage, "Eligible Warps Per Scheduler"),
                    "no_eligible_pct": value(dkey, profile, stage, "No Eligible"),
                    "stall_barrier": value(rkey, profile, stage, "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
                    "stall_long_scoreboard": value(rkey, profile, stage, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
                    "stall_wait": value(rkey, profile, stage, "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ncu_dir = args.ncu_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    drows, rrows, srows = details(ncu_dir), raw(ncu_dir), sass(ncu_dir)
    cprows = cp_summary(drows, rrows)
    write_csv(out / "ncu_key_metrics.csv", drows, list(drows[0]))
    write_csv(out / "ncu_raw_counters.csv", rrows, list(rrows[0]))
    write_csv(out / "sass_instruction_summary.csv", srows, list(srows[0]))
    write_csv(out / "ncu_cp_stage_summary.csv", cprows, list(cprows[0]))
    for name, rows in (
        ("ncu_key_metrics", drows),
        ("ncu_raw_counters", rrows),
        ("sass_instruction_summary", srows),
        ("ncu_cp_stage_summary", cprows),
    ):
        (out / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"detail_rows={len(drows)} raw_rows={len(rrows)} "
        f"sass_rows={len(srows)} cp_rows={len(cprows)}"
    )


if __name__ == "__main__":
    main()
