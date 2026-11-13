#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import math
import pathlib
import statistics
import subprocess


EXP = pathlib.Path(
    "/home/ubuntu/workspace/dyy/dyy_cutlass/"
    "dyy_test/op1/results/sf_mainloop_fusion_validation_20260727"
)
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

METRICS = {
    "duration_ms": "gpu__time_duration.sum",
    "barrier_count": "launch__barrier_count",
    "shared_mem_per_block_kb": "launch__shared_mem_per_block",
    "shared_mem_allocated_kb": "launch__shared_mem_per_block_allocated",
    "registers_per_thread": "launch__registers_per_thread",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "active_warps_per_scheduler": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "long_scoreboard_per_issue": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "barrier_stall_per_issue": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ),
    "mio_throttle_per_issue": (
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
    ),
    "wait_per_issue": (
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
    ),
    "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
    "dram_gbyte_per_s": "dram__bytes.avg.per_second",
}

INSTRUCTION_METRICS = {
    "tma_load_instructions": "sm__sass_inst_executed_op_tma_ld.sum",
    "utccp_tmem_write_bytes": "sm__mem_tensor_writes_op_utccp.sum",
    "tma_load_bytes": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "sm_instructions": "sm__inst_executed.sum",
}


def import_report(path: pathlib.Path) -> tuple[dict[str, str], dict[str, str]]:
    output = subprocess.check_output(
        [NCU, "--import", path, "--page", "raw", "--csv"],
        text=True,
    )
    rows = list(csv.reader(io.StringIO(output)))
    header = rows[0]
    units = dict(zip(header, rows[1]))
    values = dict(zip(header, rows[2]))
    return values, units


def main() -> None:
    rows: list[dict[str, object]] = []
    for variant in ("fused", "split"):
        values, units = import_report(EXP / "ncu" / f"{variant}.ncu-rep")
        inst_values, inst_units = import_report(
            EXP / "ncu" / f"{variant}_inst.ncu-rep"
        )
        row: dict[str, object] = {"variant": variant}
        for label, metric in METRICS.items():
            row[label] = values[metric]
            row[f"{label}_unit"] = units[metric]
        for label, metric in INSTRUCTION_METRICS.items():
            row[label] = inst_values[metric]
            row[f"{label}_unit"] = inst_units[metric]
        row["no_eligible_pct"] = 100.0 - float(row["issue_active_pct"])
        rows.append(row)

    output = EXP / "ncu_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(output)

    with (EXP / "event_raw.csv").open(newline="") as stream:
        event_rows = list(csv.DictReader(stream))
    pairs = []
    for repeat in range(1, 11):
        values = {
            row["variant"]: float(row["mean_ms"])
            for row in event_rows
            if int(row["repeat"]) == repeat
        }
        pairs.append(values)
    deltas = [pair["split"] - pair["fused"] for pair in pairs]
    delta_mean = statistics.mean(deltas)
    delta_stdev = statistics.stdev(deltas)
    ci95_half = 2.262 * delta_stdev / math.sqrt(len(deltas))
    fused_mean = statistics.mean(pair["fused"] for pair in pairs)
    paired_summary = {
        "pairs": len(pairs),
        "mean_delta_ms_split_minus_fused": delta_mean,
        "stdev_delta_ms": delta_stdev,
        "ci95_half_ms": ci95_half,
        "ci95_low_ms": delta_mean - ci95_half,
        "ci95_high_ms": delta_mean + ci95_half,
        "mean_delta_percent": delta_mean / fused_mean * 100.0,
    }
    with (EXP / "paired_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_summary.keys())
        writer.writeheader()
        writer.writerow(paired_summary)


if __name__ == "__main__":
    main()
