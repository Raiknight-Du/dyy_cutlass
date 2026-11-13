#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import re
import subprocess


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/packed_sf_tma_validation_20260727"
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

METRICS = {
    "gpu__time_duration.sum": "duration_ms",
    "sm__sass_inst_executed_op_tma_ld.sum": "tma_load_instructions",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum": "tma_load_bytes",
    "sm__inst_executed.sum": "sm_instructions",
    "smsp__warps_active.avg.per_cycle_active": "active_warps_per_scheduler",
    "smsp__warps_eligible.avg.per_cycle_active": "eligible_warps_per_scheduler",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_active_pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":
        "long_scoreboard_per_issue",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio":
        "barrier_stall_per_issue",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio":
        "mio_throttle_per_issue",
}


def to_float(value: str) -> float:
    return float(value.replace(",", ""))


def convert(value: float, unit: str, field: str) -> float:
    if field == "duration_ms":
        return {
            "ns": value / 1e6,
            "us": value / 1e3,
            "ms": value,
            "s": value * 1e3,
        }.get(unit, value)
    if field == "tma_load_bytes":
        return {
            "byte": value / 1e9,
            "Kbyte": value / 1e6,
            "Mbyte": value / 1e3,
            "Gbyte": value,
        }.get(unit, value)
    return value


def export(report: pathlib.Path) -> list[list[str]]:
    result = subprocess.run(
        [
            NCU,
            "--import",
            report,
            "--csv",
            "--page",
            "raw",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [
        line for line in result.stdout.splitlines()
        if line and not line.startswith("==")
    ]
    return list(csv.reader(lines))


def parse(report: pathlib.Path) -> dict[str, object]:
    rows = export(report)
    result: dict[str, object] = {}
    metric_table = next(
        (
            i for i, row in enumerate(rows)
            if "Metric Name" in row and "Metric Value" in row
        ),
        None,
    )
    if metric_table is not None:
        header = rows[metric_table]
        name_index = header.index("Metric Name")
        value_index = header.index("Metric Value")
        unit_index = header.index("Metric Unit")
        for row in rows[metric_table + 1:]:
            if len(row) <= max(name_index, value_index, unit_index):
                continue
            metric = row[name_index]
            if metric not in METRICS:
                continue
            field = METRICS[metric]
            value = convert(to_float(row[value_index]), row[unit_index], field)
            result[field] = value
        return result

    # Custom metric sets are exported as one wide CSV row:
    # header, units, values.
    header_index = next(
        i for i, row in enumerate(rows)
        if "gpu__time_duration.sum" in row
    )
    header = rows[header_index]
    units = rows[header_index + 1]
    values = rows[header_index + 2]
    for metric, field in METRICS.items():
        if metric not in header:
            continue
        index = header.index(metric)
        result[field] = convert(
            to_float(values[index]), units[index], field
        )
    return result


def main() -> None:
    event_rows = {}
    with (EXP / "event_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            event_rows[row["variant"]] = row

    rows: list[dict[str, object]] = []
    for variant in ("separate", "packed"):
        row: dict[str, object] = {
            "variant": variant,
            "event_mean_ms": float(event_rows[variant]["mean_ms"]),
            "event_effective_tib_s":
                float(event_rows[variant]["mean_effective_tib_s"]),
            "expected_tma_loads":
                int(event_rows[variant]["expected_tma_loads"]),
        }
        row.update(parse(EXP / "ncu" / f"{variant}.ncu-rep"))
        if "active_warps_per_scheduler" in row:
            active = float(row["active_warps_per_scheduler"])
            eligible = float(row["eligible_warps_per_scheduler"])
            issue = float(row["issue_active_pct"])
            row["no_eligible_pct"] = 100.0 - issue
            row["eligible_over_active_pct"] = (
                eligible / active * 100.0 if active else 0.0
            )
        rows.append(row)

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (EXP / "ncu_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(EXP / "ncu_summary.csv")


if __name__ == "__main__":
    main()
