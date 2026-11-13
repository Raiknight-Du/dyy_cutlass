#!/usr/bin/env python3
import csv
import pathlib
import re


HERE = pathlib.Path(__file__).resolve().parent
NCU = HERE / "ncu"
LABELS = ("auto8", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8")

RAW_METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "waves": "launch__waves_per_multiprocessor",
    "registers": "launch__registers_per_thread",
    "smem_total_kbyte": "launch__shared_mem_per_block",
    "smem_dynamic_kbyte": "launch__shared_mem_per_block_dynamic",
    "active_cta_cycles": "sm__ctas_active.avg",
    "elapsed_cycles": "sm__cycles_elapsed.avg",
    "sm_inst": "sm__inst_executed.sum",
    "active_warps": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
}
DETAIL_METRICS = {
    "no_eligible_pct": ("Scheduler Statistics", "No Eligible"),
    "block_limit_smem": ("Occupancy", "Block Limit Shared Mem"),
    "block_limit_registers": ("Occupancy", "Block Limit Registers"),
}


def number(value):
    value = value.replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_raw(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        next(reader)
        values = next(reader)
    raw = dict(zip(header, values))
    row = {name: number(raw[metric]) for name, metric in RAW_METRICS.items()}
    match = re.search(
        r"MainloopSm100TmaUmmaWarpSpecializedBlockScaled<"
        r"(?:\(int\))?(\d+)",
        raw["Kernel Name"],
    )
    row["stage"] = int(match.group(1)) if match else -1
    row["avg_active_cta_per_sm"] = (
        row["active_cta_cycles"] / row["elapsed_cycles"]
    )
    return row


def read_details(path):
    metrics = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            metrics[(row["Section Name"], row["Metric Name"])] = number(
                row["Metric Value"]
            )
    return {
        name: metrics.get(key, float("nan"))
        for name, key in DETAIL_METRICS.items()
    }


def main():
    event = {}
    with (HERE / "event_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            event[row["label"].lower()] = row

    rows = []
    for label in LABELS:
        row = {"label": label}
        row.update(read_raw(NCU / f"{label}_raw.csv"))
        row.update(read_details(NCU / f"{label}_details.csv"))
        row["event_us"] = float(event[label]["mean_us"])
        row["event_ci95_us"] = float(event[label]["ci95_half_us"])
        row["event_delta_pct"] = float(event[label]["delta_pct"])
        rows.append(row)

    with (HERE / "ncu_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
