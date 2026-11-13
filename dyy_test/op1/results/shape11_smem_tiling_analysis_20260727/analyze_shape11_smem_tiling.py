#!/usr/bin/env python3
import csv
import pathlib
import re


HERE = pathlib.Path(__file__).resolve().parent
NCU = HERE / "ncu"

RAW_METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "grid_size": "launch__grid_size",
    "waves": "launch__waves_per_multiprocessor",
    "registers": "launch__registers_per_thread",
    "smem_kb": "launch__shared_mem_per_block",
    "l2_mb": "lts__t_bytes.sum",
    "input_tma_mb": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "d_tma_mb": "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
    "input_tma_inst": "sm__sass_inst_executed_op_tma_ld.sum",
    "d_tma_inst": "sm__sass_inst_executed_op_tma_st.sum",
    "tensor_inst": "sm__inst_executed_pipe_tensor.sum",
    "tmem_inst": "sm__inst_executed_pipe_tmem.sum",
    "sm_inst": "sm__inst_executed.sum",
    "active_warps": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
}
DETAIL_METRICS = {
    "no_eligible_pct": ("Scheduler Statistics", "No Eligible"),
    "achieved_occupancy_pct": ("Occupancy", "Achieved Occupancy"),
    "block_limit_smem": ("Occupancy", "Block Limit Shared Mem"),
}


def number(value):
    value = value.replace(",", "").strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_raw(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        units = next(reader)
        values = next(reader)
    row = dict(zip(header, values))
    unit = dict(zip(header, units))
    result = {}
    for name, metric in RAW_METRICS.items():
        result[name] = number(row[metric])
        result[f"{name}_unit"] = unit.get(metric, "")
    kernel_name = row["Kernel Name"]
    match = re.search(
        r"MainloopSm100TmaUmmaWarpSpecializedBlockScaled<"
        r"(?:\(int\))?(\d+)",
        kernel_name,
    )
    result["stage"] = int(match.group(1)) if match else -1
    return result


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
    rows = []
    for raw_path in sorted(NCU.glob("cfg*_raw.csv")):
        stem = raw_path.name.removesuffix("_raw.csv")
        match = re.match(r"cfg(\d+)_(.+)", stem)
        row = {
            "config_index": int(match.group(1)),
            "config": match.group(2),
        }
        row.update(read_raw(raw_path))
        row.update(read_details(NCU / f"{stem}_details.csv"))
        rows.append(row)

    event = {}
    with (HERE / "event_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            event[int(row["config_index"])] = row
    for row in rows:
        timing = event[row["config_index"]]
        row["event_us"] = float(timing["mean_us"])
        row["event_delta_pct"] = float(timing["delta_pct"])

    rows.sort(key=lambda row: (row["config_index"] != 78, row["config_index"]))
    with (HERE / "ncu_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
