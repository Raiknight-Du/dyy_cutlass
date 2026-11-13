#!/usr/bin/env python3
import csv
import math
import pathlib
import re


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OUT = ROOT / "dyy_test/op1/results/bottleneck_13shape_20260724"
CONFIG_CSV = ROOT / "dyy_test/op1/results/op1_best_configs.csv"

DIMS = {
    1: (170100, 13824, 5120),
    2: (510300, 13824, 5120),
    3: (170100, 5120, 5120),
    4: (510300, 5120, 5120),
    5: (8192, 768, 4096),
    6: (16384, 768, 4096),
    7: (32768, 768, 4096),
    8: (8192, 4096, 2048),
    9: (16384, 4096, 2048),
    10: (32768, 4096, 128),
    11: (16384, 768, 128),
    12: (32768, 768, 128),
    13: (8192, 4096, 128),
}

METRICS = {
    "ncu_ms": "gpu__time_duration.sum",
    "sm_sol_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_mem_sol_pct":
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sol_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "dram_TBs": "dram__bytes.sum.per_second",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "tensor_active_pct":
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tmem_active_pct":
        "sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tma_active_pct":
        "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "active_warps_sched": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "achieved_occupancy_pct":
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    "regs_thread": "launch__registers_per_thread",
    "smem_KiB": "launch__shared_mem_per_block",
    "block_limit_smem": "launch__occupancy_limit_shared_mem",
    "block_limit_regs": "launch__occupancy_limit_registers",
    "gpc_clock_GHz": "gpc__cycles_elapsed.avg.per_second",
    "stall_long_sb": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "stall_barrier": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ),
    "stall_wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "stall_short_sb": (
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
    ),
    "stall_mio": (
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
    ),
    "stall_math": (
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
    ),
    "stall_not_selected": (
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
    ),
    "stall_no_inst": (
        "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio"
    ),
}


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def duration_ms(value, unit):
    result = as_float(value)
    if unit == "s":
        return result * 1000.0
    if unit == "us":
        return result / 1000.0
    if unit == "ns":
        return result / 1_000_000.0
    return result


def bandwidth_TBs(value, unit):
    result = as_float(value)
    if unit == "Gbyte/s":
        return result / 1000.0
    if unit == "Mbyte/s":
        return result / 1_000_000.0
    return result


def read_config_rows():
    with CONFIG_CSV.open(newline="") as stream:
        return {int(row["shape"]): row for row in csv.DictReader(stream)}


def main():
    configs = read_config_rows()
    output_rows = []
    for shape in range(1, 14):
        path = OUT / f"shape{shape:02d}_raw.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
            units = rows[0]
            raw = rows[-1]
        m, n, k = DIMS[shape]
        config = configs[shape]
        variant = config["variant"]
        ktile = int(variant.split("_K", 1)[1].split("_", 1)[0])
        cluster_match = re.search(r"_C(\d+)x(\d+)x(\d+)", variant)
        cluster_size = math.prod(int(value) for value in cluster_match.groups())
        grid = tuple(int(value) for value in re.findall(r"\d+", raw["Grid Size"]))
        blocks = math.prod(grid)
        stage_match = re.search(
            r"WarpSpecializedBlockScaled<(\d+),\s*(\d+),\s*(\d+)",
            raw["Kernel Name"],
        )
        nsys_path = OUT / f"shape{shape:02d}_nsys_stats.csv"
        nsys_lines = nsys_path.read_text().splitlines()
        nsys_header = next(
            index
            for index, line in enumerate(nsys_lines)
            if line.startswith("Time (%),Total Time (ns),Instances")
        )
        nsys_row = next(csv.reader([nsys_lines[nsys_header + 1]]))
        min_bytes = (m * k + k * n) * 0.5625 + m * n * 2.0
        row = {
            "shape": shape,
            "M": m,
            "N": n,
            "K": k,
            "variant": variant,
            "mainloop_stages": int(stage_match.group(1)),
            "K_iterations": math.ceil(k / ktile),
            "event_mean_ms": float(config["mean_ms"]),
            "event_ci95_ms": float(config["ci95_half_ms"]),
            "logical_TFLOPs": 2.0 * m * n * k / float(config["mean_ms"]) / 1e9,
            "minimum_main_bytes_GiB": min_bytes / 1024**3,
            "minimum_AI_FLOP_B": 2.0 * m * n * k / min_bytes,
            "grid": raw["Grid Size"],
            "block": raw["Block Size"],
            "blocks": blocks,
            "cluster_size": cluster_size,
            "block_waves_148SM": blocks / 148.0,
            "last_block_wave_util_pct": (
                100.0 if blocks % 148 == 0 else (blocks % 148) / 148.0 * 100.0
            ),
            "nsys_kernel_mean_ms": float(nsys_row[3]) / 1e6,
            "nsys_kernel_median_ms": float(nsys_row[4]) / 1e6,
            "nsys_kernel_min_ms": float(nsys_row[5]) / 1e6,
            "nsys_kernel_max_ms": float(nsys_row[6]) / 1e6,
        }
        for label, metric in METRICS.items():
            row[label] = as_float(raw.get(metric))
        row["ncu_ms"] = duration_ms(
            raw.get(METRICS["ncu_ms"]), units.get(METRICS["ncu_ms"])
        )
        row["dram_TBs"] = bandwidth_TBs(
            raw.get(METRICS["dram_TBs"]), units.get(METRICS["dram_TBs"])
        )
        row["no_eligible_pct"] = 100.0 - row["issue_pct"]
        output_rows.append(row)

    output = OUT / "ncu_core_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    display_columns = [
        "shape", "event_mean_ms", "logical_TFLOPs", "K_iterations",
        "mainloop_stages", "block_waves_148SM", "last_block_wave_util_pct",
        "sm_sol_pct", "tensor_active_pct", "tmem_active_pct", "dram_sol_pct",
        "dram_TBs", "l2_sol_pct", "l2_hit_pct", "achieved_occupancy_pct",
        "active_warps_sched", "eligible_warps_sched", "no_eligible_pct",
        "stall_long_sb", "stall_barrier", "stall_wait", "smem_KiB",
        "block_limit_smem",
    ]
    print(",".join(display_columns))
    for row in output_rows:
        values = []
        for column in display_columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print(",".join(values))


if __name__ == "__main__":
    main()
