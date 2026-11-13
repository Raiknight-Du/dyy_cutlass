#!/usr/bin/env python3
import csv
import math
import pathlib
import re


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/bottleneck_shape5_8_11_13_20260724"
NCU_DIR = EXP / "ncu"

DIMS = {
    5: (8192, 768, 4096),
    8: (8192, 4096, 2048),
    11: (16384, 768, 128),
    13: (8192, 4096, 128),
}

METRICS = {
    "ncu_ms": "gpu__time_duration.sum",
    "sm_sol_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_sol_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_sol_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "tensor_active_pct":
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tmem_active_pct":
        "sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "shared_active_pct":
        "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tma_active_pct":
        "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "active_warps_sched": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "regs_thread": "launch__registers_per_thread",
    "smem_KiB": "launch__shared_mem_per_block",
    "block_limit_smem": "launch__occupancy_limit_shared_mem",
    "stall_long_sb":
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_barrier":
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "stall_wait":
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "stall_mio":
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
}


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def duration_ms(value, unit):
    number = as_float(value)
    return {
        "s": number * 1000.0,
        "ms": number,
        "us": number / 1000.0,
        "ns": number / 1_000_000.0,
    }.get(unit, number)


def main():
    with (EXP / "event_summary.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    output_rows = []
    for event in events:
        shape = int(event["shape"])
        config = int(event["config_index"])
        label = event["label"]
        stem = f"shape{shape:02d}_{label}_c{config:02d}"
        with (NCU_DIR / f"{stem}_raw.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        units = rows[0]
        raw = rows[-1]
        variant = event["variant"]
        tile_match = re.search(r"M(\d+)_N(\d+)_K(\d+)", variant)
        cluster_match = re.search(r"_C(\d+)x(\d+)x(\d+)", variant)
        tile_m, tile_n, tile_k = map(int, tile_match.groups())
        cluster = tuple(map(int, cluster_match.groups()))
        grid = tuple(map(int, re.findall(r"\d+", raw["Grid Size"])))
        blocks = math.prod(grid)
        stage_match = re.search(
            r"WarpSpecializedBlockScaled<(\d+),\s*(\d+),\s*(\d+)",
            raw["Kernel Name"],
        )
        m, n, k = DIMS[shape]
        row = {
            "shape": shape,
            "label": label,
            "config_index": config,
            "variant": variant,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "cluster": "x".join(map(str, cluster)),
            "stages": int(stage_match.group(1)),
            "k_iterations": math.ceil(k / tile_k),
            "event_mean_ms": float(event["mean_ms"]),
            "event_ci95_ms": float(event["ci95_half_ms"]),
            "grid": raw["Grid Size"],
            "blocks": blocks,
            "waves_148sm": blocks / 148.0,
            "last_wave_util_pct": (
                100.0 if blocks % 148 == 0 else blocks % 148 / 148.0 * 100.0
            ),
        }
        for label_name, metric in METRICS.items():
            row[label_name] = as_float(raw.get(metric))
        row["ncu_ms"] = duration_ms(
            raw.get(METRICS["ncu_ms"]), units.get(METRICS["ncu_ms"])
        )
        row["no_eligible_pct"] = 100.0 - row["issue_pct"]
        output_rows.append(row)

    output = EXP / "comparison_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(output.read_text())


if __name__ == "__main__":
    main()
