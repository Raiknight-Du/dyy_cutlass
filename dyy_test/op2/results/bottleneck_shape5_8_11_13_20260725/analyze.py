#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

SHAPES = {
    5: {
        "logical": (8192, 768, 4096),
        "transposed": (768, 8192, 8192),
        "tile": (256, 192, 256),
        "cluster": (2, 2, 1),
        "quick_ms": 0.017175936,
    },
    8: {
        "logical": (8192, 4096, 2048),
        "transposed": (4096, 8192, 4096),
        "tile": (256, 256, 256),
        "cluster": (2, 1, 1),
        "quick_ms": 0.042603904,
    },
    11: {
        "logical": (16384, 768, 128),
        "transposed": (768, 16384, 256),
        "tile": (256, 192, 256),
        "cluster": (2, 1, 1),
        "quick_ms": 0.008989824,
    },
    13: {
        "logical": (8192, 4096, 128),
        "transposed": (4096, 8192, 256),
        "tile": (256, 256, 256),
        "cluster": (2, 1, 1),
        "quick_ms": 0.015085824,
    },
}

FIELDS = {
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_compute_memory_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_bytes_per_s": "dram__bytes.sum.per_second",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "tc_active_pct": "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tma_active_pct": "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "shared_active_pct": "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "issue_active_pct": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "active_warps_per_sched": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "issued_warps_per_sched": "smsp__issue_active.avg.per_cycle_active",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "launch__occupancy_per_block_size",
    "cluster_occupancy_pct": "launch__occupancy_cluster_gpu_pct",
    "shared_mem_bytes": "launch__shared_mem_per_block",
    "registers_per_thread": "launch__registers_per_thread",
    "waves": "launch__waves_per_multiprocessor",
    "sm_count": "device__attribute_multiprocessor_count",
    "grid_ctas": "launch__grid_size",
    "block_threads": "launch__block_size",
    "inst_executed": "smsp__inst_executed.sum",
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "short_sb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "mio_throttle": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "not_selected": "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "no_instruction": "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
}

DETAIL_FIELDS = {
    "duration_us": ("GPU Speed Of Light Throughput", "Duration"),
    "sm_frequency_ghz": ("GPU Speed Of Light Throughput", "SM Frequency"),
    "memory_throughput_pct": ("GPU Speed Of Light Throughput", "Memory Throughput"),
    "compute_throughput_pct": (
        "GPU Speed Of Light Throughput", "Compute (SM) Throughput"
    ),
    "dram_throughput_pct_detail": (
        "GPU Speed Of Light Throughput", "DRAM Throughput"
    ),
    "l1_throughput_pct_detail": (
        "GPU Speed Of Light Throughput", "L1/TEX Cache Throughput"
    ),
    "l2_throughput_pct_detail": (
        "GPU Speed Of Light Throughput", "L2 Cache Throughput"
    ),
    "one_or_more_eligible_pct": ("Scheduler Statistics", "One or More Eligible"),
    "no_eligible_pct": ("Scheduler Statistics", "No Eligible"),
    "active_warps_per_sched_detail": (
        "Scheduler Statistics", "Active Warps Per Scheduler"
    ),
    "eligible_warps_per_sched_detail": (
        "Scheduler Statistics", "Eligible Warps Per Scheduler"
    ),
    "issued_warps_per_sched_detail": (
        "Scheduler Statistics", "Issued Warp Per Scheduler"
    ),
    "warp_cycles_per_issued": (
        "Warp State Statistics", "Warp Cycles Per Issued Instruction"
    ),
    "theoretical_active_warps_per_sm": (
        "Occupancy", "Theoretical Active Warps per SM"
    ),
    "theoretical_occupancy_pct": ("Occupancy", "Theoretical Occupancy"),
    "achieved_occupancy_pct_detail": ("Occupancy", "Achieved Occupancy"),
    "achieved_active_warps_per_sm": (
        "Occupancy", "Achieved Active Warps Per SM"
    ),
    "cluster_size": ("Launch Statistics", "Cluster Size"),
    "grid_ctas_detail": ("Launch Statistics", "Grid Size"),
    "registers_per_thread_detail": ("Launch Statistics", "Registers Per Thread"),
    "dynamic_smem_kbyte": (
        "Launch Statistics", "Dynamic Shared Memory Per Block"
    ),
    "sm_count_detail": ("Launch Statistics", "# SMs"),
    "waves_detail": ("Launch Statistics", "Waves Per SM"),
    "l2_hit_rate_pct": ("Memory Workload Analysis", "L2 Hit Rate"),
    "mem_busy_pct": ("Memory Workload Analysis", "Mem Busy"),
    "mem_pipes_busy_pct": ("Memory Workload Analysis", "Mem Pipes Busy"),
}


def load_raw(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header, units, values = rows[:3]
    return (
        dict(zip(header, values)),
        dict(zip(header, units)),
    )

def load_details(path: Path):
    metrics = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            section = row["Section Name"]
            metric = row["Metric Name"]
            if section and metric and row["Metric Value"]:
                metrics[(section, metric)] = {
                    "value": number(row["Metric Value"]),
                    "unit": row["Metric Unit"],
                }
    return metrics


def load_source(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header = rows[1]
    return [
        dict(zip(header, row))
        for row in rows[2:]
        if len(row) == len(header) and row[0].startswith("0x")
    ]


def load_traffic(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header_index = next(
        index for index, row in enumerate(rows)
        if row and row[0] == "ID"
    )
    header = rows[header_index]
    values = rows[header_index + 2]
    return dict(zip(header, values))


def number(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def first_present(row, candidates):
    for name in candidates:
        if name in row and row[name] != "":
            return number(row[name])
    return None


def summarize(shape_id, config):
    row, units = load_raw(RAW / f"shape{shape_id:02d}_raw.csv")
    details = load_details(RAW / f"shape{shape_id:02d}_details.csv")
    out = {"shape": shape_id}
    for label, metric in FIELDS.items():
        out[label] = number(row.get(metric))
        if metric in units:
            out[f"{label}_unit"] = units[metric]

    out["grid"] = row.get("Grid Size")
    out["block"] = row.get("Block Size")
    out["kernel_name"] = row.get("Kernel Name")
    for label, key in DETAIL_FIELDS.items():
        item = details.get(key)
        out[label] = item["value"] if item else None
        out[f"{label}_unit"] = item["unit"] if item else None
    out["ncu_duration_ms"] = out["duration_us"] / 1000.0
    out["sm_frequency_hz"] = out["sm_frequency_ghz"] * 1.0e9
    out["shared_mem_bytes"] = out["dynamic_smem_kbyte"] * 1000.0
    out["waves"] = out["waves_detail"]
    out["registers_per_thread"] = out["registers_per_thread_detail"]
    out["grid_ctas"] = out["grid_ctas_detail"]

    m, n, k = config["logical"]
    mt, nt, kt = config["transposed"]
    tile_m, tile_n, tile_k = config["tile"]
    out["logical_m"] = m
    out["logical_n"] = n
    out["logical_k"] = k
    out["transposed_m"] = mt
    out["transposed_n"] = nt
    out["transposed_k"] = kt
    out["tile_m"] = tile_m
    out["tile_n"] = tile_n
    out["tile_k"] = tile_k
    out["cluster"] = "x".join(str(v) for v in config["cluster"])
    out["k_iterations"] = math.ceil(kt / tile_k)
    out["m_tiles"] = math.ceil(mt / tile_m)
    out["n_tiles"] = math.ceil(nt / tile_n)
    out["logical_tile_count"] = out["m_tiles"] * out["n_tiles"]
    out["quick_ms"] = config["quick_ms"]
    out["effective_tflops_quick"] = 2.0 * m * n * k / (config["quick_ms"] * 1.0e9)
    out["effective_tflops_ncu"] = 2.0 * m * n * k / (out["ncu_duration_ms"] * 1.0e9)
    out["d_bytes"] = m * n * 2
    out["d_write_tib_s_quick"] = out["d_bytes"] / (
        config["quick_ms"] * 1.0e-3
    ) / 2**40
    traffic = load_traffic(RAW / f"shape{shape_id:02d}_traffic.csv")
    out["l2_read_sectors"] = number(traffic["lts__t_sectors_op_read.sum"])
    out["l2_write_sectors"] = number(traffic["lts__t_sectors_op_write.sum"])
    out["l2_hit_sectors"] = number(traffic["lts__t_sectors_lookup_hit.sum"])
    out["l2_miss_sectors"] = number(traffic["lts__t_sectors_lookup_miss.sum"])
    out["dram_read_bytes"] = number(traffic["dram__bytes_read.sum"])
    out["dram_write_bytes"] = number(traffic["dram__bytes_write.sum"])
    out["l2_read_bytes"] = out["l2_read_sectors"] * 32
    out["l2_write_bytes"] = out["l2_write_sectors"] * 32
    out["l2_total_bytes"] = out["l2_read_bytes"] + out["l2_write_bytes"]
    out["l2_counter_hit_rate_pct"] = 100.0 * out["l2_hit_sectors"] / (
        out["l2_hit_sectors"] + out["l2_miss_sectors"]
    )
    out["l2_counter_tib_s_quick"] = out["l2_total_bytes"] / (
        config["quick_ms"] * 1.0e-3
    ) / 2**40
    out["dram_counter_tib_s_quick"] = (
        out["dram_read_bytes"] + out["dram_write_bytes"]
    ) / (config["quick_ms"] * 1.0e-3) / 2**40
    return out


def source_top(shape_id):
    rows = load_source(RAW / f"shape{shape_id:02d}_source.csv")
    metrics = [
        "Warp Stall Sampling (All Samples)",
        "stall_long_sb",
        "stall_wait",
        "stall_short_sb",
        "stall_barrier",
        "stall_no_inst",
        "stall_mio",
    ]
    output = []
    for metric in metrics:
        values = []
        for row in rows:
            value = number(row.get(metric))
            if isinstance(value, (int, float)) and value > 0:
                values.append((value, row))
        total = sum(value for value, _ in values)
        for rank, (value, row) in enumerate(
            sorted(values, key=lambda item: item[0], reverse=True)[:12], start=1
        ):
            output.append({
                "shape": shape_id,
                "metric": metric,
                "rank": rank,
                "samples": value,
                "share_pct": 100.0 * value / total if total else 0.0,
                "address": row["Address"],
                "instruction": row["Source"].strip(),
                "executed": number(row["Instructions Executed"]),
            })
    return output


def main():
    summary = [summarize(shape_id, config) for shape_id, config in SHAPES.items()]
    (ROOT / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = [
        "shape", "quick_ms", "ncu_duration_ms", "effective_tflops_quick",
        "k_iterations", "grid", "waves", "sm_frequency_hz",
        "sm_throughput_pct", "dram_throughput_pct", "l2_throughput_pct",
        "tc_active_pct", "tma_active_pct", "shared_active_pct",
        "achieved_occupancy_pct", "active_warps_per_sched",
        "eligible_warps_per_sched", "no_eligible_pct",
        "issued_warps_per_sched", "long_sb", "short_sb", "barrier",
        "mio_throttle", "wait", "not_selected", "shared_mem_bytes",
        "registers_per_thread", "inst_executed", "d_write_tib_s_quick",
        "l2_read_bytes", "l2_write_bytes", "l2_counter_hit_rate_pct",
        "l2_counter_tib_s_quick", "dram_read_bytes", "dram_write_bytes",
        "dram_counter_tib_s_quick",
    ]
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in summary)

    source = []
    contexts = []
    key_sass = []
    instruction_mix = []
    region_rows = []
    for shape_id in SHAPES:
        shape_source = source_top(shape_id)
        source.extend(shape_source)
        top_long = next(
            row for row in shape_source
            if row["metric"] == "stall_long_sb" and row["rank"] == 1
        )
        sass_rows = load_source(RAW / f"shape{shape_id:02d}_source.csv")
        pc_index = next(
            index for index, row in enumerate(sass_rows)
            if row["Address"] == top_long["address"]
        )
        contexts.append(f"## shape {shape_id}: top long_sb {top_long['address']}")
        for row in sass_rows[max(0, pc_index - 16):pc_index + 9]:
            contexts.append(
                f"{row['Address']}  {row['Source'].strip():70s} "
                f"all={row['Warp Stall Sampling (All Samples)']:>5s} "
                f"long={row['stall_long_sb']:>5s} "
                f"wait={row['stall_wait']:>5s} "
                f"short={row['stall_short_sb']:>5s}"
            )
        contexts.append("")
        for row in sass_rows:
            instruction = row["Source"].strip()
            if any(token in instruction for token in (
                "UTMALDG", "UTCOMMA", "LDTM", "STSM", "UTMASTG",
                "TRYWAIT", "SYNCS.ARRIVE",
            )):
                key_sass.append({
                    "shape": shape_id,
                    "address": row["Address"],
                    "instruction": instruction,
                    "all_samples": number(
                        row["Warp Stall Sampling (All Samples)"]
                    ),
                    "long_sb": number(row["stall_long_sb"]),
                    "short_sb": number(row["stall_short_sb"]),
                    "wait": number(row["stall_wait"]),
                    "barrier": number(row["stall_barrier"]),
                    "executed": number(row["Instructions Executed"]),
                })
        instruction_tokens = {
            "TMA load": "UTMALDG",
            "UMMA": "UTCOMMA",
            "TMEM load": "LDTM",
            "SMEM store": "STSM",
            "TMA store": "UTMASTG",
            "mbarrier trywait": "SYNCS.PHASECHK.TRANS64.TRYWAIT",
        }
        for category, token in instruction_tokens.items():
            matching = [
                row for row in sass_rows if token in row["Source"]
            ]
            instruction_mix.append({
                "shape": shape_id,
                "category": category,
                "static_instructions": len(matching),
                "executed_instructions": sum(
                    number(row["Instructions Executed"]) or 0
                    for row in matching
                ),
                "all_samples": sum(
                    number(row["Warp Stall Sampling (All Samples)"]) or 0
                    for row in matching
                ),
            })
        region_defs = [
            ("setup", 0x0000, 0x1600),
            ("producer_tma", 0x1600, 0x3200),
            ("mma_consumer", 0x3200, 0x4C00),
            ("epilogue", 0x4C00, 0x6F00),
            ("pipeline_tail", 0x6F00, 0x10000),
        ]
        for region, begin, end in region_defs:
            selected = [
                row for row in sass_rows
                if begin <= (int(row["Address"], 16) & 0xFFFF) < end
            ]
            region_rows.append({
                "shape": shape_id,
                "region": region,
                "all_samples": sum(
                    number(row["Warp Stall Sampling (All Samples)"]) or 0
                    for row in selected
                ),
                "long_sb": sum(
                    number(row["stall_long_sb"]) or 0 for row in selected
                ),
                "short_sb": sum(
                    number(row["stall_short_sb"]) or 0 for row in selected
                ),
                "wait": sum(
                    number(row["stall_wait"]) or 0 for row in selected
                ),
                "barrier": sum(
                    number(row["stall_barrier"]) or 0 for row in selected
                ),
            })
    with (ROOT / "source_top.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "shape", "metric", "rank", "samples", "share_pct",
            "address", "instruction", "executed",
        ])
        writer.writeheader()
        writer.writerows(source)
    (ROOT / "source_context.txt").write_text(
        "\n".join(contexts), encoding="utf-8"
    )
    with (ROOT / "sass_key_instructions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "shape", "address", "instruction", "all_samples",
            "long_sb", "short_sb", "wait", "barrier", "executed",
        ])
        writer.writeheader()
        writer.writerows(key_sass)
    with (ROOT / "stall_regions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "shape", "region", "all_samples", "long_sb",
            "short_sb", "wait", "barrier",
        ])
        writer.writeheader()
        writer.writerows(region_rows)
    with (ROOT / "instruction_mix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "shape", "category", "static_instructions",
            "executed_instructions", "all_samples",
        ])
        writer.writeheader()
        writer.writerows(instruction_mix)

    for row in summary:
        print(json.dumps({key: row.get(key) for key in columns}, ensure_ascii=False))


if __name__ == "__main__":
    main()
