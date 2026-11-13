#!/usr/bin/env python3
import csv
import pathlib
import statistics


EXP = pathlib.Path(__file__).resolve().parent
NCU = EXP / "ncu"

SHAPES = {
    8: {
        "m": 8192,
        "n": 4096,
        "k": 2048,
        "config": "M256_N256_K256_E64x64_C2x1x1",
        "stage": 5,
        "k_rounds": 8,
        "grid_ctas": 1024,
        "waves": 6.9189189189,
    },
    11: {
        "m": 16384,
        "n": 768,
        "k": 128,
        "config": "M128_N192_K128_E128x64_C1x1x1",
        "stage": 8,
        "k_rounds": 1,
        "grid_ctas": 512,
        "waves": 3.4594594595,
    },
    13: {
        "m": 8192,
        "n": 4096,
        "k": 128,
        "config": "M128_N256_K128_E128x128_C1x1x1",
        "stage": 5,
        "k_rounds": 1,
        "grid_ctas": 1024,
        "waves": 6.9189189189,
    },
}

SEQUENCE = (
    ("full", "完整 pure matmul"),
    ("notc", "移除 UMMA"),
    ("notma", "移除 A/B/SFA/SFB 输入 TMA 及对应 barrier"),
    ("shell_utccp", "移除输入 TMA+UMMA，保留 UTCCP/epilogue"),
    ("shell_noutccp", "再移除 SFA/SFB SMEM-to-TMEM UTCCP"),
    ("shell_nodstore", "再移除 D SMEM-to-GMEM TMA store"),
    ("shell_noepi", "再移除 TMEM load、FP32-to-BF16 和 SMEM staging"),
)

COMPARISON = {
    "notc": "full",
    "notma": "full",
    "shell_noutccp": "shell_utccp",
    "shell_nodstore": "shell_noutccp",
    "shell_noepi": "shell_nodstore",
}

METRICS = {
    "ncu_duration_us": "gpu__time_duration.sum",
    "l2_bytes_MB": "lts__t_bytes.sum",
    "input_tma_bytes_MB": (
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
    ),
    "d_tma_bytes_MB": (
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum"
    ),
    "input_tma_inst": "sm__sass_inst_executed_op_tma_ld.sum",
    "d_tma_inst": "sm__sass_inst_executed_op_tma_st.sum",
    "tensor_inst": "sm__inst_executed_pipe_tensor.sum",
    "tmem_pipe_inst": "sm__inst_executed_pipe_tmem.sum",
    "tmem_ld_MB": "sm__mem_tensor_reads_op_ldt.sum",
    "utccp_MB": "sm__mem_tensor_writes_op_utccp.sum",
    "utcmma_MB": "sm__mem_tensor_writes_op_utcmma.sum",
    "sm_inst": "sm__inst_executed.sum",
    "registers_thread": "launch__registers_per_thread",
    "smem_KB": "launch__shared_mem_per_block",
    "no_eligible_pct": "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
    "eligible_warps_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "active_warps_sched": "smsp__warps_active.avg.per_cycle_active",
    "long_sb_per_issue": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "barrier_per_issue": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ),
    "wait_per_issue": (
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
    ),
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "ctas_active_accum": "sm__ctas_active.avg",
    "cycles_active": "sm__cycles_active.avg",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_event():
    with (EXP / "event_summary.csv").open(newline="", encoding="utf-8") as stream:
        return {
            (int(row["shape"]), row["variant"]): row
            for row in csv.DictReader(stream)
        }


def read_ncu(shape, variant):
    path = NCU / f"shape{shape}_{variant}_raw.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1]


def read_empty():
    rows = []
    with (EXP / "empty_resource_event_raw.txt").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("RESULT,"):
                continue
            _, variant, sample, iterations, mean_ms = line.strip().split(",")
            rows.append(
                {
                    "variant": variant,
                    "sample": int(sample),
                    "iterations": int(iterations),
                    "mean_us": float(mean_ms) * 1000,
                }
            )
    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        values = [row["mean_us"] for row in rows if row["variant"] == variant]
        stdev = statistics.stdev(values)
        summary.append(
            {
                "variant": variant,
                "mean_us": statistics.mean(values),
                "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                "min_us": min(values),
                "max_us": max(values),
            }
        )
    return summary


def read_source_hotspots(shape):
    path = NCU / f"shape{shape}_shell_noepi_source_cuda_sass.csv"
    current_file = ""
    current_source = None
    header = None
    aggregate = {}
    with path.open(newline="", encoding="utf-16", errors="replace") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] == "File Path":
                current_file = row[1]
                current_source = None
                header = None
                continue
            if row[0] == "Line No":
                header = row
                continue
            if header is None:
                continue
            if row[0].isdigit() and len(row) > 2 and row[2] == "-":
                current_source = (current_file, int(row[0]), row[1].strip())
                continue
            if current_source is None or len(row) < 4 or not row[2].startswith("0x"):
                continue
            samples = int(row[header.index("# Samples")] or 0)
            if samples == 0:
                continue
            not_issued = int(
                row[header.index("Warp Stall Sampling (Not-issued Samples)")] or 0
            )
            item = aggregate.setdefault(
                current_source,
                {"samples": 0, "not_issued": 0, "sass": set()},
            )
            item["samples"] += samples
            item["not_issued"] += not_issued
            item["sass"].add(row[3].strip())

    result = []
    for (file_path, line, source), item in aggregate.items():
        result.append(
            {
                "shape": shape,
                "file": file_path,
                "line": line,
                "source": source,
                "samples": item["samples"],
                "not_issued_samples": item["not_issued"],
                "sass": " | ".join(sorted(item["sass"])),
            }
        )
    return sorted(result, key=lambda item: item["samples"], reverse=True)


def main():
    event = read_event()

    decomposition = []
    ncu_rows = []
    theory = []
    hotspots = []

    for shape, problem in SHAPES.items():
        for variant, description in SEQUENCE:
            row = event[(shape, variant)]
            reference = COMPARISON.get(variant)
            delta = ""
            if reference is not None:
                delta = (
                    float(event[(shape, reference)]["mean_us"])
                    - float(row["mean_us"])
                )
            decomposition.append(
                {
                    "shape": shape,
                    "config": problem["config"],
                    "variant": variant,
                    "description": description,
                    "mean_us": float(row["mean_us"]),
                    "ci95_half_us": float(row["ci95_half_us"]),
                    "exposed_reduction_us": delta,
                }
            )

            raw = read_ncu(shape, variant)
            metrics = {
                output: float(raw.get(metric, "0") or 0)
                for output, metric in METRICS.items()
            }
            metrics["active_ctas_sm"] = (
                metrics["ctas_active_accum"] / metrics["cycles_active"]
                if metrics["cycles_active"]
                else 0
            )
            ncu_rows.append(
                {
                    "shape": shape,
                    "variant": variant,
                    **metrics,
                }
            )

        full_us = float(event[(shape, "full")]["mean_us"])
        flops = 2 * problem["m"] * problem["n"] * problem["k"]
        compute_us_9pflops = flops / 9.0e15 * 1.0e6
        output_mib = problem["m"] * problem["n"] * 2 / 2**20
        for variant in ("full", "notc", "notma"):
            raw = read_ncu(shape, variant)
            l2_bytes = float(raw["lts__t_bytes.sum"]) * 1.0e6
            l2_us_20tib = l2_bytes / (20 * 2**40) * 1.0e6
            variant_compute_us = 0 if variant == "notc" else compute_us_9pflops
            lower_bound = max(variant_compute_us, l2_us_20tib)
            measured_us = float(event[(shape, variant)]["mean_us"])
            theory.append(
                {
                    "shape": shape,
                    "variant": variant,
                    "m": problem["m"],
                    "n": problem["n"],
                    "k": problem["k"],
                    "stage": problem["stage"],
                    "k_rounds": problem["k_rounds"],
                    "grid_ctas": problem["grid_ctas"],
                    "waves": problem["waves"],
                    "output_MiB": output_mib,
                    "compute_lower_us_9PF": variant_compute_us,
                    "l2_counter_MB": float(raw["lts__t_bytes.sum"]),
                    "l2_lower_us_20TiB": l2_us_20tib,
                    "roofline_lower_us": lower_bound,
                    "event_us": measured_us,
                    "event_over_lower": measured_us / lower_bound,
                    "full_event_us": full_us,
                }
            )
        hotspots.extend(read_source_hotspots(shape))

    write_csv(EXP / "decomposition_summary.csv", decomposition)
    write_csv(EXP / "empty_resource_summary.csv", read_empty())
    write_csv(EXP / "ncu_key_metrics.csv", ncu_rows)
    write_csv(EXP / "theory_summary.csv", theory)
    write_csv(EXP / "source_hotspots.csv", hotspots)


if __name__ == "__main__":
    main()
