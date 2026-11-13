#!/usr/bin/env python3
"""Rebuild the shape8 cluster-ablation summary from Event and NCU exports."""

import csv
import math
import pathlib
import re
import statistics


ROOT = pathlib.Path(__file__).resolve().parent
NCU_DIR = ROOT / "ncu"

CLUSTERS = {
    0: "C2x1x1",
    22: "C2x2x1",
    20: "C4x1x1",
}
VARIANTS = (
    "full",
    "notc",
    "notma",
    "shell_utccp",
    "shell_noutccp",
    "shell_nodstore",
    "shell_noepi",
)

METRICS = {
    "ncu_duration_us": "gpu__time_duration.sum",
    "l2_bytes_mb": "lts__t_bytes.sum",
    "tma_target_bytes_mb": (
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
    ),
    "d_store_bytes_mb": (
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum"
    ),
    "tma_load_inst": "sm__sass_inst_executed_op_tma_ld.sum",
    "d_store_inst": "sm__sass_inst_executed_op_tma_st.sum",
    "tensor_inst": "sm__inst_executed_pipe_tensor.sum",
    "tmem_inst": "sm__inst_executed_pipe_tmem.sum",
    "utccp_bytes_mb": "sm__mem_tensor_writes_op_utccp.sum",
    "sm_inst": "sm__inst_executed.sum",
    "active_cta_sum": "sm__ctas_active.sum",
    "active_cycles_sum": "sm__cycles_active.sum",
    "no_eligible_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "active_warps_per_sched": "smsp__warps_active.avg.per_cycle_active",
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
    "smem_kb": "launch__shared_mem_per_block",
    "registers_per_thread": "launch__registers_per_thread",
    "waves_per_sm": "launch__waves_per_multiprocessor",
}


def read_ncu_raw(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        next(reader)  # unit row
        value = next(reader)
    return dict(zip(header, value))


def as_float(value):
    if value in ("", "n/a", "N/A"):
        return math.nan
    return float(value)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_event():
    with (ROOT / "event_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (int(row["config_index"]), row["variant"]): {
            key: float(row[key])
            for key in ("mean_us", "ci95_half_us", "min_us", "max_us")
        }
        for row in rows
    }


def summarize_ncu():
    rows = []
    for config, cluster in CLUSTERS.items():
        for variant in VARIANTS:
            raw = read_ncu_raw(
                NCU_DIR / f"shape8_c{config}_{variant}_raw.csv"
            )
            row = {
                "config_index": config,
                "cluster": cluster,
                "variant": variant,
            }
            for output_name, metric_name in METRICS.items():
                row[output_name] = as_float(raw.get(metric_name, ""))
            cycles = row["active_cycles_sum"]
            ctas = row["active_cta_sum"]
            row["active_cta_per_sm"] = ctas / cycles if cycles else math.nan
            # Nsight reports issue-active percentage. No Eligible is its complement.
            row["no_eligible_pct"] = 100.0 - row["no_eligible_pct"]
            rows.append(row)
    write_csv(ROOT / "ncu_key_metrics.csv", rows)
    return {(row["config_index"], row["variant"]): row for row in rows}


def summarize_empty_cluster():
    pattern = re.compile(
        r"^RESULT,([^,]+),(\d+),(\d+),([0-9.eE+-]+)$"
    )
    samples = {}
    for line in (ROOT / "empty_cluster_event_raw.txt").read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        samples.setdefault(match.group(1), []).append(float(match.group(4)) * 1000)

    rows = []
    for name, values in samples.items():
        stdev = statistics.stdev(values)
        rows.append(
            {
                "case": name,
                "samples": len(values),
                "mean_us": statistics.mean(values),
                "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                "min_us": min(values),
                "max_us": max(values),
            }
        )
    write_csv(ROOT / "empty_cluster_summary.csv", rows)
    return {row["case"]: row for row in rows}


def build_comparisons(event, ncu, empty):
    rows = []
    baseline = event[(0, "full")]["mean_us"]
    for config, cluster in CLUSTERS.items():
        full = event[(config, "full")]["mean_us"]
        notc = event[(config, "notc")]["mean_us"]
        notma = event[(config, "notma")]["mean_us"]
        utccp = event[(config, "shell_utccp")]["mean_us"]
        noutccp = event[(config, "shell_noutccp")]["mean_us"]
        nodstore = event[(config, "shell_nodstore")]["mean_us"]
        noepi = event[(config, "shell_noepi")]["mean_us"]
        ncu_full = ncu[(config, "full")]
        rows.append(
            {
                "config_index": config,
                "cluster": cluster,
                "full_us": full,
                "full_vs_c2x1_pct": (full / baseline - 1.0) * 100.0,
                "notc_us": notc,
                "exposed_compute_us": full - notc,
                "notma_us": notma,
                "exposed_input_tma_us": full - notma,
                "shell_utccp_us": utccp,
                "utccp_us": utccp - noutccp,
                "d_store_us": noutccp - nodstore,
                "remaining_epilogue_us": nodstore - noepi,
                "mainloop_shell_us": noepi,
                "l2_bytes_mb": ncu_full["l2_bytes_mb"],
                "tma_target_bytes_mb": ncu_full["tma_target_bytes_mb"],
                "tma_load_inst": ncu_full["tma_load_inst"],
                "tensor_inst": ncu_full["tensor_inst"],
                "no_eligible_pct": ncu_full["no_eligible_pct"],
                "eligible_warps_per_sched": ncu_full[
                    "eligible_warps_per_sched"
                ],
                "active_warps_per_sched": ncu_full["active_warps_per_sched"],
                "long_sb_per_issue": ncu_full["long_sb_per_issue"],
                "active_cta_per_sm": ncu_full["active_cta_per_sm"],
            }
        )
    write_csv(ROOT / "cluster_comparison.csv", rows)

    empty_rows = []
    for cluster in ("C2x1", "C2x2", "C4x1"):
        sync = empty[f"{cluster}_sync"]["mean_us"]
        base = empty[f"{cluster}_empty"]["mean_us"]
        smem = empty[f"{cluster}_smem"]["mean_us"]
        empty_rows.append(
            {
                "cluster": cluster,
                "empty_us": base,
                "large_smem_us": smem,
                "cluster_sync_us": sync,
                "sync_increment_over_empty_us": sync - base,
            }
        )
    write_csv(ROOT / "empty_cluster_comparison.csv", empty_rows)
    return rows, empty_rows


def main():
    event = summarize_event()
    ncu = summarize_ncu()
    empty = summarize_empty_cluster()
    comparisons, empty_rows = build_comparisons(event, ncu, empty)

    for row in comparisons:
        print(
            f"{row['cluster']:7s} full={row['full_us']:.3f} us "
            f"NO-TC={row['notc_us']:.3f} us "
            f"NO-TMA={row['notma_us']:.3f} us "
            f"L2={row['l2_bytes_mb']:.3f} MB "
            f"NoEligible={row['no_eligible_pct']:.2f}%"
        )
    for row in empty_rows:
        print(
            f"{row['cluster']:5s} empty={row['empty_us']:.3f} us "
            f"sync={row['cluster_sync_us']:.3f} us "
            f"increment={row['sync_increment_over_empty_us']:.3f} us"
        )


if __name__ == "__main__":
    main()
