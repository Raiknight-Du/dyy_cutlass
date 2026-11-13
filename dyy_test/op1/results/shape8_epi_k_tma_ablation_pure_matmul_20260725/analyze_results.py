#!/usr/bin/env python3
import csv
import pathlib


EXP = pathlib.Path(__file__).resolve().parent
NCU = EXP / "ncu"
VARIANTS = ("full", "notc", "notma", "shell")

METRICS = {
    "ncu_duration_us": "gpu__time_duration.sum",
    "l2_bytes": "lts__t_bytes.sum",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "input_tma_bytes": (
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
    ),
    "d_tma_bytes": (
        "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum"
    ),
    "input_tma_inst": "sm__sass_inst_executed_op_tma_ld.sum",
    "d_tma_inst": "sm__sass_inst_executed_op_tma_st.sum",
    "tensor_inst": "sm__inst_executed_pipe_tensor.sum",
    "tmem_pipe_inst": "sm__inst_executed_pipe_tmem.sum",
    "sm_inst": "sm__inst_executed.sum",
    "smem_kb": "launch__shared_mem_per_block",
    "registers_thread": "launch__registers_per_thread",
    "waves_sm": "launch__waves_per_multiprocessor",
    "no_eligible_pct": "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
    "eligible_warps_sched": "smsp__warps_eligible.avg.per_cycle_active",
    "active_warps_sched": "smsp__warps_active.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "long_sb_per_issue": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
    "barrier_per_issue": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
    ),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(row, name):
    value = row.get(name, "")
    return float(value) if value not in ("", "N/A", "n/a") else None


events = read_csv(EXP / "event_summary.csv")
resources = {
    row["case"]: row for row in read_csv(EXP / "resource_summary.csv")
}
event_map = {
    (row["case"], row["variant"]): row for row in events
}
cases = list(resources)

ncu_rows = []
for case in cases:
    for variant in VARIANTS:
        rows = read_csv(NCU / f"{case}_{variant}_raw.csv")
        row = rows[-1]
        output = {"case": case, "variant": variant}
        for label, metric in METRICS.items():
            output[label] = number(row, metric)
        ncu_rows.append(output)

with (EXP / "ncu_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=ncu_rows[0].keys())
    writer.writeheader()
    writer.writerows(ncu_rows)

ncu_map = {(row["case"], row["variant"]): row for row in ncu_rows}
analysis = []
for case in cases:
    times = {
        variant: float(event_map[(case, variant)]["mean_us"])
        for variant in VARIANTS
    }
    cis = {
        variant: float(event_map[(case, variant)]["ci95_half_us"])
        for variant in VARIANTS
    }
    resource = resources[case]
    full_ncu = ncu_map[(case, "full")]
    notc_ncu = ncu_map[(case, "notc")]
    analysis.append(
        {
            "case": case,
            "tile_k": resource["tile_k"],
            "epilogue": resource["epilogue"],
            "stages": resource["stages"],
            "k_iterations": 2048 // int(resource["tile_k"]),
            "mainloop_tensor_kb": int(resource["mainloop_tensor_B"]) / 1024,
            "epilogue_tensor_kb": int(resource["epilogue_tensor_B"]) / 1024,
            "total_smem_kb": int(resource["total_shared_B"]) / 1024,
            "pair_stage_tma_kb": (
                int(resource["ab_tma_transaction_B"])
                + int(resource["sf_tma_transaction_B"])
            ) / 1024,
            "per_cta_stage_tma_kb": (
                int(resource["ab_tma_transaction_B"])
                + int(resource["sf_tma_transaction_B"])
            ) / 1024 / 2,
            "full_us": times["full"],
            "full_ci95_us": cis["full"],
            "notc_us": times["notc"],
            "notma_us": times["notma"],
            "shell_us": times["shell"],
            "tma_exposed_with_mma_us": times["full"] - times["notma"],
            "tma_isolated_no_mma_us": times["notc"] - times["shell"],
            "mma_exposed_with_tma_us": times["full"] - times["notc"],
            "mma_isolated_no_tma_us": times["notma"] - times["shell"],
            "mainloop_interaction_us": (
                (times["notc"] - times["shell"])
                + (times["notma"] - times["shell"])
                - (times["full"] - times["shell"])
            ),
            "full_sm_inst": full_ncu["sm_inst"],
            "notc_sm_inst": notc_ncu["sm_inst"],
            "full_no_eligible_pct": full_ncu["no_eligible_pct"],
            "notc_no_eligible_pct": notc_ncu["no_eligible_pct"],
            "full_long_sb": full_ncu["long_sb_per_issue"],
            "notc_long_sb": notc_ncu["long_sb_per_issue"],
            "full_input_tma_inst": full_ncu["input_tma_inst"],
            "full_input_tma_bytes": full_ncu["input_tma_bytes"],
            "full_l2_bytes": full_ncu["l2_bytes"],
            "full_l2_hit_pct": full_ncu["l2_hit_pct"],
        }
    )

with (EXP / "comparison_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=analysis[0].keys())
    writer.writeheader()
    writer.writerows(analysis)

for row in analysis:
    print(
        row["case"],
        f"stage={row['stages']}",
        f"full={row['full_us']:.3f} us",
        f"TMA exposed={row['tma_exposed_with_mma_us']:.3f} us",
        f"TMA isolated={row['tma_isolated_no_mma_us']:.3f} us",
    )
