#!/usr/bin/env python3
import csv
import pathlib


EXP = pathlib.Path(__file__).resolve().parent
NCU = EXP / "ncu"

CASES = (
    ("C2x1_none", 4, "2x1x1"),
    ("C2x2_A_SFA", 36, "2x2x1"),
    ("C4x1_B_SFB", 33, "4x1x1"),
)


def read_ncu(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1]


def f(row, key):
    return float(row[key])


def main():
    event_rows = []
    for name in ("event_summary.csv", "event_notma_summary.csv"):
        with (EXP / name).open(newline="") as stream:
            event_rows.extend(csv.DictReader(stream))
    events = {
        (row["build"], row["cluster_role"]): row
        for row in event_rows
    }

    rows = []
    for build in ("full", "notc", "notma"):
        base_l2 = None
        pending = []
        for role, config, cluster in CASES:
            raw = read_ncu(NCU / f"{build}_{role}_c{config:02d}_raw.csv")
            event = events[(build, role)]
            l2_mib = f(raw, "lts__t_bytes.sum") * 1e6 / 2**20
            if base_l2 is None:
                base_l2 = l2_mib
            pending.append(
                {
                    "build": build,
                    "cluster_role": role,
                    "cluster": cluster,
                    "config_index": config,
                    "stage": 6,
                    "grid_ctas": int(float(raw["launch__grid_size"])),
                    "k_rounds": 16,
                    "event_mean_us": float(event["mean_ms"]) * 1000,
                    "event_ci95_half_us": float(event["ci95_half_ms"]) * 1000,
                    "ncu_duration_us": f(raw, "gpu__time_duration.sum"),
                    "tma_target_mib": (
                        float(raw.get(
                            "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
                            0,
                        ) or 0) * 1e6 / 2**20
                    ),
                    "tma_multicast_mib": (
                        float(raw.get(
                            "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum",
                            0,
                        ) or 0) * 1e6 / 2**20
                    ),
                    "tma_self_mib": (
                        float(raw.get(
                            "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_self.sum",
                            0,
                        ) or 0) * 1e6 / 2**20
                    ),
                    "l2_request_mib": l2_mib,
                    "tma_load_instructions": int(
                        f(raw, "sm__sass_inst_executed_op_tma_ld.sum")
                    ),
                    "no_eligible_pct": (
                        100
                        - f(
                            raw,
                            "smsp__issue_active.avg.pct_of_peak_sustained_active",
                        )
                    ),
                    "active_warps_per_sm": f(
                        raw, "sm__warps_active.avg.per_cycle_active"
                    ),
                    "eligible_warps_per_scheduler": f(
                        raw, "smsp__warps_eligible.avg.per_cycle_active"
                    ),
                    "long_scoreboard_per_issue": f(
                        raw,
                        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                    ),
                    "smem_per_cta_kb": f(
                        raw, "launch__shared_mem_per_block_allocated"
                    ),
                    "registers_per_thread": int(
                        f(raw, "launch__registers_per_thread")
                    ),
                    "waves_per_sm": f(raw, "launch__waves_per_multiprocessor"),
                    "tensor_instructions": int(float(
                        raw.get("sm__inst_executed_pipe_tensor.sum", 0) or 0
                    )),
                    "tensor_active_cycles": int(float(
                        raw.get("sm__pipe_tensor_cycles_active.sum", 0) or 0
                    )),
                }
            )
        for row in pending:
            row["multicast_share_pct"] = (
                row["tma_multicast_mib"] / row["tma_target_mib"] * 100
                if row["tma_target_mib"] else 0
            )
            row["l2_request_delta_vs_c2x1_pct"] = (
                row["l2_request_mib"] / base_l2 - 1
            ) * 100
            rows.append(row)

    output = EXP / "comparison_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
