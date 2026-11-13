#!/usr/bin/env python3
import csv
import pathlib


EXP = pathlib.Path(__file__).resolve().parent
CASES = (
    ("N192_E64", "n192_e64_raw.csv"),
    ("N256_E64", "n256_e64_raw.csv"),
    ("N256_E128", "n256_e128_raw.csv"),
)

FIELDS = {
    "duration_us": "gpu__time_duration.sum",
    "grid_size": "launch__grid_size",
    "waves": "launch__waves_per_multiprocessor",
    "registers": "launch__registers_per_thread",
    "smem_bytes": "launch__shared_mem_per_block",
    "l2_bytes": "lts__t_bytes.sum",
    "input_tma_bytes": "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "d_tma_bytes": "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
    "input_tma_inst": "sm__sass_inst_executed_op_tma_ld.sum",
    "d_tma_inst": "sm__sass_inst_executed_op_tma_st.sum",
    "tensor_inst": "sm__inst_executed_pipe_tensor.sum",
    "tmem_pipe_inst": "sm__inst_executed_pipe_tmem.sum",
    "tmem_load_bytes": "sm__mem_tensor_reads_op_ldt.sum",
    "utccp_bytes": "sm__mem_tensor_writes_op_utccp.sum",
    "sm_inst": "sm__inst_executed.sum",
    "active_cta": "sm__ctas_active.avg",
    "active_warps": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}


def number(value):
    if value == "":
        return None
    return float(value.replace(",", ""))


def read_raw(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        units = next(reader)
        values = next(reader)
    row = {}
    for output_name, metric_name in FIELDS.items():
        row[output_name] = number(values.get(metric_name, ""))
        row[f"{output_name}_unit"] = units.get(metric_name, "")
    row["no_eligible_pct"] = (
        100.0 - row["issue_active_pct"]
        if row["issue_active_pct"] is not None
        else None
    )
    return row


def main():
    rows = []
    for label, filename in CASES:
        row = {"label": label}
        row.update(read_raw(EXP / "ncu" / filename))
        rows.append(row)

    columns = ["label"]
    for field in FIELDS:
        columns.extend((field, f"{field}_unit"))
    columns.append("no_eligible_pct")
    with (EXP / "ncu_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["label"],
            f"duration={row['duration_us']} {row['duration_us_unit']}",
            f"grid={row['grid_size']}",
            f"regs={row['registers']}",
            f"smem={row['smem_bytes']} {row['smem_bytes_unit']}",
            f"eligible={row['eligible_warps']}",
            f"no_eligible={row['no_eligible_pct']:.2f}%",
            f"long_sb={row['long_sb']}",
            f"sm_inst={row['sm_inst']}",
            f"tensor={row['tensor_inst']}",
            f"tmem={row['tmem_pipe_inst']}",
            f"ldt={row['tmem_load_bytes']}",
            f"utccp={row['utccp_bytes']}",
        )


if __name__ == "__main__":
    main()
