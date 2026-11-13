#!/usr/bin/env python3
import csv
import pathlib

from analyze_results import read_raw


EXP = pathlib.Path(__file__).resolve().parent
VARIANTS = (
    "full",
    "notc",
    "notma",
    "shell_utccp",
    "shell_noutccp",
    "shell_nodstore",
    "shell_noepi",
)


def main():
    with (EXP / "ablation_event_summary.csv").open(newline="") as stream:
        event = {row["variant"]: row for row in csv.DictReader(stream)}

    ncu_rows = []
    for variant in VARIANTS:
        row = {"variant": variant}
        row.update(
            read_raw(EXP / "ncu" / f"e64_{variant}_raw.csv")
        )
        ncu_rows.append(row)

    ncu_columns = list(ncu_rows[0].keys())
    with (EXP / "ablation_ncu_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ncu_columns)
        writer.writeheader()
        writer.writerows(ncu_rows)

    mean = {name: float(row["mean_us"]) for name, row in event.items()}
    deltas = (
        ("Full-Shell", mean["full"] - mean["shell_utccp"]),
        ("TMA without MMA", mean["notc"] - mean["shell_utccp"]),
        ("MMA without TMA", mean["notma"] - mean["shell_utccp"]),
        ("TMA with MMA", mean["full"] - mean["notma"]),
        ("MMA with TMA", mean["full"] - mean["notc"]),
        ("UTCCP", mean["shell_utccp"] - mean["shell_noutccp"]),
        ("D store", mean["shell_noutccp"] - mean["shell_nodstore"]),
        (
            "TMEM load/convert/SMEM staging",
            mean["shell_nodstore"] - mean["shell_noepi"],
        ),
    )
    with (EXP / "ablation_deltas.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("component", "exposed_us"))
        writer.writerows(deltas)

    for name, value in deltas:
        print(f"{name}: {value:.6f} us")
    for row in ncu_rows:
        print(
            row["variant"],
            f"duration={row['duration_us']}",
            f"eligible={row['eligible_warps']}",
            f"no_eligible={row['no_eligible_pct']:.2f}%",
            f"long_sb={row['long_sb']}",
            f"barrier={row['barrier']}",
            f"sm_inst={row['sm_inst']}",
        )


if __name__ == "__main__":
    main()
