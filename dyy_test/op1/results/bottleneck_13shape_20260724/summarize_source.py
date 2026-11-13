#!/usr/bin/env python3
import csv
import pathlib


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OUT = ROOT / "dyy_test/op1/results/bottleneck_13shape_20260724"
SHAPES = (1, 5, 6, 7, 8, 10, 11, 13)
METRICS = (
    "stall_long_sb (Not Issued)",
    "stall_barrier (Not Issued)",
    "stall_wait (Not Issued)",
)


def main():
    output = OUT / "source_counter_top_pcs.txt"
    with output.open("w") as result:
        for shape in SHAPES:
            path = OUT / f"shape{shape:02d}_source.csv"
            with path.open(newline="") as stream:
                next(stream)
                rows = list(csv.DictReader(stream))
            samples = sum(int(row.get("# Samples") or 0) for row in rows)
            result.write(f"\n=== SHAPE {shape} samples={samples} ===\n")
            for metric in METRICS:
                result.write(f"{metric}\n")
                ranked = sorted(
                    enumerate(rows),
                    key=lambda item: int(item[1].get(metric) or 0),
                    reverse=True,
                )
                shown = 0
                for index, row in ranked:
                    count = int(row.get(metric) or 0)
                    if not count:
                        break
                    context = " | ".join(
                        item["Source"].strip()
                        for item in rows[max(0, index - 2):index + 1]
                    )
                    result.write(
                        f" {count:6d} {row['Address']} {context}\n"
                    )
                    shown += 1
                    if shown == 5:
                        break
    print(output.read_text())


if __name__ == "__main__":
    main()
