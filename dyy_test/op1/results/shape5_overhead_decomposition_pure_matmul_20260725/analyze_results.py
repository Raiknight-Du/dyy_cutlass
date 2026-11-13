#!/usr/bin/env python3
import csv
import pathlib
import statistics


EXP = pathlib.Path(__file__).resolve().parent


def read_event_summary():
    with (EXP / "event_summary.csv").open(newline="") as stream:
        return {
            row["variant"]: {key: float(value) for key, value in row.items() if key != "variant"}
            for row in csv.DictReader(stream)
        }


def read_empty_cluster():
    rows = []
    with (EXP / "empty_cluster_event_raw.txt").open() as stream:
        for line in stream:
            if not line.startswith("RESULT,"):
                continue
            _, variant, sample, iterations, mean_ms = line.strip().split(",")
            rows.append(
                {
                    "variant": variant,
                    "sample": int(sample),
                    "iterations": int(iterations),
                    "mean_ms": float(mean_ms),
                }
            )
    summary = {}
    for variant in sorted({row["variant"] for row in rows}):
        values = [row["mean_ms"] for row in rows if row["variant"] == variant]
        stdev = statistics.stdev(values)
        summary[variant] = {
            "mean_ms": statistics.mean(values),
            "stdev_ms": stdev,
            "ci95_half_ms": 1.96 * stdev / len(values) ** 0.5,
        }
    return summary


def read_source_hotspots():
    path = EXP / "ncu/shell_noepi_source_cuda_sass.csv"
    current_file = ""
    current_source = None
    header = None
    aggregate = {}
    with path.open(newline="") as stream:
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
            if row[0].isdigit() and row[2] == "-":
                current_source = (current_file, int(row[0]), row[1].strip())
                continue
            if current_source is None or not row[2].startswith("0x"):
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

    rows = []
    for (file_path, line, source), item in aggregate.items():
        rows.append(
            {
                "file": file_path,
                "line": line,
                "source": source,
                "samples": item["samples"],
                "not_issued_samples": item["not_issued"],
                "sass": " | ".join(sorted(item["sass"])),
            }
        )
    return sorted(rows, key=lambda item: item["samples"], reverse=True)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    event = read_event_summary()
    empty = read_empty_cluster()
    hotspots = read_source_hotspots()

    sequence = [
        ("full", "完整 pure matmul"),
        ("notc", "移除 UMMA"),
        ("notma", "移除 A/B/SFA/SFB TMA"),
        ("shell_utccp", "移除 TMA+UMMA，保留 UTCCP/epilogue"),
        ("shell_noutccp", "再移除 UTCCP"),
        ("shell_nodstore", "再移除 D TMA store"),
        ("shell_noepi", "再移除 TMEM 读出、转换和 SMEM staging"),
    ]
    decomposition = []
    comparison = {
        "notc": "full",
        "notma": "full",
        "shell_noutccp": "shell_utccp",
        "shell_nodstore": "shell_noutccp",
        "shell_noepi": "shell_nodstore",
    }
    for variant, description in sequence:
        mean_us = event[variant]["mean_ms"] * 1000
        reference = comparison.get(variant)
        delta_us = (
            ""
            if reference is None
            else event[reference]["mean_ms"] * 1000 - mean_us
        )
        decomposition.append(
            {
                "variant": variant,
                "description": description,
                "mean_us": f"{mean_us:.6f}",
                "ci95_half_us": f"{event[variant]['ci95_half_ms'] * 1000:.6f}",
                "exposed_reduction_vs_previous_us": (
                    "" if delta_us == "" else f"{delta_us:.6f}"
                ),
            }
        )
    empty_rows = []
    for variant in (
        "empty_cluster_smem0",
        "empty_cluster_smem227k",
        "cluster_sync_smem227k",
    ):
        empty_rows.append(
            {
                "variant": variant,
                "mean_us": f"{empty[variant]['mean_ms'] * 1000:.6f}",
                "ci95_half_us": f"{empty[variant]['ci95_half_ms'] * 1000:.6f}",
            }
        )

    write_csv(EXP / "decomposition_summary.csv", decomposition)
    write_csv(EXP / "empty_cluster_summary.csv", empty_rows)
    write_csv(EXP / "source_hotspots.csv", hotspots)


if __name__ == "__main__":
    main()
