#!/usr/bin/env python3
import csv
import pathlib
import re
import statistics


HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE
NCU = HERE / "ncu"

RAW_METRICS = {
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
    "active_warps": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
}

DETAIL_METRICS = {
    "no_eligible_pct": ("Scheduler Statistics", "No Eligible"),
    "achieved_occupancy_pct": ("Occupancy", "Achieved Occupancy"),
    "block_limit_smem": ("Occupancy", "Block Limit Shared Mem"),
}


def number(value):
    value = value.replace(",", "").strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_raw(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        units = next(reader)
        values = next(reader)
    row = dict(zip(header, values))
    unit = dict(zip(header, units))
    result = {}
    for name, metric in RAW_METRICS.items():
        result[name] = number(row[metric]) if metric in row else float("nan")
        result[f"{name}_unit"] = unit.get(metric, "")
    return result


def read_details(path):
    metrics = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["Section Name"], row["Metric Name"])
            metrics[key] = number(row["Metric Value"])
    return {
        name: metrics.get(key, float("nan"))
        for name, key in DETAIL_METRICS.items()
    }


def collect_ncu():
    rows = []
    for path in sorted(
        list(NCU.glob("e128_*_raw.csv")) + list(NCU.glob("eauto_*_raw.csv"))
    ):
        if "_repeat_" in path.name:
            continue
        stem = path.name.removesuffix("_raw.csv")
        epilogue, variant = stem.split("_", 1)
        row = {"epilogue": epilogue, "variant": variant}
        row.update(read_raw(path))
        row.update(read_details(NCU / f"{stem}_details.csv"))
        rows.append(row)
    return rows


def collect_ncu_repeats():
    rows = []
    for path in sorted(NCU.glob("e*_full_repeat_r*_raw.csv")):
        match = re.match(r"(e128|eauto)_full_repeat_r(\d+)_raw\.csv", path.name)
        if not match:
            continue
        epilogue, repetition = match.groups()
        row = {"epilogue": epilogue, "repetition": int(repetition)}
        row.update(read_raw(path))
        row.update(
            read_details(
                NCU / f"{epilogue}_full_repeat_r{repetition}_details.csv"
            )
        )
        rows.append(row)

    summaries = []
    for epilogue in ("e128", "eauto"):
        selected = [row for row in rows if row["epilogue"] == epilogue]
        if not selected:
            continue
        summary = {"epilogue": epilogue, "repetitions": len(selected)}
        for key in (
            "duration_us",
            "sm_inst",
            "active_warps",
            "eligible_warps",
            "long_sb",
            "barrier",
            "wait",
            "no_eligible_pct",
        ):
            values = [row[key] for row in selected]
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
        summaries.append(summary)
    return rows, summaries


def collect_event():
    rows = []
    with (RESULTS / "event_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "epilogue": row["epilogue"],
                    "variant": row["variant"],
                    "mean_us": float(row["mean_us"]),
                    "ci95_half_us": float(row["ci95_half_us"]),
                }
            )
    return rows


def paired_event_deltas():
    raw = []
    with (RESULTS / "event_raw.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            raw.append(
                {
                    "sample": int(row["sample"]),
                    "epilogue": row["epilogue"],
                    "variant": row["variant"],
                    "mean_us": float(row["mean_ms"]) * 1000,
                }
            )
    table = {
        (row["sample"], row["epilogue"], row["variant"]): row["mean_us"]
        for row in raw
    }
    comparisons = (
        ("Full - Shell", "full", "shell_utccp"),
        ("TMA exposed without MMA", "notc", "shell_utccp"),
        ("MMA exposed without input TMA", "notma", "shell_utccp"),
        ("TMA exposed with MMA", "full", "notma"),
        ("MMA exposed with input TMA", "full", "notc"),
        ("UTCCP exposed", "shell_utccp", "shell_noutccp"),
        ("D store exposed", "shell_noutccp", "shell_nodstore"),
        (
            "TMEM load/convert/SMEM staging exposed",
            "shell_nodstore",
            "shell_noepi",
        ),
    )
    rows = []
    for epilogue in ("e128",):
        samples = sorted(
            {
                sample
                for sample, epi, _ in table
                if epi == epilogue
            }
        )
        for label, minuend, subtrahend in comparisons:
            values = [
                table[(sample, epilogue, minuend)]
                - table[(sample, epilogue, subtrahend)]
                for sample in samples
            ]
            stdev = statistics.stdev(values)
            rows.append(
                {
                    "epilogue": epilogue,
                    "comparison": label,
                    "mean_delta_us": statistics.mean(values),
                    "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                    "min_delta_us": min(values),
                    "max_delta_us": max(values),
                    "samples": len(values),
                }
            )
    return rows


def event_deltas(event_rows):
    table = {
        (row["epilogue"], row["variant"]): row["mean_us"]
        for row in event_rows
    }
    rows = []
    for epi in ("e128",):
        value = lambda variant: table[(epi, variant)]
        rows.extend(
            [
                {
                    "epilogue": epi,
                    "comparison": "Full - Shell",
                    "delta_us": value("full") - value("shell_utccp"),
                },
                {
                    "epilogue": epi,
                    "comparison": "TMA exposed without MMA",
                    "delta_us": value("notc") - value("shell_utccp"),
                },
                {
                    "epilogue": epi,
                    "comparison": "MMA exposed without input TMA",
                    "delta_us": value("notma") - value("shell_utccp"),
                },
                {
                    "epilogue": epi,
                    "comparison": "TMA exposed with MMA",
                    "delta_us": value("full") - value("notma"),
                },
                {
                    "epilogue": epi,
                    "comparison": "MMA exposed with input TMA",
                    "delta_us": value("full") - value("notc"),
                },
                {
                    "epilogue": epi,
                    "comparison": "TMA x MMA interaction",
                    "delta_us": (
                        value("full")
                        - value("notc")
                        - value("notma")
                        + value("shell_utccp")
                    ),
                },
                {
                    "epilogue": epi,
                    "comparison": "UTCCP exposed",
                    "delta_us": value("shell_utccp") - value("shell_noutccp"),
                },
                {
                    "epilogue": epi,
                    "comparison": "D store exposed",
                    "delta_us": value("shell_noutccp") - value("shell_nodstore"),
                },
                {
                    "epilogue": epi,
                    "comparison": "TMEM load/convert/SMEM staging exposed",
                    "delta_us": value("shell_nodstore") - value("shell_noepi"),
                },
            ]
        )
    return rows


def resource_rows():
    pattern = re.compile(r"^RESULT,([^,]+),\d+,\d+,([0-9.eE+-]+)$")
    values = {}
    for line in (RESULTS / "resource_control_raw.txt").read_text().splitlines():
        match = pattern.match(line)
        if match:
            values.setdefault(match.group(1), []).append(float(match.group(2)) * 1000)
    return [
        {
            "case": name,
            "mean_us": statistics.mean(samples),
            "stdev_us": statistics.stdev(samples),
            "min_us": min(samples),
            "max_us": max(samples),
        }
        for name, samples in values.items()
    ]


def source_hotspots():
    rows = []
    for path in sorted(NCU.glob("*_source_r*_cuda_sass.csv")):
        current_file = ""
        current_source = None
        header = None
        aggregate = {}
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
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
                long_sb = int(row[header.index("stall_long_sb (Not Issued)")] or 0)
                if long_sb == 0:
                    continue
                item = aggregate.setdefault(
                    current_source,
                    {"long_sb": 0, "sass": set()},
                )
                item["long_sb"] += long_sb
                item["sass"].add(row[3].strip())

        total = sum(item["long_sb"] for item in aggregate.values())
        parsed = sorted(
            (
                {
                    "file": source_key[0],
                    "line": source_key[1],
                    "source": source_key[2],
                    "long_sb": item["long_sb"],
                    "sass": " | ".join(sorted(item["sass"])),
                }
                for source_key, item in aggregate.items()
            ),
            key=lambda item: item["long_sb"],
            reverse=True,
        )
        for rank, item in enumerate(parsed[:16], 1):
            rows.append(
                {
                    "report": path.stem,
                    "rank": rank,
                    **item,
                    "total_long_sb": total,
                    "share_pct": 100 * item["long_sb"] / total if total else 0,
                }
            )
    return rows


def write_csv(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    event = collect_event()
    paired_deltas = paired_event_deltas()
    ncu = collect_ncu()
    ncu_repeats, ncu_repeat_summary = collect_ncu_repeats()
    deltas = event_deltas(event)
    resources = resource_rows()
    hotspots = source_hotspots()
    write_csv("ncu_summary.csv", ncu)
    write_csv("ncu_repeat_raw.csv", ncu_repeats)
    write_csv("ncu_repeat_summary.csv", ncu_repeat_summary)
    write_csv("event_deltas.csv", deltas)
    write_csv("event_paired_deltas.csv", paired_deltas)
    write_csv("resource_summary.csv", resources)
    write_csv("source_hotspots.csv", hotspots)
    for title, rows in (
        ("EVENT", event),
        ("DELTAS", deltas),
        ("RESOURCE", resources),
        ("NCU", ncu),
        ("NCU REPEATS", ncu_repeat_summary),
    ):
        print(f"\n[{title}]")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
