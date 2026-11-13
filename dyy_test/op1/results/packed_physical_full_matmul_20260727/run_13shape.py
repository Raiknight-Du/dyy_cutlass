#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/packed_physical_full_matmul_20260727"
BINARIES = {
    "baseline": EXP / "bin/op1_baseline",
    "packed": EXP / "bin/op1_packed",
}
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
REPEATS = 10
ITERATIONS = 10


def wait_idle(timeout: int = 1800) -> None:
    started = time.monotonic()
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        if not result.stdout.strip():
            return
        if time.monotonic() - started > timeout:
            raise TimeoutError("GPU did not become idle")
        time.sleep(2)


def gpu_state() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,"
            "temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    return dict(zip(
        ["pstate", "idle_sm_mhz", "mem_mhz", "idle_power_w",
         "temperature_c"],
        fields,
    ))


def parse(output: str, shape: int) -> dict[str, object]:
    result_match = re.search(
        rf"^RESULT,{shape},([^,]+),ok,(\d+),([0-9.eE+-]+)$",
        output,
        re.MULTILINE,
    )
    digest_match = re.search(
        rf"^DIGEST,{shape},([0-9a-f]+),([0-9a-f]+)$",
        output,
        re.MULTILINE,
    )
    packed_match = re.search(
        r"^PACKED_BYTES,(\d+),(\d+)$", output, re.MULTILINE)
    if result_match is None or digest_match is None:
        raise RuntimeError(f"Could not parse shape {shape}:\n{output}")
    return {
        "config": result_match.group(1),
        "iterations": int(result_match.group(2)),
        "avg_ms": float(result_match.group(3)),
        "digest_sum": digest_match.group(1),
        "digest_xor": digest_match.group(2),
        "packed_a_bytes": int(packed_match.group(1)) if packed_match else 0,
        "packed_b_bytes": int(packed_match.group(2)) if packed_match else 0,
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(1, REPEATS + 1):
        shapes = list(range(1, 14))
        shift = (repeat - 1) % len(shapes)
        shapes = shapes[shift:] + shapes[:shift]
        if repeat % 2 == 0:
            shapes.reverse()
        variant_order = (
            ("baseline", "packed") if repeat % 2
            else ("packed", "baseline")
        )
        for shape_order, shape in enumerate(shapes, 1):
            pair: dict[str, dict[str, object]] = {}
            for variant_order_index, variant in enumerate(variant_order, 1):
                wait_idle()
                state = gpu_state()
                command = [
                    BINARIES[variant],
                    f"--shape-id={shape}",
                    f"--iterations={ITERATIONS}",
                    f"--dataset-root={DATASET}",
                ]
                process = subprocess.run(
                    [str(item) for item in command],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
                if process.returncode != 0:
                    raise RuntimeError(
                        f"shape={shape} variant={variant} "
                        f"rc={process.returncode}\n{process.stdout}")
                parsed = parse(process.stdout, shape)
                pair[variant] = parsed
                row = {
                    "repeat": repeat,
                    "shape_order": shape_order,
                    "variant_order": variant_order_index,
                    "shape": shape,
                    "variant": variant,
                    **parsed,
                    **state,
                }
                rows.append(row)
                write_csv(EXP / "event_raw.csv", rows)
                print(
                    f"repeat={repeat:02d} shape={shape:02d} "
                    f"{variant}={parsed['avg_ms']:.9f} ms",
                    flush=True,
                )
            digest_keys = ("digest_sum", "digest_xor")
            if any(
                pair["baseline"][key] != pair["packed"][key]
                for key in digest_keys
            ):
                raise RuntimeError(
                    f"Output mismatch for shape {shape}: {pair}")

    summary: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for shape in range(1, 14):
        selected = [row for row in rows if int(row["shape"]) == shape]
        grouped: dict[str, list[float]] = {}
        for variant in BINARIES:
            grouped[variant] = [
                float(row["avg_ms"])
                for row in selected if row["variant"] == variant
            ]
        baseline_mean = statistics.mean(grouped["baseline"])
        packed_mean = statistics.mean(grouped["packed"])
        packed_row = next(
            row for row in selected if row["variant"] == "packed")
        summary.append({
            "shape": shape,
            "config": packed_row["config"],
            "samples": REPEATS,
            "baseline_mean_ms": baseline_mean,
            "baseline_stdev_ms": statistics.stdev(grouped["baseline"]),
            "packed_mean_ms": packed_mean,
            "packed_stdev_ms": statistics.stdev(grouped["packed"]),
            "delta_ms": packed_mean - baseline_mean,
            "delta_percent": (packed_mean / baseline_mean - 1.0) * 100.0,
            "packed_a_bytes": packed_row["packed_a_bytes"],
            "packed_b_bytes": packed_row["packed_b_bytes"],
            "digest_match": True,
        })
        deltas = []
        for repeat in range(1, REPEATS + 1):
            pair = {
                row["variant"]: row for row in selected
                if int(row["repeat"]) == repeat
            }
            deltas.append(
                float(pair["packed"]["avg_ms"]) -
                float(pair["baseline"]["avg_ms"]))
        mean_delta = statistics.mean(deltas)
        stdev_delta = statistics.stdev(deltas)
        t_975_df9 = 2.262157
        ci_half = t_975_df9 * stdev_delta / math.sqrt(REPEATS)
        paired_rows.append({
            "shape": shape,
            "pairs": REPEATS,
            "mean_delta_ms": mean_delta,
            "stdev_delta_ms": stdev_delta,
            "ci95_low_ms": mean_delta - ci_half,
            "ci95_high_ms": mean_delta + ci_half,
            "all_output_digests_match": True,
        })
    write_csv(EXP / "event_summary.csv", summary)
    write_csv(EXP / "paired_summary.csv", paired_rows)


if __name__ == "__main__":
    benchmark()
