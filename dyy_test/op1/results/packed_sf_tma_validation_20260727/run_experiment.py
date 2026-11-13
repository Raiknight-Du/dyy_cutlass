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
EXP = ROOT / "dyy_test/op1/results/packed_sf_tma_validation_20260727"
SOURCE = EXP / "packed_sf_tma_microbenchmark.cu"
BINARY = EXP / "bin/packed_sf_tma_microbenchmark"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

VARIANTS = ("separate", "packed")
REPEATS = 10
WARMUP = 5
ITERATIONS = 10


def run(
    command: list[object], *, log: pathlib.Path | None = None
) -> subprocess.CompletedProcess:
    if log is None:
        return subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    with log.open("w") as stream:
        return subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def gpu_state() -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return tuple(
        int(item.strip()) for item in result.stdout.splitlines()[0].split(",")
    )


def wait_for_idle() -> tuple[int, int]:
    for _ in range(180):
        utilization, clock, temperature = gpu_state()
        if utilization <= 2:
            return clock, temperature
        time.sleep(1)
    raise RuntimeError("GPU did not become idle")


def build() -> None:
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    command = [
        NVCC,
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-Iinclude",
        "-Itools/util/include",
        SOURCE,
        "-o",
        BINARY,
    ]
    run(command, log=EXP / "build.log")


def command(variant: str, warmup: int, iterations: int) -> list[object]:
    return [
        BINARY,
        f"--variant={variant}",
        f"--warmup={warmup}",
        f"--iterations={iterations}",
    ]


def parse_fields(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if "," not in line:
            continue
        key, value = line.split(",", 1)
        result[key.strip()] = value.strip()
    if result.get("verification") != "passed":
        raise RuntimeError(f"Verification did not pass:\n{output}")
    return result


def benchmark() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(1, REPEATS + 1):
        order = VARIANTS if repeat % 2 else tuple(reversed(VARIANTS))
        for order_index, variant in enumerate(order, start=1):
            clock, temperature = wait_for_idle()
            result = run(command(variant, WARMUP, ITERATIONS))
            fields = parse_fields(result.stdout)
            rows.append(
                {
                    "repeat": repeat,
                    "order": order_index,
                    "variant": variant,
                    "mean_ms": float(fields["mean_ms"]),
                    "effective_tib_s": float(fields["effective_tib_s"]),
                    "tma_per_stage": int(fields["tma_per_stage"]),
                    "expected_tma_loads": int(fields["expected_tma_loads"]),
                    "idle_sm_clock_mhz": clock,
                    "temperature_c": temperature,
                }
            )
            print(
                f"[EVENT] repeat={repeat:02d} order={order_index} "
                f"{variant}={rows[-1]['mean_ms']:.6f} ms",
                flush=True,
            )

    with (EXP / "event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, object]] = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        values = [float(row["mean_ms"]) for row in selected]
        bandwidth = [float(row["effective_tib_s"]) for row in selected]
        summary.append(
            {
                "variant": variant,
                "samples": len(values),
                "mean_ms": statistics.mean(values),
                "stdev_ms": statistics.stdev(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "mean_effective_tib_s": statistics.mean(bandwidth),
                "tma_per_stage": selected[0]["tma_per_stage"],
                "expected_tma_loads": selected[0]["expected_tma_loads"],
            }
        )

    separate_mean = float(summary[0]["mean_ms"])
    for row in summary:
        row["vs_separate_percent"] = (
            float(row["mean_ms"]) / separate_mean - 1.0
        ) * 100.0
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    paired: list[float] = []
    for repeat in range(1, REPEATS + 1):
        pair = {row["variant"]: row for row in rows if row["repeat"] == repeat}
        paired.append(
            float(pair["packed"]["mean_ms"]) -
            float(pair["separate"]["mean_ms"])
        )
    mean_delta = statistics.mean(paired)
    stdev_delta = statistics.stdev(paired)
    t_975_df9 = 2.262157
    ci_half = t_975_df9 * stdev_delta / math.sqrt(len(paired))
    paired_row = {
        "pairs": len(paired),
        "mean_delta_ms_packed_minus_separate": mean_delta,
        "stdev_delta_ms": stdev_delta,
        "ci95_half_ms": ci_half,
        "ci95_low_ms": mean_delta - ci_half,
        "ci95_high_ms": mean_delta + ci_half,
        "mean_delta_percent": mean_delta / separate_mean * 100.0,
    }
    with (EXP / "paired_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_row.keys())
        writer.writeheader()
        writer.writerow(paired_row)


def profile() -> None:
    metrics = [
        "gpu__time_duration.sum",
        "sm__sass_inst_executed_op_tma_ld.sum",
        "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
        "sm__inst_executed.sum",
        "smsp__warps_active.avg.per_cycle_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    ]
    (EXP / "ncu").mkdir(exist_ok=True)
    for variant in VARIANTS:
        wait_for_idle()
        report = EXP / "ncu" / variant
        ncu_command = [
            NCU,
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{variant}_tma_kernel",
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "--metrics",
            ",".join(metrics),
            "--export",
            report,
            "--force-overwrite",
        ]
        ncu_command.extend(
            [
                BINARY,
                f"--variant={variant}",
                "--warmup=2",
                "--iterations=1",
                "--no-verify",
            ]
        )
        print(f"[NCU] {variant}", flush=True)
        run(ncu_command, log=EXP / "ncu" / f"{variant}.log")


def main() -> None:
    build()
    benchmark()
    profile()
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()

