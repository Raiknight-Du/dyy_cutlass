#!/usr/bin/env python3
from __future__ import annotations

import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/sf_mainloop_fusion_validation_20260727"
SOURCE = EXP / "op1_best.cu"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

SHAPE = 1
REPEATS = 10
ITERATIONS = 10

VARIANTS = {
    "fused": {
        "binary": EXP / "bin/op1_fused",
        "include": None,
    },
    "split": {
        "binary": EXP / "bin/op1_split_sf",
        "include": EXP / "include_split",
    },
}


def run(command, *, log: pathlib.Path | None = None) -> subprocess.CompletedProcess:
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
    (EXP / "bin").mkdir(parents=True, exist_ok=True)
    for name, variant in VARIANTS.items():
        command = [
            NVCC,
            "-std=c++17",
            "-O3",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ]
        if variant["include"] is not None:
            command.append(f"-I{variant['include']}")
        command.extend(
            [
                "-Iinclude",
                "-Itools/util/include",
                "-Iexamples/common",
                "-Idyy_test/op1",
                SOURCE,
                "-o",
                variant["binary"],
            ]
        )
        print(f"[BUILD] {name}", flush=True)
        run(command, log=EXP / f"build_{name}.log")


def gemm_command(binary: pathlib.Path, iterations: int) -> list[object]:
    return [
        binary,
        f"--shape-id={SHAPE}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def parse_result(output: str) -> float:
    match = re.search(
        rf"^RESULT,{SHAPE},([^,]+),ok,(\d+),([0-9.eE+-]+)$",
        output,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Missing successful RESULT:\n{output}")
    return float(match.group(3))


def benchmark() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(1, REPEATS + 1):
        order = ("fused", "split") if repeat % 2 else ("split", "fused")
        for order_index, name in enumerate(order, start=1):
            clock, temperature = wait_for_idle()
            result = run(
                gemm_command(VARIANTS[name]["binary"], ITERATIONS)
            )
            mean_ms = parse_result(result.stdout)
            rows.append(
                {
                    "repeat": repeat,
                    "order": order_index,
                    "variant": name,
                    "iterations": ITERATIONS,
                    "mean_ms": mean_ms,
                    "idle_sm_clock_mhz": clock,
                    "temperature_c": temperature,
                }
            )
            print(
                f"[EVENT] repeat={repeat:02d} order={order_index} "
                f"{name}={mean_ms:.6f} ms",
                flush=True,
            )

    with (EXP / "event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, object]] = []
    for name in VARIANTS:
        values = [
            float(row["mean_ms"]) for row in rows if row["variant"] == name
        ]
        summary.append(
            {
                "variant": name,
                "samples": len(values),
                "mean_ms": statistics.mean(values),
                "stdev_ms": statistics.stdev(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )

    fused = next(row for row in summary if row["variant"] == "fused")
    split = next(row for row in summary if row["variant"] == "split")
    fused_mean = float(fused["mean_ms"])
    split_mean = float(split["mean_ms"])
    for row in summary:
        row["vs_fused_percent"] = (
            (float(row["mean_ms"]) / fused_mean - 1.0) * 100.0
        )
    split["delta_ms_vs_fused"] = split_mean - fused_mean
    fused["delta_ms_vs_fused"] = 0.0

    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def profile() -> None:
    (EXP / "ncu").mkdir(exist_ok=True)
    for name, variant in VARIANTS.items():
        wait_for_idle()
        report = EXP / "ncu" / name
        command = [
            NCU,
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            "--section",
            "SchedulerStats",
            "--section",
            "WarpStateStats",
            "--section",
            "MemoryWorkloadAnalysis",
            "--export",
            report,
            "--force-overwrite",
        ]
        command.extend(gemm_command(variant["binary"], 1))
        print(f"[NCU] {name}", flush=True)
        run(command, log=EXP / "ncu" / f"{name}.log")


def main() -> None:
    build()
    benchmark()
    profile()
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
