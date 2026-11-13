#!/usr/bin/env python3
import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/bottleneck_shape5_8_11_13_20260724"
SOURCE = EXP / "op1_bottleneck_probe.cu"
BINARY = ROOT / "dyy_test/op1/bin/op1_bottleneck_probe"
BUILD_LOG = EXP / "build.log"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

SAMPLES = 25
ITERATIONS = 10

CASES = (
    # shape 5: increasing N tile reduces the block grid; K512 reduces K rounds.
    (5, "n64", 98),
    (5, "n128", 30),
    (5, "n192_best", 36),
    (5, "n256", 17),
    (5, "n256_k512", 94),
    # shape 8: N128 doubles the block grid; K512 halves K rounds.
    (8, "n64", 98),
    (8, "n128", 59),
    (8, "n256_best", 0),
    (8, "n256_k128", 11),
    (8, "n256_k512", 94),
    # shape 11: K has only one iteration, so N controls block count directly.
    (11, "n64", 96),
    (11, "n128", 95),
    (11, "n192_best", 78),
    (11, "n256", 63),
    (11, "n256_2sm", 13),
    # shape 13: same one-iteration comparison at a wider problem N.
    (13, "n64", 96),
    (13, "n128", 95),
    (13, "n192", 78),
    (13, "n256_best", 63),
    (13, "n256_2sm", 13),
    (13, "n256_k256", 17),
)


def run(command, stdout=None, check=True):
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        text=True,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        check=check,
    )


def wait_for_idle():
    for _ in range(180):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        if int(result.stdout.strip().splitlines()[0]) <= 2:
            return
        time.sleep(1)
    raise RuntimeError("GPU did not become idle")


def build():
    command = [
        NVCC,
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-Iinclude",
        "-Itools/util/include",
        "-Iexamples/common",
        "-Idyy_test/op1",
        SOURCE,
        "-o",
        BINARY,
    ]
    print("[BUILD]", flush=True)
    with BUILD_LOG.open("w") as log:
        run(command, stdout=log)


def command_for(shape, config, iterations):
    return [
        BINARY,
        f"--shape-id={shape}",
        f"--config-index={config}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    raw_path = EXP / "event_raw.csv"
    summary_path = EXP / "event_summary.csv"
    rows = []
    result_pattern = re.compile(
        r"^RESULT,(\d+),([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    for shape, label, config in CASES:
        for sample in range(SAMPLES):
            wait_for_idle()
            result = subprocess.run(
                [str(item) for item in command_for(shape, config, ITERATIONS)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = result_pattern.search(result.stdout)
            if not match:
                raise RuntimeError(
                    f"missing RESULT for shape={shape} config={config}:\n"
                    f"{result.stdout}\n{result.stderr}"
                )
            rows.append(
                {
                    "shape": shape,
                    "label": label,
                    "config_index": config,
                    "variant": match.group(2),
                    "sample": sample,
                    "iterations": int(match.group(3)),
                    "mean_ms": float(match.group(4)),
                }
            )
            with raw_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"[EVENT] shape={shape} {label}", flush=True)

    summary = []
    for shape, label, config in CASES:
        selected = [
            row["mean_ms"]
            for row in rows
            if row["shape"] == shape and row["config_index"] == config
        ]
        variant = next(
            row["variant"]
            for row in rows
            if row["shape"] == shape and row["config_index"] == config
        )
        stdev = statistics.stdev(selected)
        summary.append(
            {
                "shape": shape,
                "label": label,
                "config_index": config,
                "variant": variant,
                "mean_ms": statistics.mean(selected),
                "stdev_ms": stdev,
                "ci95_half_ms": 1.96 * stdev / len(selected) ** 0.5,
                "min_ms": min(selected),
                "max_ms": max(selected),
                "samples": len(selected),
                "iterations_per_sample": ITERATIONS,
            }
        )
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    for shape, label, config in CASES:
        stem = f"shape{shape:02d}_{label}_c{config:02d}"
        report = ncu_dir / f"{stem}.ncu-rep"
        if not report.exists():
            wait_for_idle()
            print(f"[NCU] shape={shape} {label}", flush=True)
            with (ncu_dir / f"{stem}.log").open("w") as log:
                run(
                    [
                        NCU,
                        "--target-processes",
                        "all",
                        "--launch-skip",
                        "1",
                        "--launch-count",
                        "1",
                        "--cache-control",
                        "none",
                        "--clock-control",
                        "none",
                        "--section",
                        "SpeedOfLight",
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
                        "--section",
                        "InstructionStats",
                        "-o",
                        ncu_dir / stem,
                        "--force-overwrite",
                        *command_for(shape, config, 1),
                    ],
                    stdout=log,
                )
        for page in ("raw", "details"):
            output = ncu_dir / f"{stem}_{page}.csv"
            if output.exists() and output.stat().st_size:
                continue
            with output.open("w") as stream:
                run(
                    [NCU, "--import", report, "--csv", "--page", page],
                    stdout=stream,
                )


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    build()
    collect_event()
    collect_ncu()
    print("Experiment collection complete.", flush=True)


if __name__ == "__main__":
    main()
