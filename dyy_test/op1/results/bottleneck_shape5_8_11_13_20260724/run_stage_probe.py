#!/usr/bin/env python3
import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/bottleneck_shape5_8_11_13_20260724"
SOURCE = EXP / "op1_stage_probe.cu"
BINARY = ROOT / "dyy_test/op1/bin/op1_stage_probe"
AUTO_BINARY = ROOT / "dyy_test/op1/bin/op1_bottleneck_probe"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

FIXED_CASES = (
    (11, "stage1", 0),
    (11, "stage2", 1),
    (13, "stage1", 2),
    (13, "stage2", 3),
)
NCU_CASES = (
    (11, "auto", AUTO_BINARY, 78),
    (11, "stage1", BINARY, 0),
    (11, "stage2", BINARY, 1),
    (13, "auto", AUTO_BINARY, 63),
    (13, "stage1", BINARY, 2),
    (13, "stage2", BINARY, 3),
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


def command(binary, shape, config, iterations):
    return [
        binary,
        f"--shape-id={shape}",
        f"--config-index={config}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def build():
    with (EXP / "stage_probe.build.log").open("w") as log:
        subprocess.run(
            [
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
            ],
            cwd=ROOT,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def collect_event():
    pattern = re.compile(
        r"^RESULT,(\d+),([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    raw = []
    for shape, label, config in FIXED_CASES:
        for sample in range(25):
            wait_for_idle()
            result = subprocess.run(
                command(BINARY, shape, config, 10),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = pattern.search(result.stdout)
            if not match:
                raise RuntimeError(result.stdout + result.stderr)
            raw.append(
                {
                    "shape": shape,
                    "label": label,
                    "config_index": config,
                    "variant": match.group(2),
                    "sample": sample,
                    "mean_ms": float(match.group(4)),
                }
            )
        print(f"[STAGE EVENT] shape={shape} {label}", flush=True)
    with (EXP / "stage_event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw[0].keys())
        writer.writeheader()
        writer.writerows(raw)
    summary = []
    for shape, label, config in FIXED_CASES:
        values = [
            row["mean_ms"]
            for row in raw
            if row["shape"] == shape and row["config_index"] == config
        ]
        variant = next(
            row["variant"]
            for row in raw
            if row["shape"] == shape and row["config_index"] == config
        )
        stdev = statistics.stdev(values)
        summary.append(
            {
                "shape": shape,
                "label": label,
                "config_index": config,
                "variant": variant,
                "mean_ms": statistics.mean(values),
                "stdev_ms": stdev,
                "ci95_half_ms": 1.96 * stdev / len(values) ** 0.5,
            }
        )
    with (EXP / "stage_event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def collect_ncu():
    out = EXP / "ncu_stage"
    out.mkdir(exist_ok=True)
    for shape, label, binary, config in NCU_CASES:
        stem = f"shape{shape:02d}_{label}"
        report = out / f"{stem}.ncu-rep"
        wait_for_idle()
        with (out / f"{stem}.log").open("w") as log:
            subprocess.run(
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
                    "--metrics",
                    "sm__ctas_active.avg",
                    "-o",
                    out / stem,
                    "--force-overwrite",
                    *command(binary, shape, config, 1),
                ],
                cwd=ROOT,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with (out / f"{stem}_raw.csv").open("w") as stream:
            subprocess.run(
                [NCU, "--import", report, "--csv", "--page", "raw"],
                cwd=ROOT,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"[STAGE NCU] shape={shape} {label}", flush=True)


def main():
    build()
    collect_event()
    collect_ncu()


if __name__ == "__main__":
    main()
