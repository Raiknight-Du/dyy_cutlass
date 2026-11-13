#!/usr/bin/env python3
import csv
import pathlib
import random
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = (
    ROOT
    / "dyy_test/op1/results/shape11_smem_tiling_analysis_20260727"
    / "stage_fixed_retest/occupancy_pad"
)
SOURCE = EXP / "op1_shape11_stage_fixed_pad_pure_matmul.cu"
BINARY = EXP / "op1_shape11_stage_fixed_pure_matmul_pad64k"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

CASES = (
    (0, "Stage1Pad64K", "M128_N192_K128_E128x64_C1x1x1_S1"),
    (1, "Stage2Pad64K", "M128_N192_K128_E128x64_C1x1x1_S2"),
)
SAMPLES = 40
ITERATIONS = 100
RESULT_RE = re.compile(
    r"^RESULT,11,([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
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


def command(config_index, iterations):
    return [
        str(BINARY),
        "--shape-id=11",
        f"--config-index={config_index}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def build():
    with (EXP / "build.log").open("w") as stream:
        subprocess.run(
            [
                str(NVCC),
                "-std=c++17",
                "-O3",
                "-arch=sm_100a",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-DDYY_SHAPE11_FORCE_SINGLE_CTA_SMEM_PAD",
                f"-I{EXP / 'include'}",
                "-Iinclude",
                "-Itools/util/include",
                "-Iexamples/common",
                "-Idyy_test/op1",
                str(SOURCE),
                "-o",
                str(BINARY),
            ],
            cwd=ROOT,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def collect_event():
    rows = []
    rng = random.Random(0x72A1103)
    for sample in range(1, SAMPLES + 1):
        cases = list(CASES)
        rng.shuffle(cases)
        wait_for_idle()
        for order, (config_index, label, expected) in enumerate(cases, 1):
            result = subprocess.run(
                command(config_index, ITERATIONS),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = RESULT_RE.search(result.stdout)
            if not match or match.group(1) != expected:
                raise RuntimeError(result.stdout + result.stderr)
            rows.append(
                {
                    "sample": sample,
                    "order": order,
                    "config_index": config_index,
                    "label": label,
                    "iterations": int(match.group(2)),
                    "mean_ms": float(match.group(3)),
                }
            )
        print(f"[EVENT] sample {sample}/{SAMPLES}", flush=True)

    with (EXP / "event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for config_index, label, _ in CASES:
        values = [
            row["mean_ms"] * 1000
            for row in rows
            if row["label"] == label
        ]
        stdev = statistics.stdev(values)
        summary.append(
            {
                "config_index": config_index,
                "label": label,
                "mean_us": statistics.mean(values),
                "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                "stdev_us": stdev,
                "min_us": min(values),
                "max_us": max(values),
                "samples": len(values),
                "iterations": ITERATIONS,
            }
        )
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    metrics = ",".join(
        (
            "gpu__time_duration.sum",
            "sm__ctas_active.avg",
            "sm__cycles_elapsed.avg",
        )
    )
    sections = ("LaunchStats", "Occupancy", "SchedulerStats", "WarpStateStats")
    for config_index, label, _ in CASES:
        stem = label.lower()
        report = ncu_dir / f"{stem}.ncu-rep"
        wait_for_idle()
        ncu_command = [
            str(NCU),
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
            "--metrics",
            metrics,
        ]
        for section in sections:
            ncu_command.extend(["--section", section])
        ncu_command.extend(
            [
                "-o",
                str(ncu_dir / stem),
                "--force-overwrite",
                *command(config_index, 1),
            ]
        )
        with (ncu_dir / f"{stem}.log").open("w") as stream:
            subprocess.run(
                ncu_command,
                cwd=ROOT,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for page in ("raw", "details"):
            with (ncu_dir / f"{stem}_{page}.csv").open("w") as stream:
                subprocess.run(
                    [str(NCU), "--import", str(report), "--csv", "--page", page],
                    cwd=ROOT,
                    text=True,
                    stdout=stream,
                    check=True,
                )


def main():
    build()
    collect_event()
    collect_ncu()


if __name__ == "__main__":
    main()
