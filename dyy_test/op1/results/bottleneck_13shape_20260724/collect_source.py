#!/usr/bin/env python3
import pathlib
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OUT = ROOT / "dyy_test/op1/results/bottleneck_13shape_20260724"
BINARY = ROOT / "dyy_test/op1/bin/op1_best"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")
SHAPES = (1, 5, 6, 7, 8, 10, 11, 13)


def wait_for_idle():
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if int(result.stdout.strip().splitlines()[0]) <= 2:
            return
        time.sleep(1)


def run(args, stdout_path, stderr_path=None):
    with stdout_path.open("w") as stdout:
        if stderr_path:
            with stderr_path.open("w") as stderr:
                subprocess.run(
                    [str(arg) for arg in args],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=True,
                )
        else:
            subprocess.run(
                [str(arg) for arg in args],
                cwd=ROOT,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )


def main():
    for shape in SHAPES:
        stem = f"shape{shape:02d}_source"
        report = OUT / f"{stem}.ncu-rep"
        if not report.exists():
            wait_for_idle()
            print(f"[SOURCE] shape {shape}", flush=True)
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
                    "SourceCounters",
                    "--section",
                    "SchedulerStats",
                    "--section",
                    "WarpStateStats",
                    "-o",
                    OUT / stem,
                    "--force-overwrite",
                    BINARY,
                    f"--shape-id={shape}",
                    "--iterations=1",
                    f"--dataset-root={DATASET}",
                ],
                OUT / f"{stem}.log",
            )
        source_csv = OUT / f"{stem}.csv"
        if not source_csv.exists() or not source_csv.stat().st_size:
            print(f"[EXPORT] shape {shape}", flush=True)
            run(
                [NCU, "--import", report, "--csv", "--page", "source"],
                source_csv,
                OUT / f"{stem}.err",
            )


if __name__ == "__main__":
    main()
