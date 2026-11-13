#!/usr/bin/env python3
import csv
import pathlib
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OUT = ROOT / "dyy_test/op1/results/bottleneck_13shape_20260724"
BINARY = ROOT / "dyy_test/op1/bin/op1_best"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")
NSYS = pathlib.Path("/opt/nvidia/nsight-systems/2026.1.3/bin/nsys")


def run(args, stdout_path=None, stderr_path=None):
    stdout = subprocess.PIPE if stdout_path is None else stdout_path.open("w")
    stderr = subprocess.STDOUT if stderr_path is None else stderr_path.open("w")
    try:
        return subprocess.run(
            [str(arg) for arg in args],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=True,
        )
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()


def wait_for_idle():
    for _ in range(120):
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
    raise RuntimeError("GPU did not become idle")


def collect_ncu(shape):
    stem = f"shape{shape:02d}"
    report = OUT / f"{stem}_core.ncu-rep"
    if not report.exists():
        wait_for_idle()
        print(f"[NCU] shape {shape}", flush=True)
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
                OUT / f"{stem}_core",
                "--force-overwrite",
                BINARY,
                f"--shape-id={shape}",
                "--iterations=1",
                f"--dataset-root={DATASET}",
            ],
            OUT / f"{stem}_ncu.log",
        )

    for page in ("raw", "details"):
        output = OUT / f"{stem}_{page}.csv"
        if output.exists() and output.stat().st_size:
            continue
        print(f"[EXPORT] shape {shape} {page}", flush=True)
        run(
            [NCU, "--import", report, "--csv", "--page", page],
            output,
            OUT / f"{stem}_{page}.err",
        )


def collect_nsys(shape):
    stem = f"shape{shape:02d}"
    report = OUT / f"{stem}_nsys.nsys-rep"
    if not report.exists():
        wait_for_idle()
        print(f"[NSYS] shape {shape}", flush=True)
        run(
            [
                NSYS,
                "profile",
                "--trace=cuda",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=true",
                "-o",
                OUT / f"{stem}_nsys",
                BINARY,
                f"--shape-id={shape}",
                "--iterations=10",
                f"--dataset-root={DATASET}",
            ],
            OUT / f"{stem}_nsys.log",
        )

    stats = OUT / f"{stem}_nsys_stats.csv"
    if not stats.exists() or not stats.stat().st_size:
        print(f"[NSYS STATS] shape {shape}", flush=True)
        run(
            [
                NSYS,
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--report",
                "cuda_api_sum",
                "--format",
                "csv",
                report,
            ],
            stats,
            OUT / f"{stem}_nsys_stats.err",
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for shape in range(1, 14):
        collect_ncu(shape)
    for shape in range(1, 14):
        collect_nsys(shape)
    print("Collection complete.", flush=True)


if __name__ == "__main__":
    main()
