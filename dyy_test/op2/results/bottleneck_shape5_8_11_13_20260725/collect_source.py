#!/usr/bin/env python3
import pathlib
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OUT = ROOT / "dyy_test/op2/results/bottleneck_shape5_8_11_13_20260725"
BINARY = ROOT / "dyy_test/op2/bin/op2_best"
DATASET = ROOT / "dyy_test/op2/random_datasets"
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

SHAPES = {
    5: (768, 8192, 8192),
    8: (4096, 8192, 4096),
    11: (768, 16384, 256),
    13: (4096, 8192, 256),
}


def wait_for_idle(timeout_s=1800):
    deadline = time.monotonic() + timeout_s
    stable = 0
    while time.monotonic() < deadline:
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        utilization = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not apps and int(utilization.splitlines()[0]) <= 2:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        time.sleep(1)
    raise RuntimeError("GPU did not become idle")


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
    OUT.mkdir(parents=True, exist_ok=True)
    for shape, (m, n, k) in SHAPES.items():
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
                    f"--m={m}",
                    f"--n={n}",
                    f"--k={k}",
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
