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

METRICS = ",".join([
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
])


def wait_for_idle():
    stable = 0
    while stable < 2:
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
        util = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        stable = stable + 1 if not apps and int(util.splitlines()[0]) <= 2 else 0
        time.sleep(1)


for shape, (m, n, k) in SHAPES.items():
    wait_for_idle()
    output = OUT / f"shape{shape:02d}_traffic.csv"
    command = [
        str(NCU),
        "--target-processes", "all",
        "--launch-skip", "1",
        "--launch-count", "1",
        "--cache-control", "none",
        "--clock-control", "none",
        "--csv",
        "--page", "raw",
        "--metrics", METRICS,
        str(BINARY),
        f"--shape-id={shape}",
        f"--m={m}",
        f"--n={n}",
        f"--k={k}",
        f"--dataset-root={DATASET}",
        "--iterations=1",
    ]
    with output.open("w") as stdout, (
        OUT / f"shape{shape:02d}_traffic.err"
    ).open("w") as stderr:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=True,
        )
    print(f"shape {shape} complete", flush=True)
