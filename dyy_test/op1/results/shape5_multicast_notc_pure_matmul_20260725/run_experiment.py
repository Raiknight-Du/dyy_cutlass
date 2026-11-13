#!/usr/bin/env python3
import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape5_multicast_notc_pure_matmul_20260725"
SOURCE = EXP / "op1_shape5_multicast_pure_matmul.cu"
OVERLAY = EXP / "include"
BIN_DIR = EXP / "bin"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

SHAPE = 5
SAMPLES = 5
ITERATIONS = 50
CONFIGS = (
    ("C2x1_none", 4),
    ("C2x2_A_SFA", 36),
    ("C4x1_B_SFB", 33),
)
BUILDS = (
    ("full", False),
    ("notc", True),
)

METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_self.sum",
    "l1tex__tmain_requests.sum",
    "sm__sass_inst_executed_op_tma_ld.sum",
    "sm__sass_inst_executed_op_tma_st.sum",
)

SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
)


def run(command, stdout=None):
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        text=True,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        check=True,
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


def binary_for(build):
    return BIN_DIR / f"op1_shape5_multicast_{build}"


def build_all():
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    for build, use_overlay in BUILDS:
        command = [
            NVCC,
            "-std=c++17",
            "-O3",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ]
        if use_overlay:
            command.append(f"-I{OVERLAY}")
        command.extend(
            [
                "-Iinclude",
                "-Itools/util/include",
                "-Iexamples/common",
                "-Idyy_test/op1",
                SOURCE,
                "-o",
                binary_for(build),
            ]
        )
        print(f"[BUILD] {build}", flush=True)
        with (EXP / f"build_{build}.log").open("w") as log:
            run(command, stdout=log)


def command_for(build, config, iterations):
    return [
        binary_for(build),
        f"--shape-id={SHAPE}",
        f"--config-index={config}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    pattern = re.compile(
        r"^RESULT,(\d+),([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    rows = []
    for build, _ in BUILDS:
        for label, config in CONFIGS:
            for sample in range(1, SAMPLES + 1):
                wait_for_idle()
                result = subprocess.run(
                    [str(item) for item in command_for(build, config, ITERATIONS)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                match = pattern.search(result.stdout)
                if not match:
                    raise RuntimeError(
                        f"missing RESULT for {build} config {config}:\n{result.stdout}"
                    )
                rows.append(
                    {
                        "build": build,
                        "cluster_role": label,
                        "config_index": config,
                        "variant": match.group(2),
                        "sample": sample,
                        "iterations": int(match.group(3)),
                        "mean_ms": float(match.group(4)),
                    }
                )
                with (EXP / "event_raw.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"[EVENT] {build} {label}", flush=True)

    summary = []
    for build, _ in BUILDS:
        for label, config in CONFIGS:
            selected = [
                row for row in rows
                if row["build"] == build and row["config_index"] == config
            ]
            values = [row["mean_ms"] for row in selected]
            stdev = statistics.stdev(values)
            summary.append(
                {
                    "build": build,
                    "cluster_role": label,
                    "config_index": config,
                    "variant": selected[0]["variant"],
                    "mean_ms": statistics.mean(values),
                    "stdev_ms": stdev,
                    "ci95_half_ms": 1.96 * stdev / len(values) ** 0.5,
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "samples": len(values),
                    "iterations_per_sample": ITERATIONS,
                }
            )
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    for build, _ in BUILDS:
        for label, config in CONFIGS:
            wait_for_idle()
            stem = f"{build}_{label}_c{config:02d}"
            report = ncu_dir / f"{stem}.ncu-rep"
            command = [
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
                "--metrics",
                ",".join(METRICS),
            ]
            for section in SECTIONS:
                command.extend(["--section", section])
            command.extend(
                [
                    "-o",
                    ncu_dir / stem,
                    "--force-overwrite",
                    *command_for(build, config, 1),
                ]
            )
            print(f"[NCU] {build} {label}", flush=True)
            with (ncu_dir / f"{stem}.log").open("w") as log:
                run(command, stdout=log)
            for page in ("raw", "details"):
                with (ncu_dir / f"{stem}_{page}.csv").open("w") as stream:
                    run(
                        [NCU, "--import", report, "--csv", "--page", page],
                        stdout=stream,
                    )


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    build_all()
    collect_event()
    collect_ncu()
    print("Experiment collection complete.", flush=True)


if __name__ == "__main__":
    main()
