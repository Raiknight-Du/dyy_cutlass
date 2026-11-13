#!/usr/bin/env python3
import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape5_overhead_decomposition_pure_matmul_20260725"
PREV = ROOT / "dyy_test/op1/results/shape5_multicast_notc_pure_matmul_20260725"
SOURCE = EXP / "op1_shape5_overhead_pure_matmul.cu"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

CONFIG = 36
SHAPE = 5
SAMPLES = 5
ITERATIONS = 50

BUILD_OVERLAYS = {
    "shell_utccp": "include_shell_utccp",
    "shell_noutccp": "include_shell_noutccp",
    "shell_nodstore": "include_shell_nodstore",
    "shell_noepi": "include_shell_noepi",
}

REFERENCE_BINARIES = {
    "full": PREV / "bin/op1_shape5_multicast_full",
    "notc": PREV / "bin/op1_shape5_multicast_notc",
    "notma": PREV / "bin/op1_shape5_multicast_notma",
}

METRICS = (
    "gpu__time_duration.sum",
    "lts__t_bytes.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
    "sm__sass_inst_executed_op_tma_ld.sum",
    "sm__sass_inst_executed_op_tma_st.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_tmem.sum",
    "sm__mem_tensor_reads_op_ldt.sum",
    "sm__mem_tensor_writes_op_stt.sum",
    "sm__mem_tensor_writes_op_utccp.sum",
    "sm__mem_tensor_writes_op_utcmma.sum",
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


def build():
    for name, overlay in BUILD_OVERLAYS.items():
        binary = EXP / name
        if binary.exists():
            continue
        command = [
            NVCC,
            "-std=c++17",
            "-O3",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-DOVERHEAD_CONFIG36_ONLY",
            f"-I{EXP / overlay}",
        ]
        command.extend(
            [
                "-Iinclude",
                "-Itools/util/include",
                "-Iexamples/common",
                "-Idyy_test/op1",
                SOURCE,
                "-o",
                binary,
            ]
        )
        print(f"[BUILD] {name}", flush=True)
        with (EXP / f"build_{name}.log").open("w") as stream:
            run(command, stdout=stream)

    empty_binary = EXP / "empty_cluster_baseline"
    if not empty_binary.exists():
        print("[BUILD] empty_cluster_baseline", flush=True)
        with (EXP / "build_empty_cluster_baseline.log").open("w") as stream:
            run(
                [
                    NVCC,
                    "-std=c++17",
                    "-O3",
                    "-arch=sm_100a",
                    "-lineinfo",
                    EXP / "empty_cluster_baseline.cu",
                    "-o",
                    empty_binary,
                ],
                stdout=stream,
            )


def gemm_command(binary, iterations):
    return [
        binary,
        f"--shape-id={SHAPE}",
        f"--config-index={CONFIG}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    pattern = re.compile(
        r"^RESULT,(\d+),([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    binaries = dict(REFERENCE_BINARIES)
    binaries.update({name: EXP / name for name in BUILD_OVERLAYS})
    rows = []
    for name, binary in binaries.items():
        for sample in range(1, SAMPLES + 1):
            wait_for_idle()
            result = subprocess.run(
                [str(item) for item in gemm_command(binary, ITERATIONS)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = pattern.search(result.stdout)
            if not match:
                raise RuntimeError(f"missing RESULT for {name}:\n{result.stdout}")
            rows.append(
                {
                    "variant": name,
                    "sample": sample,
                    "iterations": int(match.group(3)),
                    "mean_ms": float(match.group(4)),
                }
            )
            print(f"[EVENT] {name} sample {sample}", flush=True)

    with (EXP / "event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for name in binaries:
        values = [row["mean_ms"] for row in rows if row["variant"] == name]
        stdev = statistics.stdev(values)
        summary.append(
            {
                "variant": name,
                "mean_ms": statistics.mean(values),
                "stdev_ms": stdev,
                "ci95_half_ms": 1.96 * stdev / len(values) ** 0.5,
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    wait_for_idle()
    with (EXP / "empty_cluster_event_raw.txt").open("w") as stream:
        run([EXP / "empty_cluster_baseline"], stdout=stream)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    binaries = {name: EXP / name for name in BUILD_OVERLAYS}
    for name, binary in binaries.items():
        wait_for_idle()
        report = ncu_dir / f"{name}.ncu-rep"
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
                ncu_dir / name,
                "--force-overwrite",
                *gemm_command(binary, 1),
            ]
        )
        print(f"[NCU] {name}", flush=True)
        with (ncu_dir / f"{name}.log").open("w") as stream:
            run(command, stdout=stream)
        for page in ("raw", "details"):
            with (ncu_dir / f"{name}_{page}.csv").open("w") as stream:
                run([NCU, "--import", report, "--csv", "--page", page], stdout=stream)


def main():
    build()
    if not (EXP / "event_summary.csv").exists():
        collect_event()
    elif not (EXP / "empty_cluster_event_raw.txt").exists():
        wait_for_idle()
        with (EXP / "empty_cluster_event_raw.txt").open("w") as stream:
            run([EXP / "empty_cluster_baseline"], stdout=stream)
    collect_ncu()
    print("Overhead decomposition complete.", flush=True)


if __name__ == "__main__":
    main()
