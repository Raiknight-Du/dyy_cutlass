#!/usr/bin/env python3
"""Measure which non-input paths overlap OP1 shape8 UMMA work."""

import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape8_cluster_ablation_pure_matmul_20260725"
SOURCE = EXP / "op1_shape8_cluster_ablation_pure_matmul.cu"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

SHAPE = 8
CONFIG = 0
SAMPLES = 25
ITERATIONS = 10

OVERLAYS = {
    "notma_noutccp": "include_notma_noutccp",
    "notma_noutccp_nodstore": "include_notma_noutccp_nodstore",
    "notma_noutccp_noepi": "include_notma_noutccp_noepi",
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
    "sm__mem_tensor_writes_op_utccp.sum",
    "sm__mem_tensor_writes_op_utcmma.sum",
    "sm__inst_executed.sum",
    "sm__ctas_active.avg",
    "sm__cycles_active.avg",
)

SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "InstructionStats",
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


def command_for(binary, iterations):
    return [
        binary,
        f"--shape-id={SHAPE}",
        f"--config-index={CONFIG}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def build():
    for variant, overlay in OVERLAYS.items():
        binary = EXP / "bin" / variant
        command = [
            NVCC,
            "-std=c++17",
            "-O3",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-DOVERHEAD_TARGETS_ONLY",
            f"-I{EXP / overlay}",
            "-Iinclude",
            "-Itools/util/include",
            "-Iexamples/common",
            "-Idyy_test/op1",
            SOURCE,
            "-o",
            binary,
        ]
        print(f"[BUILD] {variant}", flush=True)
        with (EXP / f"build_{variant}.log").open("w") as stream:
            run(command, stdout=stream)


def collect_event():
    pattern = re.compile(
        r"^RESULT,(\d+),([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    rows = []
    for variant in OVERLAYS:
        binary = EXP / "bin" / variant
        for sample in range(1, SAMPLES + 1):
            wait_for_idle()
            result = subprocess.run(
                command_for(binary, ITERATIONS),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = pattern.search(result.stdout)
            if not match:
                raise RuntimeError(f"missing RESULT for {variant}:\n{result.stdout}")
            rows.append(
                {
                    "shape": SHAPE,
                    "config_index": CONFIG,
                    "variant": variant,
                    "sample": sample,
                    "iterations": int(match.group(3)),
                    "mean_ms": float(match.group(4)),
                }
            )
            print(f"[EVENT] {variant} sample={sample}", flush=True)

    with (EXP / "mma_overlap_event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for variant in OVERLAYS:
        values = [
            row["mean_ms"] * 1000
            for row in rows
            if row["variant"] == variant
        ]
        stdev = statistics.stdev(values)
        summaries.append(
            {
                "shape": SHAPE,
                "config_index": CONFIG,
                "variant": variant,
                "mean_us": statistics.mean(values),
                "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                "min_us": min(values),
                "max_us": max(values),
            }
        )
    with (EXP / "mma_overlap_event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    for variant in OVERLAYS:
        stem = f"shape8_c0_{variant}"
        report = ncu_dir / f"{stem}.ncu-rep"
        wait_for_idle()
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
                *command_for(EXP / "bin" / variant, 1),
            ]
        )
        print(f"[NCU] {variant}", flush=True)
        with (ncu_dir / f"{stem}.log").open("w") as stream:
            run(command, stdout=stream)

        with (ncu_dir / f"{stem}_raw.csv").open("w") as stream:
            run(
                [
                    NCU,
                    "--import",
                    report,
                    "--csv",
                    "--page",
                    "raw",
                ],
                stdout=stream,
            )
        with (ncu_dir / f"{stem}_details.csv").open("w") as stream:
            run(
                [
                    NCU,
                    "--import",
                    report,
                    "--csv",
                    "--page",
                    "details",
                ],
                stdout=stream,
            )


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    build()
    collect_event()
    collect_ncu()
    print("[DONE] OP1 shape8 MMA-overlap matrix", flush=True)


if __name__ == "__main__":
    main()
