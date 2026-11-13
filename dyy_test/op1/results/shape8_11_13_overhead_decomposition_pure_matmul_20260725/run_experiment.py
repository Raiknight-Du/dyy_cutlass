#!/usr/bin/env python3
import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape8_11_13_overhead_decomposition_pure_matmul_20260725"
SOURCE = EXP / "op1_shape8_11_13_overhead_pure_matmul.cu"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

CASES = (
    (8, 0, "M256_N256_K256_E64x64_C2x1x1"),
    (11, 78, "M128_N192_K128_E128x64_C1x1x1"),
    (13, 63, "M128_N256_K128_E128x128_C1x1x1"),
)
SAMPLES = 25
ITERATIONS = 10

BUILD_OVERLAYS = {
    "full": None,
    "notc": "include_notc",
    "notma": "include_notma",
    "shell_utccp": "include_shell_utccp",
    "shell_noutccp": "include_shell_noutccp",
    "shell_nodstore": "include_shell_nodstore",
    "shell_noepi": "include_shell_noepi",
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


def build():
    for name, overlay in BUILD_OVERLAYS.items():
        binary = EXP / "bin" / name
        if binary.exists():
            continue
        binary.parent.mkdir(exist_ok=True)
        command = [
            NVCC,
            "-std=c++17",
            "-O3",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-DOVERHEAD_TARGETS_ONLY",
        ]
        if overlay is not None:
            command.append(f"-I{EXP / overlay}")
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

    empty_binary = EXP / "bin" / "empty_resource_baseline"
    if not empty_binary.exists():
        print("[BUILD] empty_resource_baseline", flush=True)
        with (EXP / "build_empty_resource_baseline.log").open("w") as stream:
            run(
                [
                    NVCC,
                    "-std=c++17",
                    "-O3",
                    "-arch=sm_100a",
                    "-lineinfo",
                    EXP / "empty_resource_baseline.cu",
                    "-o",
                    empty_binary,
                ],
                stdout=stream,
            )


def command_for(binary, shape, config, iterations):
    return [
        binary,
        f"--shape-id={shape}",
        f"--config-index={config}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    pattern = re.compile(
        r"^RESULT,(\d+),([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    rows = []
    for shape, config, expected_variant in CASES:
        for variant in BUILD_OVERLAYS:
            binary = EXP / "bin" / variant
            for sample in range(1, SAMPLES + 1):
                wait_for_idle()
                result = subprocess.run(
                    command_for(binary, shape, config, ITERATIONS),
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                match = pattern.search(result.stdout)
                if not match:
                    raise RuntimeError(
                        f"missing RESULT for shape={shape} {variant}:\n"
                        f"{result.stdout}"
                    )
                if match.group(2) != expected_variant:
                    raise RuntimeError(
                        f"unexpected config for shape={shape}: {match.group(2)}"
                    )
                rows.append(
                    {
                        "shape": shape,
                        "config_index": config,
                        "config": expected_variant,
                        "variant": variant,
                        "sample": sample,
                        "iterations": int(match.group(3)),
                        "mean_ms": float(match.group(4)),
                    }
                )
                print(f"[EVENT] shape={shape} {variant} sample={sample}", flush=True)

    with (EXP / "event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for shape, config, expected_variant in CASES:
        for variant in BUILD_OVERLAYS:
            values = [
                row["mean_ms"]
                for row in rows
                if row["shape"] == shape and row["variant"] == variant
            ]
            stdev = statistics.stdev(values)
            summaries.append(
                {
                    "shape": shape,
                    "config_index": config,
                    "config": expected_variant,
                    "variant": variant,
                    "mean_us": statistics.mean(values) * 1000,
                    "ci95_half_us": 1.96 * stdev / len(values) ** 0.5 * 1000,
                    "min_us": min(values) * 1000,
                    "max_us": max(values) * 1000,
                }
            )
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    wait_for_idle()
    with (EXP / "empty_resource_event_raw.txt").open("w") as stream:
        run([EXP / "bin" / "empty_resource_baseline"], stdout=stream)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    for shape, config, _ in CASES:
        for variant in BUILD_OVERLAYS:
            stem = f"shape{shape}_{variant}"
            report = ncu_dir / f"{stem}.ncu-rep"
            if not report.exists():
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
                        *command_for(EXP / "bin" / variant, shape, config, 1),
                    ]
                )
                print(f"[NCU] shape={shape} {variant}", flush=True)
                with (ncu_dir / f"{stem}.log").open("w") as stream:
                    run(command, stdout=stream)
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
    print("Shape 8/11/13 overhead decomposition complete.", flush=True)


if __name__ == "__main__":
    main()
