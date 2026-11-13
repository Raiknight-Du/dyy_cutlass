#!/usr/bin/env python3
import csv
import pathlib
import random
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape8_epi_k_tma_ablation_pure_matmul_20260725"
SOURCE = EXP / "op1_shape8_epi_k_tma_ablation_pure_matmul.cu"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

CASES = (
    (0, "K256_E64x64", "M256_N256_K256_E64x64_C2x1x1", 256, "64x64"),
    (3, "K256_E128x64", "M256_N256_K256_E128x64_C2x1x1", 256, "128x64"),
    (18, "K256_E128x128", "M256_N256_K256_E128x128_C2x1x1", 256, "128x128"),
    (100, "K128_E64x64", "M256_N256_K128_E64x64_C2x1x1", 128, "64x64"),
    (11, "K128_E128x64", "M256_N256_K128_E128x64_C2x1x1", 128, "128x64"),
    (14, "K128_E128x128", "M256_N256_K128_E128x128_C2x1x1", 128, "128x128"),
)

BUILD_OVERLAYS = {
    "full": None,
    "notc": "include_notc",
    "notma": "include_notma",
    "shell": "include_shell_utccp",
}

SAMPLES = 20
ITERATIONS = 10

METRICS = (
    "gpu__time_duration.sum",
    "lts__t_bytes.sum",
    "lts__t_sector_hit_rate.pct",
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
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "launch__shared_mem_per_block",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
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


def gpu_state():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    util, clock, temperature = [
        int(item.strip()) for item in result.stdout.splitlines()[0].split(",")
    ]
    return util, clock, temperature


def wait_for_idle():
    for _ in range(180):
        util, clock, temperature = gpu_state()
        if util <= 2:
            return clock, temperature
        time.sleep(1)
    raise RuntimeError("GPU did not become idle")


def build():
    for name, overlay in BUILD_OVERLAYS.items():
        binary = EXP / "bin" / name
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
        if overlay:
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


def command_for(binary, config, iterations):
    return [
        binary,
        "--shape-id=8",
        f"--config-index={config}",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    result_pattern = re.compile(
        r"^RESULT,8,([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
    )
    resource_pattern = re.compile(
        r"^RESOURCE,([^,]+),stages=(\d+),mainloop_tensor_B=(\d+),"
        r"mainloop_pipeline_B=(\d+),epilogue_tensor_B=(\d+),"
        r"epilogue_pipeline_B=(\d+),total_shared_B=(\d+),"
        r"ab_tma_transaction_B=(\d+),sf_tma_transaction_B=(\d+)$",
        re.MULTILINE,
    )
    rows = []
    resources = {}
    jobs = [
        (case, variant)
        for case in CASES
        for variant in BUILD_OVERLAYS
    ]
    rng = random.Random(20260725)

    for sample in range(1, SAMPLES + 1):
        rng.shuffle(jobs)
        for case, variant in jobs:
            config, key, expected_name, tile_k, epilogue = case
            clock, temperature = wait_for_idle()
            result = subprocess.run(
                command_for(EXP / "bin" / variant, config, ITERATIONS),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = result_pattern.search(result.stdout)
            if not match or match.group(1) != expected_name:
                raise RuntimeError(
                    f"bad RESULT for {key} {variant}:\n{result.stdout}"
                )
            resource_match = resource_pattern.search(result.stdout)
            if not resource_match:
                raise RuntimeError(
                    f"missing RESOURCE for {key} {variant}:\n{result.stdout}"
                )
            resources[key] = {
                "case": key,
                "config_index": config,
                "config": expected_name,
                "tile_k": tile_k,
                "epilogue": epilogue,
                "stages": int(resource_match.group(2)),
                "mainloop_tensor_B": int(resource_match.group(3)),
                "mainloop_pipeline_B": int(resource_match.group(4)),
                "epilogue_tensor_B": int(resource_match.group(5)),
                "epilogue_pipeline_B": int(resource_match.group(6)),
                "total_shared_B": int(resource_match.group(7)),
                "ab_tma_transaction_B": int(resource_match.group(8)),
                "sf_tma_transaction_B": int(resource_match.group(9)),
            }
            rows.append(
                {
                    "case": key,
                    "config_index": config,
                    "config": expected_name,
                    "tile_k": tile_k,
                    "epilogue": epilogue,
                    "variant": variant,
                    "sample": sample,
                    "iterations": int(match.group(2)),
                    "mean_ms": float(match.group(3)),
                    "idle_sm_clock_mhz": clock,
                    "temperature_c": temperature,
                }
            )
            print(
                f"[EVENT] sample={sample:02d} {key} {variant} "
                f"{float(match.group(3)) * 1000:.3f} us",
                flush=True,
            )

    with (EXP / "event_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for config, key, expected_name, tile_k, epilogue in CASES:
        for variant in BUILD_OVERLAYS:
            selected = [
                row for row in rows
                if row["case"] == key and row["variant"] == variant
            ]
            values = [row["mean_ms"] * 1000 for row in selected]
            stdev = statistics.stdev(values)
            summaries.append(
                {
                    "case": key,
                    "config_index": config,
                    "config": expected_name,
                    "tile_k": tile_k,
                    "epilogue": epilogue,
                    "variant": variant,
                    "mean_us": statistics.mean(values),
                    "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                    "min_us": min(values),
                    "max_us": max(values),
                    "mean_idle_clock_mhz": statistics.mean(
                        row["idle_sm_clock_mhz"] for row in selected
                    ),
                }
            )
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    with (EXP / "resource_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=next(iter(resources.values())).keys()
        )
        writer.writeheader()
        writer.writerows(resources[key] for _, key, *_ in CASES)


def collect_ncu():
    ncu_dir = EXP / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    for config, key, _, _, _ in CASES:
        for variant in BUILD_OVERLAYS:
            stem = f"{key}_{variant}"
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
                    *command_for(EXP / "bin" / variant, config, 1),
                ]
            )
            print(f"[NCU] {stem}", flush=True)
            with (ncu_dir / f"{stem}.log").open("w") as stream:
                run(command, stdout=stream)

            with (ncu_dir / f"{stem}_raw.csv").open("w") as stream:
                run(
                    [NCU, "--import", report, "--csv", "--page", "raw"],
                    stdout=stream,
                )
            with (ncu_dir / f"{stem}_details.csv").open("w") as stream:
                run(
                    [NCU, "--import", report, "--csv", "--page", "details"],
                    stdout=stream,
                )


def main():
    build()
    collect_event()
    collect_ncu()


if __name__ == "__main__":
    main()
