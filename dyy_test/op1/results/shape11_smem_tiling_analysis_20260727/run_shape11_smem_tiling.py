#!/usr/bin/env python3
import csv
import pathlib
import random
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape11_smem_tiling_analysis_20260727"
BINARY = ROOT / "dyy_test/op1/bin/op1_top20"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

CASES = (
    (78, "M128_N192_K128_E128x64_C1x1x1"),
    (77, "M256_N192_K128_E128x64_C2x1x1"),
    (63, "M128_N256_K128_E128x128_C1x1x1"),
    (11, "M256_N256_K128_E128x64_C2x1x1"),
)
SAMPLES = 40
ITERATIONS = 100
RESULT_RE = re.compile(
    r"^RESULT,11,([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
)

METRICS = (
    "gpu__time_duration.sum",
    "lts__t_bytes.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
    "sm__sass_inst_executed_op_tma_ld.sum",
    "sm__sass_inst_executed_op_tma_st.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_tmem.sum",
    "sm__inst_executed.sum",
)
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "InstructionStats",
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


def collect_event():
    rows = []
    order_rng = random.Random(0x72A1101)
    cases = list(CASES)
    for sample in range(1, SAMPLES + 1):
        order_rng.shuffle(cases)
        wait_for_idle()
        for order, (config_index, expected) in enumerate(cases, 1):
            result = subprocess.run(
                command(config_index, ITERATIONS),
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            match = RESULT_RE.search(result.stdout)
            if not match or match.group(1) != expected:
                raise RuntimeError(
                    f"Unexpected result for config {config_index}:\n{result.stdout}"
                )
            rows.append(
                {
                    "sample": sample,
                    "order": order,
                    "config_index": config_index,
                    "config": expected,
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
    for config_index, config in CASES:
        values = [
            row["mean_ms"] * 1000
            for row in rows
            if row["config_index"] == config_index
        ]
        stdev = statistics.stdev(values)
        summary.append(
            {
                "config_index": config_index,
                "config": config,
                "mean_us": statistics.mean(values),
                "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                "stdev_us": stdev,
                "min_us": min(values),
                "max_us": max(values),
                "samples": len(values),
                "iterations": ITERATIONS,
            }
        )
    baseline = summary[0]["mean_us"]
    for row in summary:
        row["delta_us"] = row["mean_us"] - baseline
        row["delta_pct"] = 100 * (row["mean_us"] / baseline - 1)
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def collect_ncu():
    for config_index, config in CASES:
        stem = f"cfg{config_index}_{config}"
        report = EXP / "ncu" / f"{stem}.ncu-rep"
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
            ",".join(METRICS),
        ]
        for section in SECTIONS:
            ncu_command.extend(["--section", section])
        ncu_command.extend(
            [
                "-o",
                str(EXP / "ncu" / stem),
                "--force-overwrite",
                *command(config_index, 1),
            ]
        )
        with (EXP / "ncu" / f"{stem}.log").open("w") as stream:
            subprocess.run(
                ncu_command,
                cwd=ROOT,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for page in ("raw", "details"):
            with (EXP / "ncu" / f"{stem}_{page}.csv").open("w") as stream:
                subprocess.run(
                    [str(NCU), "--import", str(report), "--csv", "--page", page],
                    cwd=ROOT,
                    text=True,
                    stdout=stream,
                    check=True,
                )


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "ncu").mkdir(exist_ok=True)
    collect_event()
    collect_ncu()
    print("Shape11 SMEM/tiling experiment complete.", flush=True)


if __name__ == "__main__":
    main()
