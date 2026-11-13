#!/usr/bin/env python3
import csv
import pathlib
import random
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape11_m128n256_e128x64_c1_analysis_20260727"
SOURCE = EXP / "op1_shape11_m128n256_ablation_pure_matmul.cu"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")

VARIANTS = {
    "full": None,
    "notc": "include_notc",
    "notma": "include_notma",
    "shell_utccp": "include_shell_utccp",
    "shell_noutccp": "include_shell_noutccp",
    "shell_nodstore": "include_shell_nodstore",
    "shell_noepi": "include_shell_noepi",
}
EPILOGUES = ("e64",)
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
    "sm__mem_tensor_reads_op_ldt.sum",
    "sm__mem_tensor_writes_op_utccp.sum",
    "sm__inst_executed.sum",
    "sm__ctas_active.avg",
)
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "InstructionStats",
)


def run(command, *, capture=False, output=None):
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        text=True,
        capture_output=capture,
        stdout=output,
        stderr=subprocess.STDOUT if output is not None else None,
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
    for epilogue in EPILOGUES:
        for variant, overlay in VARIANTS.items():
            binary = EXP / "bin" / f"{epilogue}_{variant}"
            if binary.exists():
                continue
            command = [
                NVCC,
                "-std=c++17",
                "-O3",
                "-arch=sm_100a",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-DOVERHEAD_TARGETS_ONLY",
            ]
            if epilogue == "e96":
                command.append("-DSHAPE11_EPI96")
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
            print(f"[BUILD] {epilogue} {variant}", flush=True)
            with (EXP / f"build_{epilogue}_{variant}.log").open("w") as stream:
                run(command, output=stream)

def command_for(binary, iterations):
    return [
        binary,
        "--shape-id=11",
        "--config-index=78",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    rows = []
    order_rng = random.Random(0x72A11)
    cases = [(epi, variant) for epi in EPILOGUES for variant in VARIANTS]
    for sample in range(1, SAMPLES + 1):
        order_rng.shuffle(cases)
        wait_for_idle()
        for order, (epilogue, variant) in enumerate(cases, 1):
            binary = EXP / "bin" / f"{epilogue}_{variant}"
            result = run(command_for(binary, ITERATIONS), capture=True)
            match = RESULT_RE.search(result.stdout)
            if not match:
                raise RuntimeError(
                    f"Missing result for {epilogue}/{variant}:\n{result.stdout}"
                )
            rows.append(
                {
                    "sample": sample,
                    "order": order,
                    "epilogue": epilogue,
                    "variant": variant,
                    "config": match.group(1),
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
    for epilogue in EPILOGUES:
        for variant in VARIANTS:
            values = [
                row["mean_ms"] * 1000
                for row in rows
                if row["epilogue"] == epilogue and row["variant"] == variant
            ]
            stdev = statistics.stdev(values)
            summary.append(
                {
                    "epilogue": epilogue,
                    "variant": variant,
                    "mean_us": statistics.mean(values),
                    "ci95_half_us": 1.96 * stdev / len(values) ** 0.5,
                    "stdev_us": stdev,
                    "min_us": min(values),
                    "max_us": max(values),
                    "samples": len(values),
                    "iterations": ITERATIONS,
                }
            )
    with (EXP / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

def collect_ncu():
    selected = (
        ("e64", "full"),
        ("e64", "notc"),
        ("e64", "notma"),
        ("e64", "shell_utccp"),
        ("e64", "shell_noutccp"),
        ("e64", "shell_nodstore"),
        ("e64", "shell_noepi"),
    )
    for epilogue, variant in selected:
        stem = f"{epilogue}_{variant}"
        report = EXP / "ncu" / f"{stem}.ncu-rep"
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
                    EXP / "ncu" / stem,
                    "--force-overwrite",
                    *command_for(EXP / "bin" / stem, 1),
                ]
            )
            with (EXP / "ncu" / f"{stem}.log").open("w") as stream:
                run(command, output=stream)
        for page in ("raw", "details"):
            output = EXP / "ncu" / f"{stem}_{page}.csv"
            with output.open("w") as stream:
                run([NCU, "--import", report, "--csv", "--page", page], output=stream)

    for variant in ("full", "shell_noepi"):
        for repetition in range(1, 4):
            stem = f"e64_{variant}_source_r{repetition}"
            report = EXP / "ncu" / f"{stem}.ncu-rep"
            if report.exists():
                continue
            wait_for_idle()
            with (EXP / "ncu" / f"{stem}.log").open("w") as stream:
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
                        "-o",
                        EXP / "ncu" / stem,
                        "--force-overwrite",
                        *command_for(EXP / "bin" / f"e64_{variant}", 1),
                    ],
                    output=stream,
                )


def main():
    (EXP / "bin").mkdir(parents=True, exist_ok=True)
    (EXP / "ncu").mkdir(exist_ok=True)
    build()
    collect_event()
    collect_ncu()
    print("Shape11 controlled bottleneck experiment complete.", flush=True)


if __name__ == "__main__":
    main()
