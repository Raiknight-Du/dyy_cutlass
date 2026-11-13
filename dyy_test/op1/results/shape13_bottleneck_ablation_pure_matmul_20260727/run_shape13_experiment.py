#!/usr/bin/env python3
import csv
import pathlib
import random
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape13_bottleneck_ablation_pure_matmul_20260727"
SOURCE = EXP / "op1_shape13_bottleneck_pure_matmul.cu"
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
EPILOGUES = ("e128", "eauto")
SAMPLES = 40
ITERATIONS = 100
RESULT_RE = re.compile(
    r"^RESULT,13,([^,]+),ok,(\d+),([0-9.eE+-]+)$", re.MULTILINE
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
BUILD_CASES = (
    *(("e128", variant) for variant in VARIANTS),
    ("eauto", "full"),
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
    for epilogue, variant in BUILD_CASES:
        overlay = VARIANTS[variant]
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
        if epilogue == "eauto":
            command.append("-DSHAPE13_EPIAUTO")
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

    control = EXP / "bin" / "shape13_resource_control"
    if not control.exists():
        with (EXP / "build_shape13_resource_control.log").open("w") as stream:
            run(
                [
                    NVCC,
                    "-std=c++17",
                    "-O3",
                    "-arch=sm_100a",
                    "-lineinfo",
                    "-Iinclude",
                    EXP / "shape13_resource_control.cu",
                    "-o",
                    control,
                ],
                output=stream,
            )


def command_for(binary, iterations):
    return [
        binary,
        "--shape-id=13",
        "--config-index=63",
        f"--iterations={iterations}",
        f"--dataset-root={DATASET}",
    ]


def collect_event():
    rows = []
    order_rng = random.Random(0x72A11)
    cases = list(BUILD_CASES)
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
    for epilogue, variant in BUILD_CASES:
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

    with (EXP / "resource_control_raw.txt").open("w") as stream:
        wait_for_idle()
        run([EXP / "bin" / "shape13_resource_control"], output=stream)


def collect_ncu():
    selected = (
        ("e128", "full"),
        ("e128", "notc"),
        ("e128", "notma"),
        ("e128", "shell_utccp"),
        ("e128", "shell_noutccp"),
        ("e128", "shell_nodstore"),
        ("e128", "shell_noepi"),
        ("eauto", "full"),
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
            stem = f"e128_{variant}_source_r{repetition}"
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
                        *command_for(EXP / "bin" / f"e128_{variant}", 1),
                    ],
                    output=stream,
                )


def main():
    (EXP / "bin").mkdir(parents=True, exist_ok=True)
    (EXP / "ncu").mkdir(exist_ok=True)
    build()
    collect_event()
    collect_ncu()
    print("Shape13 controlled bottleneck experiment complete.", flush=True)


if __name__ == "__main__":
    main()
