#!/usr/bin/env python3
import pathlib
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape11_bottleneck_ablation_pure_matmul_20260726"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
NCU = pathlib.Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")
METRICS = (
    "gpu__time_duration.sum",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_tmem.sum",
    "sm__sass_inst_executed_op_tma_ld.sum",
    "sm__sass_inst_executed_op_tma_st.sum",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
)
SECTIONS = ("LaunchStats", "SchedulerStats", "WarpStateStats", "InstructionStats")


def run(command, output):
    with output.open("w") as stream:
        subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            text=True,
            stdout=stream,
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


def kernel_command(epilogue):
    return [
        EXP / "bin" / f"{epilogue}_full",
        "--shape-id=11",
        "--config-index=78",
        "--iterations=1",
        f"--dataset-root={DATASET}",
    ]


def main():
    for epilogue in ("e64", "e96"):
        for repetition in range(1, 4):
            stem = f"{epilogue}_full_repeat_r{repetition}"
            report = EXP / "ncu" / f"{stem}.ncu-rep"
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
                    *kernel_command(epilogue),
                ]
            )
            run(command, EXP / "ncu" / f"{stem}.log")
            for page in ("raw", "details"):
                run(
                    [NCU, "--import", report, "--csv", "--page", page],
                    EXP / "ncu" / f"{stem}_{page}.csv",
                )

    for repetition in range(1, 4):
        stem = f"e96_full_source_r{repetition}"
        report = EXP / "ncu" / f"{stem}.ncu-rep"
        wait_for_idle()
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
                *kernel_command("e96"),
            ],
            EXP / "ncu" / f"{stem}.log",
        )
        run(
            [NCU, "--import", report, "--csv", "--page", "source"],
            EXP / "ncu" / f"{stem}_source.csv",
        )


if __name__ == "__main__":
    main()
