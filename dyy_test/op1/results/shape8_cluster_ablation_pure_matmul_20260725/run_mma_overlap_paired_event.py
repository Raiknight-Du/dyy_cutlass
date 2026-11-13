#!/usr/bin/env python3
"""Interleaved paired timing for the OP1 shape8 MMA-overlap matrix."""

import csv
import pathlib
import re
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape8_cluster_ablation_pure_matmul_20260725"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
SAMPLES = 25
ITERATIONS = 10

PAIRS = (
    ("utccp_full_epi", "notma", "shell_utccp"),
    ("no_utccp_full_epi", "notma_noutccp", "shell_noutccp"),
    (
        "no_utccp_no_dstore",
        "notma_noutccp_nodstore",
        "shell_nodstore",
    ),
    ("no_utccp_no_epi", "notma_noutccp_noepi", "shell_noepi"),
)
VARIANTS = tuple(
    variant for _, mma_on, mma_off in PAIRS for variant in (mma_on, mma_off)
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


def run_variant(variant):
    command = [
        EXP / "bin" / variant,
        "--shape-id=8",
        "--config-index=0",
        f"--iterations={ITERATIONS}",
        f"--dataset-root={DATASET}",
    ]
    result = subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    match = re.search(
        r"^RESULT,8,[^,]+,ok,(\d+),([0-9.eE+-]+)$",
        result.stdout,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"missing RESULT for {variant}:\n{result.stdout}")
    return float(match.group(2)) * 1000


def ci95(values):
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5


def main():
    rows = []
    for sample in range(1, SAMPLES + 1):
        # Rotate the fixed order so no variant always occupies the same thermal slot.
        shift = (sample - 1) % len(VARIANTS)
        order = VARIANTS[shift:] + VARIANTS[:shift]
        for variant in order:
            wait_for_idle()
            value = run_variant(variant)
            rows.append(
                {
                    "sample": sample,
                    "variant": variant,
                    "mean_us": value,
                }
            )
            print(f"[PAIRED] sample={sample} variant={variant}", flush=True)

    raw_path = EXP / "mma_overlap_paired_event_raw.csv"
    with raw_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    by_sample = {
        (row["sample"], row["variant"]): row["mean_us"] for row in rows
    }
    summary_rows = []
    delta_by_environment = {}
    for environment, mma_on, mma_off in PAIRS:
        on_values = [by_sample[(sample, mma_on)] for sample in range(1, SAMPLES + 1)]
        off_values = [
            by_sample[(sample, mma_off)] for sample in range(1, SAMPLES + 1)
        ]
        deltas = [
            on_value - off_value
            for on_value, off_value in zip(on_values, off_values)
        ]
        delta_by_environment[environment] = deltas
        summary_rows.append(
            {
                "environment": environment,
                "mma_on_variant": mma_on,
                "mma_off_variant": mma_off,
                "mma_on_mean_us": statistics.mean(on_values),
                "mma_off_mean_us": statistics.mean(off_values),
                "mma_exposed_mean_us": statistics.mean(deltas),
                "mma_exposed_ci95_half_us": ci95(deltas),
            }
        )

    with (EXP / "mma_overlap_paired_deltas.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    d0 = delta_by_environment["utccp_full_epi"]
    d1 = delta_by_environment["no_utccp_full_epi"]
    d2 = delta_by_environment["no_utccp_no_dstore"]
    d3 = delta_by_environment["no_utccp_no_epi"]
    components = (
        ("mma_exposed_with_all_remaining_work", d0),
        ("hidden_by_utccp", [b - a for a, b in zip(d0, d1)]),
        ("hidden_by_d_store", [b - a for a, b in zip(d1, d2)]),
        (
            "hidden_by_tmem_load_convert_smem_staging",
            [b - a for a, b in zip(d2, d3)],
        ),
        ("mma_exposed_without_input_or_epilogue", d3),
        ("total_hidden_by_remaining_work", [b - a for a, b in zip(d0, d3)]),
    )
    component_rows = [
        {
            "component": name,
            "mean_us": statistics.mean(values),
            "ci95_half_us": ci95(values),
        }
        for name, values in components
    ]
    with (EXP / "mma_overlap_decomposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=component_rows[0].keys())
        writer.writeheader()
        writer.writerows(component_rows)

    print("[DONE] paired MMA-overlap timing", flush=True)


if __name__ == "__main__":
    main()
