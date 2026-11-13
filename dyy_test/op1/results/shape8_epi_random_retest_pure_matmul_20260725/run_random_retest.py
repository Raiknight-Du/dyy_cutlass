#!/usr/bin/env python3
import csv
import pathlib
import random
import re
import secrets
import statistics
import subprocess
import time


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/shape8_epi_random_retest_pure_matmul_20260725"
SOURCE = EXP / "op1_shape8_epi_random_retest_pure_matmul.cu"
NVCC = pathlib.Path("/usr/local/cuda/bin/nvcc")
BINARY = EXP / "bin/op1_shape8_epi_random_retest"

CASES = (
    (0, "E64x64", "M256_N256_K256_E64x64_C2x1x1"),
    (3, "E128x64", "M256_N256_K256_E128x64_C2x1x1"),
)

PAIRS = 50
WARMUP_ITERATIONS = 200
TIMED_ITERATIONS = 100


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
    return [
        int(item.strip()) for item in result.stdout.splitlines()[0].split(",")
    ]


def wait_for_idle():
    for _ in range(180):
        util, clock, temperature = gpu_state()
        if util <= 2:
            return clock, temperature
        time.sleep(1)
    raise RuntimeError("GPU did not become idle")


def build():
    BINARY.parent.mkdir(exist_ok=True)
    command = [
        NVCC,
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-DOVERHEAD_TARGETS_ONLY",
        "-Iinclude",
        "-Itools/util/include",
        "-Iexamples/common",
        "-Idyy_test/op1",
        SOURCE,
        "-o",
        BINARY,
    ]
    print("[BUILD]", flush=True)
    with (EXP / "build.log").open("w") as stream:
        subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_case(config_index, expected_name, seed):
    clock, temperature = wait_for_idle()
    command = [
        BINARY,
        "--shape-id=8",
        f"--config-index={config_index}",
        f"--iterations={TIMED_ITERATIONS}",
        f"--warmup-iterations={WARMUP_ITERATIONS}",
        f"--random-seed={seed}",
    ]
    result = subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    match = re.search(
        r"^RESULT,8,([^,]+),ok,(\d+),([0-9.eE+-]+)$",
        result.stdout,
        re.MULTILINE,
    )
    if not match or match.group(1) != expected_name:
        raise RuntimeError(f"bad RESULT:\n{result.stdout}")
    return float(match.group(3)), clock, temperature


def ci95(values):
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5


def main():
    build()
    rows = []
    order_rng = random.SystemRandom()

    for pair in range(1, PAIRS + 1):
        seed = secrets.randbits(63)
        order = list(CASES)
        order_rng.shuffle(order)
        for order_in_pair, (config_index, label, expected_name) in enumerate(
            order, start=1
        ):
            mean_ms, clock, temperature = run_case(
                config_index, expected_name, seed
            )
            rows.append(
                {
                    "pair": pair,
                    "seed": seed,
                    "order_in_pair": order_in_pair,
                    "config_index": config_index,
                    "config": label,
                    "variant": expected_name,
                    "warmup_iterations": WARMUP_ITERATIONS,
                    "timed_iterations": TIMED_ITERATIONS,
                    "mean_ms": mean_ms,
                    "idle_sm_clock_mhz": clock,
                    "temperature_c": temperature,
                }
            )
            print(
                f"[PAIR {pair:02d}] seed={seed} order={order_in_pair} "
                f"{label} {mean_ms * 1000:.4f} us",
                flush=True,
            )

    with (EXP / "random_retest_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for _, label, expected_name in CASES:
        values = [
            row["mean_ms"] * 1000 for row in rows if row["config"] == label
        ]
        summaries.append(
            {
                "config": label,
                "variant": expected_name,
                "samples": len(values),
                "mean_us": statistics.mean(values),
                "median_us": statistics.median(values),
                "stdev_us": statistics.stdev(values),
                "ci95_half_us": ci95(values),
                "min_us": min(values),
                "max_us": max(values),
            }
        )
    with (EXP / "random_retest_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    pair_rows = []
    for pair in range(1, PAIRS + 1):
        selected = {
            row["config"]: row for row in rows if row["pair"] == pair
        }
        e64 = selected["E64x64"]["mean_ms"] * 1000
        e128 = selected["E128x64"]["mean_ms"] * 1000
        pair_rows.append(
            {
                "pair": pair,
                "seed": selected["E64x64"]["seed"],
                "e64_us": e64,
                "e128_us": e128,
                "delta_e128_minus_e64_us": e128 - e64,
                "relative_delta_pct": 100 * (e128 / e64 - 1),
                "winner": "E128x64" if e128 < e64 else "E64x64",
                "e64_order": selected["E64x64"]["order_in_pair"],
                "e128_order": selected["E128x64"]["order_in_pair"],
            }
        )
    with (EXP / "paired_deltas.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    deltas = [row["delta_e128_minus_e64_us"] for row in pair_rows]
    relative = [row["relative_delta_pct"] for row in pair_rows]
    ordered = sorted(deltas)
    trimmed = ordered[5:-5]
    paired_summary = [
        {
            "pairs": PAIRS,
            "mean_delta_us": statistics.mean(deltas),
            "median_delta_us": statistics.median(deltas),
            "trimmed_mean_delta_us": statistics.mean(trimmed),
            "stdev_delta_us": statistics.stdev(deltas),
            "ci95_half_delta_us": ci95(deltas),
            "mean_relative_delta_pct": statistics.mean(relative),
            "e128_wins": sum(row["winner"] == "E128x64" for row in pair_rows),
            "e64_wins": sum(row["winner"] == "E64x64" for row in pair_rows),
        }
    ]
    with (EXP / "paired_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=paired_summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(paired_summary)

    print(paired_summary[0], flush=True)


if __name__ == "__main__":
    main()
