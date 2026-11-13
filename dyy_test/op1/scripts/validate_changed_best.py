#!/usr/bin/env python3
import csv
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP1 = ROOT / "dyy_test/op1"
BINARY = OP1 / "bin/op1_top20"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
OUT = OP1 / "results/retest_20260724/changed_best_validation"
ITERATIONS = 1000
REPEATS = 10

CASES = {
    3: [
        ("old", 3, "M256_N256_K256_E128x64_C2x1x1"),
        ("new", 0, "M256_N256_K256_E64x64_C2x1x1"),
    ],
    5: [
        ("old", 35, "M256_N192_K256_E128x32_C4x1x1"),
        ("new", 36, "M256_N192_K256_Eauto_C2x2x1"),
    ],
    10: [
        ("old", 61, "M128_N256_K128_Eauto_C1x2x1"),
        ("new", 62, "M128_N256_K128_E128x128_C1x2x1"),
    ],
    12: [
        ("old", 78, "M128_N192_K128_E128x64_C1x1x1"),
        ("new", 63, "M128_N256_K128_E128x128_C1x1x1"),
    ],
    13: [
        ("old", 65, "M128_N256_K128_Eauto_C1x1x1"),
        ("new", 63, "M128_N256_K128_E128x128_C1x1x1"),
    ],
}


def wait_for_idle_gpu(timeout_s=1800):
    started = time.monotonic()
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
        if not result.stdout.strip():
            return
        if time.monotonic() - started > timeout_s:
            raise TimeoutError("GPU did not become idle")
        time.sleep(2)


def parse_result(output, expected_variant):
    values = []
    for line in output.splitlines():
        fields = line.split(",")
        if (len(fields) == 6 and fields[0] == "RESULT"
                and fields[2] == expected_variant and fields[3] == "ok"):
            values.append(float(fields[5]))
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one result for {expected_variant}, got {values}\n{output}")
    return values[0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = []
    for repeat in range(1, REPEATS + 1):
        for shape, cases in CASES.items():
            ordered = list(cases)
            if repeat % 2 == 0:
                ordered.reverse()
            for order, (label, config_index, variant) in enumerate(ordered, 1):
                wait_for_idle_gpu()
                command = [
                    str(BINARY), f"--shape-id={shape}",
                    f"--config-index={config_index}",
                    f"--iterations={ITERATIONS}", f"--dataset-root={DATASET}",
                ]
                process = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=3600)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"shape={shape} {label} failed:\n{process.stdout}")
                raw.append({
                    "repeat": repeat,
                    "shape": shape,
                    "order": order,
                    "candidate": label,
                    "variant": variant,
                    "config_index": config_index,
                    "iterations": ITERATIONS,
                    "avg_ms": f"{parse_result(process.stdout, variant):.9f}",
                })
                write_csv(OUT / "raw.csv", raw)
        print(f"completed repeat={repeat}", flush=True)

    samples = defaultdict(list)
    variants = {}
    for row in raw:
        key = (int(row["shape"]), row["candidate"])
        samples[key].append(float(row["avg_ms"]))
        variants[key] = row["variant"]

    summary = []
    for shape in CASES:
        old = samples[(shape, "old")]
        new = samples[(shape, "new")]
        old_mean = statistics.fmean(old)
        new_mean = statistics.fmean(new)
        winner = "new" if new_mean < old_mean else "old"
        summary.append({
            "shape": shape,
            "old_variant": variants[(shape, "old")],
            "old_mean_ms": f"{old_mean:.9f}",
            "old_stdev_ms": f"{statistics.stdev(old):.9f}",
            "new_variant": variants[(shape, "new")],
            "new_mean_ms": f"{new_mean:.9f}",
            "new_stdev_ms": f"{statistics.stdev(new):.9f}",
            "new_vs_old_pct": f"{(new_mean / old_mean - 1) * 100:.5f}",
            "winner": winner,
        })
    write_csv(OUT / "summary.csv", summary)
    for row in summary:
        print("SUMMARY," + ",".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
