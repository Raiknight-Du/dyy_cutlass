#!/usr/bin/env python3
import csv
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
DATA = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
OUT = ROOT / "dyy_test/op1/results/recheck_20260724_shape1"
PURE = ROOT / "dyy_test/op1/bin/op1_top20"
ITERATIONS = 500
REPEATS = 5

CASES = [
    {
        "path": "voidc", "config": "E64x64",
        "variant": "M256_N256_K256_E64x64_C2x1x1",
        "binary": ROOT / "dyy_test/m72a_voidc_full_legal_scan_20260720/bin/full_scan/group_0059",
        "index": 1,
    },
    {
        "path": "pure", "config": "E64x64",
        "variant": "M256_N256_K256_E64x64_C2x1x1",
        "binary": PURE, "index": 0,
    },
    {
        "path": "voidc", "config": "E128x64",
        "variant": "M256_N256_K256_E128x64_C2x1x1",
        "binary": ROOT / "dyy_test/m72a_voidc_full_legal_scan_20260720/bin/full_scan/group_0060",
        "index": 17,
    },
    {
        "path": "pure", "config": "E128x64",
        "variant": "M256_N256_K256_E128x64_C2x1x1",
        "binary": PURE, "index": 3,
    },
]


def gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    return result.stdout.strip()


def wait_idle(timeout_s=1800):
    started = time.monotonic()
    while gpu_processes():
        if time.monotonic() - started > timeout_s:
            raise TimeoutError("GPU did not become idle")
        time.sleep(2)


def gpu_state():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.sm,clocks.max.sm,power.draw,power.limit,"
            "temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True)
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    return dict(zip(
        ["pstate", "idle_sm_mhz", "max_sm_mhz", "idle_power_w",
         "power_limit_w", "temperature_c", "memory_used_mib"],
        fields,
    ))


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
    rows = []
    for repeat in range(1, REPEATS + 1):
        ordered = CASES[repeat - 1:] + CASES[:repeat - 1]
        if repeat % 2 == 0:
            ordered.reverse()
        for order, case in enumerate(ordered, 1):
            wait_idle()
            state = gpu_state()
            command = [
                str(case["binary"]), "--shape-id=1",
                f"--config-index={case['index']}",
                f"--iterations={ITERATIONS}",
                f"--dataset-root={DATA}",
            ]
            process = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=3600)
            log = OUT / (
                f"{case['config']}_{case['path']}_r{repeat}_o{order}.log")
            log.write_text(process.stdout)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed rc={process.returncode}; see {log}")
            rows.append({
                "repeat": repeat, "order": order,
                "path": case["path"], "config": case["config"],
                "variant": case["variant"], "iterations": ITERATIONS,
                "avg_ms": f"{parse_result(process.stdout, case['variant']):.9f}",
                **state,
            })
            write_csv(OUT / "raw.csv", rows)
        print(f"completed repeat={repeat}", flush=True)

    samples = defaultdict(list)
    for row in rows:
        samples[(row["config"], row["path"])].append(float(row["avg_ms"]))
    summary = []
    for config in ("E64x64", "E128x64"):
        voidc = samples[(config, "voidc")]
        pure = samples[(config, "pure")]
        voidc_mean = statistics.fmean(voidc)
        pure_mean = statistics.fmean(pure)
        summary.append({
            "config": config,
            "voidc_mean_ms": f"{voidc_mean:.9f}",
            "voidc_stdev_ms": f"{statistics.stdev(voidc):.9f}",
            "pure_mean_ms": f"{pure_mean:.9f}",
            "pure_stdev_ms": f"{statistics.stdev(pure):.9f}",
            "pure_vs_voidc_pct": f"{(pure_mean / voidc_mean - 1) * 100:.5f}",
            "pure_reaches_voidc": int(pure_mean <= voidc_mean * 1.002),
        })
    write_csv(OUT / "summary.csv", summary)
    for row in summary:
        print("SUMMARY," + ",".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
