#!/usr/bin/env python3
import csv
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP1 = ROOT / "dyy_test/op1"
BINARY = OP1 / "bin/op1_best"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
RUN = OP1 / "results/standard_retest_20260724"
RAW = RUN / "op1_best_final_raw.csv"
SUMMARY = RUN / "op1_best_final_summary.csv"
ITERATIONS = 10
REPEATS = 25

EXPECTED = {
    1: "M256_N256_K256_E64x64_C2x1x1",
    2: "M256_N256_K256_E64x64_C2x1x1",
    3: "M256_N256_K256_E64x64_C2x1x1",
    4: "M256_N256_K256_E64x64_C2x1x1",
    5: "M256_N192_K256_Eauto_C2x2x1",
    6: "M256_N192_K256_E128x16_C4x1x1",
    7: "M256_N192_K256_E128x16_C2x1x1",
    8: "M256_N256_K256_E64x64_C2x1x1",
    9: "M256_N256_K256_E64x64_C2x1x1",
    10: "M128_N256_K128_Eauto_C1x2x1",
    11: "M128_N192_K128_E128x64_C1x1x1",
    12: "M128_N192_K128_E128x64_C1x1x1",
    13: "M128_N256_K128_E128x128_C1x1x1",
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def gpu_state():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,"
            "temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True)
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    return dict(zip(
        ["pstate", "idle_sm_mhz", "mem_mhz", "idle_power_w",
         "temperature_c"],
        fields,
    ))


def parse_result(output, shape):
    rows = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[3] == "ok":
            rows.append(fields)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one result, got {len(rows)}\n{output}")
    fields = rows[0]
    if int(fields[1]) != shape or fields[2] != EXPECTED[shape]:
        raise RuntimeError(
            f"Unexpected result {fields[1:3]} for shape={shape}")
    return int(fields[4]), float(fields[5])


def main():
    raw = []
    for repeat in range(1, REPEATS + 1):
        shapes = list(range(1, 14))
        shift = (repeat - 1) % len(shapes)
        shapes = shapes[shift:] + shapes[:shift]
        if repeat % 2 == 0:
            shapes.reverse()
        for order, shape in enumerate(shapes, 1):
            wait_for_idle_gpu()
            state = gpu_state()
            command = [
                str(BINARY), f"--shape-id={shape}",
                f"--iterations={ITERATIONS}",
                f"--dataset-root={DATASET}",
            ]
            process = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=3600)
            if process.returncode != 0:
                raise RuntimeError(
                    f"shape={shape} failed rc={process.returncode}\n"
                    f"{process.stdout}")
            iterations, avg_ms = parse_result(process.stdout, shape)
            raw.append({
                "repeat": repeat,
                "shape": shape,
                "order": order,
                "variant": EXPECTED[shape],
                "iterations": iterations,
                "avg_ms": f"{avg_ms:.9f}",
                **state,
            })
            write_csv(RAW, raw)
        print(
            f"completed repeat={repeat} samples={len(raw)}", flush=True)

    samples = defaultdict(list)
    for row in raw:
        samples[int(row["shape"])].append(float(row["avg_ms"]))
    summary = []
    for shape in range(1, 14):
        values = samples[shape]
        if len(values) != REPEATS:
            raise RuntimeError(
                f"shape {shape}: expected {REPEATS}, got {len(values)}")
        stdev = statistics.stdev(values)
        stderr = stdev / math.sqrt(len(values))
        summary.append({
            "shape": shape,
            "variant": EXPECTED[shape],
            "mean_ms": f"{statistics.fmean(values):.9f}",
            "stdev_ms": f"{stdev:.9f}",
            "stderr_ms": f"{stderr:.9f}",
            "ci95_half_ms": f"{1.96 * stderr:.9f}",
            "min_ms": f"{min(values):.9f}",
            "max_ms": f"{max(values):.9f}",
            "samples": len(values),
            "warmup": 1,
            "iterations_per_sample": ITERATIONS,
        })
    write_csv(SUMMARY, summary)
    total = sum(float(row["mean_ms"]) for row in summary)
    for row in summary:
        print(
            f"FINAL,{row['shape']},{row['variant']},"
            f"{row['mean_ms']} ms,ci95={row['ci95_half_ms']} ms")
    print(f"FINAL_TOTAL,{total:.9f} ms")


if __name__ == "__main__":
    main()
