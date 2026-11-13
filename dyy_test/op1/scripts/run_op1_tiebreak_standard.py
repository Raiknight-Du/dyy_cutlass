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
BINARY = OP1 / "bin/op1_top20"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
RUN = OP1 / "results/standard_retest_20260724"
SOURCE_SUMMARY = RUN / "op1_top20_independent_summary.csv"
SOURCE_RAW = RUN / "op1_top20_independent_raw.csv"
RAW = RUN / "op1_tiebreak_additional_raw.csv"
SUMMARY = RUN / "op1_tiebreak_combined_summary.csv"
ITERATIONS = 10
ADDITIONAL_REPEATS = 20

# Clear winners (shape 7 and 9) do not need denser sampling.
MAX_RANK = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 3,
    6: 2,
    8: 2,
    10: 2,
    11: 3,
    12: 3,
    13: 2,
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


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


def parse_result(output, shape, variant):
    rows = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[3] == "ok":
            rows.append(fields)
    if len(rows) != 1 or int(rows[0][1]) != shape or rows[0][2] != variant:
        raise RuntimeError(
            f"Unexpected result for shape={shape}, variant={variant}\n{output}")
    return int(rows[0][4]), float(rows[0][5])


def main():
    candidates = defaultdict(list)
    for row in read_csv(SOURCE_SUMMARY):
        shape = int(row["shape"])
        if shape in MAX_RANK and int(row["rank"]) <= MAX_RANK[shape]:
            candidates[shape].append(row)
    for shape, expected in MAX_RANK.items():
        if len(candidates[shape]) != expected:
            raise RuntimeError(
                f"shape {shape}: expected {expected} candidates, "
                f"got {len(candidates[shape])}")

    additional = []
    shapes_base = sorted(candidates)
    for repeat in range(1, ADDITIONAL_REPEATS + 1):
        shapes = list(shapes_base)
        shift = (repeat - 1) % len(shapes)
        shapes = shapes[shift:] + shapes[:shift]
        if repeat % 2 == 0:
            shapes.reverse()
        for shape in shapes:
            ordered = list(candidates[shape])
            candidate_shift = (repeat - 1) % len(ordered)
            ordered = ordered[candidate_shift:] + ordered[:candidate_shift]
            if repeat % 2 == 0:
                ordered.reverse()
            for order, row in enumerate(ordered, 1):
                wait_for_idle_gpu()
                command = [
                    str(BINARY), f"--shape-id={shape}",
                    f"--config-index={row['config_index']}",
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
                iterations, avg_ms = parse_result(
                    process.stdout, shape, row["variant"])
                additional.append({
                    "repeat": repeat,
                    "shape": shape,
                    "order": order,
                    "variant": row["variant"],
                    "config_index": row["config_index"],
                    "iterations": iterations,
                    "avg_ms": f"{avg_ms:.9f}",
                })
                write_csv(RAW, additional)
        print(
            f"completed additional repeat={repeat} "
            f"samples={len(additional)}", flush=True)

    selected = {
        (shape, row["variant"])
        for shape, rows in candidates.items()
        for row in rows
    }
    samples = defaultdict(list)
    config_index = {}
    for row in read_csv(SOURCE_RAW) + additional:
        key = (int(row["shape"]), row["variant"])
        if key in selected:
            samples[key].append(float(row["avg_ms"]))
            config_index[key] = row["config_index"]

    summary = []
    for key, values in samples.items():
        expected = 5 + ADDITIONAL_REPEATS
        if len(values) != expected:
            raise RuntimeError(
                f"{key}: expected {expected} samples, got {len(values)}")
        stdev = statistics.stdev(values)
        stderr = stdev / math.sqrt(len(values))
        summary.append({
            "shape": key[0],
            "rank": 0,
            "variant": key[1],
            "config_index": config_index[key],
            "mean_ms": f"{statistics.fmean(values):.9f}",
            "stdev_ms": f"{stdev:.9f}",
            "stderr_ms": f"{stderr:.9f}",
            "ci95_half_ms": f"{1.96 * stderr:.9f}",
            "min_ms": f"{min(values):.9f}",
            "max_ms": f"{max(values):.9f}",
            "samples": len(values),
            "iterations_per_sample": ITERATIONS,
        })
    summary.sort(key=lambda row: (int(row["shape"]), float(row["mean_ms"])))
    rank = defaultdict(int)
    for row in summary:
        rank[int(row["shape"])] += 1
        row["rank"] = rank[int(row["shape"])]
    write_csv(SUMMARY, summary)
    for row in summary:
        if int(row["rank"]) == 1:
            print(
                f"TIEBREAK_BEST,{row['shape']},{row['variant']},"
                f"{row['mean_ms']} ms,ci95={row['ci95_half_ms']} ms")


if __name__ == "__main__":
    main()
