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
CANDIDATES = OP1 / "results/op1_top20_candidates.csv"
RUN = OP1 / "results/standard_retest_20260724"
RAW = RUN / "op1_top20_independent_raw.csv"
SUMMARY = RUN / "op1_top20_independent_summary.csv"
ITERATIONS = 10
REPEATS = 5


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


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_result(output, expected_shape, expected_variant):
    values = []
    for line in output.splitlines():
        fields = line.split(",")
        if (len(fields) == 6 and fields[0] == "RESULT"
                and fields[3] == "ok"):
            values.append(fields)
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one result, got {len(values)}\n{output}")
    fields = values[0]
    if int(fields[1]) != expected_shape or fields[2] != expected_variant:
        raise RuntimeError(
            f"Unexpected result {fields[1:3]}, expected "
            f"{expected_shape},{expected_variant}")
    return int(fields[4]), float(fields[5])


def main():
    source = read_csv(CANDIDATES)
    by_shape = defaultdict(list)
    for row in source:
        by_shape[int(row["shape"])].append(row)
    if any(len(by_shape[shape]) != 20 for shape in range(1, 14)):
        raise RuntimeError("Expected exactly 20 candidates per shape")

    raw = []
    for repeat in range(1, REPEATS + 1):
        shapes = list(range(1, 14))
        shape_shift = (repeat - 1) % len(shapes)
        shapes = shapes[shape_shift:] + shapes[:shape_shift]
        if repeat % 2 == 0:
            shapes.reverse()

        for shape in shapes:
            candidates = list(by_shape[shape])
            shift = ((repeat - 1) * 7 + (shape - 1) * 3) % len(candidates)
            candidates = candidates[shift:] + candidates[:shift]
            if repeat % 2 == 0:
                candidates.reverse()

            for order, row in enumerate(candidates, 1):
                wait_for_idle_gpu()
                command = [
                    str(BINARY), f"--shape-id={shape}",
                    f"--config-index={row['config_index']}",
                    f"--iterations={ITERATIONS}",
                    f"--dataset-root={DATASET}",
                ]
                started = time.monotonic()
                process = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=3600)
                wall_s = time.monotonic() - started
                if process.returncode != 0:
                    log = RUN / "logs" / (
                        f"independent_failed_s{shape:02d}_r{repeat}_"
                        f"o{order}.log")
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(process.stdout)
                    raise RuntimeError(
                        f"Command failed rc={process.returncode}; see {log}")
                iterations, avg_ms = parse_result(
                    process.stdout, shape, row["variant"])
                raw.append({
                    "repeat": repeat,
                    "shape": shape,
                    "order": order,
                    "variant": row["variant"],
                    "config_index": row["config_index"],
                    "iterations": iterations,
                    "avg_ms": f"{avg_ms:.9f}",
                    "wall_s": f"{wall_s:.3f}",
                })
                write_csv(RAW, raw)
            print(
                f"completed repeat={repeat} shape={shape} "
                f"samples={len(raw)}", flush=True)

    samples = defaultdict(list)
    metadata = {}
    for row in raw:
        key = (int(row["shape"]), row["variant"])
        samples[key].append(float(row["avg_ms"]))
        metadata[key] = row

    summary = []
    for key, values in samples.items():
        if len(values) != REPEATS:
            raise RuntimeError(
                f"{key}: expected {REPEATS} samples, got {len(values)}")
        row = metadata[key]
        summary.append({
            "shape": key[0],
            "rank": 0,
            "variant": key[1],
            "config_index": row["config_index"],
            "mean_ms": f"{statistics.fmean(values):.9f}",
            "stdev_ms": f"{statistics.stdev(values):.9f}",
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
                f"INDEPENDENT_BEST,{row['shape']},{row['variant']},"
                f"{row['mean_ms']} ms")


if __name__ == "__main__":
    main()
