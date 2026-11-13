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
RUN = OP1 / "results/standard_retest_20260724"
SOURCE_SUMMARY = RUN / "op1_top20_summary.csv"
RAW = RUN / "op1_top3_raw.csv"
SUMMARY = RUN / "op1_top3_summary.csv"
ITERATIONS = 10
REPEATS = 5


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wait_for_idle_gpu(timeout_s=1800):
    start = time.monotonic()
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
        if not result.stdout.strip():
            return
        if time.monotonic() - start > timeout_s:
            raise TimeoutError("GPU did not become idle")
        time.sleep(5)


def parse_result(output, variant):
    found = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[2] == variant and fields[3] == "ok":
            found.append(float(fields[5]))
    if len(found) != 1:
        raise RuntimeError(f"Expected one result for {variant}, got {found}\n{output}")
    return found[0]


def main():
    source = read_csv(SOURCE_SUMMARY)
    top3 = defaultdict(list)
    for row in source:
        if int(row["pure_matmul_rank"]) <= 3:
            top3[int(row["shape"])].append(row)
    if any(len(top3[shape]) != 3 for shape in range(1, 14)):
        raise RuntimeError("Source summary does not contain three candidates per shape")

    raw = []
    for repeat in range(1, REPEATS + 1):
        for shape in range(1, 14):
            candidates = sorted(top3[shape], key=lambda row: int(row["pure_matmul_rank"]))
            shift = (repeat - 1) % len(candidates)
            candidates = candidates[shift:] + candidates[:shift]
            if repeat % 2 == 0:
                candidates.reverse()
            for order, row in enumerate(candidates, 1):
                wait_for_idle_gpu()
                command = [
                    str(BINARY), f"--shape-id={shape}",
                    f"--config-index={row['config_index']}",
                    f"--iterations={ITERATIONS}", f"--dataset-root={DATASET}",
                ]
                process = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=3600)
                log = RUN / "logs" / (
                    f"validate_shape{shape:02d}_r{repeat}_o{order}_{row['variant']}.log")
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(process.stdout)
                if process.returncode != 0:
                    raise RuntimeError(f"Command failed rc={process.returncode}; see {log}")
                avg_ms = parse_result(process.stdout, row["variant"])
                raw.append({
                    "repeat": repeat, "shape": shape, "order": order,
                    "variant": row["variant"], "config_index": row["config_index"],
                    "iterations": ITERATIONS, "avg_ms": f"{avg_ms:.9f}",
                })
                write_csv(RAW, raw, list(raw[0]))
            print(f"validated repeat={repeat} shape={shape}", flush=True)

    source_map = {(int(row["shape"]), row["variant"]): row for row in source}
    samples = defaultdict(list)
    for row in raw:
        samples[(int(row["shape"]), row["variant"])].append(float(row["avg_ms"]))
    summary = []
    for key, values in samples.items():
        prior = source_map[key]
        summary.append({
            "shape": key[0], "pure_matmul_rank": 0, "variant": key[1],
            "config_index": prior["config_index"],
            "old_event_rank": prior["old_event_rank"],
            "old_nsys_rank": prior["old_nsys_rank"],
            "pure_matmul_mean_ms": f"{statistics.fmean(values):.9f}",
            "pure_matmul_stdev_ms": f"{statistics.stdev(values):.9f}",
            "pure_matmul_min_ms": f"{min(values):.9f}",
            "pure_matmul_max_ms": f"{max(values):.9f}",
            "samples": len(values), "iterations_per_sample": ITERATIONS,
        })
    summary.sort(key=lambda row: (int(row["shape"]), float(row["pure_matmul_mean_ms"])))
    rank = defaultdict(int)
    for row in summary:
        rank[int(row["shape"])] += 1
        row["pure_matmul_rank"] = rank[int(row["shape"])]
    write_csv(SUMMARY, summary, list(summary[0]))
    for row in summary:
        if int(row["pure_matmul_rank"]) == 1:
            print(f"VALIDATED_BEST,{row['shape']},{row['variant']},{row['pure_matmul_mean_ms']} ms")


if __name__ == "__main__":
    main()
