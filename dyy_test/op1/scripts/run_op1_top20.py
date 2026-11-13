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
RAW = OP1 / "results/op1_top20_pure_matmul_raw.csv"
SUMMARY = OP1 / "results/op1_top20_pure_matmul_summary.csv"
ITERATIONS = 50
REPEATS = 5


def wait_for_idle_gpu(timeout_s=1800):
    started = time.monotonic()
    while True:
        check = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
        if not check.stdout.strip():
            return
        if time.monotonic() - started > timeout_s:
            raise TimeoutError("GPU did not become idle")
        time.sleep(5)


def load_candidates():
    with CANDIDATES.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_results(output, expected_shape):
    rows = []
    for line in output.splitlines():
        if not line.startswith("RESULT,"):
            continue
        fields = line.split(",")
        if len(fields) != 6 or fields[3] != "ok":
            continue
        shape = int(fields[1])
        if shape != expected_shape:
            raise RuntimeError(f"Unexpected shape {shape}, expected {expected_shape}")
        rows.append((fields[2], int(fields[4]), float(fields[5])))
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw, candidates):
    samples = defaultdict(list)
    for row in raw:
        samples[(row["variant"], int(row["shape"]))].append(float(row["avg_ms"]))
    old = {(int(row["shape"]), row["variant"]): row for row in candidates}
    summary = []
    for (shape, variant), prior in old.items():
        values = samples[(variant, shape)]
        if len(values) != REPEATS:
            raise RuntimeError(f"{variant} shape {shape}: expected {REPEATS} samples, got {len(values)}")
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values)
        summary.append({
            "shape": shape,
            "pure_matmul_rank": 0,
            "variant": variant,
            "config_index": prior["config_index"],
            "old_event_rank": prior["old_event_rank"],
            "old_nsys_rank": prior["old_nsys_rank"],
            "old_event_avg_ms": prior["old_event_avg_ms"],
            "old_nsys_kernel_avg_ms": prior["old_nsys_kernel_avg_ms"],
            "pure_matmul_mean_ms": f"{mean:.9f}",
            "pure_matmul_stdev_ms": f"{stdev:.9f}",
            "pure_matmul_min_ms": f"{min(values):.9f}",
            "pure_matmul_max_ms": f"{max(values):.9f}",
            "samples": len(values),
            "iterations_per_sample": ITERATIONS,
        })
    summary.sort(key=lambda row: (int(row["shape"]), float(row["pure_matmul_mean_ms"])))
    ranks = defaultdict(int)
    for row in summary:
        ranks[int(row["shape"])] += 1
        row["pure_matmul_rank"] = ranks[int(row["shape"])]
    return summary


def main():
    candidates = load_candidates()
    expected = defaultdict(set)
    for row in candidates:
        expected[int(row["shape"])].add(row["variant"])
    raw = []
    for repeat in range(1, REPEATS + 1):
        for shape in range(1, 14):
            wait_for_idle_gpu()
            command = [
                str(BINARY), f"--shape-id={shape}", f"--iterations={ITERATIONS}",
                f"--dataset-root={DATASET}",
            ]
            process = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=3600)
            log = OP1 / "logs" / f"top20_shape{shape:02d}_r{repeat}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(process.stdout)
            if process.returncode != 0:
                raise RuntimeError(f"Command failed rc={process.returncode}; see {log}")
            parsed = parse_results(process.stdout, shape)
            got = {variant for variant, _, _ in parsed}
            if got != expected[shape]:
                raise RuntimeError(
                    f"shape {shape} repeat {repeat}: "
                    f"missing={expected[shape]-got}, extra={got-expected[shape]}")
            for variant, iterations, avg_ms in parsed:
                raw.append({
                    "repeat": repeat, "shape": shape, "variant": variant,
                    "iterations": iterations, "avg_ms": f"{avg_ms:.9f}",
                })
            write_csv(RAW, raw, ["repeat", "shape", "variant", "iterations", "avg_ms"])
            print(f"completed repeat={repeat} shape={shape}", flush=True)

    summary = summarize(raw, candidates)
    write_csv(SUMMARY, summary, list(summary[0]))
    for best in (row for row in summary if int(row["pure_matmul_rank"]) == 1):
        print(f"BEST,{best['shape']},{best['variant']},{best['pure_matmul_mean_ms']} ms")


if __name__ == "__main__":
    main()
