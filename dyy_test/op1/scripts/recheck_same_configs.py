#!/usr/bin/env python3
import csv
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OLD = ROOT / "dyy_test/m72a_m84a_voidc_baseline_best_20260722/matmul/M_72a_pure_matmul_variants"
NEW = ROOT / "dyy_test/op1/bin/op1_top20"
DATA = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"
OUT = ROOT / "dyy_test/op1/results/recheck_20260723"
ITERATIONS = 500
REPEATS = 5

CONFIGS = {
    "E128x64": {
        "variant": "M256_N256_K256_E128x64_C2x1x1",
        "old_index": 1,
        "new_index": 3,
    },
    "E64x64": {
        "variant": "M256_N256_K256_E64x64_C2x1x1",
        "old_index": 5,
        "new_index": 0,
    },
}


def wait_idle():
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
        if not result.stdout.strip():
            return
        time.sleep(2)


def parse_result(output, variant):
    values = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[2] == variant and fields[3] == "ok":
            values.append(float(fields[5]))
    if len(values) != 1:
        raise RuntimeError(f"Expected one result for {variant}, got {values}\n{output}")
    return values[0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for shape in (1, 3):
        for config_name, config in CONFIGS.items():
            for build, binary, index_key in (
                ("old", OLD, "old_index"), ("new", NEW, "new_index")):
                cases.append((shape, config_name, config, build, binary, config[index_key]))

    rows = []
    for repeat in range(1, REPEATS + 1):
        ordered = cases[repeat - 1:] + cases[:repeat - 1]
        if repeat % 2 == 0:
            ordered.reverse()
        for order, (shape, config_name, config, build, binary, index) in enumerate(ordered, 1):
            wait_idle()
            command = [
                str(binary), f"--shape-id={shape}", f"--config-index={index}",
                f"--iterations={ITERATIONS}", f"--dataset-root={DATA}",
            ]
            process = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=3600)
            log = OUT / f"shape{shape}_{config_name}_{build}_r{repeat}_o{order}.log"
            log.write_text(process.stdout)
            if process.returncode != 0:
                raise RuntimeError(f"Command failed rc={process.returncode}; see {log}")
            rows.append({
                "repeat": repeat, "order": order, "shape": shape,
                "config": config_name, "variant": config["variant"], "build": build,
                "iterations": ITERATIONS,
                "avg_ms": f"{parse_result(process.stdout, config['variant']):.9f}",
            })
            write_csv(OUT / "same_config_raw.csv", rows)
        print(f"completed repeat={repeat}", flush=True)

    samples = defaultdict(list)
    for row in rows:
        samples[(int(row["shape"]), row["config"], row["build"])].append(float(row["avg_ms"]))
    summary = []
    for shape in (1, 3):
        for config_name in CONFIGS:
            old = samples[(shape, config_name, "old")]
            new = samples[(shape, config_name, "new")]
            old_mean = statistics.fmean(old)
            new_mean = statistics.fmean(new)
            summary.append({
                "shape": shape, "config": config_name,
                "old_mean_ms": f"{old_mean:.9f}",
                "old_stdev_ms": f"{statistics.stdev(old):.9f}",
                "new_mean_ms": f"{new_mean:.9f}",
                "new_stdev_ms": f"{statistics.stdev(new):.9f}",
                "new_vs_old_pct": f"{(new_mean / old_mean - 1) * 100:.5f}",
            })
    write_csv(OUT / "same_config_summary.csv", summary)
    for row in summary:
        print("SUMMARY," + ",".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
