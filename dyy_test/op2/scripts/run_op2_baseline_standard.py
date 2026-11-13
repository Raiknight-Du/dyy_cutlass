#!/usr/bin/env python3
import csv
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2"
BINARY = OP2 / "bin/op2_baseline"
DATASET = OP2 / "random_datasets"
RUN = OP2 / "results/standard_search_20260724"
RAW = RUN / "op2_baseline_raw.csv"
SUMMARY = RUN / "op2_baseline_summary.csv"
VARIANT = "S2_M256_N128_K256_Eauto_C2x1x1"
ITERATIONS = 10
REPEATS = 5

SHAPES = {
    1: (13824, 170100, 10240),
    2: (13824, 510300, 10240),
    3: (5120, 170100, 10240),
    4: (5120, 510300, 10240),
    5: (768, 8192, 8192),
    6: (768, 16384, 8192),
    7: (768, 32768, 8192),
    8: (4096, 8192, 4096),
    9: (4096, 16384, 4096),
    10: (4096, 32768, 256),
    11: (768, 16384, 256),
    12: (768, 32768, 256),
    13: (4096, 8192, 256),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wait_for_idle_gpu(timeout_s: int = 1800) -> None:
    deadline = time.monotonic() + timeout_s
    stable = 0
    while time.monotonic() < deadline:
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout.strip()
        state = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout.strip()
        if not apps and int(state) <= 1:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        time.sleep(1)
    raise TimeoutError("GPU did not become idle")


def main() -> None:
    raw = read_csv(RAW)
    completed = {
        (int(row["repeat"]), int(row["shape"])) for row in raw
    }
    for repeat in range(1, REPEATS + 1):
        shapes = list(SHAPES)
        shift = (repeat - 1) % len(shapes)
        shapes = shapes[shift:] + shapes[:shift]
        if repeat % 2 == 0:
            shapes.reverse()
        for order, shape in enumerate(shapes, 1):
            if (repeat, shape) in completed:
                continue
            wait_for_idle_gpu()
            m, n, k = SHAPES[shape]
            command = [
                str(BINARY),
                f"--m={m}",
                f"--n={n}",
                f"--k={k}",
                f"--shape-id={shape}",
                f"--iterations={ITERATIONS}",
                f"--dataset-root={DATASET}",
            ]
            process = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
            fields = [
                line.split(",")
                for line in process.stdout.splitlines()
                if line.startswith("RESULT,")
            ]
            if (
                process.returncode != 0
                or len(fields) != 1
                or fields[0][2] != VARIANT
                or fields[0][3] != "ok"
            ):
                raise RuntimeError(process.stdout)
            raw.append(
                {
                    "repeat": repeat,
                    "shape": shape,
                    "order": order,
                    "variant": VARIANT,
                    "iterations": int(fields[0][4]),
                    "avg_ms": f"{float(fields[0][5]):.9f}",
                }
            )
            completed.add((repeat, shape))
            write_csv(RAW, raw)
        print(f"completed repeat={repeat} samples={len(raw)}", flush=True)

    samples = defaultdict(list)
    for row in raw:
        samples[int(row["shape"])].append(float(row["avg_ms"]))
    summary = []
    for shape in SHAPES:
        values = samples[shape]
        if len(values) != REPEATS:
            raise RuntimeError(f"shape {shape}: got {len(values)} samples")
        stdev = statistics.stdev(values)
        stderr = stdev / math.sqrt(len(values))
        summary.append(
            {
                "shape": shape,
                "variant": VARIANT,
                "mean_ms": f"{statistics.fmean(values):.9f}",
                "stdev_ms": f"{stdev:.9f}",
                "ci95_half_ms": f"{1.96 * stderr:.9f}",
                "samples": len(values),
                "warmup": 1,
                "iterations_per_sample": ITERATIONS,
            }
        )
    write_csv(SUMMARY, summary)


if __name__ == "__main__":
    main()
