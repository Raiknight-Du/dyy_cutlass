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
BINARY = OP2 / "bin/op2_best"
DATASET = OP2 / "random_datasets"
CONFIGS = OP2 / "results/op2_best_configs.csv"
RUN = OP2 / "results/standard_search_20260724"
RAW = RUN / "op2_best_final_raw.csv"
SUMMARY = RUN / "op2_best_final_summary.csv"
ITERATIONS = 10
REPEATS = 25

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
    if not rows:
        return
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


def gpu_state() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    return dict(
        zip(
            [
                "pstate",
                "idle_sm_mhz",
                "mem_mhz",
                "idle_power_w",
                "temperature_c",
            ],
            fields,
        )
    )


def parse_result(
    output: str, shape: int, expected_variant: str
) -> tuple[int, float]:
    rows = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[3] == "ok":
            rows.append(fields)
    if (
        len(rows) != 1
        or int(rows[0][1]) != shape
        or rows[0][2] != expected_variant
    ):
        raise RuntimeError(
            f"Unexpected result for shape={shape}, "
            f"variant={expected_variant}\n{output}"
        )
    return int(rows[0][4]), float(rows[0][5])


def main() -> None:
    expected = {
        int(row["shape"]): row["variant"] for row in read_csv(CONFIGS)
    }
    if sorted(expected) != list(SHAPES):
        raise RuntimeError("Expected one best configuration for every shape")

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
            key = (repeat, shape)
            if key in completed:
                continue
            wait_for_idle_gpu()
            state = gpu_state()
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
            if process.returncode != 0:
                raise RuntimeError(
                    f"shape={shape} failed rc={process.returncode}\n"
                    f"{process.stdout}"
                )
            iterations, avg_ms = parse_result(
                process.stdout, shape, expected[shape]
            )
            raw.append(
                {
                    "repeat": repeat,
                    "shape": shape,
                    "order": order,
                    "variant": expected[shape],
                    "iterations": iterations,
                    "avg_ms": f"{avg_ms:.9f}",
                    **state,
                }
            )
            completed.add(key)
            write_csv(RAW, raw)
        print(f"completed repeat={repeat} samples={len(raw)}", flush=True)

    samples = defaultdict(list)
    for row in raw:
        samples[int(row["shape"])].append(float(row["avg_ms"]))
    summary = []
    for shape in SHAPES:
        values = samples[shape]
        if len(values) != REPEATS:
            raise RuntimeError(
                f"shape {shape}: expected {REPEATS}, got {len(values)}"
            )
        stdev = statistics.stdev(values)
        stderr = stdev / math.sqrt(len(values))
        summary.append(
            {
                "shape": shape,
                "variant": expected[shape],
                "mean_ms": f"{statistics.fmean(values):.9f}",
                "stdev_ms": f"{stdev:.9f}",
                "stderr_ms": f"{stderr:.9f}",
                "ci95_half_ms": f"{1.96 * stderr:.9f}",
                "min_ms": f"{min(values):.9f}",
                "max_ms": f"{max(values):.9f}",
                "samples": len(values),
                "warmup": 1,
                "iterations_per_sample": ITERATIONS,
            }
        )
    write_csv(SUMMARY, summary)
    total = sum(float(row["mean_ms"]) for row in summary)
    for row in summary:
        print(
            f"FINAL,{row['shape']},{row['variant']},"
            f"{row['mean_ms']} ms,ci95={row['ci95_half_ms']} ms"
        )
    print(f"FINAL_TOTAL,{total:.9f} ms")


if __name__ == "__main__":
    main()
