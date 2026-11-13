#!/usr/bin/env python3
import csv
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2"
GROUP_BIN = OP2 / "search_space/pure_matmul_groups/bin"
DATASET = OP2 / "random_datasets"
CANDIDATES = (
    OP2 / "search_space/pure_matmul_groups/results/broad_scan_top20.csv"
)
RUN = OP2 / "results/standard_search_20260724"
RAW = RUN / "op2_top20_independent_raw.csv"
SUMMARY = RUN / "op2_top20_independent_summary.csv"
GPU_STATE = RUN / "op2_top20_gpu_state.csv"
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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gpu_snapshot() -> tuple[str, str]:
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
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
            "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()
    return apps, state


def wait_for_idle_gpu(timeout_s: int = 1800) -> str:
    deadline = time.monotonic() + timeout_s
    stable = 0
    last_state = ""
    while time.monotonic() < deadline:
        apps, state = gpu_snapshot()
        last_state = state
        util = int(state.split(",")[0].strip())
        if not apps and util <= 1:
            stable += 1
            if stable >= 2:
                return state
        else:
            stable = 0
        time.sleep(1)
    raise TimeoutError(f"GPU did not become idle; last state: {last_state}")


def parse_result(
    output: str, expected_shape: int, expected_variant: str
) -> tuple[int, float]:
    values = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[3] == "ok":
            values.append(fields)
    if len(values) != 1:
        raise RuntimeError(f"Expected one result, got {len(values)}\n{output}")
    fields = values[0]
    if int(fields[1]) != expected_shape or fields[2] != expected_variant:
        raise RuntimeError(
            f"Unexpected result {fields[1:3]}, expected "
            f"{expected_shape},{expected_variant}"
        )
    return int(fields[4]), float(fields[5])


def main() -> None:
    source = read_csv(CANDIDATES)
    by_shape = defaultdict(list)
    for row in source:
        by_shape[int(row["shape"])].append(row)
    if any(len(by_shape[shape]) != 20 for shape in SHAPES):
        raise RuntimeError("Expected exactly 20 candidates per shape")

    raw = read_csv(RAW)
    gpu_rows = read_csv(GPU_STATE)
    completed = {
        (int(row["repeat"]), int(row["shape"]), row["variant"]) for row in raw
    }

    for repeat in range(1, REPEATS + 1):
        shapes = list(SHAPES)
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

            m, n, k = SHAPES[shape]
            for order, row in enumerate(candidates, 1):
                key = (repeat, shape, row["variant"])
                if key in completed:
                    continue
                idle_state = wait_for_idle_gpu()
                command = [
                    str(GROUP_BIN / row["group"]),
                    f"--m={m}",
                    f"--n={n}",
                    f"--k={k}",
                    f"--shape-id={shape}",
                    f"--config-index={row['config_index']}",
                    f"--iterations={ITERATIONS}",
                    f"--dataset-root={DATASET}",
                ]
                started = time.monotonic()
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
                wall_s = time.monotonic() - started
                if process.returncode != 0:
                    log = RUN / "logs" / (
                        f"independent_failed_s{shape:02d}_r{repeat}_"
                        f"o{order}.log"
                    )
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(process.stdout)
                    raise RuntimeError(
                        f"Command failed rc={process.returncode}; see {log}"
                    )
                iterations, avg_ms = parse_result(
                    process.stdout, shape, row["variant"]
                )
                raw.append(
                    {
                        "repeat": repeat,
                        "shape": shape,
                        "order": order,
                        "variant": row["variant"],
                        "group": row["group"],
                        "config_index": row["config_index"],
                        "iterations": iterations,
                        "avg_ms": f"{avg_ms:.9f}",
                        "wall_s": f"{wall_s:.3f}",
                    }
                )
                gpu_rows.append(
                    {
                        "repeat": repeat,
                        "shape": shape,
                        "variant": row["variant"],
                        "idle_state": idle_state,
                    }
                )
                completed.add(key)
                write_csv(RAW, raw)
                write_csv(GPU_STATE, gpu_rows)
            print(
                f"completed repeat={repeat} shape={shape} samples={len(raw)}",
                flush=True,
            )

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
                f"{key}: expected {REPEATS} samples, got {len(values)}"
            )
        row = metadata[key]
        summary.append(
            {
                "shape": key[0],
                "rank": 0,
                "variant": key[1],
                "group": row["group"],
                "config_index": row["config_index"],
                "mean_ms": f"{statistics.fmean(values):.9f}",
                "stdev_ms": f"{statistics.stdev(values):.9f}",
                "min_ms": f"{min(values):.9f}",
                "max_ms": f"{max(values):.9f}",
                "samples": len(values),
                "iterations_per_sample": ITERATIONS,
            }
        )
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
                f"{row['mean_ms']} ms"
            )


if __name__ == "__main__":
    main()
