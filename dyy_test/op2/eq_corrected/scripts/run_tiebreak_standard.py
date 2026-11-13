#!/usr/bin/env python3
import csv
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from op2_eq_shapes import SHAPES


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2_eq_corrected"
GROUP_BIN = ROOT / "dyy_test/op2_eq/search_space/pure_matmul_groups/bin"
DATASET = OP2 / "random_datasets"
RUN = OP2 / "results/standard_search_20260727"
SOURCE_SUMMARY = RUN / "op2_eq_corrected_top20_independent_summary.csv"
SOURCE_RAW = RUN / "op2_eq_corrected_top20_independent_raw.csv"
RAW = RUN / "op2_eq_corrected_tiebreak_additional_raw.csv"
SUMMARY = RUN / "op2_eq_corrected_tiebreak_combined_summary.csv"
WINNERS = RUN / "op2_eq_corrected_final_winners.csv"
ITERATIONS = 10
ADDITIONAL_REPEATS = 20
MAX_TIEBREAK_CANDIDATES = 4
RELATIVE_WINDOW = 0.005

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


def parse_result(
    output: str, shape: int, variant: str
) -> tuple[int, float]:
    rows = []
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) == 6 and fields[0] == "RESULT" and fields[3] == "ok":
            rows.append(fields)
    if len(rows) != 1 or int(rows[0][1]) != shape or rows[0][2] != variant:
        raise RuntimeError(
            f"Unexpected result for shape={shape}, variant={variant}\n{output}"
        )
    return int(rows[0][4]), float(rows[0][5])


def choose_candidates(
    summary_rows: list[dict[str, str]]
) -> dict[int, list[dict[str, str]]]:
    by_shape = defaultdict(list)
    for row in summary_rows:
        by_shape[int(row["shape"])].append(row)
    selected = {}
    for shape, rows in by_shape.items():
        rows.sort(key=lambda row: int(row["rank"]))
        best = rows[0]
        best_mean = float(best["mean_ms"])
        best_se = float(best["stdev_ms"]) / math.sqrt(int(best["samples"]))
        candidates = [best]
        for row in rows[1:MAX_TIEBREAK_CANDIDATES]:
            mean = float(row["mean_ms"])
            stderr = float(row["stdev_ms"]) / math.sqrt(int(row["samples"]))
            uncertainty = 2.0 * math.sqrt(best_se * best_se + stderr * stderr)
            window = max(RELATIVE_WINDOW * best_mean, uncertainty)
            if mean - best_mean <= window:
                candidates.append(row)
        selected[shape] = candidates
    return selected


def main() -> None:
    independent_summary = read_csv(SOURCE_SUMMARY)
    candidates = choose_candidates(independent_summary)
    selected = {
        (shape, row["variant"])
        for shape, rows in candidates.items()
        if len(rows) > 1
        for row in rows
    }

    additional = read_csv(RAW)
    completed = {
        (int(row["repeat"]), int(row["shape"]), row["variant"])
        for row in additional
    }
    shapes_base = sorted(
        shape for shape, rows in candidates.items() if len(rows) > 1
    )

    for repeat in range(1, ADDITIONAL_REPEATS + 1):
        shapes = list(shapes_base)
        shift = (repeat - 1) % len(shapes) if shapes else 0
        shapes = shapes[shift:] + shapes[:shift]
        if repeat % 2 == 0:
            shapes.reverse()
        for shape in shapes:
            ordered = list(candidates[shape])
            candidate_shift = (repeat - 1) % len(ordered)
            ordered = ordered[candidate_shift:] + ordered[:candidate_shift]
            if repeat % 2 == 0:
                ordered.reverse()
            m, n, k = SHAPES[shape]
            for order, row in enumerate(ordered, 1):
                key = (repeat, shape, row["variant"])
                if key in completed:
                    continue
                wait_for_idle_gpu()
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
                    process.stdout, shape, row["variant"]
                )
                additional.append(
                    {
                        "repeat": repeat,
                        "shape": shape,
                        "order": order,
                        "variant": row["variant"],
                        "group": row["group"],
                        "config_index": row["config_index"],
                        "iterations": iterations,
                        "avg_ms": f"{avg_ms:.9f}",
                    }
                )
                completed.add(key)
                write_csv(RAW, additional)
        print(
            f"completed additional repeat={repeat} samples={len(additional)}",
            flush=True,
        )

    samples = defaultdict(list)
    metadata = {}
    for row in read_csv(SOURCE_RAW) + additional:
        key = (int(row["shape"]), row["variant"])
        if key in selected:
            samples[key].append(float(row["avg_ms"]))
            metadata[key] = row

    combined = []
    for key, values in samples.items():
        expected = 5 + ADDITIONAL_REPEATS
        if len(values) != expected:
            raise RuntimeError(
                f"{key}: expected {expected} samples, got {len(values)}"
            )
        stdev = statistics.stdev(values)
        stderr = stdev / math.sqrt(len(values))
        row = metadata[key]
        combined.append(
            {
                "shape": key[0],
                "rank": 0,
                "variant": key[1],
                "group": row["group"],
                "config_index": row["config_index"],
                "mean_ms": f"{statistics.fmean(values):.9f}",
                "stdev_ms": f"{stdev:.9f}",
                "stderr_ms": f"{stderr:.9f}",
                "ci95_half_ms": f"{1.96 * stderr:.9f}",
                "min_ms": f"{min(values):.9f}",
                "max_ms": f"{max(values):.9f}",
                "samples": len(values),
                "iterations_per_sample": ITERATIONS,
            }
        )
    combined.sort(key=lambda row: (int(row["shape"]), float(row["mean_ms"])))
    rank = defaultdict(int)
    for row in combined:
        rank[int(row["shape"])] += 1
        row["rank"] = rank[int(row["shape"])]
    write_csv(SUMMARY, combined)

    independent_best = {
        int(row["shape"]): row
        for row in independent_summary
        if int(row["rank"]) == 1
    }
    combined_best = {
        int(row["shape"]): row for row in combined if int(row["rank"]) == 1
    }
    winners = []
    for shape in SHAPES:
        source = combined_best.get(shape, independent_best[shape])
        winners.append(
            {
                "shape": shape,
                "variant": source["variant"],
                "group": source["group"],
                "config_index": source["config_index"],
                "selection_mean_ms": source["mean_ms"],
                "selection_stdev_ms": source["stdev_ms"],
                "selection_samples": source["samples"],
                "selection_stage": (
                    "tiebreak_25" if shape in combined_best else "independent_5"
                ),
            }
        )
    write_csv(WINNERS, winners)
    for row in winners:
        print(
            f"FINAL_WINNER,{row['shape']},{row['variant']},"
            f"{row['selection_mean_ms']} ms,{row['selection_stage']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
