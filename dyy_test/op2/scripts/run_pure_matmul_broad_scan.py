#!/usr/bin/env python3
import csv
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op2/search_space/pure_matmul_groups"
BIN = EXP / "bin"
LOGS = EXP / "logs/broad_scan"
RESULTS = EXP / "results"
DATASETS = ROOT / "dyy_test/op2/random_datasets"
ITERATIONS = 10

SHAPES = [
    (1, 13824, 170100, 10240),
    (2, 13824, 510300, 10240),
    (3, 5120, 170100, 10240),
    (4, 5120, 510300, 10240),
    (5, 768, 8192, 8192),
    (6, 768, 16384, 8192),
    (7, 768, 32768, 8192),
    (8, 4096, 8192, 4096),
    (9, 4096, 16384, 4096),
    (10, 4096, 32768, 256),
    (11, 768, 16384, 256),
    (12, 768, 32768, 256),
    (13, 4096, 8192, 256),
]


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


def wait_for_idle(timeout_s: int = 900) -> str:
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


def parse_results(stdout: str, group: str, shape_id: int) -> list[dict[str, str]]:
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("RESULT,"):
            continue
        parts = line.split(",")
        if len(parts) != 6:
            raise RuntimeError(f"Malformed RESULT line: {line}")
        _, reported_shape, variant, status, iterations, runtime_ms = parts
        if int(reported_shape) != shape_id:
            raise RuntimeError(f"Shape mismatch in result: {line}")
        rows.append(
            {
                "shape": str(shape_id),
                "group": group,
                "variant": variant,
                "status": status,
                "iterations": iterations,
                "runtime_ms": runtime_ms,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "shape",
        "group",
        "variant",
        "status",
        "iterations",
        "runtime_ms",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_top20(rows: list[dict[str, str]]) -> None:
    ranked = []
    for shape_id, *_ in SHAPES:
        valid = [
            row
            for row in rows
            if int(row["shape"]) == shape_id and row["status"] == "ok"
        ]
        valid.sort(key=lambda row: float(row["runtime_ms"]))
        for rank, row in enumerate(valid[:20], 1):
            ranked.append(
                {
                    "shape": shape_id,
                    "rank": rank,
                    "variant": row["variant"],
                    "group": row["group"],
                    "config_index": variant_index[row["variant"]][1],
                    "broad_runtime_ms": row["runtime_ms"],
                }
            )
    with (RESULTS / "broad_scan_top20.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "shape",
                "rank",
                "variant",
                "group",
                "config_index",
                "broad_runtime_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(ranked)


def load_manifest() -> dict[str, tuple[str, int]]:
    path = EXP / "src/manifest.csv"
    mapping = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            mapping[row["variant"]] = (
                row["group_name"],
                int(row["config_index"]),
            )
    if len(mapping) != 1080:
        raise RuntimeError(f"Expected 1080 variants, found {len(mapping)}")
    return mapping


variant_index = load_manifest()


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    groups = sorted(path.stem for path in BIN.glob("group_[0-9][0-9][0-9][0-9]"))
    if len(groups) != 14:
        raise RuntimeError(f"Expected 14 group binaries, found {len(groups)}")

    rows = []
    raw_path = RESULTS / "broad_scan_raw.csv"
    state_path = RESULTS / "broad_scan_gpu_state.csv"
    state_rows = []
    total_runs = len(SHAPES) * len(groups)
    completed = 0

    for shape_id, m, n, k in SHAPES:
        offset = (shape_id - 1) % len(groups)
        ordered_groups = groups[offset:] + groups[:offset]
        if shape_id % 2 == 0:
            ordered_groups.reverse()

        for group in ordered_groups:
            completed += 1
            idle_state = wait_for_idle()
            command = [
                str(BIN / group),
                f"--m={m}",
                f"--n={n}",
                f"--k={k}",
                f"--shape-id={shape_id}",
                f"--iterations={ITERATIONS}",
                f"--dataset-root={DATASETS}",
            ]
            started = time.monotonic()
            process = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=1800,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            )
            elapsed = time.monotonic() - started
            log_path = LOGS / f"shape{shape_id:02d}_{group}.log"
            log_path.write_text(process.stdout)
            parsed = parse_results(process.stdout, group, shape_id)
            rows.extend(parsed)
            state_rows.append(
                {
                    "shape": shape_id,
                    "group": group,
                    "idle_state": idle_state,
                    "process_returncode": process.returncode,
                    "wall_seconds": f"{elapsed:.6f}",
                    "result_count": len(parsed),
                }
            )
            write_csv(raw_path, rows)
            with state_path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "shape",
                        "group",
                        "idle_state",
                        "process_returncode",
                        "wall_seconds",
                        "result_count",
                    ],
                )
                writer.writeheader()
                writer.writerows(state_rows)
            print(
                f"[{completed}/{total_runs}] shape{shape_id:02d} {group}: "
                f"rc={process.returncode} results={len(parsed)} wall={elapsed:.1f}s",
                flush=True,
            )

    write_top20(rows)
    expected = len(SHAPES) * len(variant_index)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {len(rows)}")
    failures = [row for row in rows if row["status"] != "ok"]
    print(
        f"broad scan complete: rows={len(rows)} failures={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
