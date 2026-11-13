#!/usr/bin/env python3
import csv
import os
import subprocess
import time
from pathlib import Path

from op2_eq_shapes import SHAPES


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2_eq_corrected"
SHARED_SEARCH = ROOT / "dyy_test/op2_eq/search_space/pure_matmul_groups"
BIN = SHARED_SEARCH / "bin"
LOGS = OP2 / "search_space/logs/broad_scan"
RESULTS = OP2 / "search_space/results"
DATASETS = OP2 / "random_datasets"
ITERATIONS = 10

SHAPE_ROWS = [(shape, *dimensions) for shape, dimensions in SHAPES.items()]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_top20(rows: list[dict[str, str]]) -> None:
    ranked = []
    for shape_id, *_ in SHAPE_ROWS:
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
    path = SHARED_SEARCH / "src/manifest.csv"
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
expected_results_by_group = {}
for group, _ in variant_index.values():
    expected_results_by_group[group] = expected_results_by_group.get(group, 0) + 1


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    groups = sorted(path.stem for path in BIN.glob("group_[0-9][0-9][0-9][0-9]"))
    if len(groups) != 14:
        raise RuntimeError(f"Expected 14 group binaries, found {len(groups)}")

    raw_path = RESULTS / "broad_scan_raw.csv"
    state_path = RESULTS / "broad_scan_gpu_state.csv"
    rows = read_csv(raw_path)
    state_rows = read_csv(state_path)
    completed_keys = {
        (int(row["shape"]), row["group"])
        for row in state_rows
        if int(row["process_returncode"]) == 0
        and int(row["result_count"])
        == expected_results_by_group.get(row["group"], -1)
    }
    total_runs = len(SHAPE_ROWS) * len(groups)
    completed = len(completed_keys)

    for shape_id, m, n, k in SHAPE_ROWS:
        offset = (shape_id - 1) % len(groups)
        ordered_groups = groups[offset:] + groups[:offset]
        if shape_id % 2 == 0:
            ordered_groups.reverse()

        for group in ordered_groups:
            if (shape_id, group) in completed_keys:
                continue
            rows = [
                row
                for row in rows
                if not (
                    int(row["shape"]) == shape_id
                    and row["group"] == group
                )
            ]
            state_rows = [
                row
                for row in state_rows
                if not (
                    int(row["shape"]) == shape_id
                    and row["group"] == group
                )
            ]
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
            expected_group_results = expected_results_by_group[group]
            if process.returncode != 0 or len(parsed) != expected_group_results:
                raise RuntimeError(
                    f"shape {shape_id} {group} failed: rc={process.returncode}, "
                    f"results={len(parsed)}/{expected_group_results}; "
                    f"see {log_path}"
                )
            completed_keys.add((shape_id, group))

    write_top20(rows)
    expected = len(SHAPE_ROWS) * len(variant_index)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {len(rows)}")
    failures = [row for row in rows if row["status"] != "ok"]
    print(
        f"broad scan complete: rows={len(rows)} failures={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
