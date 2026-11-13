#!/usr/bin/env python3
import csv
import hashlib
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op2/search_space/pure_matmul_groups"
SOURCES = EXP / "src"
BIN = EXP / "bin"
LOGS = EXP / "logs/build"
RESULTS = EXP / "results"
WORKERS = int(os.environ.get("BUILD_WORKERS", "2"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path) -> tuple[str, str, float, str, str]:
    name = source.stem
    binary = BIN / name
    stamp = BIN / f"{name}.sha256"
    source_hash = digest(source)
    if (
        binary.exists()
        and stamp.exists()
        and stamp.read_text().strip() == source_hash
    ):
        return name, "reused", 0.0, source_hash, ""

    command = [
        "/usr/local/cuda/bin/nvcc",
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "--expt-relaxed-constexpr",
        "-diag-suppress=20012",
        "-Iinclude",
        "-Itools/util/include",
        "-Iexamples/common",
        "-Idyy_test/op2",
        str(source),
        "-o",
        str(binary),
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
    elapsed = time.monotonic() - started
    (LOGS / f"{name}.log").write_text(process.stdout)
    if process.returncode == 0:
        stamp.write_text(source_hash + "\n")
        return name, "ok", elapsed, source_hash, ""
    error = next(
        (line.strip() for line in process.stdout.splitlines() if "error:" in line),
        process.stdout[-1200:].replace("\n", " "),
    )
    return name, "failed", elapsed, source_hash, error


def write_status(rows: list[tuple[str, str, float, str, str]]) -> None:
    with (RESULTS / "build_status.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["group", "status", "elapsed_s", "source_sha256", "error"]
        )
        writer.writerows(sorted(rows))


def main() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = sorted(SOURCES.glob("group_*.cu"))
    if len(sources) != 14:
        raise RuntimeError(f"Expected 14 group sources, found {len(sources)}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(build, source) for source in sources]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            write_status(rows)
            print(
                f"[{index}/{len(sources)}] {row[0]} {row[1]} {row[2]:.1f}s",
                flush=True,
            )
    raise SystemExit(1 if any(row[1] == "failed" for row in rows) else 0)


if __name__ == "__main__":
    main()
