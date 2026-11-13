#!/usr/bin/env python3

import subprocess
from pathlib import Path

from op2_eq_shapes import SHAPES


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2_eq_corrected"
BINARY = OP2 / "bin/op2_eq_corrected_baseline"
DATASET_ROOT = OP2 / "random_datasets"


def shape_directory(m: int, n: int, k: int) -> Path:
    return DATASET_ROOT / f"M{m}_N{n}_K{k}"


def dataset_complete(m: int, n: int, k: int) -> bool:
    directory = shape_directory(m, n, k)
    return all(
        (directory / name).is_file()
        for name in ("A.bin", "B.bin", "SFA.bin", "SFB.bin")
    )


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    for shape_id, (m, n, k) in SHAPES.items():
        if dataset_complete(m, n, k):
            print(f"[{shape_id:02d}/13] reuse M={m} N={n} K={k}", flush=True)
            continue

        print(f"[{shape_id:02d}/13] generate M={m} N={n} K={k}", flush=True)
        command = [
            str(BINARY),
            f"--m={m}",
            f"--n={n}",
            f"--k={k}",
            f"--shape-id={shape_id}",
            "--iterations=0",
            f"--dataset-root={DATASET_ROOT}",
            "--generate-dataset",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"dataset generation failed for shape {shape_id}: "
                f"exit {completed.returncode}"
            )
        if not dataset_complete(m, n, k):
            raise RuntimeError(f"dataset files missing for shape {shape_id}")


if __name__ == "__main__":
    main()
