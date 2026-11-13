#!/usr/bin/env python3

import pathlib
import re
import subprocess


ROOT = pathlib.Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
EXP = ROOT / "dyy_test/op1/results/packed_physical_full_matmul_20260727"
DATASET = ROOT / "dyy_test/m72a_voidc_matmul_20260719/random_datasets"


def digest(binary: pathlib.Path, shape: int) -> tuple[str, str]:
    process = subprocess.run(
        [
            str(binary),
            f"--shape-id={shape}",
            "--iterations=0",
            f"--dataset-root={DATASET}",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=3600,
    )
    match = re.search(
        rf"^DIGEST,{shape},([0-9a-f]+),([0-9a-f]+)$",
        process.stdout,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(process.stdout)
    return match.group(1), match.group(2)


def main():
    baseline = EXP / "bin/op1_baseline"
    packed = EXP / "bin/op1_packed"
    for shape in range(1, 14):
        baseline_digest = digest(baseline, shape)
        packed_digest = digest(packed, shape)
        status = "PASS" if baseline_digest == packed_digest else "FAIL"
        print(shape, status, baseline_digest, packed_digest, flush=True)
        if status == "FAIL":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
