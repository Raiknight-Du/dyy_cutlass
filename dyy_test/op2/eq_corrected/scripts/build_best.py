#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/ubuntu/workspace/dyy/dyy_cutlass")
OP2 = ROOT / "dyy_test/op2_eq_corrected"
SCRIPTS = OP2 / "scripts"
SOURCE = OP2 / "op2_eq_corrected_best.cu"
BINARY = OP2 / "bin/op2_eq_corrected_best"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, str(SCRIPTS / "generate_best.py")])
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "/usr/local/cuda-13.0/bin/nvcc",
            "-std=c++17",
            "-O3",
            "-arch=sm_100a",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-Iinclude",
            "-Itools/util/include",
            "-Iexamples/common",
            f"-I{OP2}",
            str(SOURCE),
            "-o",
            str(BINARY),
        ]
    )


if __name__ == "__main__":
    main()
