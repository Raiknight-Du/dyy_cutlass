#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/native_cuda
FLASHINFER_REPO=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/upstreams/flashinfer
PYTHON=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python
OUT="$ROOT/validation/tcgen05"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin
export TORCH_CUDA_ARCH_LIST=10.0a
export FLASHINFER_REPO
export OP12_NATIVE_BACKEND=tcgen05

mkdir -p "$OUT"
cd "$ROOT"

"$PYTHON" -m pytest tests/test_correctness.py -q \
  > "$OUT/pytest_tcgen05.log" 2>&1

for trial in 1 2 3; do
  "$PYTHON" benchmarks/benchmark_native_gdn.py \
    --suite all --implementation native --trial "$trial" \
    --warmup 20 --iterations 100 \
    > "$OUT/benchmark_tcgen05_trial${trial}.jsonl" \
    2> "$OUT/benchmark_tcgen05_trial${trial}.stderr.log"
done

