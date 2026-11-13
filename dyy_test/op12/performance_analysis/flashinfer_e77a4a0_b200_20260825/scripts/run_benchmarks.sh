#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12
OUT=$ROOT/performance_analysis/flashinfer_e77a4a0_b200_20260825
export FLASHINFER_REPO=$ROOT/upstreams/flashinfer
export PYTHONPATH=$FLASHINFER_REPO

mkdir -p "$OUT/raw/bench" "$OUT/tables" "$OUT/logs"

for trial in 1 2 3; do
  "$ROOT/.venv/bin/python" "$OUT/scripts/benchmark_flashinfer_gdn.py" \
    --suite all --trial "$trial" --warmup 20 --iterations 100 \
    >"$OUT/raw/bench/trial_${trial}.jsonl" \
    2>"$OUT/logs/bench_trial_${trial}.stderr.log"
done
