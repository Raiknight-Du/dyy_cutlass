#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/native_cuda
FLASHINFER_REPO=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/upstreams/flashinfer
PYTHON=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
OUT="$ROOT/validation/ncu_tcgen05"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin
export TORCH_CUDA_ARCH_LIST=10.0a
export FLASHINFER_REPO
export NATIVE_GDN_REPO="$ROOT"
export OP12_NATIVE_BACKEND=tcgen05

mkdir -p "$OUT"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  cases=("$@")
else
  cases=(
    noncp_underfilled_fp16
    noncp_nearfull_fp16
    noncp_underfilled_bf16
    cp_s8192_fp16
    cp_s16384_fp16
    cp_s32768_fp16
    cp_s8192_bf16
    cp_s16384_bf16
    cp_s32768_bf16
  )
fi

for case_name in "${cases[@]}"; do
  prefix="$OUT/tcgen05_${case_name}"
  "$NCU" \
    --set full \
    --target-processes all \
    --nvtx \
    --nvtx-include "OP12_GDN_${case_name}/" \
    --import-source yes \
    --source-folders "$FLASHINFER_REPO" \
    --force-overwrite \
    --export "$prefix" \
    "$PYTHON" benchmarks/profile_native_gdn.py --case "$case_name" \
    > "${prefix}.run.log" 2>&1

  "$NCU" --import "${prefix}.ncu-rep" --page details --print-details all --csv \
    > "${prefix}_details.csv" 2> "${prefix}_details.stderr.log"
  "$NCU" --import "${prefix}.ncu-rep" --page raw --csv \
    > "${prefix}_raw.csv" 2> "${prefix}_raw.stderr.log"
  "$NCU" --import "${prefix}.ncu-rep" --page source --print-source sass --csv \
    > "${prefix}_sass.csv" 2> "${prefix}_sass.stderr.log"
done

