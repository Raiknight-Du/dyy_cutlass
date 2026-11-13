#!/usr/bin/env bash
set -euo pipefail
set -x
cd "$(dirname "$0")"
repo=/home/ubuntu/workspace/dyy/dyy_cutlass
cuda=/usr/local/cuda-13.0/bin/nvcc
common=(-std=c++17 -O3 -arch=sm_100a --expt-relaxed-constexpr --expt-extended-lambda --threads 2 -lineinfo -Xptxas=-v)
includes=(-I"$repo/include" -I"$repo/tools/util/include" -I"$repo/examples/common")
"$cuda" "${common[@]}" "${includes[@]}" reuse_gemm_large.cu -o bin/reuse_large_baseline >build/reuse_large_baseline.log 2>&1 &
baseline_pid=$!
"$cuda" "${common[@]}" -DPACKED_INPUT -Ioverlay "${includes[@]}" reuse_gemm_large.cu build/reuse_packer.o -o bin/reuse_large_packed >build/reuse_large_packed.log 2>&1 &
packed_pid=$!
wait "$baseline_pid"
wait "$packed_pid"
sha256sum reuse_gemm_large.cu bin/reuse_large_* >build/reuse_large_SHA256SUMS.txt
