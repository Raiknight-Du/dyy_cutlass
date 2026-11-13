#!/usr/bin/env bash
set -euo pipefail
set -x
cd "$(dirname "$0")"
repo=/home/ubuntu/workspace/dyy/dyy_cutlass
cuda=/usr/local/cuda-13.0/bin/nvcc
common=(-std=c++17 -O3 -arch=sm_100a --expt-relaxed-constexpr --expt-extended-lambda --threads 2 -lineinfo -Xptxas=-v)
includes=(-I"$repo/include" -I"$repo/tools/util/include" -I"$repo/examples/common")
mkdir -p bin build overlay/cutlass/gemm/collective
install -m 644 sm100_blockscaled_mma_warpspecialized_packed.hpp overlay/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp
"$cuda" "${common[@]}" "${includes[@]}" -c reuse_packer.cu -o build/reuse_packer.o >build/packer.log 2>&1 &
packer_pid=$!
"$cuda" "${common[@]}" "${includes[@]}" reuse_gemm.cu -o bin/reuse_baseline >build/baseline.log 2>&1 &
baseline_pid=$!
wait "$packer_pid"
"$cuda" "${common[@]}" -DPACKED_INPUT -Ioverlay "${includes[@]}" reuse_gemm.cu build/reuse_packer.o -o bin/reuse_packed >build/packed.log 2>&1
wait "$baseline_pid"
sha256sum reuse_gemm.cu reuse_packer.cu bin/reuse_baseline bin/reuse_packed > build/SHA256SUMS.txt
git -C "$repo" rev-parse HEAD >build/cutlass_commit.txt
"$cuda" --version >build/nvcc.txt
