#!/usr/bin/env bash
set -euo pipefail
set -x
cd "$(dirname "$0")"
repo=/home/ubuntu/workspace/dyy/dyy_cutlass
cuda=/usr/local/cuda-13.0/bin/nvcc
common=(-std=c++17 -O3 -arch=sm_100a --expt-relaxed-constexpr --expt-extended-lambda --threads 2 -lineinfo -Xptxas=-v)
includes=(-I"$repo/include" -I"$repo/tools/util/include" -I"$repo/examples/common")
for variant in baseline_late packed_late; do
  dir="overlay_$variant/cutlass/gemm/collective"
  mkdir -p "$dir"
  install -m 644 "sm100_blockscaled_mma_warpspecialized_$variant.hpp" "$dir/sm100_blockscaled_mma_warpspecialized.hpp"
done
"$cuda" "${common[@]}" "${includes[@]}" intervention_gemm.cu -o bin/intervention_baseline >build/intervention_baseline.log 2>&1 &
baseline_pid=$!
"$cuda" "${common[@]}" -DPACKED_INPUT -Ioverlay "${includes[@]}" intervention_gemm.cu build/reuse_packer.o -o bin/intervention_packed >build/intervention_packed.log 2>&1 &
packed_pid=$!
wait "$baseline_pid"
wait "$packed_pid"
"$cuda" "${common[@]}" -Ioverlay_baseline_late "${includes[@]}" intervention_gemm.cu -o bin/intervention_baseline_late >build/intervention_baseline_late.log 2>&1 &
baseline_pid=$!
"$cuda" "${common[@]}" -DPACKED_INPUT -Ioverlay_packed_late "${includes[@]}" intervention_gemm.cu build/reuse_packer.o -o bin/intervention_packed_late >build/intervention_packed_late.log 2>&1 &
packed_pid=$!
wait "$baseline_pid"
wait "$packed_pid"
sha256sum intervention_gemm.cu sm100_blockscaled_mma_warpspecialized_*late.hpp bin/intervention_* >build/intervention_SHA256SUMS.txt
