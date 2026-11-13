#!/usr/bin/env bash
set -euo pipefail
set -x
cd "$(dirname "$0")"
out="$1"
mkdir "$out"
ncu=/usr/local/cuda-13.0/bin/ncu
metrics=gpu__time_duration.sum,sm__sass_inst_executed_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,launch__shared_mem_per_block,sm__warps_active.avg.pct_of_peak_sustained_active
"$ncu" --version >"$out/ncu_version.txt"
for arm in baseline packed baseline_late packed_late; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -qE '^[0-9]'; then
    echo "GPU occupied before Nsight Compute" >&2
    exit 75
  fi
  timeout 75 "$ncu" --clock-control none --cache-control none \
    --kernel-name-base function --kernel-name 'regex:device_kernel' --launch-count 1 \
    --metrics "$metrics" --force-overwrite --export "$out/$arm" \
    "$PWD/bin/intervention_$arm" --shape-id=7 --seed-index=2011112 --profile >"$out/$arm.log" 2>&1
  "$ncu" --import "$out/$arm.ncu-rep" --page raw --csv >"$out/$arm.csv"
done
