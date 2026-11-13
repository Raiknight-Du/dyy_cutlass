#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12
OUT=$ROOT/performance_analysis/flashinfer_e77a4a0_b200_20260825
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
PY=$ROOT/.venv/bin/python
DRIVER=$OUT/scripts/profile_flashinfer_gdn.py
export FLASHINFER_REPO=$ROOT/upstreams/flashinfer
export PYTHONPATH=$FLASHINFER_REPO

mkdir -p "$OUT/raw/ncu" "$OUT/logs"

FULL_SECTIONS=(
  --section LaunchStats
  --section SpeedOfLight
  --section Occupancy
  --section MemoryWorkloadAnalysis
  --section MemoryWorkloadAnalysis_Chart
  --section SchedulerStats
  --section WarpStateStats
  --section InstructionStats
  --section SourceCounters
  --section SpeedOfLight_HierarchicalTensorRooflineChart
)

LIGHT_SECTIONS=(
  --section LaunchStats
  --section SpeedOfLight
  --section Occupancy
  --section MemoryWorkloadAnalysis
  --section SchedulerStats
  --section WarpStateStats
)

profile_one() {
  local case_name=$1
  local mode=$2
  local report=$OUT/raw/ncu/$case_name
  local -a sections
  if [[ $mode == full ]]; then
    sections=("${FULL_SECTIONS[@]}")
  else
    sections=("${LIGHT_SECTIONS[@]}")
  fi
  "$NCU" --force-overwrite --target-processes all --profile-from-start off \
    --replay-mode kernel "${sections[@]}" --export "$report" \
    "$PY" "$DRIVER" --case "$case_name" \
    >"$OUT/logs/ncu_${case_name}.stdout.log" \
    2>"$OUT/logs/ncu_${case_name}.stderr.log"

  "$NCU" --import "$report.ncu-rep" --page details --csv \
    >"${report}_details.csv" 2>"${report}_details.stderr.log"
  "$NCU" --import "$report.ncu-rep" --page raw --csv \
    >"${report}_raw.csv" 2>"${report}_raw.stderr.log"
  "$NCU" --import "$report.ncu-rep" --page source --print-source sass --csv \
    >"${report}_sass.csv" 2>"${report}_sass.stderr.log" || true
}

profile_one noncp_underfilled_fp16 full
profile_one noncp_nearfull_fp16 full
profile_one cp_underfilled_fp16 full
profile_one noncp_underfilled_bf16 light
