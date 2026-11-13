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

SECTIONS=(
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

profile_one() {
  local case_name=$1
  local report=$OUT/raw/ncu/$case_name
  if [[ -s "$report.ncu-rep" && ${FORCE:-0} != 1 ]]; then
    echo "SKIP existing $case_name"
    return
  fi
  echo "PROFILE $case_name"
  "$NCU" --force-overwrite --target-processes all --profile-from-start off \
    --replay-mode kernel "${SECTIONS[@]}" --export "$report" \
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

for dtype in fp16 bf16; do
  for seq in 8192 16384 32768; do
    profile_one "cp_s${seq}_${dtype}"
  done
done
