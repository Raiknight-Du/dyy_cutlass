#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/workspace/dyy/dyy_cutlass
BIN="$ROOT/dyy_test/op1/bin/op1_any_mnk"
DATASET="$ROOT/dyy_test/op1/random_datasets_any_mnk"
OUT="$ROOT/dyy_test/op1/results/shape_8192_cube_20260804"
RAW="$OUT/op1_shape8192_raw.csv"

mkdir -p "$OUT"
printf 'repeat,pstate,sm_mhz,mem_mhz,power_w,temp_c,variant,avg_ms\n' > "$RAW"

wait_for_idle_gpu() {
  local stable=0
  while (( stable < 2 )); do
    local apps util
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [[ -z "$apps" && "$util" -le 1 ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 1
  done
}

for repeat in $(seq 1 10); do
  wait_for_idle_gpu
  state=$(nvidia-smi \
    --query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader,nounits | head -1 | tr -d ' ')
  output=$(cd "$ROOT" && "$BIN" \
    --m=8192 --n=8192 --k=8192 \
    --shape-id=8192 --config-index=0 --iterations=10 \
    --dataset-root="$DATASET")
  printf '%s\n' "$output" > "$OUT/run_${repeat}.log"
  result=$(printf '%s\n' "$output" | awk -F, '$1=="RESULT" && $4=="ok" {print; exit}')
  if [[ -z "$result" ]]; then
    printf 'Missing RESULT for repeat %s\n%s\n' "$repeat" "$output" >&2
    exit 1
  fi
  variant=$(printf '%s\n' "$result" | cut -d, -f3)
  avg_ms=$(printf '%s\n' "$result" | cut -d, -f6)
  printf '%s,%s,%s,%s\n' "$repeat" "$state" "$variant" "$avg_ms" >> "$RAW"
  printf 'repeat=%s avg_ms=%s\n' "$repeat" "$avg_ms"
done

awk -F, '
  NR>1 {
    x=$8; sum+=x; sum2+=x*x; count++;
    if (count==1 || x<min) min=x;
    if (count==1 || x>max) max=x;
  }
  END {
    mean=sum/count;
    stdev=sqrt((sum2-sum*sum/count)/(count-1));
    printf "samples,%d\nmean_ms,%.9f\nstdev_ms,%.9f\nmin_ms,%.9f\nmax_ms,%.9f\n", count, mean, stdev, min, max;
  }
' "$RAW" | tee "$OUT/op1_shape8192_summary.csv"
