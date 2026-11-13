#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/gdn_paper_supp_20261112/probe
mkdir -p results
exec 9>/home/ubuntu/gdn_paper_supp_20261112/gpu.lock
flock -n 9
for i in {1..3}; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader
  other=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [[ -n "$other" ]]; then
    printf 'GPU busy: %s\n' "$other" >&2
    exit 75
  fi
  sleep 1
done

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used \
  --format=csv --loop-ms=250 >results/gpu_telemetry.csv &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true' EXIT

for process in 1 2 3; do
  sets=(448 7168)
  if (( process % 2 == 0 )); then sets=(7168 448); fi
  for tiles in "${sets[@]}"; do
    printf '%s process=%s tiles=%s timing\n' "$(date -Iseconds)" "$process" "$tiles"
    out="results/timing_${tiles}_p${process}.csv"
    timeout 90 ./same_buffer_tma "$tiles" 20 8 timing >"$out.part"
    mv "$out.part" "$out"
  done
done
for tiles in 448 7168; do
  out="results/trace_${tiles}.csv"
  timeout 60 ./same_buffer_tma "$tiles" 1 1 trace >"$out.part"
  mv "$out.part" "$out"
done

ncu=/opt/nvidia/nsight-compute/2026.2.1/ncu
metrics=gpu__time_duration.sum,sm__sass_inst_executed_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,launch__registers_per_thread
for tiles in 448 7168; do
  for arm in 0 1 2 3; do
    echo "$(date -Iseconds) ncu tiles=$tiles arm=$arm"
    rep="results/ncu_${tiles}_a${arm}"
    timeout 90 "$ncu" -f --cache-control none --clock-control none \
      -k 'regex:probe<' --launch-count 1 --metrics "$metrics" \
      -o "$rep" ./same_buffer_tma "$tiles" 1 1 "$arm" >"$rep.log" 2>&1
    "$ncu" --import "$rep.ncu-rep" --page raw --csv >"$rep.csv"
  done
done
echo "$(date -Iseconds) completed"

