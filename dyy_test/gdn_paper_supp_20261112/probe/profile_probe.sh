#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/gdn_paper_supp_20261112/probe
exec 9>/home/ubuntu/gdn_paper_supp_20261112/gpu.lock
flock -n 9
ncu=/opt/nvidia/nsight-compute/2026.2.1/ncu
metrics=gpu__time_duration.sum,sm__sass_inst_executed_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,launch__registers_per_thread
for tiles in 448 7168; do
  for arm in 0 1 2 3; do
    rep="results/counters_${tiles}_a${arm}"
    [[ -s "$rep.csv" ]] && continue
    other=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$other" ]]; then
      printf 'GPU busy: %s\n' "$other" >&2
      exit 75
    fi
    echo "$(date -Iseconds) ncu tiles=$tiles arm=$arm"
    timeout 90 "$ncu" -f --cache-control none --clock-control none \
      -k probe --launch-count 1 --metrics "$metrics" \
      -o "$rep" ./same_buffer_tma "$tiles" 1 1 "$arm" >"$rep.log" 2>&1
    "$ncu" --import "$rep.ncu-rep" --page raw --csv >"$rep.csv.part"
    mv "$rep.csv.part" "$rep.csv"
  done
done

