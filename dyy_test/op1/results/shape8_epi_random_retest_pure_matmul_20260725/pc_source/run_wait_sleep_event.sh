#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/workspace/dyy/dyy_cutlass

exp=dyy_test/op1/results/shape8_epi_random_retest_pure_matmul_20260725
out=$exp/pc_source/wait_sleep_event_raw.csv

printf 'pair,seed,order,label,sleep_ns,mean_ms\n' > "$out"

for i in $(seq 1 15); do
  seed=$((725641 + i * 1009))
  if ((i % 2 == 0)); then
    cases=(ns128 ns0)
  else
    cases=(ns0 ns128)
  fi

  order=0
  for label in "${cases[@]}"; do
    order=$((order + 1))
    if [[ "$label" == ns0 ]]; then
      ns=0
    else
      ns=128
    fi

    line=$(
      "$exp/bin/op1_shape8_wait_$label" \
        --shape-id=8 \
        --config-index=3 \
        --iterations=100 \
        --warmup-iterations=200 \
        --random-seed="$seed" |
        grep '^RESULT,'
    )
    ms=$(printf '%s\n' "$line" | awk -F, '{print $6}')
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$i" "$seed" "$order" "$label" "$ns" "$ms" >> "$out"
    printf 'pair=%02d %s %s ms\n' "$i" "$label" "$ms"
  done
done
