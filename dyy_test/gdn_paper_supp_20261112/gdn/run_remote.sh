#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
python=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python
mode=${1:?pilot|formal|nsys|ncu}
length=${2:?sequence length}
partitions=${3:?partition count}
trial=${4:?process index}
arms=${5:-B0,K2,K3,K23}
name="${mode}_S${length}_P${partitions}_t${trial}"
meta="results/${name}_logs"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/cache/$name"
mkdir -p results "$TMPDIR"
mkdir "$meta"
expected_uuid=GPU-6abbea3c-b8d3-7f4b-ca76-85d612930fdf
exec 9>/home/ubuntu/gdn_paper_supp_20261112/gpu.lock
flock -w 12 9
nvidia-smi -L > "$meta/device.txt"
grep -F "$expected_uuid" "$meta/device.txt"
for attempt in 1 2 3; do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)
    printf '%s %s\n' "$(date -Iseconds)" "$apps" >> "$meta/compute_processes_pre.log"
    if [[ -n "$apps" ]]; then
        echo "BUSY: $apps" >&2
        exit 75
    fi
    sleep 0.25
done
{
    date -Iseconds
    hostname
    nvidia-smi
    /usr/local/bin/nsys --version
    /usr/local/cuda-13.0/bin/ncu --version
} > "$meta/platform.log" 2>&1
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv -lms 100 > "$meta/gpu.csv" 9>&- &
gpu_logger=$!
(
    while true; do
        printf '%s\n' "$(date -Iseconds)"
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
        sleep 0.25
    done
) > "$meta/compute_processes.log" 9>&- &
app_logger=$!
trap 'kill "$gpu_logger" "$app_logger" 2>/dev/null || true' EXIT
args=("$python" -B src/k3trace/run_chain.py --out "results/$name" --s "$length" --partitions "$partitions" --trial "$trial" --arms "$arms")
case "$mode" in
    pilot)
        args+=(--samples 2 --warmup 2)
        ;;
    formal)
        args+=(--samples 20 --warmup 20)
        ;;
    nsys)
        args=(/usr/local/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --export=sqlite --output="results/${name}_trace" "${args[@]}" --samples 20 --warmup 20)
        ;;
    ncu)
        metrics=l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_global_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum,dram__bytes_write.sum,launch__shared_mem_per_block_dynamic,launch__registers_per_thread
        args=(/usr/local/cuda-13.0/bin/ncu --target-processes all --clock-control none --cache-control none --replay-mode kernel --graph-profiling node --kernel-name 'regex:CPDeltaRuleFixupUtcmmaSm100|K3OptCoalescedStore' --launch-skip 8 --launch-count 4 --metrics "$metrics" -o "results/${name}_counters" "${args[@]}" --samples 1 --warmup 1)
        ;;
    *)
        printf '%s\n' 'Unexpected mode' >&2
        exit 2
        ;;
esac
printf '%q ' "${args[@]}" > "$meta/command.sh"
printf '\n' >> "$meta/command.sh"
set +e
timeout 8m "${args[@]}" > >(tee "$meta/stdout.log" 9>&-) 2> >(tee "$meta/stderr.log" 9>&- >&2)
status=$?
set -e
printf '%s\n' "$status" > "$meta/exit_status"
exit "$status"
