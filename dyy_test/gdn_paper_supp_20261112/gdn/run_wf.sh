set -eu
cd /home/ubuntu/gdn_paper_supp_20261112
exec 9>gpu.lock
flock -w 25 9
ncu=/opt/nvidia/nsight-compute/2026.2.1/ncu
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo "GPU busy"; exit 75; fi
cd gdn
name=wf_S32768_P8_t51
mkdir -p "results/${name}_logs" "cache/$name"
export TMPDIR="$PWD/cache/$name" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
metrics=l1tex__data_pipe_lsu_wavefronts.sum,l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read.sum,l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_write.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,launch__shared_mem_per_block_dynamic,launch__registers_per_thread
args=("$ncu" --target-processes all --clock-control none --cache-control none --replay-mode kernel --graph-profiling node --kernel-name 'regex:CPDeltaRuleFixupUtcmmaSm100|K3OptCoalescedStore' --launch-skip 8 --launch-count 4 --metrics "$metrics" -o "results/${name}_counters" /home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python -B src/k3trace/run_chain.py --out "results/$name" --s 32768 --partitions 8 --trial 51 --arms B0,K3 --samples 1 --warmup 1)
printf '%q ' "${args[@]}" >"results/${name}_logs/command.sh"
timeout 90 "${args[@]}" >"results/${name}_logs/stdout.log" 2>"results/${name}_logs/stderr.log"
"$ncu" --import "results/${name}_counters.ncu-rep" --page raw --csv >"results/${name}_counters.csv"
