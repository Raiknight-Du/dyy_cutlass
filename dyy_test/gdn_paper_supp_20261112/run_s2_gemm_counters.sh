#!/bin/bash
set -u
cd /home/ubuntu/gdn_paper_supp_20261112
while kill -0 2393423 2>/dev/null && ! grep -q 'END-OF-QUEUE' extension_1113.log; do sleep 2; done
exec 9>gpu.lock
flock -w 20 9
ncu=/opt/nvidia/nsight-compute/2026.2.1/ncu
out="$PWD/environment/s2_gemm_counters"
mkdir -p "$out"
metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_inst_executed_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,launch__registers_per_thread,launch__shared_mem_per_block_dynamic
cd op2
for k in 512 3072; do
  for mode in 0 1; do
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo "GPU busy"; exit 75; fi
    echo "$(date -Iseconds) S2 K=$k mode=$mode"
    timeout 90 python3 - "$k" "$mode" "$out" "$ncu" "$metrics" <<'PY' 
import sys
import subprocess
from pathlib import Path
from run import HOLDOUT, gemm_args
k, mode, out, ncu, metrics = sys.argv[1:]
case = next(case for case in HOLDOUT if case[0] == 8 and case[3] == int(k))
prefix = Path(out) / f"s2_K{k}_mode{mode}"
cmd = [ncu, "--cache-control", "none", "--clock-control", "none",
       "-k", "regex:device_kernel<GemmUniversal", "--launch-count", "1", "--metrics", metrics,
       "-o", str(prefix), "./bin/op2_dense_reference", *gemm_args(case, 0, 1, 1, 1),
       f"--profile-mode={mode}"]
prefix.with_suffix(".command").write_text(repr(cmd))
with prefix.with_suffix(".log").open("w") as stream:
    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=80)
with prefix.with_suffix(".csv").open("w") as stream:
    subprocess.run([ncu, "--import", str(prefix.with_suffix(".ncu-rep")), "--page", "raw", "--csv"],
                   stdout=stream, check=True)
PY
  done
done
echo "$(date -Iseconds) END-OF-QUEUE"
