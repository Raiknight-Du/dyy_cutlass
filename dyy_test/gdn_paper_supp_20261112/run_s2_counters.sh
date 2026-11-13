set -eu
cd /home/ubuntu/gdn_paper_supp_20261112
exec 9>gpu.lock
flock -w 20 9
ncu=/opt/nvidia/nsight-compute/2026.2.1/ncu
out="$PWD/environment/s2_all_kernels"
mkdir -p "$out"
cd op2
for k in 512 3072; do
  if [[ -s "$out/s2_K${k}.ncu-rep" ]]; then continue; fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo "GPU busy"; exit 75; fi
  python3 - "$k" "$out" "$ncu" <<'PY'
import sys
import subprocess
from pathlib import Path
from run import HOLDOUT, gemm_args
k, out, ncu = sys.argv[1:]
case = next(case for case in HOLDOUT if case[0] == 8 and case[3] == int(k))
prefix = Path(out) / f"s2_K{k}"
metrics = "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_inst_executed_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,launch__registers_per_thread,launch__shared_mem_per_block_dynamic"
cmd = [ncu, "--cache-control", "none", "--clock-control", "none",
       "-k", "device_kernel", "--launch-count", "5", "--metrics", metrics,
       "-o", str(prefix), "./bin/op2_dense_reference", *gemm_args(case, 0, 1, 1, 1)]
prefix.with_suffix(".command").write_text(repr(cmd))
with prefix.with_suffix(".log").open("w") as stream:
    subprocess.run(cmd, stdout=stream, stderr=stream, check=True, timeout=100)
with prefix.with_suffix(".csv").open("w") as stream:
    subprocess.run([ncu, "--import", str(prefix.with_suffix(".ncu-rep")), "--page", "raw", "--csv"],
                   stdout=stream, check=True)
PY
done
