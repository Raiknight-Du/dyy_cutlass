"""Single authorized hardware-only diagnostic; never a rate/model input."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent;NATIVE=Path('/home/ubuntu/workspace/op12/k4_10h_native_r3_20261026')
assert os.environ.get('K1_GPU_SESSION')
out=ROOT/'r3_v1_hardware_diagnostic01';out.mkdir(exist_ok=False)
binary=NATIVE/'compile02_hostfix/ldg_service';sha=hashlib.sha256(binary.read_bytes()).hexdigest()
assert sha=='bdeabbb53a7a4313d734aa8db7db4f04f567fae39a1fc6880bdf6caf59a30531'
auth=out/'DIAGNOSTIC_ADMISSION.json';auth.write_text(json.dumps(dict(round='R3',binary_sha256=sha,approved_modes=['NCU'],max_N=512,
    allowed_configurations=['S_G128_LOAD'],authorized_attempts=1,tool_diagnostic_only=True,effective_model_value=None,model_enabled=False),indent=2))
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
command=[ncu,'--target-processes','all','--replay-mode','application','--cache-control','none','--clock-control','none',
 '--profile-from-start','off','--launch-count','1','--export',str(out/'hardware_only'),
 '--section','LaunchStats','--metrics','dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum',
 sys.executable,str(NATIVE/'profile_one.py'),'--binary',str(binary),'--gate',str(NATIVE/'STATIC_GATE.json'),
 '--authorization',str(auth),'--config','S_G128_LOAD','--out-root',str(out/'replays')]
(out/'COMMAND.json').write_text(json.dumps(command,indent=2))
q=subprocess.run([sys.executable,str(ROOT/'gpu_telemetry_sidecar.py'),'--out',str(out/'telemetry'),'--command-json',str(out/'COMMAND.json'),'--interval-ms','100'],timeout=600)
record=dict(returncode=q.returncode,binary_sha256=sha,sections=['LaunchStats'],SourceCounters=False,InstructionStats=False,
 CUDA_DISABLE_PTX_JIT='1, unchanged in v1 profile_one',clock_management=False,tool_diagnostic_only=True,effective_model_value=None,model_enabled=False)
(out/'DIAGNOSTIC_RESULT.json').write_text(json.dumps(record,indent=2))
if (out/'hardware_only.ncu-rep').exists():
    with (out/'hardware_only_raw.csv').open('x') as f:
        subprocess.run([ncu,'--import',str(out/'hardware_only.ncu-rep'),'--page','raw','--csv','--print-units','base'],stdout=f,check=True)
print(json.dumps(record),flush=True)
raise SystemExit(q.returncode)
