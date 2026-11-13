"""Exactly one approved R3 counter action, with independent reason telemetry."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
NATIVE=Path('/home/ubuntu/workspace/op12/k4_10h_native_r3_20261026')
assert os.environ.get('K1_GPU_SESSION')
assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
auth=ROOT/'R3_NCU_SINGLE_ADMISSION.json'
admission=json.loads(auth.read_text());assert admission['allowed_configurations']==['S_G128_LOAD'] and admission['authorized_NCU_invocations']==1
binary=NATIVE/'compile02_hostfix/ldg_service'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==admission['binary_sha256']
out=ROOT/'r3_ncu_S_G128_LOAD_01';out.mkdir(exist_ok=False)
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
metrics=json.loads((ROOT/'NCU_METRIC_SELECTION.json').read_text())['metrics']
extra=['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum',
 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum','lts__t_sectors_srcunit_tex_op_read.sum',
 'lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum','lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_gds_op_ld.sum']
query=[ncu,'--query-metrics','--chips','gb100','--query-metrics-mode','suffix','--csv','--metrics',','.join(x.rsplit('.',1)[0] for x in extra)]
q=subprocess.run(query,text=True,capture_output=True,check=True,timeout=120)
(out/'EXTRA_METRIC_QUERY.csv').write_text(q.stdout)
available={x['Metric name'] for x in csv.DictReader(q.stdout.splitlines())};assert all(x in available for x in extra)
metrics=list(dict.fromkeys(metrics+extra));(out/'METRICS.json').write_text(json.dumps(metrics,indent=2))
command=[ncu,'--target-processes','all','--replay-mode','application','--cache-control','none','--clock-control','none',
 '--profile-from-start','off','--launch-count','1','--export',str(out/'S_G128_LOAD'),
 '--section','LaunchStats','--section','SourceCounters','--section','InstructionStats','--metrics',','.join(metrics),
 sys.executable,str(NATIVE/'profile_one.py'),'--binary',str(binary),'--gate',str(NATIVE/'STATIC_GATE.json'),
 '--authorization',str(auth),'--config','S_G128_LOAD','--out-root',str(out/'replays')]
(out/'PROFILE_COMMAND.json').write_text(json.dumps(command,indent=2))
(out/'INVOCATION.json').write_text(json.dumps(dict(start_epoch=time.time(),configuration='S_G128_LOAD',N=512,authorized_invocations=1,
   binary_sha256=admission['binary_sha256'],clock_management=False,model_enabled=False),indent=2))
supervisor=[sys.executable,str(ROOT/'gpu_telemetry_sidecar.py'),'--out',str(out/'telemetry'),'--command-json',str(out/'PROFILE_COMMAND.json'),'--interval-ms','100']
done=subprocess.run(supervisor,timeout=1800)
if done.returncode:raise SystemExit(done.returncode)
assert (out/'S_G128_LOAD.ncu-rep').exists()
for page in ['raw','source']:
    export=[ncu,'--import',str(out/'S_G128_LOAD.ncu-rep'),'--page',page,'--csv','--print-units','base']
    if page=='source':export += ['--print-source','sass']
    with (out/('S_G128_LOAD_'+page+'.csv')).open('x') as f:
        subprocess.run(export,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
(out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,formal_NCU_invocations=1,cache_control='none',clock_control='none',replay_mode='application',model_enabled=False),indent=2))
print('R3_SINGLE_NCU_COMPLETE',flush=True)
