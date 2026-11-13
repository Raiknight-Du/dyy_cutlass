"""Nsight Compute metric/section metadata only: never profiles a process."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
commands=[]
def run(name,argv):
    q=subprocess.run(argv,text=True,capture_output=True,timeout=120)
    (a.out/(name+'.txt')).write_text(q.stdout+q.stderr)
    commands.append(dict(name=name,command=argv,returncode=q.returncode))
    (a.out/'COMMANDS.json').write_text(json.dumps(commands,indent=2))
    return q.stdout
base=run('GB100_BASE_METRICS',[ncu,'--query-metrics','--chips','gb100','--query-metrics-mode','base','--csv'])
rows=list(csv.DictReader(base.splitlines()))
selected=[x for x in rows if x['Metric name'].startswith('l1tex__') and any(y in x['Metric name'] for y in ['data_pipe_','data_bank_','cycles_elapsed','cycles_active','throughput','tma_ld','tma_st'])]
relevant=[x['Metric name']+' | '+x['Metric description'] for x in selected]
(a.out/'RELEVANT_BASE_ROWS.txt').write_text('\n'.join(relevant))
bases=[x['Metric name'] for x in selected]
if bases:
    run('GB100_RELEVANT_SUFFIXES',[ncu,'--query-metrics','--chips','gb100','--query-metrics-mode','suffix','--csv','--metrics',','.join(bases)])
for name in ['SharedMemoryConflicts.py','MemoryWorkloadAnalysis_Tables.section','MemoryWorkloadAnalysis_Chart.section','MemoryWorkloadAnalysis.section','PmSampling.section']:
    source=Path('/opt/nvidia/nsight-compute/2026.2.1/sections')/name
    if source.exists():
        (a.out/name).write_bytes(source.read_bytes())
manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.out.iterdir() if p.is_file()]
(a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('\n'.join(relevant[:70]))
print('READ_ONLY_METRIC_QUERY_NO_GPU_KERNEL',len(relevant),'relevant rows')
