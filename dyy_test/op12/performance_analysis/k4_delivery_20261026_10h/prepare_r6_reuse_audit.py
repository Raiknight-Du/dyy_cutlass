"""Read old evidence and source only. No CUDA/torch import or device context."""
from pathlib import Path
import ast,hashlib,json
O=Path(__file__).resolve().parent;old=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_init_layout_matched_20261024_2_run01')
out=O/'r6_cpu_reuse_audit01';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['run_matched.py','run_ncu.py','APPROVAL.json','CONTRACT.json','SOURCE_DIFF.json']
snap={n:sha(old/n) for n in files}
for n in files:(out/n).write_bytes((old/n).read_bytes())
source=(old/'run_matched.py').read_text();ast.parse(source)
approval=json.loads((old/'APPROVAL.json').read_text());expected={k:approval['hashes'][k] for k in ['B0','V0']}
audits=[];first={}
for role in ['B0','V0']:
    for f in sorted((old/'NCU'/(role+'_replays')).glob('*/COMPILE_AUDIT.json')):
        d=json.loads(f.read_text())
        for k,v in expected.items():assert d[k]['cubin_sha256']==v
        audits.append(dict(profile_role=role,path=str(f),hashes={k:d[k]['cubin_sha256'] for k in expected}))
        if role not in first:
            parent=f.parent;first[role]={}
            for n in ['IDENTITY.json','ADDRESS_BINDINGS.json','GRAPHS.json','PROFILE_COMPLETE.json','CORRECTNESS.json','ENVIRONMENT.json','COMPILE_AUDIT.json']:
                p=parent/n
                if p.exists():
                    first[role][n]=json.loads(p.read_text());(out/(role+'_'+n)).write_bytes(p.read_bytes())
            for k in expected:
                p=Path(d[k]['cubin']);assert sha(p)==expected[k]
                assert 'REG:168 STACK:24' in d[k]['resources']
assert all(any(r['profile_role']==k for r in audits) for k in expected)
report=dict(status='CPU_READ_ONLY_REUSE_CANDIDATE_NO_GPU_ADMISSION',old_root=str(old),source_hashes=snap,
 old_replay_compile_audits=audits,expected_CUBIN_hashes=expected,first_replay_evidence=first,
 no_CUDA_context_created=True,old_sources_modified=False,
 constraints=['Old run_ncu.py hardcodes old/NCU and mkdir(exist_ok=False); cannot rerun that orchestrator in place.',
 'Use new R6 orchestrator with original run_matched.py --mode profile --profile B0|V0 --out NEW; original script, sources, inputs and kernels remain unchanged.',
 'Original run_matched compiles both B0/V0 each replay and asserts old APPROVAL exact hashes before target NVTX action; full gold/correctness and CP graph warmup remain original.',
 'No GPU compile or experimental invocation is authorized by this audit.'],
 prediction_only=dict(Data_read_WF_B0_minus_V0=1376256,basis='112 fixed CTA * (16384 - 4096); original initial16CTA and all other native clients presumed unchanged pending Source/perPC and hardware checks',model_input=False),
 model_enabled=False)
(out/'REUSE_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(status=report['status'],replay_audits=len(audits),hashes=expected)))
