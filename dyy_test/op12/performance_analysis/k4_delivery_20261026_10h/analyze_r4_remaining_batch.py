from pathlib import Path
import json,subprocess,sys
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r4_20261026');root=O/'r4_ncu_remaining01'
records=[]
for mode,case in enumerate(['CONTROL','SHADER_ONLY','TC_ONLY']):
    out=root/case
    subprocess.run([sys.executable,str(N/'analyze_ncu.py'),'--source',str(out/(case+'_source.csv')),'--raw',str(out/(case+'_raw.csv')),'--mode',str(mode),'--out',str(out/'COUNTER_AUDIT.json')],check=True)
    subprocess.run([sys.executable,str(O/'check_r4_ncu_identity.py'),'--run',str(out),'--config',case,'--engineering',str(N)],check=True)
    d=json.loads((out/'COUNTER_AUDIT.json').read_text());assert d['status']=='TARGET_PC_COUNTS_PASS'
    assert d['TC_hardware_gate_status']=='CHECKED' and len(d['TC_hardware_checks'])==4
    records.append(dict(configuration=case,PC_checks=len(d['checks']),TC_checks=d['TC_hardware_checks'],hardware_inventory=d['hardware_inventory'],sync_inventory=d['synchronization_source_inventory']))
(root/'BATCH_COUNTER_AUDIT.json').write_text(json.dumps(dict(status='ALL3_PC_IDENTITY_RESOURCE_PASS',cases=records,model_enabled=False),indent=2))
print('ALL3_PC_IDENTITY_RESOURCE_PASS',flush=True)
