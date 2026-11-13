from pathlib import Path
import hashlib,json,subprocess,sys
O=Path(__file__).resolve().parent;analyzer=Path('/home/ubuntu/workspace/op12/k4_10h_native_r5_20261026/analyze_rates.py')
assert hashlib.sha256(analyzer.read_bytes()).hexdigest()=='fa2a21f6e23e578d38f164fe73325c0a3cf068c79b1aad94297146063c4db085'
for name in ['r5_graph_formal01','r5_graph_formal_ext16384_01']:
    root=O/name;p=root/'RATE_ANALYSIS.json';before=json.loads(p.read_text());backup=root/'RATE_ANALYSIS_PRE_JOINT_CLEAR.json'
    if not backup.exists():backup.write_bytes(p.read_bytes())
    subprocess.run([sys.executable,str(analyzer),'--input',str(root),'--out',str(p)],check=True)
    after=json.loads(p.read_text());assert before['input_SHA256']==after['input_SHA256'] and before['status']==after['status']
    for old,new in zip(before['records'],after['records']):
        for key in ['configuration','domain','control_corrected','trials','mean_tail_slope','cross_process_range_over_mean','pass_gate','admission_role']:assert old[key]==new[key],(name,key)
        if after['status']!='CONDITIONAL_SLOPES_PASS':assert all(v is None for k,v in new.items() if k.startswith('conditional_'))
    (root/'JOINT_ADMISSION_REFRESH.json').write_text(json.dumps(dict(analyzer_SHA256=hashlib.sha256(analyzer.read_bytes()).hexdigest(),statistics_unchanged=True,old_derived_preserved=backup.name,raw_files_unchanged=True,joint_conditional_fields_cleared_on_failure=True,model_enabled=False),indent=2))
    subprocess.run([sys.executable,str(O/'inspect_r5_formal.py'),'--run',str(root)],stdout=subprocess.DEVNULL,check=True)
