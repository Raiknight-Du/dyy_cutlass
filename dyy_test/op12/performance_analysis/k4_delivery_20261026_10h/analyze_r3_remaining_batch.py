from pathlib import Path
import json,subprocess,sys
O=Path(__file__).resolve().parent;run=O/'r3_v2_ncu_remaining01'
summary=[]
for case in ['S_G1_LOAD','S_G1_CONTROL','S_G128_CONTROL','V_G1_LOAD','V_G1_CONTROL','V_G128_LOAD','V_G128_CONTROL']:
    q=subprocess.run([sys.executable,str(O/'analyze_r3_ncu.py'),'--run',str(run/case),'--engineering',str(O/'r3_ldg_service_v2'),'--config',case],text=True,capture_output=True)
    (run/case/'ANALYSIS.stdout').write_text(q.stdout);(run/case/'ANALYSIS.stderr').write_text(q.stderr)
    assert q.returncode==0,(case,q.stderr)
    d=json.loads((run/case/'COUNTER_AUDIT.json').read_text());assert d['status']=='DYNAMIC_COUNTS_PASS_SASS_MATCH',d['native_SASS_comparison_mismatches']
    metrics=d['metrics'];chosen={k:v for k,v in metrics.items() if k.endswith('.sum') and any(x in k for x in ['dram__bytes','lts__t_sectors','data_pipe_lsu_wavefronts','data_bank_conflicts_pipe_lsu_mem_gds','t_requests_pipe_lsu_mem_global_op_ld','t_sectors_pipe_lsu_mem_global_op_ld'])}
    summary.append(dict(configuration=case,status=d['status'],load_PC_count=d['load_PC_count'],load_totals=d['load_totals'],metrics=chosen))
(run/'BATCH_COUNTER_AUDIT.json').write_text(json.dumps(dict(status='ALL7_DYNAMIC_COUNTS_SASS_PASS',cases=summary,model_enabled=False),indent=2))
print('ALL7_DYNAMIC_COUNTS_SASS_PASS',flush=True)
