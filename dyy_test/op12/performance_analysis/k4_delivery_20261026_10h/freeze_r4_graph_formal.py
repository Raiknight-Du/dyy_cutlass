from pathlib import Path
import json,subprocess,sys
O=Path(__file__).resolve().parent;root=O/'r4_graph_native_edges_formal01'
subprocess.run([sys.executable,str(O/'audit_r4_formal_raw.py'),'--run',str(root)],check=True)
graphs=[]
for trial in range(3):
    files=list((root/f'trial{trial}'/'GRAPH_NODES').glob('N*.json'));assert len(files)==28
    for f in files:
        d=json.loads(f.read_text());assert d['status']=='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS'
        n=d['native_owned_v2_edges'];assert n['count_query_returncode']==n['data_query_returncode']==n['NULL_edgeData_returncode']==0
        assert all(e['from_port']==e['to_port']==e['type']==0 and not any(e['reserved']) for e in n['edges'])
    graphs.append(dict(trial=trial,certificates=len(files),all_native_default_edges_and_NULL_lossless=True))
(root/'GRAPH_NODE_AUDIT_SUMMARY.json').write_text(json.dumps(dict(status='ALL84_GRAPH_TOPOLOGY_NATIVE_DEFAULT_EDGE_GATES_PASS',records=graphs,model_enabled=False),indent=2))
rate=json.loads((root/'RATE_ANALYSIS.json').read_text());assert rate['status']=='RETEST_REQUIRED'
(root/'BRANCH_STOP.json').write_text(json.dumps(dict(status='STOP_STABLE_RATE_BRANCH_JOINT_GATE_FAILURE',all_original_points_retained=True,N16384_extension_used=False,
 failed=[dict(configuration=r['configuration'],domain=r['domain'],control_corrected=r['control_corrected'],role=r['admission_role']) for r in rate['records'] if not r['pass_gate']],
 native_only_pass_does_not_override_joint_gate=True,old_direct_cohort_merged=False,stable_rate_admitted=False,model_enabled=False),indent=2))
subprocess.run([sys.executable,str(O/'analyze_reason_telemetry.py'),'--telemetry',str(O/'r4_graph_native_edges_formal01_telemetry')],check=True)
subprocess.run([sys.executable,str(O/'pack_evidence.py'),'--name','R4_GRAPH_FORMAL01_EVIDENCE','--items',str(root.name),'r4_graph_native_edges_formal01_telemetry','guard_r4_graph_native_edges_formal01','R4_GRAPH_FORMAL_ADMISSION.json','R4_GRAPH_FORMAL_COMMAND.json','R4_GRAPH_FORMAL_BATCH_COMMAND.json','R4_GRAPH_FORMAL01_AUTO_RESTORE_VERIFICATION.json','r4_graph_event_native_edges'],check=True)
