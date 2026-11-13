from pathlib import Path
import csv,json,subprocess,sys
O=Path(__file__).resolve().parent
run=O/'r4_graph_screen01';sup=O/'r4_graph_screen01_supervision'
ev=list(csv.DictReader((run/'EVENT_RAW.csv').open()));assert not ev
stop=dict(status='STOPPED_EDGE_TYPE_BINDING_ASSERT_BEFORE_GRAPH_INSTANTIATE_REPLAY',samples=0,graph_replays=0,
 captured_DOTs=len(list((run/'GRAPH_NODES').glob('*.dot'))),node_audit_JSONs=len(list((run/'GRAPH_NODES').glob('*.json'))),
 direct_lazy_invocations=json.loads((run/'GRAPH_TIMING_PROTOCOL.json').read_text())['direct_lazy_invocations'],
 reported_edge_types=[36,0],full_original_tuple_preserved=False,kernel_params_gate_reached=False,kernel_changed=False,model_enabled=False)
(run/'STOP_RECORD.json').write_text(json.dumps(stop,indent=2))
groups={
 'R6_NCU01_EVIDENCE':['r6_ncu01','guard_r6_ncu01','R6_NCU_COMMAND.json','R6_NCU01_AUTO_RESTORE_VERIFICATION.json','RUNNER_NCU_LEDGER.jsonl'],
 'R5_PREPARE01_EVIDENCE':['r5_actual_prepare01','r5_actual_prepare01_telemetry','guard_r5_prepare01','R5_PREPARE_ADMISSION.json','R5_PREPARE_COMMAND.json'],
 'R4_GRAPH_SCREEN01_FAILED_EVIDENCE':['r4_graph_screen01','r4_graph_screen01_supervision','guard_r4_graph_screen01','R4_GRAPH_SCREEN_ADMISSION.json','r4_graph_event_v2']}
for name,items in groups.items():subprocess.run([sys.executable,str(O/'pack_evidence.py'),'--name',name,'--items',*items],check=True,timeout=300)
