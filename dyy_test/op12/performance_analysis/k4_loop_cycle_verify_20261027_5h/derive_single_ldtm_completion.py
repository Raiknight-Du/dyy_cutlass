"""Derive conditional ONE-instruction ready endpoint, without K4 timing.

The exact native NOP pair preserves all nonload bytes and counts. The first
consumer waits on the load's only write scoreboard at native distance q=2.
Thus Pload-Pnop is the exposed ready wait, not total chain latency. The
unchanged five/five/thirteen-cycle feedback suffix and outer-loop 0.375cy
are independently visible in the NOP slope. No whole helper is a node.
"""
from pathlib import Path
import json,statistics,hashlib
R=Path(__file__).resolve().parent;D=R/'derived'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
controls=read(D/'DEP_NOP_CONTROL_WAIT_POSITION.json')
reviews=[read(D/'DEP_NOP_SLOPE_REVIEW_ORIGINAL.json')['bounds'][0],read(D/'DEP_NOP_SLOPE_REVIEW_EXTENSION.json')['bounds'][0]]
result=[]
for c,b in zip(controls,reviews):
 assert b['strict_gate'] and len(c['loads'])==16
 qs={v['first_wait']['q_before_wait'] for v in c['loads']}
 assert qs=={2}
 for i,z in enumerate(c['loads']):
  assert z['WB']==0 and z['following'][2]['waitmask']==1
  assert z['following'][1]['SASS']=='NOP'
  assert [x['stall'] for x in z['following'][:5]]==([1,1,5,5,13] if i<15 else [1,1,5,1,6])
 load=statistics.mean(b['P_load']);nop=statistics.mean(b['P_nop']);delta=load-nop;q=2
 assert abs(nop-(2+5+5+13+6/16))<1e-9
 result.append(dict(subtype=b['subtype'],kind=b['kind'],Pload=load,Pnop=nop,exposed_wait=delta,
  issue_to_first_consumer_wait=q,conditional_completion=round(q+delta,9),
  ordinary_feedback_suffix_cycles=23,net_loop_boundary_overhead_per_chain=6/16,
  formula='L_conditional = q + (Pload-Pnop); NOP slope matches q+23+6/16; final unrolled suffix differs but is byte-identical between pair.',
  scope='Single LDTM; 4KiB register payload/warp, initialized TMEM, native RAW address feedback. One/four active warps crosschecked. Not a universal load latency or full K4 phase cost.',
  assumptions=['Identical nonload bytes and executed counts, verified by NCU/native patch audit.',
   'Load-specific WB0 is first waited by LOP3 at native q=2; no intervening producer or outstanding older WB0 in the serialized chain.',
   'This is ready-endpoint conditional calibration, distinct from throughput. K4 grouping/queue and address-phase applicability remain explicit.'],
  strict_gate=True,model_enabled=False,effective_model_value=None))
out=dict(method='Exact native one-load/NOP difference plus independently decoded wait position; no K4 observation input.',records=result,
 previous_loose_bounds='Preserved in original review files; [25,50]/[73,98] were protocol bounds before decoding q, not four independent physical parameters.')
(D/'SINGLE_LDTM_COMPLETION_DERIVATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
