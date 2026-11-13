"""Same-action counter accounting; never uses replay time as Event timing."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
def number(s):
    if s is None or str(s).strip() in ['','n/a','nan','N/A']:return None
    try:return float(str(s).replace(',',''))
    except ValueError:return None
cases={}
for case in ['REF','FRAME0','FRAME7']:
    if not (a.run/(case+'_raw.csv')).exists():continue
    raw=list(csv.reader((a.run/(case+'_raw.csv')).open()))
    assert len(raw)==3,'Must have exactly one selected action'
    metrics=dict(zip(raw[0],raw[2]));units=dict(zip(raw[0],raw[1]))
    src=list(csv.reader((a.run/(case+'_source.csv')).open()))
    hdr=src[1];data=[dict(zip(hdr,r)) for r in src[2:] if len(r)==len(hdr)]
    inst=collections.Counter();pred=collections.Counter();pref=[];trace_clocks=[];gate_waits=[];shared=collections.defaultdict(float)
    first=next(int(r['Address'],16) for r in data if r.get('Address','').startswith('0x'))
    target_fields=[k for k in hdr if ('Wavefront' in k or 'Sector' in k) and ('Shared' in k or 'Global' in k or 'Local' in k)]
    for row in data:
        text=row.get('Source','').strip()
        if not text or not row.get('Address','').startswith('0x'):continue
        stripped=re.sub(r'^@!?\w+\s+','',text)
        op=stripped.split()[0].split('.')[0]
        executed=number(row.get('Instructions Executed')) or 0
        predicate=number(row.get('Predicated-On Thread Instructions Executed')) or 0
        inst[op]+=executed;pred[op]+=predicate
        if 'UTMACCTL.PF' in text:pref.append(dict(relative_PC=hex(int(row['Address'],16)-first),source=text,warp_instructions=executed,predicated_threads=predicate))
        relative=hex(int(row['Address'],16)-first)
        if 'SR_CLOCKLO' in text:trace_clocks.append(dict(relative_PC=relative,source=text,warp_instructions=executed,predicated_threads=predicate))
        if case!='REF' and relative in ['0x9d50','0xbec0','0xe620']:
            gate_waits.append(dict(relative_PC=relative,source=text,warp_instructions=executed,predicated_threads=predicate))
        for field in target_fields:
            shared[field]+=number(row.get(field)) or 0
    def m(base,suffix='.sum'):
        value=number(metrics.get('l1tex__'+base+suffix));assert value is not None,base+suffix
        return value
    residuals=dict(
       LSU_minus_mem_children=m('data_pipe_lsu_wavefronts')-m('data_pipe_lsu_wavefronts_mem_lgds')-m('data_pipe_lsu_wavefronts_mem_shared')-m('data_pipe_lsu_wavefronts_mem_surface'),
       LSU_minus_cmd_children=m('data_pipe_lsu_wavefronts')-m('data_pipe_lsu_wavefronts_cmd_read')-m('data_pipe_lsu_wavefronts_cmd_write'),
       LGDS_minus_cmd_children=m('data_pipe_lsu_wavefronts_mem_lgds')-m('data_pipe_lsu_wavefronts_mem_lgds_cmd_read')-m('data_pipe_lsu_wavefronts_mem_lgds_cmd_write'),
       SHARED_minus_cmd_children=m('data_pipe_lsu_wavefronts_mem_shared')-m('data_pipe_lsu_wavefronts_mem_shared_cmd_read')-m('data_pipe_lsu_wavefronts_mem_shared_cmd_write'),
       SHARED_minus_op_children=m('data_pipe_lsu_wavefronts_mem_shared')-sum(m('data_pipe_lsu_wavefronts_mem_shared_op_'+x) for x in ['ld','st','atom','misc','umemsets','ipa']),
       LSU_read_minus_mem_read_children=m('data_pipe_lsu_wavefronts_cmd_read')-m('data_pipe_lsu_wavefronts_mem_lgds_cmd_read')-m('data_pipe_lsu_wavefronts_mem_shared_cmd_read'),
       LSU_write_minus_mem_write_children=m('data_pipe_lsu_wavefronts_cmd_write')-m('data_pipe_lsu_wavefronts_mem_lgds_cmd_write')-m('data_pipe_lsu_wavefronts_mem_shared_cmd_write')-m('data_pipe_lsu_wavefronts_mem_surface'),
       TC_minus_shared=m('data_pipe_tc_wavefronts')-m('data_pipe_tc_wavefronts_mem_shared'),
       TC_shared_minus_explicit_subtypes=m('data_pipe_tc_wavefronts_mem_shared')-sum(m('data_pipe_tc_wavefronts_mem_shared_op_'+x) for x in ['utccp','utcmma_matrix_a','utcmma_matrix_b_scope_1cta','utcmma_matrix_b_scope_2cta']),
       TEX_minus_mem_children=m('data_pipe_tex_wavefronts')-m('data_pipe_tex_wavefronts_mem_surface')-m('data_pipe_tex_wavefronts_mem_texture'))
    read=m('data_pipe_lsu_wavefronts_mem_shared_op_ld');write=m('data_pipe_lsu_wavefronts_mem_shared_op_st')
    cr=m('data_bank_conflicts_pipe_lsu_mem_shared_op_ld');cw=m('data_bank_conflicts_pipe_lsu_mem_shared_op_st')
    nativekeys=['LDG','LDL','STL','LDTM','STTM','UTCHMMA','UTMALDG','UTMASTG','STG','CS2R','SYNCS','NANOSLEEP']
    selected={k:dict(value=number(v),unit=units[k]) for k,v in metrics.items() if k.startswith(('l1tex__','dram__','lts__')) and number(v) is not None}
    cases[case]=dict(kernel=metrics['Kernel Name'],ID=metrics['ID'],selected_actions=1,
        native_warp_instructions={k:inst[k] for k in nativekeys},native_predicated_threads={k:pred[k] for k in nativekeys},
        descriptor_prefetch=pref,prefetch_dynamic_warp_total=sum(x['warp_instructions'] for x in pref),
        trace_clock_reads=trace_clocks,trace_clock_predicated_threads=sum(x['predicated_threads'] for x in trace_clocks),
        new_ready_gate_waits=gate_waits,
        source_counter_sums=dict(shared),hardware_metrics=selected,parent_child_residuals=residuals,
        shared_actual_minus_conflict=dict(read=read-cr,write=write-cw,total=read+write-cr-cw,
             source_shared_actual=shared.get('L1 Wavefronts Shared'),source_shared_ideal=shared.get('L1 Wavefronts Shared Ideal')),
        NCU_replay_passes=number(metrics.get('profiler__replayer_passes')),timing_used=False)
report=dict(cases=cases,scope='Each equality is evaluated within a single selected K4 action',
    claim_limit='One NCU action can contain metrics from different application replay passes; deterministic instruction/TC counts and variable hardware poll/arbitration counts must be distinguished. No replay timing or stable throughput. Actual LSU WF may already contain arbitration replay; do not add bank conflicts or double count TC/TMA-caused arbitration.',model_enabled=False)
(a.run/'NCU_RECONCILIATION.json').write_text(json.dumps(report,indent=2))
print(json.dumps({case:{k:v for k,v in rec.items() if k in ['native_warp_instructions','prefetch_dynamic_warp_total','parent_child_residuals','source_counter_sums','shared_actual_minus_conflict','NCU_replay_passes']} for case,rec in cases.items()},indent=2))
