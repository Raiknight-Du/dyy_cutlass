"""Offline, isolated schedule controls; preserve all arithmetic and transfers."""
from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent
p=R/'inputs/gated_delta_net_cp_prefill.py'
s=p.read_text(encoding='utf-8');tree=ast.parse(s)
c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CPDeltaRulePrefillTcgen05Sm100')
f=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='compute_group_1_chunk')
method='\n'.join(s.splitlines()[f.lineno-2:f.end_lineno])+'\n'
ks='            ks_handle = cg1_shared_acc_consumer.wait_and_advance()\n'
qs='            qs_handle = q_state_acc_consumer.wait_and_advance()\n'
assert method.count(ks)==method.count(qs)==1
anchor='        tRT_rV = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)\n'
assert method.count(anchor)==1
out=[]
for name,moved in [('KSJOIN',[ks]),('QSJOIN',[ks,qs]),('GATEJOIN',[ks,qs])]:
    body=method
    for line in moved:body=body.replace(line,'')
    defaults='        ks_handle = cg1_shared_acc_consumer.current_handle()\n'
    if name in ['QSJOIN','GATEJOIN']:defaults+='        qs_handle = q_state_acc_consumer.current_handle()\n'
    selected_anchor='        for k in cutlass.range_constexpr(cute.size(tTR_tCcShared)):\n' if name=='GATEJOIN' else anchor
    assert body.count(selected_anchor)==1
    body=body.replace(selected_anchor,defaults+'        if valid_state:\n'+''.join(moved)+selected_anchor)
    text='\n'.join(s.splitlines()[:28])+'''\n# Private mechanism test: V gather after selected completed-MMA waits.
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import tcgen05
from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp_prefill import CPDeltaRulePrefillTcgen05Sm100
class K4VJoin(CPDeltaRulePrefillTcgen05Sm100):
'''+body
    ast.parse(text)
    (R/f'variant_{name}.py').write_text(text,encoding='utf-8')
    out.append(dict(variant=name,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
      moved_waits=[x.strip() for x in moved],all_math_and_TMA_and_shared_operations_preserved=True,
      causal_question='Does removing V LDSM overlap with KS or both KS/QS reduce shared read arbitration?',
      prediction='SourceWF unchanged. QSJOIN must place V LDSM after KS/QS done; if V is main origin of read excess, hardware shared read conflict falls. Time may increase from lost overlap.',
      falsifier='No meaningful excess decrease with confirmed native placement, or other work/resources change enough to invalidate attribution.',
      model_enabled=False))
(R/'V_JOIN_PREDICTION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print([(x['variant'],x['moved_waits']) for x in out])
