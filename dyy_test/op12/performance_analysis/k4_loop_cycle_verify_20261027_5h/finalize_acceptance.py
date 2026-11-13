from pathlib import Path
import json,datetime,hashlib
R=Path(__file__).resolve().parent;D=R/'derived'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
m=read(D/'FINAL_CYCLE_MODEL.json');verify=read(R/'LOCAL_EVIDENCE_VERIFICATION.json');audit=read(D/'FINAL_CPU_AUDIT.json')
assert verify['status']==audit['status']=='PASS'
assert all(x['all_cases_within10'] for x in m['comparison'])
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='DELIVERED_WITH_SCOPE',
 scope='OriginalFP16K4 B1H16D128, three targetS/C, completeCP cache-history target return cadence, 64-token body block only.',
 stop_reason='Requested body-slope +/-10% achieved for allthree shapes; independent mechanism/count evidence exists; remaining uncertainty is disclosed. No whole-kernel90% or complete causalGap closure claimed.',
 observed_slopes=[x['observed_cycle_slope'] for x in m['comparison']],conditional_model_cycle_range=m['main_range'],
 signed_error_ranges=[[x['signed_error_min'],x['signed_error_max']] for x in m['comparison']],
 main_evidence=['CG0 all132 sharedPCs: max intrinsic address conflict1; source/active predicates match.',
  'Six shared families isolated/concurrentTC:24 accepted NCU; read interference, write/read overlap.',
  'CompleteCP same-work waiting-order intervention:12 NCU, bitwise outputs/state, unchanged native work.',
  'OneLDTM exactNOP pair:8NCU, strict1% slopes, native firstconsumer q2, conditional ready26/74.',
  'SameCUBIN PACK/IADD3:6NCU and strict1% dual-domain slopes; shared ALU service condition.',
  'Actualdevice maxIPC4/SM and1/scheduler; aggregate issue queue included; CPU resource overlap checks pass.'],
 limitations=m['unresolved']+['Grouped/FIFO issue schedule can overestimate actual per-SMSP warp-skew schedule; signed residual255..294cycles/block is not a positive unmodeled cost.',
  'Model/observation agreement is not the proportion of the original causalGap explained.'],
 original_upstream_clean=read(R/'ENVIRONMENT_FINAL.json')['upstream_clean'],formal_workbook_overwritten=False,
 model_enabled=False,effective_model_value=None,evidence_verification=verify,
 deliverables=[dict(file=n,sha256=hashlib.sha256((R/n).read_bytes()).hexdigest()) for n in ['K4_LOOP_CYCLE_ANALYSIS_CN.md','K4_LOOP_CYCLE_REVIEW_20261027.xlsx','K4_LOOP_NATIVE_DAG.drawio','K4_LOOP_EVIDENCE_20261027.tar.gz']])
(R/'FINAL_ACCEPTANCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=out['status'],scope=out['scope'],model_range=m['main_range'],files_verified=verify['files_verified']),ensure_ascii=False))
