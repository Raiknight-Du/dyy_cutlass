from pathlib import Path
import csv,collections,json,statistics,argparse
O=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=O/'r4_formal01');a=p.parse_args();root=a.run;records=[]
for trial in range(3):
    run=root/f'trial{trial}';done=json.loads((run/'completion.json').read_text());meta=json.loads((run/'ACTUAL_METADATA_GATE.json').read_text())
    assert done['completed'] and done['correctness'] and done['samples']==30 and done['warmup']==20 and done['invocations']==840
    assert meta['CPU_text_exact_match'] and meta['max_CTA_per_SM']==1 and meta['actual_CUBIN_SHA256']=='5ebdc94be5cda307ee64cb975209e11ed68ef25a420cfc3598b00c15368cc066'
    ev=list(csv.DictReader((run/'EVENT_RAW.csv').open()));ct=list(csv.DictReader((run/'CTA_CLOCK_RAW.csv').open()));assert len(ev)==840 and len(ct)==107520
    by=collections.defaultdict(list)
    for r in ct:
        assert int(r['smid_begin'])==int(r['smid_end']) and int(r['end_cycle'])-int(r['begin_cycle'])==int(r['cycles'])>0
        by[int(r['invocation'])].append(r)
    unique=collections.Counter()
    for r in ev:
        rows=by[int(r['invocation'])];assert len(rows)==128 and {int(x['cta']) for x in rows}==set(range(128))
        cy=[int(x['cycles']) for x in rows];sms={int(x['smid_begin']) for x in rows}
        assert r['valid']=='1' and float(r['cta_cycle_median'])==statistics.median(cy) and float(r['cta_cycle_max'])==max(cy) and int(r['unique_SMs'])==len(sms)
        unique[len(sms)]+=1
    records.append(dict(trial=trial,launches=len(ev),CTA_records=len(ct),unique_SMs_per_launch_histogram=dict(unique)))
(root/'RAW_ENDPOINT_AUDIT.json').write_text(json.dumps(dict(status='ALL2520_LAUNCH_322560_CTA_ENDPOINTS_VALID',records=records,model_enabled=False),indent=2));print('ALL2520_LAUNCH_322560_CTA_ENDPOINTS_VALID')
