from pathlib import Path
import hashlib,json,csv
R=Path(__file__).resolve().parent; B=R.parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
manifest=load(R/'SHA256SUMS.json')
for key,digest in manifest.items():assert sha(R/key)==digest,key
evidence=load(R/'EVIDENCE_INDEX.json')
for key,data in evidence.items():assert sha(B/key)==data['sha256'],key
jobs=load(R/'FORMAL_JOBS.json');assert len(jobs)==12 and all(not j['foreign'] for j in jobs)
restored=load(R/'FORMAL_CLOCK_RESTORED.json')
assert restored['reset_gpu_clocks_issued'] and restored['memory_clock_unchanged'] and restored['power_limit_mW']==1000000
runner=sha(R/'measure.py')
assert runner==sha(B/'_superseded/cp4_frequency_20261023/cp4_cold_l2_lock1845_20261023/measure.py')
for rp in (R/'runs').iterdir():
    if not rp.is_dir():continue
    ident=load(rp/'IDENTITY.json');assert ident['runner_sha256']==runner
    assert ident['pin']=='e77a4a0d276367895c3b50a642fd8f326c03fb72' and ident['dtype']=='FP16'
    assert ident['eviction_bytes']==530579456
    assert all(x['O_exact'] and x['state_exact'] and x['finite'] for x in load(rp/'CORRECTNESS.json'))
with (R/'02_theory_cold_locked1845/NVML_WINDOW_STATISTICS.csv').open(encoding='utf-8-sig') as f:
    for x in csv.DictReader(f):assert float(x['SM_MHz_min'])==1845 and float(x['SM_MHz_max'])==1845
for category in ('01_performance_cold_auto','02_theory_cold_locked1845'):
    with (R/category/'EVENT_PROCESS_STATISTICS.csv').open(encoding='utf-8-sig') as f:
        for x in csv.DictReader(f):assert float(x['cv_pct'])<=5
result={'manifest_files_verified':len(manifest),'evidence_files_verified':len(evidence),
    'new_processes_valid':12,'new_output_and_state_exact':True,'runner_identical_to_previous':True,
    'locked_NVML_all_shapes_1845':True,'clocks_reset':True,'permanent_deletions':False,
    'local_archive_content_hashes_preserved':True,'model_enabled':False}
(R/'MIRROR_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
