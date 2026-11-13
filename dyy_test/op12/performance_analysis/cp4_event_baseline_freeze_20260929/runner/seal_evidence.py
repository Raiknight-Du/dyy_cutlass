#!/usr/bin/env python3
"""Exclusive-create SHA manifest and portable evidence archive; tensors stay on B200."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile

root=Path(__file__).resolve().parents[1]
base=json.loads((root/'frozen_baseline.json').read_text())
assert base['baseline_id']=='OP12_CP4_EVENT_FP16_B1H16_P8_20260929_V1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
assert sha(root/'frozen_baseline.json')=='f84842965b4aa3245000ded3e91c764344837abe1bee801c8fee154419b6db3f'
assert sha(root/'PROTOCOL_CN.md')==base['protocol_sha256']
assert sha(root/'runner/measure_cp4_event.py')==base['runner_sha256']
book=root/'outputs/cp4_event_freeze_0929/OP12_CP4_EVENT_FROZEN_BASELINE_20260929.xlsx'
ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(book) as z:
    main=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    cells={c.attrib['r']:c for c in main.findall('.//x:c',ns)}
    for i,shape in enumerate(base['shapes'],6):
        expected=[*shape['stage_us'].values(),shape['stage_sum_us'],shape['native_chain_us']]
        for col,value in zip('CDEFGH',expected):
            c=cells[f'{col}{i}']
            assert c.find('x:f',ns) is not None
            assert abs(float(c.find('x:v',ns).text)-value)<1e-8
    for name in z.namelist():
        if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
            tree=ET.fromstring(z.read(name))
            assert not any(c.get('t')=='e' for c in tree.findall('.//x:c',ns)),name
repo=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/upstreams/flashinfer')
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==base['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root).as_posix()
    if rel.startswith('delivery/') or rel in ['FREEZE_MANIFEST.json','ARCHIVE_RECEIPT.json'] or '__pycache__' in rel:
        continue
    if p.name=='inputs.pt':
        s=int(p.parent.name[1:]);shape=next(x for x in base['shapes'] if x['S']==s)
        h=sha(p);assert h==shape['frozen_input_storage_sha256']
        files.append({'path':rel,'bytes':p.stat().st_size,'sha256':h,'location':'B200_ONLY'})
    else:files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p),'location':'PORTABLE'})
manifest={'baseline_id':base['baseline_id'],'sealed_utc':datetime.now(timezone.utc).isoformat(),
    'baseline_sha256':sha(root/'frozen_baseline.json'),'file_count':len(files),
    'checks':{'formal_processes':9,'prefix_event_observations':4500,'native_batch_observations':45,
        'graph_kernel_order':True,'output_exact_finite':True,'source_clean':True,
        'xlsx_formula_cache_reconciled':True,'xlsx_formula_errors':0,'visual_QA':'main lead verified 5 rendered ranges'},
    'files':files}
with (root/'FREEZE_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2,ensure_ascii=False);f.write('\n')
delivery=root/'delivery';delivery.mkdir(exist_ok=True)
archive=delivery/'RAW_EVENT_EVIDENCE_20260929.tar.gz'
if archive.exists():raise RuntimeError('Archive exists; do not overwrite')
with tarfile.open(archive,'w:gz') as tar:
    for rec in files:
        if rec['location']=='PORTABLE':tar.add(root/rec['path'],arcname=rec['path'],recursive=False)
    tar.add(root/'FREEZE_MANIFEST.json',arcname='FREEZE_MANIFEST.json',recursive=False)
receipt={'baseline_id':base['baseline_id'],'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
    'manifest_sha256':sha(root/'FREEZE_MANIFEST.json'),'portable_files':sum(x['location']=='PORTABLE' for x in files)+1,
    'tensor_files_B200_only':3,'tensor_sha_verified':True,'upstream_clean':True}
with (root/'ARCHIVE_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
