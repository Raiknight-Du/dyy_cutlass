"""Offline comparison of kept actual artifacts; no CUDA imports."""
import argparse
import collections
import difflib
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--actual',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def read(root):
    sass=next(root.glob('*.sass')).read_text();ptx=next(root.glob('*.ptx')).read_text();cub=next(root.glob('*.cubin'))
    ins=[]
    for line in sass.splitlines():
        m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?);',line)
        if m:
            text=m[2].strip();op=re.sub(r'^@!?\w+\s+','',text).split()[0]
            ins.append(dict(pc=m[1],text=text,op=op))
    raw=cub.read_bytes();shoff=int.from_bytes(raw[40:48],'little');shsize=int.from_bytes(raw[58:60],'little');shnum=int.from_bytes(raw[60:62],'little');stridx=int.from_bytes(raw[62:64],'little')
    sh=[raw[shoff+i*shsize:shoff+(i+1)*shsize] for i in range(shnum)]
    strhdr=sh[stridx];so=int.from_bytes(strhdr[24:32],'little');sn=int.from_bytes(strhdr[32:40],'little');strings=raw[so:so+sn]
    text=[]
    for hdr in sh:
        no=int.from_bytes(hdr[:4],'little');name=strings[no:strings.index(b'\0',no)].decode()
        off=int.from_bytes(hdr[24:32],'little');size=int.from_bytes(hdr[32:40],'little')
        if name.startswith('.text.'):
            text.append(dict(name=name,size=size,sha256=hashlib.sha256(raw[off:off+size]).hexdigest()))
    return dict(sass=sass,ptx=ptx,cubin_sha256=hashlib.sha256(raw).hexdigest(),instructions=ins,text_sections=text)
b,f=read(a.before),read(a.actual)
norm=lambda x:re.sub(r'\.L_x_\d+','LABEL',x['text'])
sm=difflib.SequenceMatcher(None,[norm(x) for x in b['instructions']],[norm(x) for x in f['instructions']],autojunk=False)
diff=[]
for tag,i,j,k,l in sm.get_opcodes():
    if tag!='equal':diff.append(dict(kind=tag,before=b['instructions'][i:j],actual=f['instructions'][k:l]))
bc=collections.Counter(x['op'] for x in b['instructions']);fc=collections.Counter(x['op'] for x in f['instructions'])
report=dict(before=str(a.before),actual=str(a.actual),hashes=[b['cubin_sha256'],f['cubin_sha256']],instructions=[len(b['instructions']),len(f['instructions'])],
    text_sections_before=b['text_sections'],text_sections_actual=f['text_sections'],
    executable_text_equal=[(x['size'],x['sha256']) for x in b['text_sections']]==[(x['size'],x['sha256']) for x in f['text_sections']],
    exact_instruction_text_equal=b['instructions']==f['instructions'],label_normalized_instruction_equal=not diff,
    extra_actual=dict(fc-bc),removed_actual=dict(bc-fc),differences=diff,
    before_ptx_entry=b['ptx'][b['ptx'].index('.visible .entry'):b['ptx'].index('.visible .entry')+5000],
    actual_ptx_entry=f['ptx'][f['ptx'].index('.visible .entry'):f['ptx'].index('.visible .entry')+5000],
    model_enabled=False)
a.out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['differences','before_ptx_entry','actual_ptx_entry']},indent=2))
print('FIRST_DIFFERENCES',json.dumps(diff[:4],indent=2))
