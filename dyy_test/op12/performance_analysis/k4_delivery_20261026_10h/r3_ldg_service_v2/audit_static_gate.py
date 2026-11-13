"""Audit the actual executable's SASS/resources and ELF text with stdlib only."""
from pathlib import Path
import collections,csv,hashlib,json,re,struct
O=Path(__file__).resolve().parent; B=O/'compile01_cuda130'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sections(p):
    b=p.read_bytes();assert b[:6]==b'\x7fELF\x02\x01'
    off=struct.unpack_from('<Q',b,40)[0];size,num,si=struct.unpack_from('<HHH',b,58)
    heads=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*size) for i in range(num)]
    s=heads[si];strings=b[s[4]:s[4]+s[5]]
    return {strings[h[0]:].split(b'\0',1)[0].decode():b[h[4]:h[4]+h[5]] for h in heads}
text=lambda p:{k:v for k,v in sections(p).items() if k.startswith('.text.')}
actual=text(B/'ldg_service.sm_100a.cubin')
assert not (B/'ptx_list.stdout').read_text().strip()
src=(O/'ldg_service.cu').read_text(encoding='utf-8');ptx=(B/'audit.compute_100a.ptx').read_text()
assert 'ld.volatile' not in ptx
assert 'ld.global.L1::no_allocate.v4.b32' in ptx and 'ld.global.L1::no_allocate.b32' in ptx
resource=(B/'actual_resource.txt').read_text();sass=(B/'actual_executable.sass').read_text()
functions={};name=None
for line in sass.splitlines():
    m=re.search(r'Function\s*:\s*(\S+)',line)
    if m:name=m[1];functions[name]=[]
    m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:(@!?\w+)\s+)?([\w.]+)\s*(.*?);',line)
    if m and name:functions[name].append({'pc':int(m[1],16),'predicate':m[2],
             'opcode':m[3],'operands':m[4]})
records=[];pc_rows=[];allseq=[]
for vector in [False,True]:
    marker='ILb1E' if vector else 'ILb0E';fn=next(k for k in functions if marker in k)
    rows=functions[fn];loads=[x for x in rows if x['opcode'].startswith('LDG')]
    expected='LDG.E.NA.128' if vector else 'LDG.E.NA'
    assert len(loads)==(32 if vector else 128) and all(x['opcode']==expected for x in loads)
    back=[]
    for x in rows:
        if x['opcode'].startswith('BRA'):
            target=re.findall(r'0x([0-9a-f]+)',x['operands'])
            if target:
                t=int(target[-1],16)
                if t<x['pc'] and t<=loads[0]['pc']<=loads[-1]['pc']<=x['pc']:back.append((t,x['pc']))
    assert len(back)==1
    lo,hi=back[0];loop=[x for x in rows if lo<=x['pc']<=hi]
    forbidden=('STG','LDS','STS','LDSM','LDL','STL','LDTM','STTM','UTCHMMA','HMMA','BAR','WARPSYNC')
    assert not any(x['opcode'].startswith(forbidden) for x in loop)
    assert not any(x['opcode'].startswith(('LDL','STL','STTM','LDTM','UTCHMMA','HMMA','LDS','STS')) for x in rows)
    data={};offsets=[]
    for x in loads:
        dest=int(re.match(r'R(\d+)',x['operands'])[1]);width=4 if vector else 1
        for j in range(width):
            assert dest+j not in data;data[dest+j]={len(data)}
        off=re.search(r'\+0x([0-9a-f]+)',x['operands']);offsets.append(int(off[1],16) if off else 0)
    assert len(data)==128 and sorted(offsets)==list(range(0,512,16 if vector else 4))
    # No consumption of the loaded values occurs before the loop backedge.
    for x in loop:
        if x['opcode'].startswith('LDG'):continue
        used={int(v) for v in re.findall(r'\bR(\d+)\b',x['operands'])}
        assert not used.intersection(data),(x,used.intersection(data))
    taint={k:set(v) for k,v in data.items()}
    tail=[x for x in rows if x['pc']>hi];certificate=None
    for x in tail:
        if x['opcode']=='LOP3.LUT':
            args=[a.strip() for a in x['operands'].split(',')]
            assert args[4] in ['0x96','0x3c']
            dest=int(re.match(r'R(\d+)',args[0])[1])
            taint[dest]=set().union(*(taint.get(int(z),set()) for z in re.findall(r'\bR(\d+)\b',','.join(args[1:4]))))
        if x['opcode'].startswith('ISETP') and '-0x21524111' in x['operands']:
            rr=[int(v) for v in re.findall(r'\bR(\d+)\b',x['operands'])]
            assert len(rr)==1 and len(taint[rr[0]])==128
            certificate=x['pc'];break
    assert certificate is not None
    clocks=[x for x in rows if x['opcode']=='CS2R' and 'SR_CLOCKLO' in x['operands']]
    bars=[x for x in rows if x['opcode'].startswith('BAR.SYNC')]
    assert len(clocks)==2 and len(bars)==2
    assert clocks[0]['pc']<bars[0]['pc']<lo<hi<certificate<bars[1]['pc']<clocks[1]['pc']
    assert all(x['pc']>clocks[1]['pc'] for x in rows if x['opcode'].startswith('STG'))
    rmatch=re.search(re.escape(fn)+r':\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',resource)
    assert rmatch and [int(v) for v in rmatch.groups()[1:]]==[0,0,0]
    d={'variant':'vector' if vector else 'scalar','symbol':fn,'REG':int(rmatch[1]),
      'STACK':0,'LOCAL':0,'LDL':0,'STL':0,'native_load_opcode':expected,
      'static_load_PC_count':len(loads),'data_live_values':128,
      'data_registers':sorted(data),'loop_start_PC':hex(lo),'loop_backedge_PC':hex(hi),
      'load_byte_offsets_in_native_issue_order':offsets,
      'loop_nonload_opcodes':dict(collections.Counter(x['opcode'] for x in loop if not x['opcode'].startswith('LDG'))),
      'fixed_certificate_PC':hex(certificate),'clock_PCs':[hex(x['pc']) for x in clocks],
      'fixed_barrier_PCs':[hex(x['pc']) for x in bars],
      'all_checksum_inputs_data_dependent':True,'control':'same symbol and runtime uniform skip branch, same loop address/control and allocated resources',
      'PTX_volatile':False,'CXX_asm_volatile':True,
      'issue_ready_note':'Clock duration includes fixed start barrier and final data-dependent certificate/barrier; no first-burst latency claim.'}
    records.append(d)
    for x in loads:
        for grid in [1,128]:
            pc_rows.append({'variant':d['variant'],'symbol':fn,'pc':hex(x['pc']),
              'opcode':expected,'grid':grid,'N':512,'expected_warp_issued':4*512*grid,
              'expected_pred_on_lane_instructions':128*512*grid,
              'expected_Source_L2_theoretical_sectors':32*4*512*grid,
              'load_control_expected':0,'actual_NCU':None})
    allseq.extend(dict(variant=d['variant'],pc=hex(x['pc']),predicate=x['predicate'],opcode=x['opcode'],
                       operands=x['operands'],scope='loop' if lo<=x['pc']<=hi else 'fixed') for x in rows)
gate={'status':'STATIC_PASS_PENDING_LEAD_GPU_ADMISSION','GPU_admitted':False,
 'remote_root':'/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_delivery_20261026_10h/r3_ldg_service_v2',
 'actual_executable_relative':'compile01_cuda130/ldg_service','actual_executable_sha256':sha(B/'ldg_service'),
 'actual_CUBIN_relative':'compile01_cuda130/ldg_service.sm_100a.cubin','actual_CUBIN_sha256':sha(B/'ldg_service.sm_100a.cubin'),
 'source_sha256':sha(O/'ldg_service.cu'),'toolkit':'nvcc13.0.88 native-only sm_100a -O3 -lineinfo; separate PTX audit',
 'kernel_revision_count':2,'application_fatbin_native_only':True,'address_mapping':'slot-major ((iteration&7)*gridDim.x+CTA)*65536+thread*512+col*4','CTA_stride_B':65536,'timing_protocol_changed':False,'v1_preserved':True,
 'native_text_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in actual.items()},
 'dynamic_SMEM_B':225280,'dynamic_SMEM_access':False,'runtime_occupancy_validation_required':1,
 'thread0_record_semantics':'Compiler broadcasts clock reads to active threads; only thread0 pair is output. Start clock precedes full CTA barrier; end follows dependent certificate and full CTA barrier.',
 'records':records,'model_enabled':False,'full_prediction':None}
(O/'STATIC_GATE.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding='utf-8')
for name,rows in [('EXPECTED_LDG_PC_N512.csv',pc_rows),('ACTUAL_SASS_SCOPE.csv',allseq)]:
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'status':gate['status'],'resources':[(x['variant'],x['REG'],x['data_live_values']) for x in records],
                  'CUBIN':gate['actual_CUBIN_sha256'],'executable':gate['actual_executable_sha256']}))

