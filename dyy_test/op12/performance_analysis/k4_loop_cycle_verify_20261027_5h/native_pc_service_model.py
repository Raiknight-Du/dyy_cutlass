"""Resource-constrained *conditional* native-program service model.
No target timing is read until after scenarios are evaluated.
Native work/rate only. Unknown completion lags are symbolic outside the
numeric service scenario, not zero-cost claims about real hardware.
"""
from pathlib import Path
import csv,json,math,heapq,statistics
from collections import defaultdict,Counter
R=Path(__file__).resolve().parent;D=R/'derived'
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
EXT=rows(R/'inputs/EXTENDED_NATIVE_PER_PC.csv')
SIM={int(r['relative_pc'],16):r for r in rows(R/'inputs/SIMT_NATIVE_PER_PC.csv')}
SH={int(r['relative_pc'],16):r for r in rows(R/'inputs/EXPLICIT_SHARED_NATIVE_PER_PC.csv')}
CTRL={int(r['pc'],16):{k:int(r[k]) for k in ['stall','wrbar','rdbar','waitmask']} for r in rows(D/'NATIVE_SCHEDULE_PC.csv')}

def native_ops(lo,hi,hybrid):
 ans=[]
 for r in EXT:
  pc=int(r['relative_pc'],16)
  if not lo<=pc<=hi:continue
  op=r['opcode'];n=float(r['dynamic_warp_instructions_full_action'])/(8192*4)
  if n<.9:continue # fixed/tail branches do not become once-per-block work
  n=min(n,1) if op.startswith(('BRA','SYNCS')) else n
  if n!=1:raise AssertionError((r['relative_pc'],n,op))
  # Four lockstep warps occupy one issue slot in each SMSP. Every retained
  # instruction has at least one group issue cycle, including false lanes.
  resource='ISSUE';work=4.;rate=4.;dur=1;reason='Four successful warp issues / four SMSP schedulers; extra polling/ready latency UNKNOWN'
  if pc in SH:
   resource='LSU_Data|SMEM_Read' if op.startswith(('LDS','LDSM')) else 'LSU_Data|SMEM_Write'
   work=float(SH[pc]['Source_address_model_wavefronts_full_action'])/8192;rate=1
   dur=max(1,math.ceil(work/rate));reason='Enumerated instruction-address WF / NCU Data capacity, same-clock condition'
  elif op.startswith(('LDTM','STTM')):
   reg=32 if '.x32' in op or '16dp256bit.x8' in op else 16
   work=128*reg*4;rate=256 if hybrid and '16dp256bit' in op else 512
   resource='TMEM';dur=math.ceil(work/rate);reason='Explicit native register bytes / subtype-conditioned service; no raw latency included'
  elif pc in SIM:
   z=SIM[pc];work=float(z['scalar_work_full_action'])/8192
   if 'MUFU.EX2' in op or 'MUFU.LG2' in op:
    resource='SFU';rate=16
   elif op.startswith('F2FP'):
    resource='ALU_PACK';rate=128 # scalar conversion inputs/results, two per PACK
   elif op.startswith('HADD2.F32'):
    resource='FP';rate=64
   elif op.startswith(('HADD2','HFMA2','HMUL2')):
    resource='HALF';rate=64
   elif op.startswith(('FMUL','FADD','FFMA')):
    resource='FP';rate=128
   else:raise AssertionError(('UNMAPPED_SIMT',op,z))
   dur=max(1,math.ceil(work/rate));reason='Native per-PC scalar work/rate; one issue floor; fixed generic rates are explicit conditions'
  elif op.startswith(('ISETP','UISETP','VIMNMX','VIADD','IMAD','IADD')) and not op.startswith(('UIMAD','UIADD','UISETP')):
   resource='ALU_PACK';work=float(r['pred_on_lane_instructions_full_action'])/8192;rate=64
   dur=max(1,math.ceil(work/rate));reason='Known generic32 integer work/64; exact unknown subtypes retain issue-only floor'
  if work==0:resource='ISSUE';dur=1;reason='Predicated-off group still issues; no data/compute service charged'
  ans.append(dict(pc=pc,pc_hex=hex(pc),opcode=op,SASS=r['SASS'],resource=resource,work=work,rate=rate,cycles=dur,reason=reason,control=CTRL[pc]))
 return ans

class Engine:
 def __init__(self):self.t=0;self.q=[];self.serial=0;self.resources=defaultdict(float);self.ready=set();self.waiters=defaultdict(list);self.log=[]
 def at(self,t,fun):self.serial+=1;heapq.heappush(self.q,(t,self.serial,fun))
 def signal(self,k):
  self.ready.add(k)
  for fn in self.waiters.pop(k,[]):self.at(self.t,fn)
 def spawn(self,name,gen):
  def step():
   try:z=next(gen)
   except StopIteration:return
   if z[0]=='wait':
    if z[1] in self.ready:self.at(self.t,step)
    else:self.waiters[z[1]].append(step)
   elif z[0]=='signal':self.signal(z[1]);self.at(self.t,step)
   elif z[0]=='work':
    _,res,cy,label=z;parts=res.split('|');start=max([self.t]+[self.resources[p] for p in parts]);end=start+cy
    for p in parts:self.resources[p]=end
    self.log.append((name,label,res,self.t,start,end));self.at(end,step)
   elif z[0]=='async':
    _,res,cy,label,done=z;parts=res.split('|');start=max([self.t]+[self.resources[p] for p in parts]);end=start+cy
    for p in parts:self.resources[p]=end
    self.log.append((name,label,res,self.t,start,end));self.at(end,lambda:self.signal(done));self.at(self.t+1,step)
   else:raise ValueError(z)
  self.at(0,step)
 def run(self):
  while self.q:self.t,_,f=heapq.heappop(self.q);f()
  if self.waiters:raise RuntimeError(('DEADLOCK',list(self.waiters)[:15]))

def simulate(hybrid,N=128,bank_TC=False,qkrate=4096,asynchronous=False,smem_lag=0,tmem_lag=0,full_issuer=False,instruction_issue=False):
 g0=native_ops(0x2330,0x5470,hybrid);g1=native_ops(0x6fb0,0x94d0,hybrid)
 if asynchronous:
  from encoded_async_engine import EncodedEngine
  e=EncodedEngine(smem_lag,tmem_lag)
 else:e=Engine()
 ret=[]
 def run_group(role,ops):
  for c in range(N):
   for op in ops:
    p=op['pc']
    if role=='CG0':
     if p==0x2400 and c>=3:yield 'wait',f'Ainv.free.{c-3}'
     if p==0x3e50 and c>=2:yield 'wait',f'QKsmem.free.{c-2}'
     if p==0x3e70:yield 'wait',f'QK.done.{c}'
    else:
     if p==0x7060 and c:yield 'wait',f'dS.done.{c-1}'
     if p==0x7c60:yield 'wait',f'KS.done.{c}'
     if p==0x8310:yield 'wait',f'QS.done.{c}'
     if p==0x8830:yield 'wait',f'NV.done.{c}'
     if p==0x90b0:yield 'wait',f'O.done.{c}'
    if asynchronous:yield 'op',dict(op,chunk=c)
    else:yield 'work',op['resource'],op['cycles'],f'{c}:{op["pc_hex"]}'
    if role=='CG0':
     if p==0x3e40:yield 'signal',f'Ainv.ready.{c}'
     if p==0x5440:yield 'signal',f'QKacc.free.{c}'
     if p==0x5450:yield 'signal',f'QKweights.ready.{c}'
    else:
     marks={0x79e0:'A.ready',0x7a50:'base.ready',0x8250:'KSacc.free',0x8300:'VKS.ready',0x8820:'QSscaled.ready',
       0x9070:'NVinput.ready',0x9080:'Delta.ready',0x94c0:'CG1.return'}
     if p in marks:
      yield 'signal',f'{marks[p]}.{c}'
      if p==0x94c0:ret.append((c,e.t))
 def packet(name,c,atoms,flops,accwait=None):
  # Separate asynchronous native MMA atoms, shared single TC service engine.
  for a in range(atoms):
   if bank_TC:
    # Explicit sensitivity only. A real bank-coupling law is not assumed
    # from an arbitrary sum of different pipeline counters.
    yield 'work','SMEM_Read',({'QK':256,'KS':128,'QS':128,'NV':64,'O':64,'dS':128}[name]/atoms),f'{c}:{name}.operand.{a}'
   yield 'async','TC',flops/atoms/(qkrate if name=='QK' else 8192),f'{c}:{name}.atom{a}',f'{name}.atom{a}.done.{c}'
  yield 'signal',f'{name}.submitted.{c}'
  yield 'wait',f'{name}.atom{atoms-1}.done.{c}'
  yield 'signal',f'{name}.done.{c}'
 def w8():
  for c in range(N):
   if c>=2:yield 'wait',f'QKacc.free.{c-2}'
   # Execute packet as a separate issuer so next submissions do not wait
   # for the previous packet's result; only its submit prefix.
   e.spawn(f'QK{c}',packet('QK',c,8,1048576))
   yield 'wait',f'QK.submitted.{c}'
 def w10():
  for c in range(N):
   yield 'wait',f'A.ready.{c}'
   for name,atoms,flop in [('KS',8,2097152),('QS',8,2097152),('NV',4,1048576),('O',4,1048576),('dS',4,2097152)]:
    if name=='NV':
     yield 'wait',f'VKS.ready.{c}';yield 'wait',f'Ainv.ready.{c}';yield 'wait',f'KSacc.free.{c}'
    if name=='O':
     yield 'wait',f'NVinput.ready.{c}';yield 'wait',f'QSscaled.ready.{c}';yield 'wait',f'QKweights.ready.{c}'
    if name=='dS':yield 'wait',f'Delta.ready.{c}';yield 'wait',f'base.ready.{c}'
    e.spawn(f'{name}{c}',packet(name,c,atoms,flop));yield 'wait',f'{name}.submitted.{c}'
    if name=='NV':
     # Release Ainv only when its queued MMA reads are complete.
     def release_a(cc=c):
      yield 'wait',f'NV.done.{cc}';yield 'signal',f'Ainv.free.{cc}'
     e.spawn('Ainv.release',release_a())
    if name=='O':
     def release_q(cc=c):
      yield 'wait',f'O.done.{cc}';yield 'signal',f'QKsmem.free.{cc}'
     e.spawn('QKsmem.release',release_q())
 def atom(name,c,a,atoms,flops):
  if bank_TC:yield 'work','SMEM_Read',({'QK':256,'KS':128,'QS':128,'NV':64,'O':64,'dS':128}[name]/atoms),f'{c}:{name}.operand.{a}'
  yield 'async','TC',flops/atoms/(qkrate if name=='QK' else 8192),f'{c}:{name}.atom{a}',f'{name}.atom{a}.done.{c}'
 def commit(name,c,last_atom,release=None):
  yield 'wait',f'{name}.atom{last_atom}.done.{c}';yield 'signal',f'{name}.done.{c}'
  if release:yield 'signal',f'{release}.{c}'
 def w8_async():
  for c in range(N):
   if c>=2:
    yield 'wait',f'QKacc.free.{c-2}'
    yield 'wait',f'QS.done.{c-2}' # Q ring has two slots and dual users
   if c>=3:yield 'wait',f'dS.done.{c-3}' # K ring has three slots
   for a in range(8):
    if instruction_issue:yield 'issue',1,f'{c}:QK.issuer.{a}'
    e.spawn(f'QK.{c}.{a}',atom('QK',c,a,8,1048576));yield 'delay',1
   e.spawn('QK.commit',commit('QK',c,7))
 def w10_native():
  stream=[r for r in EXT if 0xadb0<=int(r['relative_pc'],16)<=0xb8c0 and float(r['dynamic_warp_instructions_full_action'])>=8192]
  data={'KS':(8,2097152,7),'QS':(8,2097152,7),'NV':(4,1048576,3),'O':(4,1048576,3),'dS':(4,2097152,3)}
  waits={0xaeb0:'A.ready',0xb2c0:'KSacc.free',0xb2e0:'VKS.ready',0xb330:'Ainv.ready',
    0xb470:'QSscaled.ready',0xb4c0:'QKweights.ready',0xb4e0:'NVinput.ready',0xb670:'base.ready',0xb690:'Delta.ready'}
  commits={0xb0c0:'KS',0xb290:'QS',0xb450:'NV',0xb640:'O',0xb840:'dS'}
  for c in range(N):
   counts=Counter()
   for row in stream:
    pc=int(row['relative_pc'],16);op=row['opcode'];ctrl=CTRL[pc]
    if pc in waits:yield 'wait',f'{waits[pc]}.{c}'
    if instruction_issue:yield 'issue',1,f'{c}:W10.{hex(pc)}'
    if op=='UTCHMMA':
     name='KS' if pc<0xb170 else 'QS' if pc<0xb3f0 else 'NV' if pc<0xb580 else 'O' if pc<0xb7e0 else 'dS'
     a=counts[name];counts[name]+=1;atoms,flop,_=data[name]
     e.spawn(f'{name}.{c}.{a}',atom(name,c,a,atoms,flop))
    if pc in commits:
     name=commits[pc];release={'NV':'Ainv.free','O':'QKsmem.free'}.get(name)
     e.spawn(f'{name}.commit',commit(name,c,data[name][2],release))
    # Native encoded minimum *issue spacing*, not a measured instruction
    # latency or a substitution of full MMA packet time.
    yield 'delay',max(1,ctrl['stall'])
   assert dict(counts)=={k:v[0] for k,v in data.items()},counts
 e.spawn('CG0',run_group('CG0',g0));e.spawn('CG1',run_group('CG1',g1))
 e.spawn('W8',w8_async() if full_issuer else w8());e.spawn('W10',w10_native() if full_issuer else w10());e.run()
 intervals=[ret[i][1]-ret[i-1][1] for i in range(16,N-16)]
 return dict(hybrid=hybrid,bank_TC=bank_TC,QK_math_rate=qkrate,asynchronous=asynchronous,full_issuer=full_issuer,smem_lag=smem_lag,tmem_lag=tmem_lag,mean_cycle=statistics.mean(intervals),min_cycle=min(intervals),max_cycle=max(intervals),
  per_block_known_service_sum_CG0=sum(z['cycles'] for z in g0),per_block_known_service_sum_CG1=sum(z['cycles'] for z in g1),
  terminal=e.t,ret=ret,resource_work=Counter({k:sum(z[-1]-z[-2] for z in e.log if k in z[2].split('|'))/N for k in e.resources}),
  assumptions=['Fourwarp group instructions processed in native PC order, blocking at modeled resource service, unknown extra completion lags excluded.',
   'All external Q/K/V/T/gate inputs assumed ready. No TMA pipeline constraint in this scenario.',
   'MMA atoms share one FIFO service server. Actual queue arbitration not asserted. Native resource scenario, not a hardware absolute floor.'],model_enabled=False),g0,g1

if __name__=='__main__':
 result=[]
 for hy in [False,True]:
  for bank,qr in [(False,4096),(True,4096),(True,8192)]:
   x,g0,g1=simulate(hy,bank_TC=bank,qkrate=qr);result.append(x)
   if not bank:
    for name,rs in [('CG0',g0),('CG1',g1)]:
     with (D/f'NATIVE_PC_SERVICE_{name}_HY{int(hy)}.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
 (D/'NATIVE_PC_SERVICE_MODELS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps([{k:v for k,v in x.items() if k not in ['ret','assumptions']} for x in result],ensure_ascii=False))
