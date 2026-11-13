"""Alternative to serial per-PC service: compiled stall/scoreboard controls.
This checks whether matching from a blocking service schedule is robust.
Memory service, completion lag and program issue spacing stay separate.
"""
from collections import defaultdict
import heapq
class EncodedEngine:
 def __init__(self,smem_lag=0,tmem_lag=0):
  self.t=0;self.q=[];self.serial=0;self.resources=defaultdict(float);self.ready=set();self.waiters=defaultdict(list);self.log=[]
  self.smem_lag=smem_lag;self.tmem_lag=tmem_lag
 def at(self,t,f):self.serial+=1;heapq.heappush(self.q,(t,self.serial,f))
 def signal(self,k):
  self.ready.add(k)
  for f in self.waiters.pop(k,[]):self.at(self.t,f)
 def spawn(self,name,gen):
  bars=[0.]*6
  def step():
   try:z=next(gen)
   except StopIteration:return
   kind=z[0]
   if kind=='wait':
    if z[1] in self.ready:self.at(self.t,step)
    else:self.waiters[z[1]].append(step)
   elif kind=='signal':self.signal(z[1]);self.at(self.t,step)
   elif kind=='op':
    o=z[1];res=o['resource'];parts=res.split('|');ctrl=o['control'];mask=ctrl['waitmask']
    ready=max([self.t]+[bars[i] for i in range(6) if mask&(1<<i)])
    start=max([ready]+[self.resources[p] for p in parts]);end=start+o['cycles']
    for p in parts:self.resources[p]=end
    lag=0
    if o['opcode'].startswith(('LDS','LDSM')):lag=self.smem_lag
    if o['opcode'].startswith('LDTM'):
     lag=self.tmem_lag.get('16dp' if '16dp' in o['opcode'] else '32dp',0) if isinstance(self.tmem_lag,dict) else self.tmem_lag
    wb,rb=ctrl['wrbar'],ctrl['rdbar']
    if wb<6:bars[wb]=max(bars[wb],end+lag)
    if rb<6:bars[rb]=max(bars[rb],end)
    self.log.append((name,f'{o["chunk"]}:{o["pc_hex"]}',res,ready,start,end))
    self.at(start+max(1,ctrl['stall']),step)
   elif kind=='delay':self.at(self.t+z[1],step)
   elif kind in ['work','async']:
    _,res,cy,label,*done=z;parts=res.split('|');start=max([self.t]+[self.resources[p] for p in parts]);end=start+cy
    for p in parts:self.resources[p]=end
    self.log.append((name,label,res,self.t,start,end))
    if kind=='work':self.at(end,step)
    else:self.at(end,lambda:self.signal(done[0]));self.at(self.t+1,step)
   else:raise ValueError(z)
  self.at(0,step)
 def run(self):
  while self.q:self.t,_,f=heapq.heappop(self.q);f()
  if self.waiters:raise RuntimeError(('DEADLOCK',list(self.waiters)[:12]))
