from pathlib import Path
import csv,json,statistics,sys,math
from collections import defaultdict
R=Path(__file__).resolve().parent;B=R/(sys.argv[1] if len(sys.argv)>1 else 'dep_nop_formal')
def fit(pp,field):
 xx=[p['N'] for p in pp];yy=[p[field] for p in pp];xm,ym=statistics.mean(xx),statistics.mean(yy)
 b=sum((x-xm)*(y-ym) for x,y in zip(xx,yy))/sum((x-xm)**2 for x in xx);a=ym-b*xm
 return dict(slope=b,intercept=a,R2=1-sum((y-a-b*x)**2 for x,y in zip(xx,yy))/sum((y-ym)**2 for y in yy))
def quant(v,q):
 a=sorted(v);p=q*(len(a)-1);i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
by=defaultdict(list)
for p in sorted(B.glob('k*_t*_m-1/RAW.csv')):
 for r in csv.DictReader(p.open()):by[tuple(int(r[z]) for z in ['kind','warps','mode','trial','N'])].append(r)
points=[]
for (k,w,m,t,n),rr in by.items():
 p=dict(kind=k,warps=w,mode=m,trial=t,N=n,samples=len(rr),valid=len(rr)>=30)
 for f in ['clock_cycles','event_us']:
  v=[float(r[f]) for r in rr];med=statistics.median(v);cv=statistics.pstdev(v)/statistics.mean(v);spread=(quant(v,.95)-quant(v,.05))/med
  p[f]=med;p[f+'_CV']=cv;p[f+'_spread']=spread;p['valid']=p['valid'] and cv<=.05 and spread<=.10
 points.append(p)
summary=[];fits=[]
for k,w,m in sorted({(p['kind'],p['warps'],p['mode']) for p in points}):
 s=dict(kind=k,warps=w,mode=m,model_enabled=False,strict_gate=True)
 for f in ['clock_cycles','event_us']:
  tail=[];ok=True
  for t in range(3):
   pp=sorted([p for p in points if (p['kind'],p['warps'],p['mode'],p['trial'])==(k,w,m,t)],key=lambda p:p['N'])
   if len(pp)<4:ok=False;continue
   a,b=[fit(pp[j:j+3],f) for j in [len(pp)-4,len(pp)-3]]
   change=abs(b['slope']/a['slope']-1)
   ok=ok and all(p['valid'] for p in pp[-4:]) and a['R2']>=.99 and b['R2']>=.99 and change<.01
   tail.append(b['slope']);fits.append(dict(kind=k,warps=w,mode=m,trial=t,field=f,first=a,last=b,adjacent_change=change))
  cross=(max(tail)-min(tail))/statistics.mean(tail) if len(tail)==3 else None
  ok=ok and cross is not None and cross<.01
  s[f+'_gate']=ok;s[f+'_tail_slopes']=tail;s[f+'_cross_change']=cross;s['strict_gate']=s['strict_gate'] and ok
 summary.append(s)
bounds=[]
for k in [0,1]:
 rr={s['mode']:s for s in summary if s['kind']==k and s['warps']==1}
 if len(rr)!=2:continue
 c0=[x/16 for x in rr[0]['clock_cycles_tail_slopes']];c1=[x/16 for x in rr[1]['clock_cycles_tail_slopes']]
 # P_load=L+C, P_nop=1+C in the strict one-RAW-chain model; keep
 # interval [P_load-P_nop+1,P_load] instead of declaring a bare latency.
 low=min(a-b+1 for a,b in zip(c1,c0));high=max(c1)
 bounds.append(dict(kind=k,subtype='32x32b.x32' if k==0 else '16x256b.x8',P_load=c1,P_nop=c0,
    conditional_completion_lower=low,conditional_completion_upper=high,
    formula='L in [P_load-P_nop+1,P_load], exact nonload native control; lower bound assigns all remaining no-load cost to independent chain overhead.',
    strict_gate=all(s['strict_gate'] for s in rr.values()),model_enabled=False))
out=dict(points=points,summary=summary,fits=fits,bounds=bounds,model_enabled=False)
(B/'DEP_NOP_SLOPE_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(summary=[dict(kind=s['kind'],w=s['warps'],m=s['mode'],gate=s['strict_gate'],cycle_per_read=[x/16 for x in s['clock_cycles_tail_slopes']]) for s in summary],bounds=bounds),ensure_ascii=False))
