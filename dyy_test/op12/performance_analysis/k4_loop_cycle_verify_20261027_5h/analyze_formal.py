"""Strict1% gate on native-cycle and independent Event slopes.
Probe packet timing remains validation, never a K4 theoretical node weight.
"""
from pathlib import Path
from collections import defaultdict
import csv,json,statistics,sys,math
R=Path(__file__).resolve().parent
root=R/(sys.argv[1] if len(sys.argv)>1 else 'formal_v2')
def pct(v,q):
 s=sorted(v);k=q*(len(s)-1);i=int(k);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(k-i)
def fit(xs,ys):
 xm,ym=statistics.mean(xs),statistics.mean(ys);b=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs);a=ym-b*xm
 return dict(slope=b,intercept=a,R2=1-sum((y-a-b*x)**2 for x,y in zip(xs,ys))/sum((y-ym)**2 for y in ys))
raw=[]
for p in sorted(root.glob('kind*_t*/EVENT_RAW.csv')):
 for r in csv.DictReader(p.open()):r.update(source=str(p));raw.append(r)
by=defaultdict(list)
for r in raw:by[int(r['kind']),int(r['mode']),int(r['trial']),int(r['N'])].append(r)
points=[]
for (kind,mode,tr,n),rr in by.items():
 z=dict(kind=kind,mode=mode,trial=tr,N=n,samples=len(rr))
 z['valid']=all(r['valid']=='1' for r in rr)
 for field in ['cta_cycle_median','event_us']:
  vals=[float(r[field]) for r in rr];med=statistics.median(vals);cv=statistics.pstdev(vals)/statistics.mean(vals);spread=(pct(vals,.95)-pct(vals,.05))/med
  z[field]=med;z[field+'_CV']=cv;z[field+'_spread']=spread
  z['valid']=z['valid'] and cv<=.05 and spread<=.10
 points.append(z)
allfits=[];summary=[]
for kind,mode in sorted(set((x['kind'],x['mode']) for x in points)):
 passed=True;rec=dict(kind=kind,mode=mode,rate_adopted=False,model_enabled=False)
 for field in ['cta_cycle_median','event_us']:
  tail=[];passe=True
  for tr in range(3):
   pp=sorted([x for x in points if (x['kind'],x['mode'],x['trial'])==(kind,mode,tr)],key=lambda x:x['N'])
   if len(pp)<4:passe=False;continue
   w=[]
   for j in [len(pp)-4,len(pp)-3]:
    q=pp[j:j+3];f=fit([x['N'] for x in q],[x[field] for x in q]);f.update(kind=kind,mode=mode,trial=tr,field=field,points=[x['N'] for x in q]);w.append(f);allfits.append(f)
    passe=passe and f['R2']>=.99 and f['slope']>0 and all(x['valid'] for x in q)
   change=abs(w[1]['slope']/w[0]['slope']-1);passe=passe and change<.01
   tail.append(w[-1]['slope']);rec[f'{field}_trial{tr}_adjacent_change']=change
  cross=max(tail)/min(tail)-1 if len(tail)==3 and min(tail)>0 else None
  passe=passe and cross is not None and cross<.01;passed=passed and passe
  rec[field+'_tail_slopes']=tail;rec[field+'_cross_change']=cross;rec[field+'_pass']=passe
 rec['strict_gate']=passed
 summary.append(rec)
out=dict(scope='Matched native family loop + control + final completion, not bare primitive rate or K4 phase',points=points,fits=allfits,summary=summary,model_enabled=False)
(root/'FORMAL_SLOPE_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if points:
 with (root/'POINT_MEDIANS.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(points[0]));w.writeheader();w.writerows(points)
print(json.dumps([dict(kind=x['kind'],mode=x['mode'],strict=x['strict_gate'],cycles=x['cta_cycle_median_tail_slopes'],event=x['event_us_tail_slopes'],crosscycle=x['cta_cycle_median_cross_change'],crossevent=x['event_us_cross_change']) for x in summary]))
