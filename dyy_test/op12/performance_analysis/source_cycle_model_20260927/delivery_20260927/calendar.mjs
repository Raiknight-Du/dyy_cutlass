/** Source DAG + explicitly assumed nonpreemptive resource calendar. No timing input. */
export function schedule(jobs,rates,{etas={},arbitration='priority',latencies={}}={}){
  const pending=jobs.map((x,i)=>({...x,index:i,dependency:x.dependency??x.predecessors??[]}));
  const byId=new Map(pending.map(x=>[x.id,x]));
  if(byId.size!==pending.length)throw Error('Duplicate node id');
  for(const x of pending)for(const p of x.dependency)if(!byId.has(p))throw Error(`Missing predecessor ${p} for ${x.id}`);
  const completed=new Map(),running=[],busy=new Map(),ordered=[];
  let time=0;
  while(completed.size<jobs.length){
    for(const [r,x] of busy)if(x.service_end<=time)busy.delete(r);
    for(let i=running.length-1;i>=0;i--)if(running[i].end<=time){const x=running.splice(i,1)[0];completed.set(x.id,x);}
    let added;
    do{
      added=false;
      const ready=pending.filter(x=>!x.started&&x.dependency.every(p=>completed.has(p))&&!busy.has(x.resource));
      ready.sort((a,b)=>(arbitration==='priority'?(a.priority??0)-(b.priority??0):0)||a.index-b.index);
      for(const x of ready){
        if(busy.has(x.resource))continue;
        const demand=x.demand??0,rate=(rates[x.resource]??1)*(etas[x.resource]??1);
        x.cycles=demand===0?0:Math.ceil(demand/rate);
        // Response-only tails do not consume the modeled engine service capacity.
        x.start=time;x.service_end=time+x.cycles;x.end=x.service_end+(latencies[x.id]??0);x.started=true;
        ordered.push(x);added=true;
        if(x.end===time)completed.set(x.id,x);else {running.push(x);if(x.cycles>0)busy.set(x.resource,x);}
      }
    }while(added);
    if(completed.size===jobs.length)break;
    if(!running.length)throw Error('Cyclic dependencies / no progress');
    const events=running.flatMap(x=>[x.service_end,x.end]).filter(t=>t>time);
    time=Math.min(...events);
  }
  const final=ordered.map(x=>({...x}));
  const finish=Math.max(0,...final.map(x=>x.end));
  const resource_work={};
  for(const x of final)resource_work[x.resource]=(resource_work[x.resource]??0)+x.cycles;
  validate(final);
  return {makespan_cycles:finish,resource_work,nodes:final};
}
export function validate(nodes){
  const ids=new Map(nodes.map(x=>[x.id,x]));
  for(const x of nodes)for(const p of x.dependency??[])if(ids.get(p).end>x.start)throw Error(`RAW violation ${p} -> ${x.id}`);
  const groups=new Map();for(const x of nodes){if(!groups.has(x.resource))groups.set(x.resource,[]);groups.get(x.resource).push(x);}
  for(const [r,a] of groups){a.sort((x,y)=>x.start-y.start||x.end-y.end);let last=0;for(const x of a)if(x.cycles>0){if(x.start<last)throw Error(`Overlap ${r}:${x.id}`);last=x.service_end??x.end;}}
  return true;
}
