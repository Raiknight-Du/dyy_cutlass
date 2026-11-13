/** Offline structural preview of the MCP-authored Draw.io graphs. No upload. */
import fs from 'node:fs';import path from 'node:path';import {chromium} from 'playwright';
const root=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const xml=fs.readFileSync(path.join(root,process.argv[2]??'OP12_SOURCE_CYCLE_DAGS_20260927.drawio'),'utf8');
const dir=path.join(root,process.argv[3]??'figures');fs.mkdirSync(dir,{recursive:true});
const attr=s=>Object.fromEntries([...s.matchAll(/([\w]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
const decode=s=>s.replaceAll('&quot;','"').replaceAll('&gt;','>').replaceAll('&lt;','<').replaceAll('&amp;','&');
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const pages=[...xml.matchAll(/<diagram\b([^>]*)>([\s\S]*?)<\/diagram>/g)];
const browser=await chromium.launch({headless:true,executablePath:'C:/Users/27194/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe'});
const view=await browser.newPage({viewport:{width:1320,height:800}});
try{for(let idx=0;idx<pages.length;idx++){
  const content=pages[idx][2];const cells=[...content.matchAll(/<mxCell\b([^>]*?)(?:\/\s*>|>([\s\S]*?)<\/mxCell>)/g)].map(m=>({a:attr(m[1]),g:attr((m[2]??'').match(/<mxGeometry\b([^>]*)/)?.[1]??'')}));
  const nodes=cells.filter(x=>x.a.vertex==='1').map(x=>({id:x.a.id,label:decode(x.a.value??''),style:x.a.style,x:+x.g.x,y:+x.g.y,w:+x.g.width,h:+x.g.height}));const ids=new Map(nodes.map(x=>[x.id,x]));
  const boxes=nodes.filter(n=>n.id!=='title'&&n.id!=='subtitle');const blocked=new Set();
  for(const n of boxes)for(let x=Math.floor((n.x-7)/10);x<=Math.ceil((n.x+n.w+7)/10);x++)for(let y=Math.floor((n.y-7)/10);y<=Math.ceil((n.y+n.h+7)/10);y++)blocked.add(`${x},${y}`);
  function route(a,b){
    const dx=b.x+b.w/2-(a.x+a.w/2),dy=b.y+b.h/2-(a.y+a.h/2);
    let pa,pb;if(Math.abs(dx)>=Math.abs(dy)){pa=[dx>0?a.x+a.w+20:a.x-20,a.y+a.h/2];pb=[dx>0?b.x-20:b.x+b.w+20,b.y+b.h/2]}else{pa=[a.x+a.w/2,dy>0?a.y+a.h+20:a.y-20];pb=[b.x+b.w/2,dy>0?b.y-20:b.y+b.h+20]}
    const start=pa.map(v=>Math.round(v/10)),goal=pb.map(v=>Math.round(v/10));const key=p=>p.join(',');const sk=key(start),gk=key(goal),open=[{p:start,g:0,f:0}],cost=new Map([[sk,0]]),prev=new Map();
    let reached=false;
    while(open.length){open.sort((x,y)=>x.f-y.f||x.g-y.g);const u=open.shift();const uk=key(u.p);if(uk===gk){reached=true;break;}if(u.g!==cost.get(uk))continue;
      for(const [ox,oy]of[[1,0],[-1,0],[0,1],[0,-1]]){const v=[u.p[0]+ox,u.p[1]+oy],vk=key(v);if(v[0]<2||v[0]>126||v[1]<7||v[1]>62)continue;if(blocked.has(vk)&&vk!==gk&&vk!==sk)continue;const ng=u.g+1;if(ng<(cost.get(vk)??Infinity)){cost.set(vk,ng);prev.set(vk,uk);open.push({p:v,g:ng,f:ng+Math.abs(v[0]-goal[0])+Math.abs(v[1]-goal[1])});}}
    }
    if(!reached)throw Error(`No obstacle-free route ${a.id}->${b.id}`);
    const points=[goal];let k=gk;while(k!==sk){k=prev.get(k);points.push(k.split(',').map(Number))}points.reverse();
    const simple=points.filter((p,i)=>i===0||i===points.length-1||((p[0]-points[i-1][0])!==(points[i+1][0]-p[0])||(p[1]-points[i-1][1])!==(points[i+1][1]-p[1])));
    const pointsPx=simple.map(p=>p.map(v=>v*10));
    const ba=Math.abs(dx)>=Math.abs(dy)?[dx>0?a.x+a.w:a.x,pointsPx[0][1]]:[pointsPx[0][0],dy>0?a.y+a.h:a.y];
    const bb=Math.abs(dx)>=Math.abs(dy)?[dx>0?b.x:b.x+b.w,pointsPx.at(-1)[1]]:[pointsPx.at(-1)[0],dy>0?b.y:b.y+b.h];
    return [ba,...pointsPx,bb];
  }
  const edges=cells.filter(c=>c.a.edge==='1').map(c=>{const a=ids.get(c.a.source),b=ids.get(c.a.target),style=c.a.style??'',stroke=style.match(/strokeColor=([^;]+)/)?.[1]??'#536477';return `<polyline points="${route(a,b).map(p=>p.join(',')).join(' ')}" fill="none" stroke="${stroke}" ${style.includes('dashed=1')?'stroke-dasharray="7 5"':''} stroke-width="2" marker-end="url(#arrow)"/>`}).join('');
  const verts=nodes.map(n=>{
    const fill=n.style.match(/fillColor=([^;]+)/)?.[1]??'none';const text=n.style.includes('text;');const size=n.id==='title'?24:15;
    const limit=Math.max(9,Math.floor((n.w-20)/size)),lines=[];let chars='';let units=0;for(const ch of n.label.match(/[A-Za-z0-9_]+|[^A-Za-z0-9_]/gu)??[]){const u=/^[\x00-\x7F]+$/.test(ch)?ch.length*.58:1;if(units+u>limit&&chars){lines.push(chars.trim());chars='';units=0}chars+=ch;units+=u;}if(chars)lines.push(chars.trim());
    const sy=n.y+(n.h-lines.length*(size+6))/2+size;
    return `${text?'':`<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="14" fill="${fill}" stroke="#748094" stroke-width="1.5"/>`}<text x="${text?n.x+5:n.x+n.w/2}" y="${sy}" text-anchor="${text?'start':'middle'}" font-size="${size}" fill="#17365D">${lines.map((s,i)=>`<tspan x="${text?n.x+5:n.x+n.w/2}" dy="${i?size+6:0}">${esc(s)}</tspan>`).join('')}</text>`;
  }).join('');
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="800" viewBox="0 0 1320 800"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#536477"/></marker></defs><rect width="1320" height="800" fill="white"/><g font-family="Microsoft YaHei,Arial,sans-serif">${edges}${verts}<text x="40" y="780" font-size="12" fill="#64748B">本地结构预览 · 与Draw.io源图同节点、同依赖；路由由本地预览器生成 · 未上传第三方</text></g></svg>`;
  const prefix=path.join(dir,`${String(idx+1).padStart(2,'0')}_dag`);fs.writeFileSync(prefix+'.svg',svg);await view.setContent(svg);await view.screenshot({path:prefix+'.png'});console.log(prefix+'.png');
}}finally{await browser.close()}
