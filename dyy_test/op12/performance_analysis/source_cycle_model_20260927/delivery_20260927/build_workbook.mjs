import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const root=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const model=JSON.parse(await fs.readFile(path.join(root,'approved_model.json'),'utf8'));
const sanity=JSON.parse(await fs.readFile(path.join(root,'event_sanity_check.json'),'utf8'));
const out=path.join(root,'outputs','source_model_20260927');await fs.mkdir(out,{recursive:true});
const wb=Workbook.create();const sheets=new Map();
const names=['Summary','Assumptions','Work and Traffic','Pipeline Summary','K1 Stages','K2 Stages','K3 Stages','K4 Stages',...model.calendars.map(c=>c.sheet),'Sensitivity','Event Sanity','Review'];
for(const name of names){const s=wb.worksheets.add(name);s.showGridLines=false;s.tabColor='#17365D';s.freezePanes.freezeRows(5);sheets.set(name,s);}
const fmt={font:{name:'Microsoft YaHei',size:10,color:'#1F2937'},wrapText:true,verticalAlignment:'center'};
function base(s,n,last='L'){
  s.getRange(`A1:${last}${n}`).format=fmt;s.getRange(`A1:${last}${n}`).format.rowHeight=28;
  s.getRange(`A1:${last}${n}`).format.columnWidth=16;
  s.getRange(`A1:${last}1`).merge();s.getRange('A1').values=[[s.name==='Summary'?'OP12 · 四kernel源驱动依赖感知cycle模型':s.name]];
  s.getRange(`A1:${last}1`).format={fill:'#17365D',font:{name:'Microsoft YaHei',bold:true,size:16,color:'#FFFFFF'},rowHeight:40};
  s.getRange(`A2:${last}3`).merge();s.getRange('A2').values=[['条件化理想资源服务模型；不是实测拟合，也不是唯一物理时延。零响应尾部≠硬件延迟为零。固定commit e77a4a0d…；FP16/BF16输入，FP32 state。']];s.getRange(`A2:${last}3`).format={fill:'#FFF2CC',wrapText:true,rowHeight:30};
}
function header(s,row,labels){s.getRangeByIndexes(row-1,0,1,labels.length).values=[labels];s.getRangeByIndexes(row-1,0,1,labels.length).format={fill:'#1F4E78',font:{bold:true,color:'#FFFFFF',name:'Microsoft YaHei',size:10},wrapText:true,rowHeight:36};}
function val(s,cell,value){s.getRange(cell).values=[[value]];}function formula(s,cell,value){s.getRange(cell).formulas=[[value]];}
const a=sheets.get('Assumptions');base(a,50,'F');header(a,5,['变量','值','单位 / 范围','分类','证据 / 推导','解释']);
const assumptions=[
 ['SM',148,'SM/GPU','SOURCE FACT','static_artifacts/manifest.json','设备启用SM数'],
 ['f_ref',1965000000,'Hz','SOURCE FACT / device metadata','硬件快照SM1965MHz','只用于参考周期换算，不视为每轮锁频'],
 ['dense FP16/BF16 peak',2250000000000000,'FLOP/s/GPU','OFFICIAL HARDWARE FACT','NVIDIA Blackwell datasheet p8 HGX B200：4.5P sparse / 2','公布峰值，不保证小tile实际吞吐'],
 ['dense TF32 peak',1100000000000000,'FLOP/s/GPU','OFFICIAL HARDWARE FACT','NVIDIA Blackwell datasheet p8：2.2P sparse / 2','公布值经四舍五入'],
 ['HBM peak',7700000000000,'B/s/GPU','OFFICIAL HARDWARE FACT','NVIDIA Blackwell datasheet p8：7.7 TB/s','GMEM请求不保证均命中HBM'],
 ['SMEM service',128,'B/(SM cycle)','MODELING ASSUMPTION','统一理想服务场景','全SM共享；不按计算组复制'],
 ['TMEM explicit-copy service',512,'B/(SM cycle)','MODELING ASSUMPTION','统一理想服务场景','显式ld/st共享；不重复计MMA内部流量'],
 ['Conversion service',64,'scalar elements/(SM cycle)','MODELING ASSUMPTION','统一理想服务场景','未验证CVT issue/依赖与eligible warp'],
 ['SIMT service',128,'scalar results/(SM cycle)','MODELING ASSUMPTION','统一理想服务场景','一FMA是一个结果、两FLOP；SFU/shuffle另留符号'],
 ['rho_HMMA',1,'比例','MODELING ASSUMPTION','理想HMMA等效全SM peak','不是官方HMMA准确吞吐'],
 ['eta64 FP16',1,'比例','MODELING ASSUMPTION','乐观M64 group1效率；另测0.5敏感性','PTX half-datapath不能直接证明精确0.5'],
 ['eta64 TF32',1,'比例','MODELING ASSUMPTION','乐观M64 group1效率；另测0.5敏感性','与FP16效率单独控制'],
 ['HBM allocation divisor',148,'份/GPU','MODELING ASSUMPTION','每活跃SM保留1/148全卡带宽','不是欠并行时HBM必须按SM覆盖率下降的事实'],
 ['Unpriced response tail',0,'cycle；理想数值场景','MODELING ASSUMPTION','所有未知硬件响应/issue/barrier保留符号','此数值用于服务模型，不声称未知成本已建模'],
 ['D',128,'elements','SOURCE FACT','固定工作负载 / 源码self.D','head size'],
 ['algorithm chunk',64,'tokens','SOURCE FACT','源码BLK / chunk_size','不是CP partition长度'],
 ['heads',16,'heads','SOURCE FACT','指定工作负载','B=1，P=8'],
 ['partitions per head',8,'partitions','SOURCE FACT','S/C：三组均8','K3恒定工作量'],
 ['R16',null,'FLOP/(SM cycle)','MODELING ASSUMPTION','GPU peak / (148*f_ref)','频率等效归一化，不是公开每周期指令常数'],
 ['RTF32',null,'FLOP/(SM cycle)','MODELING ASSUMPTION','GPU peak / (148*f_ref)','同上'],
 ['B_GPU',null,'B/(GPU cycle)','MODELING ASSUMPTION','HBM peak / f_ref','全卡共同容量'],
 ['B_alloc',null,'B/(active SM cycle)','MODELING ASSUMPTION','B_GPU / allocation divisor','CTA共享所在SM该份额'],
 ['R_HMMA',null,'FLOP/(SM cycle)','MODELING ASSUMPTION','R16*rho_HMMA','rho敏感性'],
 ['U',null,'FLOP','SOURCE FACT','2*D*chunk*chunk','单128×64×64 GEMM'],
 ['FP16/BF16 byte size',2,'B/element','SOURCE FACT','dtype contract','Q/K/V/O/T'],
 ['FP32 byte size',4,'B/element','SOURCE FACT','dtype contract','M/N/fixed/initial/final'],
 ['resident K1 CTA ceiling',8,'CTA/SM','SOURCE FACT / untimed driver query','cuOccupancyMaxActiveBlocksPerMultiprocessor，dynamicSMEM24960B','容量8已确认；实际调度采用同步cohort假设'],
 ['resident K2/K3/K4 ceiling',1,'CTA/SM','SOURCE FACT + resource ceiling','K2/K4 REG168*384；K3 SMEM>160KiB','实际驻留还受TMEM/launch限制']
];
a.getRange(`A6:F${5+assumptions.length}`).values=assumptions;
for(const [row,f] of [[24,'=B8/(B6*B7)'],[25,'=B9/(B6*B7)'],[26,'=B10/B7'],[27,'=B26/B18'],[28,'=B24*B15'],[29,'=2*B20*B21*B21']])formula(a,`B${row}`,f);
a.getRange('B6:B33').format.numberFormat='#,##0.000';a.getRange('B6:B23').format.font.color='#0000FF';a.getRange('B30:B34').format.font.color='#0000FF';a.getRange('D6:D33').format.font.color='#7C2D12';a.getRange('E6:F33').format.columnWidth=42;a.getRange('A6:A33').format.columnWidth=28;a.getRange('A6:F33').format.rowHeight=52;
a.getRange('A34:F34').values=[['K1 shuffle dispatch proxy',4,'warp shuffle instructions/(SM cycle)','MODELING ASSUMPTION','仅K1 shuffle服务；不是完整frontend建模','其他warp issue、predicated-lane粒度和RAW响应留符号']];
a.getRange('A35:F35').values=[['K1 inverse HMMA',null,'FLOP/block','SOURCE FACT / derived','8→16 +16→32 +32→64','实际广播重复计算计入，不是唯一数学元素数量']];formula(a,'B35','=4*2*(2*16*8*8)+2*2*(2*16*16*16)+2*(2*32*32*32)');
val(a,'A37','官方来源');a.getRange('A38:F39').merge();val(a,'A38','https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf');a.getRange('A40:F41').merge();val(a,'A40','https://docs.nvidia.com/cuda/archive/12.9.1/blackwell-tuning-guide/index.html#occupancy');a.getRange('A42:F43').merge();val(a,'A42','https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions');
const refs={HBM:'Assumptions!$B$27',SMEM:'Assumptions!$B$11',TMEM:'Assumptions!$B$12',CVT:'Assumptions!$B$13',SIMT:'Assumptions!$B$14',TC16:'Assumptions!$B$24',TF32:'Assumptions!$B$25',HMMA:'Assumptions!$B$28',ISSUE:'Assumptions!$B$34',control:'1'};
const timelineRefs={};
for(const c of model.calendars){
  const s=sheets.get(c.sheet);base(s,c.nodes.length+10,'L');header(s,5,['源节点 / substage','resource','原始量','单位','rate','service cy','pred / reuse','前资源job','start cy','end cy','源位置','重叠 / 假设']);
  s.getRange('A4:F4').values=[['HBM divisor',c.HBM_divisor??148,'TF32 eta倍率',c.eta_TF32??1,'M64 FP16 eta倍率',c.eta_FP16??1]];
  for(const cell of ['B4','D4','F4'])s.getRange(cell).format.font.color='#0000FF';
  const rowBy=new Map(c.nodes.map((n,i)=>[n.id,i+6]));const lastBy={};const rows=[];const forms=[];
  for(let i=0;i<c.nodes.length;i++){
    const n=c.nodes[i],r=i+6,preds=n.dependency??[];
    const prev=lastBy[n.resource];if(n.demand>0)lastBy[n.resource]=r;
    rows.push([n.id,n.resource,n.demand,n.unit??'',null,null,preds.join(';'),prev??0,null,null,n.source_location??'',n.overlap_rule??'源码依赖+固定resource仲裁；未知尾部单列']);
    const rf=n.resource==='HBM'?'Assumptions!$B$26/$B$4':n.resource==='TC16'&&n.eta64?`${refs.TC16}*Assumptions!$B$16*$F$4`:n.resource==='TF32'?`${refs.TF32}*Assumptions!$B$17*$D$4`:refs[n.resource]??'1';
    const dep=preds.map(p=>{if(!rowBy.has(p))throw Error('Missing pred');const pr=rowBy.get(p);if(pr>=r)throw Error('Calendar not topo-sorted');return `J${pr}`});
    if(prev)dep.push(`J${prev}`);
    forms.push([`=${rf}`,`=IF(C${r}=0,0,ROUNDUP(C${r}/E${r},0))`,null,null,`=MAX(0${dep.length?','+dep.join(','):''})`,`=I${r}+F${r}`]);
  }
  s.getRange(`A6:L${5+rows.length}`).values=rows;s.getRange(`E6:J${5+rows.length}`).formulas=forms;
  const end=5+c.nodes.length;s.getRange(`A${end+3}:D${end+3}`).merge();val(s,`A${end+3}`,'条件调度makespan（cycle）');formula(s,`F${end+3}`,`=MAX(J6:J${end})`);
  timelineRefs[c.key]=`'${c.sheet}'!F${end+3}`;
  s.getRange(`C6:F${end}`).format.numberFormat='#,##0.00';s.getRange(`I6:J${end}`).format.numberFormat='#,##0';
  s.getRange(`C6:C${end}`).format.font.color='#0000FF';
  for(const [col,width]of[['A',34],['G',46],['K',45],['L',58]])s.getRange(`${col}1:${col}${end+3}`).format.columnWidth=width;
  s.getRange(`A6:L${end}`).format.rowHeight=42;
}
// Pipeline terms link to exact source-calendar node finishes, never Event cells.
const pipeline=sheets.get('Pipeline Summary');base(pipeline,22,'L');header(pipeline,5,['kernel','S','长度/单位','首段 / 尾cohort','稳态unit cy','重复次数','边界项 cy','drain cy','重建cycle','完整日历cycle','差值cycle','公式含义']);
const nodeFinish=(key,id)=>{const c=model.calendars.find(c=>c.key===key),idx=c.nodes.findIndex(n=>n.id===id);if(idx<0)throw Error('Missing pipeline node '+key+'/'+id);return `'${c.sheet}'!J${idx+6}`;};
for(let i=0;i<3;i++)for(let k=0;k<4;k++){
 const r=6+4*i+k,L=[16,32,64][i],q=6+i,S=[8192,16384,32768][i],kernel='K'+(k+1);val(pipeline,`A${r}`,kernel);formula(pipeline,`B${r}`,`=Summary!A${q}`);
 if(k===0){const ct=Math.ceil(16*S/64/148),tail=ct%8;val(pipeline,`C${r}`,'8CTA共享cohort');formula(pipeline,`D${r}`,tail?`=${timelineRefs['K1_batch_'+tail]}`:'=0');formula(pipeline,`E${r}`,`=${timelineRefs.K1_batch_8}`);formula(pipeline,`F${r}`,`=ROUNDDOWN(Summary!J${q}/Assumptions!$B$32,0)`);formula(pipeline,`G${r}`,'=0');formula(pipeline,`H${r}`,'=0');val(pipeline,`L${r}`,'尾cohort + 完整cohort次数×共享服务；全卡HBM floor单列在Summary');}
 if(k===1){const key='K2_'+L,p=id=>nodeFinish(key,id);val(pipeline,`C${r}`,'64token block');formula(pipeline,`D${r}`,`=${p('n_update_0_done')}`);formula(pipeline,`E${r}`,`=${p('n_update_2_done')}-${p('n_update_1_done')}`);formula(pipeline,`F${r}`,`=Summary!C${q}-1`);formula(pipeline,`G${r}`,`=${p('n_update_1_done')}-${p('n_update_0_done')}-E${r}+${p('n_update_'+(L-1)+'_done')}-${p('n_update_'+(L-3)+'_done')}-2*E${r}`);formula(pipeline,`H${r}`,`=${timelineRefs[key]}-${p('n_update_'+(L-1)+'_done')}`);val(pipeline,`L${r}`,'W + (L−1)I + 首后块/末两块边界修正 + M/N output drain；非实测回归');}
 if(k===2){const key='K3_8',p=id=>nodeFinish(key,id);val(pipeline,`C${r}`,'partition P=8');formula(pipeline,`D${r}`,`=${p('workspace.hbm.3')}`);formula(pipeline,`E${r}`,`=${p('store.end.1')}-${p('store.end.0')}`);formula(pipeline,`F${r}`,'=Assumptions!B23-1');formula(pipeline,`G${r}`,`=${p('store.end.7')}-${p('store.end.6')}`);formula(pipeline,`H${r}`,`=${timelineRefs[key]}-${p('store.end.7')}`);val(pipeline,`L${r}`,'initial workspace + p0…p6七个间隔 + p7 tail间隔 + 重复fixed/final drain');}
 if(k===3){const key='K4_'+L,p=b=>nodeFinish(key,'o_store_done_'+b);val(pipeline,`C${r}`,'2-block chunk pair');formula(pipeline,`D${r}`,`=${p(1)}`);formula(pipeline,`E${r}`,`=${p(5)}-${p(3)}`);formula(pipeline,`F${r}`,`=Summary!C${q}/2-1`);formula(pipeline,`G${r}`,`=${p(3)}-${p(1)}-E${r}+${p(L-1)}-${p(L-5)}-2*E${r}`);formula(pipeline,`H${r}`,`=${timelineRefs[key]}-${p(L-1)}`);val(pipeline,`L${r}`,'first pair + later pairs的steady服务 + 第二/末两pair边界修正；不是独立state块');}
 formula(pipeline,`I${r}`,`=D${r}+E${r}*F${r}+G${r}+H${r}`);formula(pipeline,`J${r}`,`=Summary!${['D','E','F','G'][k]}${q}`);formula(pipeline,`K${r}`,`=I${r}-J${r}`);
}
pipeline.getRange('D6:K17').format.numberFormat='#,##0';pipeline.getRange('A6:L17').format.rowHeight=60;pipeline.getRange('L1:L22').format.columnWidth=56;
pipeline.getRange('A19:L20').merge();val(pipeline,'A19','各项是源图调度时戳的公式差值，不是实测拆分。K1按最忙SM同步cohort；K2按block；K3按partition；K4按chunk pair。固定三个L及本仲裁下成立，不外推短L/尾块。');
for(const k of ['K1','K2','K3','K4']){
  const s=sheets.get(k+' Stages'),stages=model.stage_tables[k];base(s,stages.length+10,'L');header(s,5,['串行stage','substage / 操作','shape','data / work量','资源','rate','服务cycle','前驱 / RAW','可并行部分','overlap规则','源码位置','证据 / 硬件假设']);
  s.getRange(`A6:L${5+stages.length}`).values=stages.map(n=>[n.stage,n.operation,n.shape,n.demand,n.resource,null,null,n.dependency,n.parallel,n.overlap,n.source,n.confidence]);
  for(let i=0;i<stages.length;i++){const n=stages[i],r=i+6;formula(s,`F${r}`,`=${refs[n.resource]??'1'}`);formula(s,`G${r}`,`=IF(D${r}=0,0,ROUNDUP(D${r}/F${r},0))`);}
  s.getRange(`A6:L${5+stages.length}`).format.rowHeight=110;s.getRange(`D6:G${5+stages.length}`).format.numberFormat='#,##0.00';s.getRange(`G6:G${5+stages.length}`).format.numberFormat='#,##0';
  s.getRange(`D6:D${5+stages.length}`).format.font.color='#0000FF';
  for(const col of ['B','H','I','J','K','L'])s.getRange(`${col}1:${col}${5+stages.length}`).format.columnWidth=32;
}
const work=sheets.get('Work and Traffic');base(work,25,'L');header(work,5,['S','C','L=C/64','P','K1 grid','K2 grid','K3 grid','K4 grid','K1 MMA FLOP','K2 MMA FLOP','K3 TF32 FLOP','K4 MMA FLOP']);
for(let i=0;i<3;i++){
 const r=i+6,S=[8192,16384,32768][i],C=[1024,2048,4096][i];val(work,`A${r}`,S);val(work,`B${r}`,C);
 const fs=[`=B${r}/Assumptions!$B$21`,`=A${r}/B${r}`,`=Assumptions!$B$22*A${r}/Assumptions!$B$21`,`=Assumptions!$B$22*D${r}`,`=Assumptions!$B$22*2`,`=F${r}`,`=E${r}*(Assumptions!$B$29+Assumptions!$B$35)`,`=F${r}*(9*C${r}-2)*Assumptions!$B$29`,`=G${r}*D${r}*2*64*Assumptions!$B$20*Assumptions!$B$20`,`=H${r}*C${r}*9*Assumptions!$B$29`];work.getRange(`C${r}:L${r}`).formulas=[fs];
}
header(work,11,['S','K1请求 B','K2请求 B','K3请求 B','K4请求 B','合计请求 B','说明']);
for(let i=0;i<3;i++){const r=12+i,q=6+i;work.getRange(`A${r}:F${r}`).formulas=[[`=A${q}`,`=E${q}*B18`,`=F${q}*(C${q}*B19+B20)`,`=G${q}*B21`,`=H${q}*(B22+C${q}*B23)`,`=SUM(B${r}:E${r})`]];val(work,`G${r}`,'逻辑bulk请求，不是NCU DRAM bytes；K1 beta重复lane读取不当HBM重复字节。');}
header(work,17,['payload 项','B/CTA或块','组成 / 输入输出']);
for(const [r,label,f,note]of[
 [18,'K1每CTA','=Assumptions!B21*Assumptions!B20*Assumptions!B30+Assumptions!B21*Assumptions!B31+Assumptions!B21^2*Assumptions!B30','K读16KiB + beta读256B + T写8KiB；metadata8B另计'],
 [19,'K2每块','=2*Assumptions!B21*Assumptions!B20*Assumptions!B30+Assumptions!B21^2*Assumptions!B30+Assumptions!B21*Assumptions!B31','K/V各16KiB、T8KiB、alpha256B'],
 [20,'K2结束写','=2*Assumptions!B20^2*Assumptions!B31','M/N各64KiB FP32'],
 [21,'K3 P8每CTA','=2*64*Assumptions!B20*Assumptions!B31+Assumptions!B23*(2*64*Assumptions!B20*Assumptions!B31+Assumptions!B20^2*Assumptions!B31)+2*64*Assumptions!B20*Assumptions!B31','initial读/写；每p M/N读+fixed写；最后重复fixed+final写'],
 [22,'K4初始读','=Assumptions!B20^2*Assumptions!B31','initial或fixed state 64KiB FP32'],
 [23,'K4每块','=4*Assumptions!B21*Assumptions!B20*Assumptions!B30+Assumptions!B21^2*Assumptions!B30+Assumptions!B21*Assumptions!B31','Q/K/V读+O写各16KiB；T8KiB；alpha256B，无final写']]){val(work,`A${r}`,label);formula(work,`B${r}`,f);work.getRange(`C${r}:L${r}`).merge();val(work,`C${r}`,note);work.getRange(`A${r}:L${r}`).format.rowHeight=46;}
work.getRange('A6:L14').format.numberFormat='#,##0';work.getRange('G12:L14').merge(true);work.getRange('A12:L14').format.rowHeight=54;
const summary=sheets.get('Summary');base(summary,39,'L');header(summary,5,['S','C','L','K1 cycle','K2 cycle','K3 cycle','K4 cycle','四阶段合计','μs @ f_ref','K1最忙SM CTA','K1驻留batch','模型性质']);
for(let i=0;i<3;i++){
  const r=6+i,q=6+i,S=[8192,16384,32768][i],L=[16,32,64][i];
  summary.getRange(`A${r}:C${r}`).formulas=[[`='Work and Traffic'!A${q}`,`='Work and Traffic'!B${q}`,`='Work and Traffic'!C${q}`]];
  for(const[k,col]of[['K2','E'],['K3','F'],['K4','G']])formula(summary,`${col}${r}`,`=${timelineRefs[k+'_'+L]??timelineRefs[k+'_8']}`);
  formula(summary,`J${r}`,`=ROUNDUP('Work and Traffic'!E${q}/Assumptions!$B$6,0)`);formula(summary,`K${r}`,`=ROUNDUP(J${r}/Assumptions!$B$32,0)`);
  formula(summary,`D${r}`,model.k1_excel_formulas[i]);formula(summary,`H${r}`,`=SUM(D${r}:G${r})`);formula(summary,`I${r}`,`=H${r}/Assumptions!$B$7*1000000`);val(summary,`L${r}`,'条件化服务调度；未知响应/launch另加');
}
summary.getRange('D6:H8').format.numberFormat='#,##0';summary.getRange('I6:I8').format.numberFormat='0.000';summary.getRange('A6:L8').format.rowHeight=55;
header(summary,11,['核对 / 限制','内容']);
const notes=[['核心结论','三组shape：K1 grid变大；K2/K4单CTA块数16→32→64；K3 P恒8所以cycle恒定。'],['周期定义','service=ceil(work/rate)。serial相加；不同资源且无依赖时并行；join=max(predecessor end,resource free)。'],['GPU调度','K2/K4 grid128、K3 grid32，一CTA/SM场景一波；K1最多8 CTA/SM，但共享吞吐不能复制8份。'],['HBM口径','1/148带宽配额为模型假设；grid32并不证明HBM可用带宽仅32/148。缓存与TMA响应均未定。'],['不确定性','每个未公开的response、issue、barrier、SFU、cache项保留符号。数值0仅理想场景，不能称已解释实测差值。'],['时间换算','μs=cycle/f_Hz×10^6。历史cycle若频率不同不能直接比较；必须先还原Event μs。'],['可编辑性','蓝色输入来自源码/官方/假设，其他派生格由公式计算。timeline改变rate后保持固定仲裁顺序，仍为合法场景但不是重新优化的调度。'],['证据文件','完整DAG JSON、静态PTX/CUBIN/SASS、Draw.io源图、综合报告位于同级目录。'],['版本边界','不修改上游固定源码；不纳入FP8实验；未用任何目标kernel时延拟合常数。']];
for(let i=0;i<notes.length;i++){const r=12+i;val(summary,`A${r}`,notes[i][0]);summary.getRange(`B${r}:L${r}`).merge();val(summary,`B${r}`,notes[i][1]);summary.getRange(`A${r}:L${r}`).format.rowHeight=46;}
header(summary,23,['S','K2资源忙时下界','K2依赖倍率','K3资源忙时下界','K3依赖倍率','K4资源忙时下界','K4依赖倍率','同一模型口径的解释']);
for(let i=0;i<3;i++){const r=24+i,L=[16,32,64][i],q=6+i;formula(summary,`A${r}`,`=A${q}`);for(const[k,fcol,pcol,ccol]of[['K2','B','C','E'],['K3','D','E','F'],['K4','F','G','G']]){const cal=model.calendars.find(c=>c.key===(k==='K3'?'K3_8':k+'_'+L));const last=5+cal.nodes.length;const sums=['HBM','SMEM','TMEM','CVT','SIMT','TC16','TF32'].map(res=>`SUMIF('${cal.sheet}'!B6:B${last},"${res}",'${cal.sheet}'!F6:F${last})`);formula(summary,`${fcol}${r}`,`=MAX(${sums.join(',')})`);formula(summary,`${pcol}${r}`,`=${ccol}${q}/${fcol}${r}`);}summary.getRange(`H${r}:L${r}`).merge();val(summary,`H${r}`,'分包服务与固定rate场景内，最忙资源busy sum是容量下界；调度值/容量值展示依赖与复用增加的周期，不是物理依赖罚时实测。');summary.getRange(`A${r}:L${r}`).format.rowHeight=66;}
summary.getRange('B24:G26').format.numberFormat='#,##0.000';
const sens=sheets.get('Sensitivity');base(sens,model.sensitivity.length+9,'H');header(sens,5,['kernel','L/P','变量','基准 /148 cy','替代 cy','差值 cy','性质','解释']);
sens.getRange(`A6:H${5+model.sensitivity.length}`).values=model.sensitivity.map(x=>[x.kernel,x.L,x.variable,null,null,null,'固定仲裁的公式场景；不是置信区间',x.note]);
for(let i=0;i<model.sensitivity.length;i++){const x=model.sensitivity[i],r=i+6;formula(sens,`D${r}`,`=${timelineRefs[x.base_key]}`);formula(sens,`E${r}`,`=${timelineRefs[x.alternative_key]}`);formula(sens,`F${r}`,`=E${r}-D${r}`);}
sens.getRange(`A6:H${5+model.sensitivity.length}`).format.rowHeight=65;sens.getRange(`C1:C${model.sensitivity.length+5}`).format.columnWidth=34;sens.getRange(`G1:H${model.sensitivity.length+5}`).format.columnWidth=48;
const eventSheet=sheets.get('Event Sanity');base(eventSheet,28,'L');header(eventSheet,5,['S','kernel','源模型 cycle','模型 μs','Event μs','Event参考cycle','残差 μs','残差参考cycle','Event/模型','trial范围 μs','最高trial CV%','限制']);
for(let i=0;i<sanity.cases.length;i++)for(let j=0;j<4;j++){
 const c=sanity.cases[i],stage=c.stages[j],r=6+i*4+j,summaryCol=['D','E','F','G'][j];
 val(eventSheet,`A${r}`,c.S);val(eventSheet,`B${r}`,stage.kernel);formula(eventSheet,`C${r}`,`=Summary!${summaryCol}${6+i}`);formula(eventSheet,`D${r}`,`=C${r}/Assumptions!$B$7*1000000`);
 val(eventSheet,`E${r}`,stage.event_median_of_three_trial_medians_us);formula(eventSheet,`F${r}`,`=E${r}*Assumptions!$B$7/1000000`);formula(eventSheet,`G${r}`,`=E${r}-D${r}`);formula(eventSheet,`H${r}`,`=F${r}-C${r}`);formula(eventSheet,`I${r}`,`=E${r}/D${r}`);
 val(eventSheet,`J${r}`,`${Math.min(...stage.trial_medians_us).toFixed(3)}–${Math.max(...stage.trial_medians_us).toFixed(3)}`);val(eventSheet,`K${r}`,Math.max(...stage.trial_cv_percent));val(eventSheet,`L${r}`,'仅事后检查；重复输入/缓存/调用包装；高CV不能作为紧误差验证');
}
eventSheet.getRange('C6:H17').format.numberFormat='#,##0.000';eventSheet.getRange('C6:C17').format.numberFormat='#,##0';eventSheet.getRange('I6:K17').format.numberFormat='0.000';eventSheet.getRange('E6:E17').format.font.color='#0000FF';eventSheet.getRange('J6:K17').format.font.color='#0000FF';eventSheet.getRange('L1:L28').format.columnWidth=38;eventSheet.getRange('A6:L17').format.rowHeight=60;
eventSheet.getRange('A19:L20').merge();val(eventSheet,'A19','模型冻结后才读取Event；所有理论sheet不引用本页。固定FP16、B1/H16/D128、C1024/2048/4096，initial/final开启。stage独立wrapper：2 warmup/20 events；整条CP：20 warmup/100 events，3轮。Event按3轮median的median。');eventSheet.getRange('A19:L20').format.rowHeight=40;
eventSheet.getRange('A22:L23').merge();val(eventSheet,'A22','来源：../evidence_gap_ablation_20260920/cp_stage_same_round/trial1.json、trial2.json、trial3.json。SHA与冻结时间见event_sanity_check.json。独立阶段时延之和不替代真实端到端Event；不使用NCU时延。');eventSheet.getRange('A22:L23').format.rowHeight=40;
const review=sheets.get('Review');base(review,model.review.length+10,'E');header(review,5,['级别','问题','修正 / 处理','证据','状态']);review.getRange(`A6:E${model.review.length+5}`).values=model.review.map(x=>[x.severity,x.issue,x.correction,x.evidence,x.status]);review.getRange(`A6:E${model.review.length+5}`).format.rowHeight=80;review.getRange('B1:D30').format.columnWidth=50;
wb.recalculate();
const inspection=await wb.inspect({kind:'region',sheetId:'Summary',range:'A5:L8',maxChars:5000,tableMaxCols:12});console.log(inspection.ndjson);
await fs.writeFile(path.join(out,'inspection_summary.txt'),inspection.ndjson);
const errors=[];for(const [name,s]of sheets){const v=s.getUsedRange().values;for(let r=0;r<v.length;r++)for(let c=0;c<v[r].length;c++)if(typeof v[r][c]==='string'&&/^#(REF!|DIV\/0!|VALUE!|NAME\?|NUM!|N\/A|SPILL!)/.test(v[r][c]))errors.push({name,row:r+1,col:c+1,value:v[r][c]});}
if(errors.length)throw Error(JSON.stringify(errors));
if(pipeline.getRange('K6:K17').values.some(row=>row[0]!==0))throw Error('Source pipeline reconstruction mismatch');
const formulaChecks=model.calendars.map(c=>({key:c.key,expected:c.expected_cycles,actual:sheets.get(c.sheet).getRange(timelineRefs[c.key].split('!')[1]).values[0][0]}));
if(formulaChecks.some(x=>x.actual!==x.expected))throw Error('Calendar formula mismatch '+JSON.stringify(formulaChecks.filter(x=>x.actual!==x.expected)));
// Input-change test: verify derived tensor rate updates; restore before export.
const before=a.getRange('B24').values[0][0];val(a,'B8',2300000000000000);wb.recalculate();const changed=a.getRange('B24').values[0][0];if(before===changed)throw Error('Formula sensitivity failed');val(a,'B8',2250000000000000);wb.recalculate();
await fs.writeFile(path.join(out,'validation.json'),JSON.stringify({formula_errors:errors,calendar_formula_checks:formulaChecks,derived_rate_input_change_passed:true,primary_model:model.id,timing_fit:false},null,2));
for(const name of names){const s=sheets.get(name);const last=name==='Assumptions'?'F':name==='Review'?'E':name==='Sensitivity'?'H':'L';const lastRow=name==='Summary'?20:name==='Assumptions'?17:Math.min(14,s.getUsedRange().values.length);const p=await wb.render({sheetName:name,range:`A1:${last}${lastRow}`,scale:1,format:'png'});await fs.writeFile(path.join(out,`${name.replaceAll(' ','_')}.png`),new Uint8Array(await p.arrayBuffer()));console.log('rendered '+name);}
const file=path.join(out,'OP12_FOUR_KERNEL_SOURCE_CYCLE_MODEL_20260927.xlsx');await (await SpreadsheetFile.exportXlsx(wb)).save(file);console.log('OUTPUT='+file);
const resourceInspection=await wb.inspect({kind:'region',sheetId:'Summary',range:'A23:L26',maxChars:6000,tableMaxCols:12});await fs.writeFile(path.join(out,'inspection_resource_floors.txt'),resourceInspection.ndjson);
for(const [sheetName,range,name]of [['Summary','A23:L26','Summary_resource_floors'],['K2 Stages','A30:L44','K2_packet_detail'],['Event Sanity','A1:L23','Event_full_sanity']]){const p=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await p.arrayBuffer()));}
