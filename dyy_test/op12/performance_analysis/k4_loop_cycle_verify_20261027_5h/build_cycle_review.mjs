import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const R=path.dirname(fileURLToPath(import.meta.url));
const p=JSON.parse(await fs.readFile(path.join(R,'derived/FINAL_DELIVERY_DATA.json'),'utf8'));
const wb=Workbook.create();
const names=['循环周期对照','步骤资源预算','原生PC账单','NCU计数对账','独立实验证据','参数与边界'];
const S=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
function put(s,r,c,v){s.getCell(r-1,c-1).values=[[v]];}
function fx(s,r,c,v){s.getCell(r-1,c-1).formulas=[[v]];}
function setup(s,title,h,rows=60){
 s.showGridLines=false;s.getRangeByIndexes(0,0,rows,h.length).format.font={name:'Microsoft YaHei',size:10};
 s.getRangeByIndexes(0,0,rows,h.length).format.verticalAlignment='center';
 put(s,2,1,title);s.getRangeByIndexes(1,0,1,h.length).format.font={name:'Microsoft YaHei',size:15,bold:true};
 s.getRangeByIndexes(4,0,1,h.length).values=[h];s.getRangeByIndexes(4,0,1,h.length).format={fill:'#17365D',font:{name:'Microsoft YaHei',size:10,color:'#FFFFFF',bold:true},rowHeight:36,wrapText:true};
 s.getRangeByIndexes(5,0,rows-5,h.length).format.rowHeight=24;s.getRangeByIndexes(0,0,rows,h.length).format.columnWidth=17;
 s.getRange('A:A').format.columnWidth=27;
}
const rates=S['参数与边界'];setup(rates,'K4 条件资源速率与未量化项',['资源/参数','数值','单位','证据类型','适用范围'],51);
rates.getRange(`A6:E${5+p.parameters.length}`).values=p.parameters;
rates.getRange('A:A').format.columnWidth=30;rates.getRange('C:C').format.columnWidth=32;rates.getRange('D:D').format.columnWidth=38;rates.getRange('E:E').format.columnWidth=78;
rates.getRange(`A6:E${5+p.parameters.length}`).format.wrapText=true;rates.getRange(`A6:E${5+p.parameters.length}`).format.rowHeight=46;
put(rates,21,1,'单 warp 读取就绪的条件标定');rates.getRange('A23:E23').values=[['子型','条件 cy','同条件 cy','控制与采样','解释']];
for(let i=0;i<2;i++){const k=['32dp','16dp'][i],r=24+i,b=p.model.completion_bounds[k];rates.getRange(`A${r}:E${r}`).values=[[k,b[0],b[1],'原生 LDTM→NOP 配对；严格1%门槛通过','首个consumer wait在q=2；单读RAW条件标定，不是整阶段或普适硬件延迟']];}
put(rates,28,1,'计算公式');put(rates,28,2,'L = (Pload − Pnop) + q；q=2由原生控制字独立解码');
put(rates,30,1,'来源');put(rates,30,2,'NVIDIA PTX ISA / tcgen05.wait、fence、commit');put(rates,30,5,'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html');
put(rates,31,2,'NVIDIA NCU Guide / pipeline、wavefront、仲裁');put(rates,31,5,'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html');
put(rates,32,2,'CUDA Best Practices 13.0.1 / CC10.0 arithmetic table');put(rates,32,5,'https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html');
put(rates,34,1,'未量化项');
p.model.unresolved.forEach((x,i)=>{put(rates,35+i,1,i+1);put(rates,35+i,2,x);rates.getRange(`B${35+i}:E${35+i}`).merge();rates.getRange(`B${35+i}:E${35+i}`).format.wrapText=true;rates.getRange(`A${35+i}:E${35+i}`).format.rowHeight=42;});
put(rates,42,1,'正式模型启用');put(rates,42,2,false);put(rates,43,1,'effective_model_value');put(rates,43,2,'null');
const pcs=S['原生PC账单'];const ph=['warp组','阶段','PC','原生指令','工作量','计费单位','资源','适用速率','服务cy','编译stall','wait mask','说明'];
setup(pcs,'原版每64-token块的逐PC工作量及条件服务预算',ph,p.native.length+6);pcs.freezePanes.freezeRows(5);pcs.freezePanes.freezeColumns(3);
let out=[];
for(const r of p.native){
 const u=r.resource==='TMEM'?'显式寄存器payload B':r.resource.includes('SMEM')?'地址模型 WF':r.resource==='ISSUE'?'warp issue槽':r.opcode.startsWith('F2FP')?'有效标量转换输入/结果':'原生lane或标量结果';
 out.push([r.role,r.phase,'PC '+r.pc_hex,r.SASS,r.work,u,r.resource,r.rate,null,Number(r.stall),Number(r.waitmask),r.reason]);
}
pcs.getRangeByIndexes(5,0,out.length,ph.length).values=out;
pcs.getRangeByIndexes(5,8,out.length,1).formulas=out.map((_,i)=>[`=MAX(1,ROUNDUP(E${i+6}/H${i+6},0))`]);
pcs.getRange('B:B').format.columnWidth=35;pcs.getRange('D:D').format.columnWidth=88;pcs.getRange('F:G').format.columnWidth=28;pcs.getRange('L:L').format.columnWidth=85;
pcs.getRange(`D6:D${5+out.length}`).format.wrapText=true;pcs.getRange(`L6:L${5+out.length}`).format.wrapText=true;pcs.getRange(`A6:L${5+out.length}`).format.rowHeight=40;
pcs.getRange(`E6:I${5+out.length}`).setNumberFormat('#,##0.0');
const b=S['步骤资源预算'];const bh=['步骤 / native PC范围','参与者','TMEM cy','shared读 cy','shared写 cy','ALU/转换 cy','FP32 cy','FP16 cy','SFU cy','IMAD cy','issue最低槽'];
setup(b,'分步骤资源服务账：不能将各列或各行直接相加为墙钟',bh,36);
for(let i=0;i<p.phase.length;i++){
 const x=p.phase[i],r=i+6;put(b,r,1,x.phase+' '+x.PC_begin+'–'+x.PC_end);put(b,r,2,x.phase.startsWith('CG0')?'W0–3':'W4–7');
 const cats=['TMEM','LSU_Data|SMEM_Read','LSU_Data|SMEM_Write','ALU_PACK','FP','HALF','SFU','FMA_HEAVY','ISSUE'];
 for(let j=0;j<cats.length;j++)fx(b,r,j+3,`=SUMIFS('原生PC账单'!$I$6:$I$${5+out.length},'原生PC账单'!$B$6:$B$${5+out.length},"${x.phase}",'原生PC账单'!$G$6:$G$${5+out.length},"${cats[j]}")`);
}
b.getRange('A:A').format.columnWidth=62;b.getRange('C:K').format.columnWidth=17;b.getRange('C6:K14').setNumberFormat('#,##0');
put(b,17,1,'MMA的算术与隐式shared服务（独立列账，不与shader WF重试重复计费）');
b.getRange('A19:I19').values=[['GEMM','M','N','K','atom/块','FLOP','TC算术cy','shared字节','shared WF']];b.getRange('A19:I19').format.fill='#DCE6F1';
for(let i=0;i<p.mma.length;i++){const x=p.mma[i],r=i+20;b.getRange(`A${r}:E${r}`).values=[[x.name,x.M,x.N,x.K,x.atoms]];fx(b,r,6,`=2*B${r}*C${r}*D${r}`);fx(b,r,7,`=F${r}/'参数与边界'!$B$6`);put(b,r,8,x.shared_A_B+x.shared_B_B);fx(b,r,9,`=H${r}/128`);}
put(b,27,1,'总量（仅同资源求和）');for(const c of ['E','F','G','H','I'])b.getRange(`${c}27`).formulas=[[`=SUM(${c}20:${c}25)`]];
put(b,30,1,'HBM全冷逻辑代理cy');fx(b,30,3,`=(3*16384+8192+16384+256)/'参数与边界'!$B$17`);
put(b,31,1,'T来自L2时的HBM逻辑代理cy');fx(b,31,3,`=(3*16384+16384+256)/'参数与边界'!$B$17`);
put(b,33,1,'TMA另有448 write WF +128 read WF/块；W11 gate另有4 write WF。');
put(b,35,1,'预算表是服务量；实际节点完成和循环周期由Python DAG重新传播。未知完成延迟不填零。');
const nc=S['NCU计数对账'];setup(nc,'完整CP前史中的原版K4：原生量与NCU逐项核对',['指标','每块','每CTA固定','总块数','CTA数','模型推导','NCU观测','差值'],24);
for(let i=0;i<p.source_checks.length;i++){const x=p.source_checks[i],r=i+6;nc.getRange(`A${r}:E${r}`).values=[[x.metric,x.per_block,x.fixed_per_CTA,x.blocks,x.CTAs]];fx(nc,r,6,`=B${r}*D${r}+C${r}*E${r}`);put(nc,r,7,x.observed);fx(nc,r,8,`=F${r}-G${r}`);}
nc.getRange('A:A').format.columnWidth=35;nc.getRange('B:H').format.columnWidth=19;nc.getRange('B6:H10').setNumberFormat('#,##0');
put(nc,13,1,'同工作量原版/等待顺序对照；下列为3份NCU均值，不用replay duration推导提速');
nc.getRange('A15:H15').values=[['路径','Source WF','TC WF','硬件shared读','硬件shared写','读冲突','写冲突','仲裁计数']];nc.getRange('A15:H15').format.fill='#DCE6F1';
for(let i=0;i<p.vjoin.summary.length;i++){const x=p.vjoin.summary[i];nc.getRange(`A${16+i}:H${16+i}`).values=[[x.variant,x.Source_WF,x.TC,x.read,x.write,x.read_conflict,x.write_conflict,x.arbitration]];}
nc.getRange('B16:H19').setNumberFormat('#,##0');put(nc,22,1,'Source WF为地址模型；硬件WF含仲裁重试。不能把HW重试与TC操作数字节再次相加收费。');
put(nc,25,1,'原始全action流量摊销（含固定项/缓存前史；不是内层计时窗口的带宽）');
nc.getRange('A27:F27').values=[['指标','全action均值','原单位','摊销 B/块','条件B/cy代理','条件服务cy']];nc.getRange('A27:F27').format.fill='#DCE6F1';
for(let i=0;i<p.traffic.length;i++){const x=p.traffic[i],r=i+28;nc.getRange(`A${r}:E${r}`).values=[[x.metric,x.total,x.unit,null,x.rate]];fx(nc,r,4,`=B${r}*${x.metric==='L2 total sectors'?32:1}/8192`);if(x.rate!==null)fx(nc,r,6,`=D${r}/E${r}`);}
nc.getRange('B28:F31').setNumberFormat('#,##0.00');nc.getRange('C28:C31').format.wrapText=true;nc.getRange('A28:F31').format.rowHeight=36;
const ex=S['独立实验证据'];setup(ex,'独立对照与适用边界',['探针','模式','native cy/N','Event µs/N','严格1%','用途'],51);
const mm=JSON.parse(await fs.readFile(path.join(R,'derived/MATRIX_STRICT_SLOPE_REVIEW.json'),'utf8'));
let rr=6;
for(const x of mm.summary){ex.getRange(`A${rr}:F${rr}`).values=[['矩阵family '+x.kind,x.mode, x.cta_cycle_median_tail_slopes.reduce((a,b)=>a+b,0)/3,x.event_us_tail_slopes.reduce((a,b)=>a+b,0)/3,x.strict_gate,'指令族与控制包，仅验证读/写重叠规律']];rr++;}
rr=17;put(ex,rr,1,'ALU相同CUBIN（只统计活跃warp）');rr+=2;
for(const x of p.ALU_mix.summary){ex.getRange(`A${rr}:F${rr}`).values=[['PACK / IADD3',x.mode,x.clock_cycles_tail_slopes.reduce((a,b)=>a+b,0)/3,x.event_us_tail_slopes.reduce((a,b)=>a+b,0)/3,x.strict_gate,'mode1/2各自；mode3并行。不是K4阶段时延。']];rr++;}
rr=25;put(ex,rr,1,'LDTM↔NOP：32x32b.x32 / 16x256b.x8');
ex.getRange('A27:F27').values=[['子型','Pload cy/链','Pnop cy/链','q 原生cy','条件L cy','用途']];
for(let i=0;i<2;i++){const x=[p.completion_original.bounds[0],p.completion_extension.bounds[0]][i],r=28+i;ex.getRange(`A${r}:C${r}`).values=[[x.subtype,x.P_load.reduce((a,b)=>a+b,0)/3,x.P_nop.reduce((a,b)=>a+b,0)/3]];put(ex,r,4,2);fx(ex,r,5,`=B${r}-C${r}+D${r}`);put(ex,r,6,'单条LDTM的RAW就绪标定；只加已计服务之外的余项，不把整条链时长入模。');}
ex.getRange('A:A').format.columnWidth=39;ex.getRange('B:E').format.columnWidth=19;ex.getRange('F:F').format.columnWidth=72;ex.getRange('C6:D21').setNumberFormat('0.0000');
put(ex,33,1,'已保留失败轮');put(ex,34,1,'运行时predicate控制组改变了R2UR/NOP控制量，已排除。原生NOP配对通过。');
put(ex,36,1,'早期ALU汇总混入inactive warp，native包络偏小；修正host汇总，不改CUBIN。');
put(ex,38,1,'个别Event首次未过1%；在预声明N≤16384内独立扩点，未放宽门禁。');
const s=S['循环周期对照'];setup(s,'K4 每64-token循环块周期：同调用返回斜率对照',['Shape','观测 cy/块','条件模型低','条件模型高','低端偏差','高端偏差','调用数','10%范围'],42);s.tabColor='#17365D';
for(let i=0;i<p.comparison.length;i++){const x=p.comparison[i],r=i+6;s.getRange(`A${r}:D${r}`).values=[['S'+x.S+' / C'+x.C,x.observed_cycle_slope,x.conditional_model_min,x.conditional_model_max]];fx(s,r,5,`=(C${r}-B${r})/B${r}`);fx(s,r,6,`=(D${r}-B${r})/B${r}`);put(s,r,7,x.calls);fx(s,r,8,`=IF(AND(C${r}>=0.9*B${r},D${r}<=1.1*B${r}),"范围内","范围外")`);}
s.getRange('A:A').format.columnWidth=31;s.getRange('B:D').format.columnWidth=19;s.getRange('E:F').format.columnWidth=19;s.getRange('G:H').format.columnWidth=16;s.getRange('B6:D8').setNumberFormat('#,##0.0');s.getRange('E6:F8').setNumberFormat('0.00%');
s.getRange('H6:H8').format.horizontalAlignment='center';
put(s,11,1,'范围');put(s,11,2,'B1/H16/D128 FP16，P=8；主体循环，不含INIT、首尾和完整kernel。');s.getRange('B11:H11').merge();s.getRange('B11:H11').format.wrapText=true;s.getRange('A11:H11').format.rowHeight=34;
put(s,12,1,'观测');put(s,12,2,'同次调用CG1返回点拟合；每shape 3进程×60调用。插桩局部偏差仍未完全界定。');s.getRange('B12:H12').merge();s.getRange('B12:H12').format.wrapText=true;s.getRange('A12:H12').format.rowHeight=38;
const ch=s.charts.add('bar',s.getRange('A5:D8'));ch.setPosition('A15','I30');ch.title='稳态循环的条件预算与观测';ch.titleTextStyle.typeface='Microsoft YaHei';ch.titleTextStyle.fontSize=13;ch.legend={position:'top',textStyle:{typeface:'Microsoft YaHei',fontSize:10}};ch.xAxis={axisType:'textAxis',textStyle:{typeface:'Microsoft YaHei',fontSize:10}};ch.yAxis={numberFormatCode:'#,##0',numberFormatSourceLinked:false,textStyle:{typeface:'Microsoft YaHei',fontSize:10}};
ch.yAxis={min:0,max:3500};
['#343C44','#2B78B8','#5DA09B'].forEach((c,i)=>{ch.series.items[i].fill=c;ch.series.items[i].line={fill:c,width:1,style:'solid'};});
put(s,33,1,'模型性质');put(s,33,2,'原生工作量/条件速率 + scoreboard/RAW + buffer就绪传播。不是阶段实测时间求和。');
put(s,35,1,'调度结果');put(s,35,2,'循环周期为Python依赖调度快照；修改速率后须重跑调度脚本，再刷新本表。');
put(s,37,1,'正式参数');put(s,37,2,'model_enabled=false；未经用户审核不覆盖正式模型。');
put(s,39,1,'验收含义');put(s,39,2,'10%仅对应此主体比较；不等于解释90%的原始Gap或完整kernel性能。');
for(const r of [33,35,37,39]){s.getRange(`B${r}:H${r}`).merge();s.getRange(`B${r}:H${r}`).format.wrapText=true;s.getRange(`A${r}:H${r}`).format.rowHeight=35;}
put(s,42,1,'历史4-reader流速率敏感性：246.376B/cy，不自动替换主情景256');
s.getRange('A44:F44').values=[['Shape','敏感性低 cy','敏感性高 cy','低端偏差','高端偏差','说明']];s.getRange('A44:F44').format.fill='#DCE6F1';
for(let i=0;i<3;i++){const x=p.comparison[i],r=45+i;s.getRange(`A${r}:C${r}`).values=[['S'+x.S,x.sensitivity_min,x.sensitivity_max]];fx(s,r,4,`=(B${r}-B${i+6})/B${i+6}`);fx(s,r,5,`=(C${r}-B${i+6})/B${i+6}`);put(s,r,6,'保留超估符号');}
s.getRange('B45:C47').setNumberFormat('#,##0.00');s.getRange('D45:E47').setNumberFormat('0.00%');s.getRange('A44:H47').format.font={name:'Microsoft YaHei',size:10};s.getRange('A44:H47').format.rowHeight=24;
wb.recalculate();
if(!s.getRange('H6:H8').values.flat().every(v=>v==='范围内'))throw new Error('Comparison gate failed');
if(nc.getRange('H6:H10').values.flat().some(v=>v!==0))throw new Error('Counter reconciliation failed');
// Demonstrate work/rate formula recalculation without altering DAG snapshots.
const before=pcs.getRange('I6').values[0][0],oldRate=pcs.getRange('H6').values[0][0];put(pcs,6,8,oldRate/2);wb.recalculate();
if(pcs.getRange('I6').values[0][0]<before)throw new Error('Rate sensitivity did not propagate');put(pcs,6,8,oldRate);wb.recalculate();
const inspect=await wb.inspect({kind:'table',range:'循环周期对照!A5:H8',include:'values,formulas',tableMaxRows:5,tableMaxCols:8,maxChars:4000});await fs.writeFile(path.join(R,'derived/WORKBOOK_INSPECT.ndjson'),inspect.ndjson);
const err=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000});await fs.writeFile(path.join(R,'derived/WORKBOOK_ERROR_SCAN.ndjson'),err.ndjson);
for(const [name,range] of [['循环周期对照','A1:I48'],['步骤资源预算','A1:K35'],['原生PC账单','A1:L13'],['NCU计数对账','A1:H32'],['独立实验证据','A1:F39'],['参数与边界','A1:E26']]){
 const blob=await wb.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(path.join(R,'derived',`preview_${name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
for(const [label,name,range] of [['PC_middle','原生PC账单','A680:L689'],['PC_last','原生PC账单',`A${out.length-4}:L${out.length+5}`],['unknowns','参数与边界','A28:E43']]){
 const blob=await wb.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(path.join(R,'derived',`preview_${label}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(R,'K4_LOOP_CYCLE_REVIEW_20261027.xlsx'));console.log(JSON.stringify({sheets:names,nativeRows:p.native.length,range:p.model.main_range,output:'K4_LOOP_CYCLE_REVIEW_20261027.xlsx'}));
