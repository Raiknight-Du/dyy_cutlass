import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root=path.dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(await fs.readFile(path.join(root,'validated_process_data.json'),'utf8'));
const runs=data.processes;
const out=path.join(root,'outputs/cp4_event_freeze_0929');
await fs.mkdir(out,{recursive:true});
const wb=Workbook.create();
const main=wb.worksheets.add('Frozen Baseline');
const raw=wb.worksheets.add('Event Evidence');
const blue='#17365D',body='#24354B';
function put(s,addr,v){s.getRange(addr).values=[[v]];}
function formula(s,addr,v){s.getRange(addr).formulas=[[v]];}
function basic(s,end){
  s.showGridLines=false;
  s.getRange(`A1:J${end}`).format.font={name:'Arial',size:11,color:body};
  s.getRange(`A1:J${end}`).format.rowHeight=25;
  s.getRange(`A1:J${end}`).format.verticalAlignment='center';
  s.getRange(`A1:J${end}`).format.columnWidth=17;
}
function head(s,row,labels){
  const range=s.getRange(`A${row}:${String.fromCharCode(64+labels.length)}${row}`);
  range.values=[labels];range.format.fill=blue;
  range.format.font.color='#FFFFFF';range.format.font.bold=true;
  range.format.horizontalAlignment='center';range.format.wrapText=true;
  range.format.rowHeight=39;
}
function title(s,text){put(s,'A2',text);s.getRange('A2').format.font.size=16;s.getRange('A2').format.font.bold=true;s.getRange('A2').format.rowHeight=31;}
function stripes(s,a,b,last){for(let r=a;r<=b;r++)if((r-a)%2===0)s.getRange(`A${r}:${last}${r}`).format.fill='#F3F6FA';}
basic(main,34);basic(raw,139);
main.tabColor=blue;
title(main,'OP12 CP四kernel · 冻结Event基准');
title(raw,'累计Event观测 · 每进程差分与原生CP对照');
put(main,'A3','FP16 / B1 H16 D128 / FP32 state / µs；正式9进程，4500个累计Event观测。');
main.getRange('A3').format.font.italic=true;
head(main,5,['S (token)','C (token)','K1 T (µs)','K2 M/N (µs)','K3 Fixup (µs)','K4 Final (µs)','阶段之和 (µs)','原生CP E2E (µs)']);
head(raw,5,['S','Trial','K1 (µs)','K2 (µs)','K3 (µs)','K4 (µs)','阶段和 (µs)','Native (µs)','Probe差异 %','闭合差异 %']);
head(raw,18,['S','Trial','Probe','样本数','均值 (µs)','P5 (µs)','P95 (µs)','CV (%)','图总时长/调用 (µs)','证据相对路径']);
head(raw,67,['S','Trial','Repeat','原生图总时长 (µs)','每调用均值 (µs)']);
put(raw,'A117','CP/图');put(raw,'B117',400);
put(raw,'A118','样本/probe');put(raw,'B118',100);
put(raw,'A119','进程/shape');put(raw,'B119',3);
for(let i=0;i<runs.length;i++){
  const run=runs[i], su=run.summary, r=6+i, p=19+i*5, n=68+i*5;
  put(raw,`A${r}`,run.S);put(raw,`B${r}`,run.trial);
  for(let k=1;k<=4;k++)formula(raw,`${String.fromCharCode(66+k)}${r}`,`=E${p+k}-E${p+k-1}`);
  formula(raw,`G${r}`,`=SUM(C${r}:F${r})`);
  formula(raw,`H${r}`,`=MEDIAN(E${n}:E${n+4})`);
  put(raw,`I${r}`,su.marker_check_pct_max);
  formula(raw,`J${r}`,`=(G${r}/H${r}-1)*100`);
  for(let k=0;k<=4;k++){
    const pr=su.probes[`P${k}`];
    raw.getRange(`A${p+k}:J${p+k}`).values=[[run.S,run.trial,`P${k}`,pr.n,pr.mean_us,pr.p5_us,pr.p95_us,pr.cv_pct,pr.batch_per_call_mean_us,`${run.path}/summary.json`]];
  }
  for(let k=0;k<5;k++){
    const v=run.native_batch_samples[k];
    raw.getRange(`A${n+k}:D${n+k}`).values=[[run.S,run.trial,v.repeat,v.batch_us]];
    formula(raw,`E${n+k}`,`=D${n+k}/$B$117`);
  }
}
for(let i=0;i<3;i++){
  const shape=data.baseline.shapes[i], r=6+i, start=6+i*3;
  put(main,`A${r}`,shape.S);put(main,`B${r}`,shape.C);
  for(let c=3;c<=6;c++){const l=String.fromCharCode(64+c);formula(main,`${l}${r}`,`=MEDIAN('Event Evidence'!${l}${start}:${l}${start+2})`);}
  formula(main,`G${r}`,`=SUM(C${r}:F${r})`);
  formula(main,`H${r}`,`=MEDIAN('Event Evidence'!H${start}:H${start+2})`);
}
stripes(main,6,8,'H');stripes(raw,6,14,'J');stripes(raw,19,63,'J');stripes(raw,68,112,'E');
main.getRange('C6:H8').setNumberFormat('0.000');main.getRange('A6:B8').setNumberFormat('#,##0');
main.getRange('A6:H8').format.rowHeight=30;
raw.getRange('C6:J14').setNumberFormat('0.000');
raw.getRange('E19:I63').setNumberFormat('0.000');
raw.getRange('D68:E112').setNumberFormat('0.000');
raw.getRange('A6:B112').setNumberFormat('0');
main.getRange('A:A').format.columnWidth=12;main.getRange('B:B').format.columnWidth=12;
main.getRange('C:F').format.columnWidth=19;main.getRange('G:H').format.columnWidth=23;
raw.getRange('A:A').format.columnWidth=12;raw.getRange('B:C').format.columnWidth=10;
raw.getRange('D:H').format.columnWidth=18;raw.getRange('I:I').format.columnWidth=24;raw.getRange('J:J').format.columnWidth=53;
raw.freezePanes.freezeRows(5);
put(main,'A11','K1–K4为累计Event均值差的跨进程中位数，不是同一调用内四个直接时间戳。');
put(main,'A12','原生CP E2E：无内部Event的400次四kernel图，batch总时长/400；不含JIT、分配与Python。');
put(main,'A13','阶段和不强制缩放到E2E；差异含计时标记、缓存前史及频率/统计口径残差。');
put(main,'A15','唯一权威：frozen_baseline.json；工作簿为可追溯计算视图，不覆盖旧版测量。');
put(main,'A16',data.baseline.baseline_id);
head(main,19,['S','Grid K1','Grid K2','Grid K3','Grid K4','P','Probe差异上限 %','阶段和/E2E差异 %']);
for(let i=0;i<3;i++){
  const sh=data.baseline.shapes[i],r=20+i,start=6+i*3;
  main.getRange(`A${r}:F${r}`).values=[[sh.S,sh.grid.K1,sh.grid.K2,sh.grid.K3,sh.grid.K4,sh.P]];
  formula(main,`G${r}`,`=MAX('Event Evidence'!I${start}:I${start+2})`);
  formula(main,`H${r}`,`=(G${6+i}/H${6+i}-1)*100`);
}
main.getRange('A20:F22').setNumberFormat('#,##0');main.getRange('G20:H22').setNumberFormat('0.00');
put(main,'A25','K3（S32768）三进程范围：25.591–28.011 µs；小幅消融需复测与漂移门禁。');
put(main,'A26','消融：同源固定提交、同输入SHA/轮转/C/dtype/state、同完整链与probe方法；使用冻结绝对基准。');
put(main,'A27','同轮B0只检查漂移，不能替代冻结基准；旧孤立K3 / NCU replay / IKET时延不可直接相减。');
put(raw,'A122','前缀定义');put(raw,'C122','P0：空Event对；P1：K1；P2：K1+K2；P3：K1+K2+K3；P4：完整四kernel。');
put(raw,'A123','差分定义');put(raw,'C123','Ki=mean(Pi)-mean(Pi-1)；不同probe不是同一次调用的四阶段时间线。');
put(raw,'A124','来源');put(raw,'C124','本目录 formal/*/prefix_event_samples.csv、summary.json、native_batch_samples.json；原始样本未删点。');
put(raw,'A125','版本');put(raw,'C125',data.baseline.source_commit);
put(raw,'A126','方法文档');put(raw,'C126','PROTOCOL_CN.md；真实K1→K2→K3→K4，单stream，4 rotating QKV，无cache prime/flush。');
put(raw,'A127','时钟');put(raw,'C127','未锁频；完整环境和采样温度/时钟/功耗见 formal/*/environment.json；不换算为精确cycle。');
put(raw,'A128','官方Event依据');put(raw,'C128','https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html');
put(raw,'A129','官方Graph依据');put(raw,'C129','https://docs.pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html');
put(raw,'A131','原始数据');put(raw,'C131','validated_process_data.json 保留每进程统计；冻结JSON SHA见 FREEZE_MANIFEST.json。');
// One final calculation pass; no assumptions fitted to target latency.
wb.recalculate();
const check=await wb.inspect({kind:'table',range:'Frozen Baseline!A5:H8',include:'values,formulas',tableMaxRows:8,tableMaxCols:8});
await fs.writeFile(path.join(out,'formula_inspection.ndjson'),check.ndjson);
const errs=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'final formula scan'});
await fs.writeFile(path.join(out,'formula_error_scan.ndjson'),errs.ndjson);
const values=main.getRange('C6:H8').values;
for(let i=0;i<3;i++){
  const sh=data.baseline.shapes[i],expected=[...Object.values(sh.stage_us),sh.stage_sum_us,sh.native_chain_us];
  for(let j=0;j<6;j++)if(Math.abs(values[i][j]-expected[j])>1e-8)throw new Error(`Baseline reconciliation failed ${i},${j}`);
}
for(const [name,sheetName,range] of [
  ['frozen_baseline','Frozen Baseline','A1:H28'],
  ['process_and_prefix','Event Evidence','A1:J29'],
  ['remaining_prefix','Event Evidence','A30:J64'],
  ['native_batch','Event Evidence','A66:J96'],
  ['native_and_sources','Event Evidence','A97:J132']]){
  if(process.argv.includes('--review-title-only') && name!=='process_and_prefix')continue;
  const preview=await wb.render({sheetName,range,scale:1.35,format:'png'});
  await fs.writeFile(path.join(out,`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(out,'OP12_CP4_EVENT_FROZEN_BASELINE_20260929.xlsx'));
console.log(JSON.stringify({status:'PASS',sheets:2,rows_in_source:9,filename:'OP12_CP4_EVENT_FROZEN_BASELINE_20260929.xlsx',values}));
