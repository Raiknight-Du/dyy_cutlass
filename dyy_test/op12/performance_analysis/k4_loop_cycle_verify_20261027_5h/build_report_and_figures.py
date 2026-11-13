from pathlib import Path
import json,csv,statistics,math,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
R=Path(__file__).resolve().parent;D=R/'derived';F=R/'figures';F.mkdir(exist_ok=True)
p=json.loads((D/'FINAL_DELIVERY_DATA.json').read_text(encoding='utf-8'));m=p['model']
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():fm.fontManager.addfont(str(font));plt.rcParams['font.family']=fm.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.axisbelow':True,'svg.fonttype':'none','axes.unicode_minus':False})
sc=m['scenarios'];cmp=p['comparison']
labels=[f"观测 S{x['S']} / C{x['C']}" for x in cmp]+['条件DAG：主情景256B/cy','读流246.376B/cy敏感性','原生工作＋shared读耦合，未加读取完成范围','原生工作，暂不考虑TC操作数读耦合','全冷逻辑GMEM服务代理','仅TC算术服务','显式TMEM混合子型服务','shader＋TC shared读服务']
low=[x['observed_cycle_slope'] for x in cmp]+[m['main_range'][0],cmp[0]['sensitivity_min'],sc[2]['mean_cycle'],sc[1]['mean_cycle'],2312,1152,1120,1384]
high=[x['observed_cycle_slope'] for x in cmp]+[m['main_range'][1],cmp[0]['sensitivity_max']]+low[5:]
assert len(labels)==len(low)==len(high)
fig,ax=plt.subplots(figsize=(12,6.9));colors=['#38424A']*3+['#2778B8','#6B98BF','#D17734','#D17734']+['#2D9A79']*4
for i,(a,b,c) in enumerate(zip(low,high,colors)):
 ax.barh(i,a,color=c,height=.66)
 if b>a:ax.barh(i,b-a,left=a,color=c,alpha=.35,height=.66,hatch='//')
 ax.text(b+22,i,f'{a:,.0f}' if a==b else f'{a:,.0f}–{b:,.0f}',va='center',fontsize=9)
ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,3900);ax.grid(axis='x',alpha=.23);ax.set_xlabel('Cycles / 64-token block / CTA')
ax.set_title('K4：服务下界、条件依赖模型与同调用返回斜率',loc='left',fontweight='bold',pad=16)
fig.text(.02,.025,'蓝色范围为独立单读就绪标定＋原生调度＋共享资源情景；非置信区间或绝对硬件上下界。\n含全SM四warp/cycle发射约束；观测为插桩返回斜率。未将阶段实测时间填入理论节点。',fontsize=9,color='#525B63')
fig.subplots_adjust(left=.43,right=.96,top=.9,bottom=.15)
fig.savefig(F/'K4_LOOP_CYCLE_COMPARISON.png',dpi=180);fig.savefig(F/'K4_LOOP_CYCLE_COMPARISON.svg');plt.close(fig)
v=p['vjoin']['summary'];fig,axs=plt.subplots(1,2,figsize=(10.7,4.6));names=[x['variant'] for x in v]
for ax,key,title in zip(axs,['read','read_conflict'],['硬件shared读WF','shared读冲突计数']):
 y=[x[key]/1e6 for x in v];ax.bar(names,y,color=['#3C464D','#5A93B7','#40849F','#327879']);ax.set_ylabel('百万次 / 完整K4');ax.set_title(title);ax.set_ylim(0,max(y)*1.17);ax.grid(axis='y',alpha=.23)
 for i,z in enumerate(y):ax.text(i,z+.025,f'{z:.3f}',ha='center',fontsize=10)
fig.suptitle('同工作量，仅移动CG1等待位置：仲裁计数显著变化',x=.02,ha='left',fontweight='bold')
fig.text(.02,.015,'B1/H16/S32768/D128/FP16/C4096；每变体3份完整CP前史NCU。\nSource shared WF=7,178,496、TC WF=6,291,456及主要原生工作量均不变；此图不表示性能提速。',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.92]);fig.savefig(F/'K4_SHARED_ARBITRATION_CONTROL.png',dpi=180);fig.savefig(F/'K4_SHARED_ARBITRATION_CONTROL.svg');plt.close(fig)
# A resource schedule is a model output, never an observed trace. Keep one
# representative block plus neighbouring overlap, not a fabricated path.
logs=list(csv.DictReader((D/'REPRESENTATIVE_MODEL_SERVICE.csv').open(encoding='utf-8-sig')))
recs=[(int(z['node'].split(':')[0]),z) for z in logs if z['node'].split(':')[0].isdigit()]
chunk=32;t0=min(float(z['request_cycle']) for c,z in recs if c==chunk and z['actor']=='CG1');t1=t0+4000
roles=['CG0','CG1','SMEM_Read','TMEM','TC'];palette={'CG0':'#6794AD','CG1':'#DB8C47','SMEM_Read':'#477C7B','TMEM':'#8662A1','TC':'#3C6A92'}
fig,ax=plt.subplots(figsize=(12,3.9))
for i,role in enumerate(roles):
 for z in logs:
  start,end=float(z['service_start']),float(z['service_end'])
  match=(z['actor']==role if role in ['CG0','CG1'] else role in z['resource'].split('|'))
  if match and start<t1 and end>t0:ax.broken_barh([(max(start,t0)-t0,min(end,t1)-max(start,t0))],(i-.32,.64),facecolors=palette[role])
ax.set_yticks(range(len(roles)),roles);ax.invert_yaxis();ax.set_xlim(0,4000);ax.grid(axis='x',alpha=.2);ax.set_xlabel('模型相对cycle（含相邻块重叠）');ax.set_title('原生PC顺序＋资源服务＋就绪约束：代表性调度',loc='left',fontweight='bold');
fig.text(.015,.02,'此图来自Python条件调度器，不是IKET或NCU采样。CG0/CG1只画服务片段，空白包含等待与可重叠issue。',fontsize=9);fig.tight_layout(rect=[0,.08,1,.98]);fig.savefig(F/'K4_CONDITIONAL_RESOURCE_SCHEDULE.png',dpi=180);fig.savefig(F/'K4_CONDITIONAL_RESOURCE_SCHEDULE.svg');plt.close(fig)
# Editable diagram source. Computation-only, with explicit resource edges.
root=ET.Element('mxGraphModel',{'page':'1','pageWidth':'1520','pageHeight':'800','adaptiveColors':'auto'});r=ET.SubElement(root,'root');ET.SubElement(r,'mxCell',id='0');ET.SubElement(r,'mxCell',id='1',parent='0')
def node(k,text,col,row,color='#DAE8FC'):
 z=ET.SubElement(r,'mxCell',id=k,value=text,style=f'rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor={color};strokeColor=#738395;',vertex='1',parent='1');ET.SubElement(z,'mxGeometry',x=str(col*180+40),y=str(row*120+40),width='140',height='60',**{'as':'geometry'})
def edge(k,a,b,dashed=False):
 z=ET.SubElement(r,'mxCell',id=k,source=a,target=b,style='edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=classic;'+('dashed=1;strokeColor=#75818D;' if dashed else 'strokeColor=#344B61;'),edge='1',parent='1');ET.SubElement(z,'mxGeometry',relative='1',**{'as':'geometry'})
for z in [('T','T门控变换\nCG0；shared/ALU/SFU',0,0),('QK','QK MMA\n8atom / TC+shared',1,0),('WQK','QK后处理/发布\nCG0；LDTM/PACK/STSM',2,0),
 ('STATE','state读回、PACK\n与decay算术交错',0,2),('PUB','A发布 / wait::st\n保留就绪端点',1,2),('KS','KS → QS\nMMA顺序发射、异步完成',2,2),('RES','KS residual\nV gather / VKS发布',3,2),('NV','NV MMA\n等待VKS与Ainv',4,2),('DELTA','NV/Delta处理\nLDTM/PACK/STTM',5,2),('DS','dS MMA\n生成下一块state',6,2),
 ('BASE','decayed state发布\nSTTM / wait::st',1,3),('QS','QS后处理/发布\n等待QS完成',3,3),('O','O MMA与暂存\n等待QK权重/NV/QS',5,0),('R','返回 / 下一块\n不可只累加warp阶段时间',7,2),
 ('READ','shared读资源\nshader616+TC768 WF/块',2,4),('TMEM','显式TMEM资源\n按子型记账，RAW另列',4,4),('ALU','ALU/浮点/SFU\n原生工作量及issue/scoreboard',6,4)]:node(*z)
for i,(a,b) in enumerate([('T','NV'),('QK','WQK'),('WQK','O'),('STATE','PUB'),('STATE','BASE'),('PUB','KS'),('KS','RES'),('KS','QS'),('RES','NV'),('NV','DELTA'),('QS','O'),('DELTA','O'),('DELTA','DS'),('BASE','DS'),('DS','R'),('O','R')]):edge('e'+str(i),a,b)
for i,(a,b) in enumerate([('READ','QK'),('READ','RES'),('TMEM','PUB'),('TMEM','DELTA'),('ALU','T'),('ALU','DELTA')]):edge('d'+str(i),a,b,True)
for el in r:
 if el.get('value'):el.set('value',el.get('value').replace('\n','&#xa;'))
xml=ET.tostring(root,encoding='unicode').replace('&amp;#xa;','&#xa;');(D/'K4_LOOP_NATIVE_DAG.mxGraphModel.xml').write_text(xml,encoding='utf-8')
mx=ET.Element('mxfile',{'host':'app.diagrams.net'});dg=ET.SubElement(mx,'diagram',id='k4-loop-native',name='K4循环依赖与资源');dg.append(root);ET.ElementTree(mx).write(R/'K4_LOOP_NATIVE_DAG.drawio',encoding='utf-8',xml_declaration=True)
rows=[]
for x in cmp:rows.append(f"|{x['S']} / {x['C']}|{x['observed_cycle_slope']:.1f}|{x['conditional_model_min']:.0f}–{x['conditional_model_max']:.0f}|{x['signed_error_min']:+.2%}～{x['signed_error_max']:+.2%}|")
phase_rows=[]
for x in p['phase']:phase_rows.append('|'+x['phase']+'|'+x['PC_begin']+'–'+x['PC_end']+'|'+' / '.join(f"{k}:{x[k]:g}" for k in ['TMEM','shared_read','shared_write','ALU_PACK','FP32','FP16','SFU','IMAD','issue_only_floor'] if x[k])+'|')
mma_rows=[f"|{x['name']}|{x['M']}×{x['N']}×{x['K']}|{x['atoms']}|{x['FLOP']//8192}|{x['shared_A_B']+x['shared_B_B']:,}|{x['shared_WF']:g}|" for x in p['mma']]
counter_rows=[f"|{x['metric']}|{x['per_block']}×8192＋{x['fixed_per_CTA']}×128|{x['predicted']:,}|{x['observed']:,.0f}|" for x in p['source_checks']]
vrows=[f"|{x['variant']}|{x['Source_WF']:,.0f}|{x['TC']:,.0f}|{x['read']:,.0f}|{x['read_conflict']:,.0f}|" for x in v]
texts=f'''# K4 稳态循环周期与shared/LSU依赖核验

## 结论与验收范围

**本轮得到从原生工作量、具名条件速率和依赖传播计算的主体循环预算：3430.25–3435.25 cycles/64-token块。三组shape相对同调用返回斜率的偏差为+8.03%至+9.36%，满足本轮±10%的主体目标，但模型有明确的正偏差。** 这是条件资源/队列近似，不是整阶段实测时间求和，不把目标Gap反推成资源参数。保留旧文件，正式模型未自动启用。

|S / CP partition C|观测 cy/块|条件模型 cy/块|带符号相对偏差|
|---|---:|---:|---:|
{chr(10).join(rows)}

固定B1/H16/D128、FP16、P=8，算法块64token，单CTA递推16/32/64块、grid128、384线程。观测按每次调用中CG1返回点对块号独立OLS拟合，再取3进程×60调用的斜率均值；S8192/S16384/S32768分别使用返回点4..11/8..23/8..55。不拼接不同warp均值、不用NCU replay时延、不用请求1845MHz把µs换成cycle。

理论调度也在上述相同返回点窗口内拟合；本次最终两种FP16/FP32逻辑共享场景在三个窗口均得到3430.25–3435.25cy。外部初始就绪条件仍属模型假设，未用实测INIT时间校准。

比较对象是**插桩返回节拍**。已有插桩局部偏差未完全界定，所以本轮不宣称“原版全kernel性能已解释90%”，不覆盖INIT和首尾。模型/观测比值也不是原始Gap的因果解释比例。机器台账另保留相对旧2296-cycle服务情景的区别，不能把数值预算上移本身视为物理原因已经逐项分离。

![循环预算与观测](figures/K4_LOOP_CYCLE_COMPARISON.png)

历史4-reader流的246.376B/cy仅作独立敏感性，不替换256主情景。它给出3436.625–3441.625cy/块；S32768最大值相对3141.20cy的偏差为**+9.57%**，明确保留超估。此敏感性也仍在±10%内。范围来自明确条件场景，不是统计置信区间或绝对硬件上下界。

最终复核发现早期2869–3131cy版本遗漏CG0/CG1/W8/W10之间的总发射容量约束，因此撤回其“匹配”结论。该版文档、表格和payload保存在`history/pre_issue_capacity`供审计，不作为最终模型。补齐发射约束后，原先未经端点解码的宽TMEM协议边界会产生超出10%的高端情景；进一步仅根据独立原语的原生wait位置推导26/74cy条件就绪值，才得到上表。既不保留遗漏资源的有利数值，也不按目标挑选宽区间低端。

## 1. MMA隐式读shared：该新增什么账

**是隐式操作数读取，不是额外生成LDS指令。** tcgen05的SS/TS操作数按descriptor从shared读取；因此要在shared读资源中补齐TC客户端需求，但不能增加shader的LSU指令数，更不能把已含仲裁重试的硬件LSU WF再与同一TC需求双重收费。

|GEMM|M×N×K|atom/块|纯算术cy，8192FLOP/cy|隐式shared B/块|TC客户端WF/块|
|---|---|---:|---:|---:|---:|
{chr(10).join(mma_rows)}
|合计||36|1152|98,304|768|

shader地址模型另有读616、写260WF/块。TC implicit read为96KiB/128B=768WF/块，其中QK同时读A/B，其他五组主要为shared B；显式TMEM字节另立账。原版的专用`l1tex__data_pipe_tc_wavefronts.sum`=6,291,456=768×8192，与上述分子精确一致。**本轮MMA算术统一8192FLOP/SM-cycle参考，不凭“测到半速”认定固定一半硬件通路**；QK的128算术cy与256操作数读WF是两种不同资源需求。

TMA还涉及每块448 shared-write WF和128 shared-read WF；W11 gate另有4 write WF。它们已列入全量资源账。TMA与所有shared客户端是否完全共用同一服务通路尚无充分匹配数据，不能把四类计数无条件全加成一个“LSU时间”。

## 2. 地址核查、独立实验与原版干预

CG0全部132个shared PC完成逐lane地址/谓词枚举，动态pred-on及Source WF全部匹配，未发现原生地址自身的额外bank conflict。ring stage仅改变已对齐基址，目标三shape均为完整64-token块。独立探针覆盖CG0/CG1代表性的LDS、LDSM、STSM六类序列；不是声称每个PC都测了独立吞吐。

24份NCU显示：只改变是否并行运行SS-QK，目标shader Source WF不变，读指令族的硬件shared读WF增加；例如CG0 LDSM64→320WF/轮、CG1 LDSM128→约384WF/轮，TC读恒为256WF/轮。STSM写WF基本不增加。严格1%扩点数据中，LDSM并行周期近似两种读需求之和；STSM与TC读并行则大部分可覆盖。因此**shared读耦合有实证，不能采用“所有shared读写都同池串行相加”的模型**。

为了定位原版，保持完整冷K1→K2→K3→K4缓存前史，只移动CG1等待位置：KSJOIN在V读取前等KS；QSJOIN先等KS和QS再读V；GATEJOIN把等待移到gate/V读取前。12份有效NCU，输出/state位精确一致，MMA/TMA/TMEM/转换/主要数学动态工作量及Source WF一致；原版自身已有的local指令保留，没有新增spill。

|原版/变体|Source shared WF|TC WF|硬件shared读WF，3份均值|读冲突计数，3份均值|
|---|---:|---:|---:|---:|
{chr(10).join(vrows)}

![原版等待位置对照](figures/K4_SHARED_ARBITRATION_CONTROL.png)

这是CG1 gate/V读取与未完成MMA的跨客户端仲裁证据；改变了其他工作交错，故不能把全部计数差唯一分配到某条LDS PC，也不能当成该步骤独立时长或实际优化收益。历史Source总数少512WF来自固定控制/atomic统计；本批同CUBIN条件统一为7,178,496，循环仍为876WF/块。

ALU补充对照使用同一CUBIN、32条独立链、4+4warp，保留4096 PACK lane操作和4096三实源IADD3 lane操作/N。修正为只统计活跃warp后，单独PACK约65.375cy/N、IADD3约68.25cy/N，并行约128.625cy/N；对应Event斜率全部通过严格1%扩点门槛，6份NCU将两类目标工作精确记入ALU。它支持本混合条件下合并服务需求，不升级为普适裸硬件速率；IMAD按官方pipe定义排除在此ALU项之外。早期把inactive warp混入中位数的结果已降级，原始数据保留。

## 3. 原生预算如何传播成每块cycle

每条已执行的native指令先由原生工作量和适用速率形成服务需求；PACK按两个标量输入/结果计，FMUL2按两个FP32结果计，ISETP按执行lane而非布尔true数计。保留同warp真实PC顺序、原生控制字中的最小issue间距和scoreboard wait；编译stall不是硬件RAW latency。

```text
ready_i = max(上一个issue允许时刻, RAW/scoreboard就绪, pipeline/buffer可用)
service_start_i = max(ready_i, 对应资源可接收时刻)
service_end_i = service_start_i + work_i / conditional_rate_i
consumer_ready_i = service_end_i + 已独立限定的完成延迟余项
```

另加**全SM每cycle最多4条warp指令**的聚合发射资源。来源为3份原版NCU的device元数据：`device__attribute_max_ipc_per_multiprocessor=4`、`max_ipc_per_scheduler=1`，见`DEVICE_ISSUE_CAPACITY.json`。CG0/CG1的一条四warp组指令占1个SM-cycle，W8/W10的一条issuer指令占1/4。数据服务与发射占用分开，可重叠但不能各自超额发射。调度器是四warp锁步组与FIFO资源队列的近似；它没有还原每个SMSP的独立warp交错，因此不是绝对硬件下界或逐cycle硬件模拟器。该组级限制可能偏保守，须与正偏差一起报告。

控制字、数据RAW、实际异步MMA提交/完成、Ainv三槽、QK两槽、Q/K ring及单槽state/QS反馈均保留。native issuer W10的描述符/寻址/控制也入模型。unknown谓词/选择等容量只保留可证明issue下界，不填一个“Gap税”。

特别核实了`STTM → wait::st → mbarrier.arrive → consumer`。原版PTX在2710、2846、3158、3302、3591行有`tcgen05.wait::st`，不能把发布flag发射就当数据已完成。模型在可证明完成端点加入已有store服务完成的依赖边；MEMBAR/shared发布同样保留。相关边不能再另加一份“store阶段耗时”。

**TMEM吞吐与读结果就绪分开。** 16dp256显式read以及QS write沿用已核条件流256B/cy，其他具名子型保留512假设；本项总显式服务为1120cy/块，仍不是墙钟。新单指令实验仅含LDTM、wait及最小地址RAW链。逐字节替换16个LDTM为NOP，并关闭这16个指令自己的WB/RB，其他native字节/动态计数不变；实际寄存器、输入、SM ID和计时端点保留。

|指令子型|有读Pload cy/单次原语链|NOP对照Pnop cy/链|首wait距离q|条件就绪cy|
|---|---:|---:|---:|---:|
|32x32b.x32|49.375|25.375|2|26|
|16x256b.x8|97.375|25.375|2|74|

**端点推导：** 原生LDTM设置WB0，后面一条NOP，再由LOP3首次等待WB0；两条间距各为1，故`q=2`。16个展开位置、两种子型均如此，没有中间WB0生产者。NOP与有读版本的非load字节及执行计数相同；普通后缀为5+5+13cy，循环边界额外贡献在配对中相消。于是`Δ=Pload−Pnop=max(0,L−q)`，推出`L=q+Δ=26/74cy`。这是对单条LDTM结果就绪端点的**条件标定**，不是把49.375/97.375整条反馈链作为K4节点。

用于DAG的额外完成部分为`max(0,L−Bwarp/rate)`，以免重复收取已计的读服务量。四warp组服务先按总字节排队，最后结果消费再等待上述余项。原版短burst、TMEM相位/列位置、四warp并发仍属适用条件，不声称全地址通用延迟常数。早期[25,50]/[73,98]只是在尚未解码首wait时的宽协议边界，旧审阅JSON保留；不是为配合K4目标而任选一个端点。

独立统计门禁均保存≥4个递增N、20预热、30样本、3进程；native/Event两域相邻尾部三点窗和跨进程斜率变化严格<1%、R²≥0.99、CV及P5–P95波动检查。未过的初轮保留，最多扩到N16384，不用放宽门槛来填平目标Gap。

![条件资源调度](figures/K4_CONDITIONAL_RESOURCE_SCHEDULE.png)

以下只是各native PC服务预算的分步骤汇总；**不把它们直接串行相加或逐阶段取max后称为最终关键路径**。

|步骤|PC范围|资源服务cy（仅非零已记项）|
|---|---|---|
{chr(10).join(phase_rows)}

完整1384条PC账单、控制位及公式见工作簿/CSV。CG0和CG1共同的ALU/SMEM/TMEM与总发射服务按资源约束传播；FP16/FP32逻辑共享两种场景形成3430.25–3435.25，而非一个反调出来的点。16dp短读的消费依赖、shared读耦合和CG1发布/递推是比单纯1152cy Tensor算术服务更有影响的组合。改变边会改变FIFO交错，因此情景差值不能当作可加的独立步骤耗时。

## 4. NCU一致性与尚未量化的成本

|指标|原生分子公式|推导|NCU原始观测|
|---|---|---:|---:|
{chr(10).join(counter_rows)}

原版硬件shared读/写WF的额外量与仲裁counter接近，但两者定义不完全相同，保留固定项/atomic及其他服务差异。模型用唯一的操作数读服务和队列依赖解释阻塞，不把retry WF当额外算法字节。没有用stall百分比乘kernel总cycle分摊阶段时间。逻辑HBM、L2请求、Source地址WF、硬件shared/TC WF也没有混为同一物理流量。

原版B0三份NCU的全action均值为DRAM读472,293,973B、写142,104,405B，L2总sector约31,285,207。按8192个完整块仅作工作量摊销，DRAM读写约74,999.8B/块；按用户32B/SM-cycle代理为2343.7cy。L2请求约122,207.8B/块，按64代理为1909.5cy。两者均低于主体条件模型周期，未形成容量矛盾。这里包含INIT/descriptor固定项及缓存前史；不能把全action计数除以内层trace时间冒称实测带宽，也不能把额外DRAM写直接归为K4新计算或已证实的某一脏写回来源。

主要剩余项：稀疏EX2的精确服务粒度、部分谓词/选择和uniform容量、真实队列仲裁、短MMA完成通知、原版local/L1返回，以及TMA/gate/O的晚到暴露。独立共享实验支持读耦合，但未唯一确定每个native warp的实际队列优先级。外部Q/K/V/T/gate在数值主体DAG中按稳态已就绪情景处理；其非零工作量、全冷2312cy/T来自L2时2056cy代理和TMA共享服务另列，并未以未知=0声称已完整建模。

**带符号残差是模型高于观测约255–294cy/块，不能命名成“尚缺的正开销”。** 四warp锁步、聚合issue分配、FIFO队列和实际warp错峰的差异可能使模型偏保守；外部TMA晚到、通知及local返回则可能增加实际时间。这些影响可能相互抵消，尚不能逐项分离，故不宣称完整因果Gap解释率。**本轮停止于三个同范围主体窗口均在±10%内、主要机制证据成立；不进一步开发优化，不宣称全kernel或每个剩余cycle都解释了。** 若要进一步降低正偏差，下一步应先在CPU调度器中拆分四warp并按SMSP约束发射，而不是添加正残差常数或再测整阶段时间。

## 5. 复算与证据

- `K4_LOOP_CYCLE_REVIEW_20261027.xlsx`：循环对照、步骤服务、逐PC公式、NCU对账、独立试验、参数边界。修改速率后须先重跑Python调度，循环周期列为调度快照。
- `derive_single_ldtm_completion.py` + `recompute_final_model.py` + `native_pc_service_model.py` + `encoded_async_engine_capacity.py`：独立推导单读就绪、先求条件调度，最后才读取目标观测，禁止反调参数。旧`encoded_async_engine.py`保留供历史复算，最终入口使用容量约束版。
- `derived/FINAL_CYCLE_MODEL.json`、`FINAL_SHAPE_COMPARISON.csv`、`STEP_RESOURCE_NATIVE_LEDGER.csv`、`PHASE_RESOURCE_BUDGET.csv`：机器可读推导及每PC工作。
- `cg0_address_audit.py`、`derived/CG0_LANE_ADDRESSES.csv`：CG0地址枚举；`MATCHED_COUNTER_SUMMARY_FINAL.json`及原始24份NCU：指令族隔离/并行验证。
- 原始新证据：`raw_evidence`；远端目录 `/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_loop_cycle_verify_20261027_5h/`。目录中的失败轮、源码、PTX/SASS/CUBIN、Source/硬件CSV、Event/clock样本全部保留。
- 旧条件速率的独立审阅快照在`inputs/historical_reviews`，原始路径和hash列入`derived/FINAL_DELIVERY_DATA.json`；历史目标trace未覆盖。
- 所有候选均`model_enabled=false`、`effective_model_value=null`，不覆盖`K4理论分析.xlsx`，不修改FlashInfer固定源码。

公开口径：[NVIDIA NCU pipeline/WF定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines)、[NVIDIA PTX tcgen05同步与shared访问](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions)、[CUDA CC10.0吞吐表](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions)、[CUTLASS Blackwell相对Hopper吞吐](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)。
'''
(R/'K4_LOOP_CYCLE_ANALYSIS_CN.md').write_text(texts,encoding='utf-8')
print(json.dumps(dict(figures=3,drawio_nodes=17,comparison=m['main_range'],report='K4_LOOP_CYCLE_ANALYSIS_CN.md'),ensure_ascii=False))
