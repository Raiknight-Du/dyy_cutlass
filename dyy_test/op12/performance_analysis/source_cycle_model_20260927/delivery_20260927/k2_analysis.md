# K2 M/N source-only cycle model

Pinned source e77a4a0d276367895c3b50a642fd8f326c03fb72; B200 148 SM. Untimed, source-derived conditional resource calendar. No target kernel timing or NCU readings were inspected or fitted.

## Roles and block graph

The CTA has 12 warps: 0-3 CG0; 4-7 CG1; 8 transfer MMA; 9 TMA; 10 alpha; 11 state MMA. It processes L=C/64 tiles. The grid has 128 CTAs and is modeled as one balanced wave on 128 SMs (20 idle). Clean static metadata is 384 threads, 168 registers/thread (64,512/CTA), one CTA/SM and 512 TMEM columns.

Each block executes X=K*T (U=1,048,576 TC FLOP), Y=N*K^T (2U), M update (2U), and N update (2U). The first-block Y executes even though N was zero-initialized, so first block is 7U. Later blocks add Z=M*K^T (2U), hence 9U. CTA total is (9L-2)U. GMEM tensor payload is exactly 41,216L+131,072 B; two cu_seqlens int32 reads add 8 B separately in the HBM calendar. X and Y alias one TMEM accumulator; X materialization precedes Y overwrite, and next X waits prior N update.

The source helpers are packetized at 4,096 FP32 values / 16 KiB per packet: Ld16x256b/Repetition8 helpers use an inferred 64x64 logical view (X materialization, Z conversion, processY), while Ld32x32b/Repetition32 helpers use an inferred 128x32 view (M/N conversion, M/N scale, output store). Counts follow pinned helper partition loops; geometries are source-layout inferences, not clean-CUBIN/PTX proof. Packet counts are X/Z/processY 2 each and M/N conversion, scale, and output 4 each. In processY, V conversion and alpha accesses are separate resource packets after V LDSM and may overlap; FP32 math joins both. This allows a possible within-expression overlap without claiming same-warp issue timing; no cross-subtile streaming is modeled.

| L | TC FLOP/CTA | CVT elements | SIMT result slots | FP32 add/sub/mul FLOP | SMEM logical B | TMEM explicit B | GMEM logical B | Primary cycles | Serial-packet sensitivity cycles | Bglobal/128 sensitivity cycles | HBM GPU floor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 148897792 | 1024000 | 794144 | 924192 | 1740800 | 9551872 | 790528 | 46619 | 45587 | 46343 | 25823 |
| 32 | 299892736 | 2072576 | 1588288 | 1848384 | 3481600 | 18989056 | 1449984 | 83643 | 84435 | 84631 | 47364 |
| 64 | 601882624 | 4169728 | 3176576 | 3696768 | 6963200 | 37863424 | 2768896 | 157691 | 162131 | 161207 | 90447 |

processY has one multiply plus one FMA per output: 16,384 result slots but 24,576 semantic FLOP per 8,192 outputs, plus 8,192 V half-to-FP32 and 8,192 FP32-to-half conversions. Alpha has 64 log2 and 128 exp2 SFU calls; 10 scan shuffles, one carry shuffle, and one terminal-lane shuffle. Its 482 scalar result slots are 64 raw-plus-epsilon adds, 290 prefix/carry adds, 64 end-log subtractions, and 64 unary negations; ordinary FP32 add/sub FLOP is 418 and unary negation is reported separately. SFU/shuffle rates remain unknown and their numeric service is zero in the ideal calendar.

SMEM counts are logical source requests: per block K/V/T TMA writes 40,960 B; X shared write 16,384 B; V LDSM 16,384 B; per-token alpha 32,768 B; two coefficient vectors 1,024 B logical (32 B physical broadcast estimate); and AlphaProcessor 256 B raw write +256 B read +768 B writes. The primary does not subtract the broadcast estimate. TC operand feed is not charged again to generic SMEM/TMEM.

Corrected source edges: each block's X materialization waits the prior block's M scale; M coefficient reads wait on X materialization in block 0 and Z conversion-ready later; N coefficient reads wait on N conversion and Y MMA; processY's V/alpha inputs occur only after those prerequisites. N global output is not gated by the M done token; CTA completion waits for both outputs.

## Cycle interpretation

Primary calendar uses HBM=Bglobal/148 per representative CTA and integer ceil service per operation packet. TMA issue order is K,V,T per block, while independent staged packets may overlap: each packet has HBM then SMEM readiness, but the next packet HBM does not wait for prior SMEM completion. The serial-packet sensitivity adds that wait. Because the deterministic earliest-ready scheduler is a list heuristic, not a makespan optimizer, the extra ordering can change arbitration and can produce a numerically shorter span; the two values are distinct conditional scenarios, not monotone bounds. Unknown issue/response/barrier/SFU/shuffle/memory tails remain symbolic; their ideal numeric extra is zero. No scenario is a measured prediction or rigorous hardware lower bound.

The exact unrolled async-primary job and event calendars for L=16/32/64, with per-node shape, bytes/FLOP/result demand, dependencies, priority, start/end and source locations, are in k2_model.json. That file also contains serial-packet and HBM-share sensitivities and a separate whole-GPU capacity floor.

Source map: setup=gated_delta_net_cp.py:1331-1340; seq=gated_delta_net_cp.py:1152-1153; tma=gated_delta_net_cp.py:153-189,1460-1500; alphaLoad=gated_delta_net_cp.py:109-127,192-224; alphaProc=alpha.py:18-63; init=gated_delta_net_cp.py:227-258; scale=gated_delta_net_cp.py:260-293; store=gated_delta_net_cp.py:295-324; x=gated_delta_net_cp.py:326-374; convert=gated_delta_net_cp.py:376-431; zConvert=gated_delta_net_cp.py:433-488; y=gated_delta_net_cp.py:491-570; transfer=gated_delta_net_cp.py:573-671; state=gated_delta_net_cp.py:673-776; cg0=gated_delta_net_cp.py:777-849; cg1=gated_delta_net_cp.py:851-918; storage=gated_delta_net_cp.py:1032-1068,1181-1329; caller=gdn_cp_prefill.py:283-289; kernel=gated_delta_net_cp.py:1342-1517.
