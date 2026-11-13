# K4 — source-only CP final prefill DAG and cycle model

This report analyzes only the verified local source selected by the lead. It does not use measured target timings, old performance conclusions, or instruction-latency constants inferred from timing. Numeric rows are **IDEAL RESOURCE SCENARIO A**, with lead-approved common capacity assumptions; they are not exact B200 predictions.

The primary fully unrolled causal resource calendars give **57,560 /107,032 /205,976 modeled cycles** for L16/32/64 under the approved conservative source-service scenario (29.293 /54.469 /104.822 µs at the reference clock). These are conditional execution-plan results, with no measured target timings. They retain legal producer/consumer overlap and serialize only demands assigned to the same modeled resource. Section10 supplies the resource calendar and sensitivity details.

The machine-readable companion is [k4_model.json](./k4_model.json). Its 26 stage records each include the requested common schema: `serial / stage / cycles / substage / parallelwork / dependency / datavolume / resource / throughput / cycleformula / substepcycles / overlap / source / hardwareassumption / confidence`. The `serial` number is an inventory identifier; the legal topological schedule is given separately. Async instruction issue order must not be mistaken for completion order.

## 1. Scope, identity and target dispatch

Source `S` = [gated_delta_net_cp_prefill.py](E:/Users/27194/Documents/Cutlass/op12_flashinfer_gdn_b200_analysis_20260825/ablation_20260908/source_original/gated_delta_net_cp_prefill.py); SHA256 `7e7c0308004801adf71eff9494fb7a0b0c56fa986cee5c83d244f31f115c3ebc`.
Wrapper `W` = [gdn_cp_prefill.py](E:/Users/27194/Documents/Cutlass/op12_flashinfer_gdn_b200_analysis_20260825/ablation_20260908/source_original/gdn_cp_prefill.py); SHA256 `8dd9577f8b950f03f69ce22fdaf3b84acf8ef016c70d8d40c6cfdc9f2b2493b5`.
The pinned revision is `e77a4a0d276367895c3b50a642fd8f326c03fb72`, supplied by the lead and corroborated by its clean untimed compilation metadata. This subanalysis records source hashes and uses no historical kernel timings.

B=1, Hq=Hk=Hv=16, Dk=Dv=128, io=FP16 or BF16, FP32 state, P=8, C=1024/2048/4096. One CTA handles one head and one CP chunk. Grid is `(H*P,B,1)=(128,1,1)` [S824–866, S878–892]; L=C/64=16/32/64. All chunks are full and even in 64-token blocks. Pair loop count is L/2. There are no invalid/padded token blocks for these cases.

User initial and final state are enabled in the complete public pipeline. **The public K4 call nevertheless passes its own state-output argument as `None`** [W1096–1124], because K3 fixup receives `output_state=state` [W1103]. Therefore K4 loads an explicit initial/fixed state but does **not** write final FP32 state to GMEM. Treating user final-state enablement as K4 final-state-store enablement would misattribute 1 MiB of writes.

For CP index p=0, K4 loads the user initial-state workspace. For p>0, it loads `fixed_state[cp_chunk_idx-1]` [S2193–2215]. This is 64 KiB per CTA: 1 MiB from the initial workspace for 16 CTAs and 7 MiB from fixed-state entries for 112 CTAs.

`self.use_initial_state=True` is forced [S190–196], independently of `needs_initial_state`. For the target, the very first 64-token block executes KS and QS. There is no peeled first-block zero-state GEMM omission.

T is already the signed, beta-folded inverse operand [S1603, W70–81]. This K4 source only loads and applies a gate sandwich to T; it never computes Wkk or an inverse. The inverse/HMMA source is therefore outside this kernel's work ledger.

## 2. Twelve warps and storage

| Warps | Source role / strict per-block order | Registers per thread |
|---|---|---:|
| 0–3 | CG0: transformed T/Ainv, then gamma-scaled QK materialization | 224 |
| 4–7 | CG1: state publication, state decay, gate expansion, residual, QS scale, NV/Delta publication, O staging | 256 |
| 8 | QK issuer, two ordinary blocks per pair | 24 |
| 9 | TMA issuer: K → Q → V → T | 24 |
| 10 | tensor issuer: KS → QS → NV → Ointra → dS | 24 |
| 11 | four gate warmup tiles, then next gate → blocking O store | 24 |

These are the actual dispatch conditions [S201–238, S1245–1590]. The gate helper's old docstring says “warp10”; actual `gate_warp_id=epilogue_warp_id=11` and dispatch establish warp11. CTA=384 threads, one-CTA instruction group, occupancy target1 [S234–247]. Register allocation sums to 64,512 32-bit registers per CTA; this is a source allocation budget, not a measured occupancy claim.

SMEM stages are Q2, K3, V3, T2, Ainv3, QK2, O2 and gate5 [S293–314]. S/K/QS accumulators have the TMEM stages below. K transpose is a view over the same K allocation, not another K transfer [S1014–1019].

| TMEM column interval [start,end) | View / lifetime | Logical payload |
|---|---|---|
| [0,128) | FP32 recurrent S; old, decayed, then updated in place | 128×128, 64 KiB |
| [128,192) | FP32 QS, scaled QS, then final O | 128×64, 32 KiB |
| [192,256) | io-dtype undecayed state operand for KS/QS | 128×128, 32 KiB |
| [256,384) | two CG0 QK accumulator slots | logical64×64 FP32 per slot; 16 KiB payload each |
| [384,448) | one CG1 accumulator reused for KS then NV | 128×64 FP32, 32 KiB |
| [448,480) | fixed io slot0: VKS, then NV | 128×64, 16 KiB |
| [480,512) | fixed io slot1: Delta/decayV | 128×64, 16 KiB |

Offsets are computed in source [S316–336]. The QK producer constructs a **128×64 fragment view** for its allocation [S1754–1762], while the QK instruction is the native M=64 operation and the consumer only reads the logical64×64 result [S1811–1831]. The larger storage view does not imply a padded M=128 QK GEMM. TMEM column addressing/reservation is not the same quantity as logical payload bytes.

## 3. Exact arithmetic and instruction shapes

Use the source orientation: S[v,k], Q[t,k], K[t,k], V[v,t]. Let c_t=Σ_{u≤t}log2(alpha_u+epsilon), gamma_t=exp2(c_t), gamma_end=gamma_63. “io” denotes FP16/BF16 rounding. State stays FP32 between blocks, but KS/QS use a freshly rounded io state.

- State publication: `Sio=round_io(Sprev)`, before changing the FP32 state.
- T transform: for t≤s, `B_Ainv[s,t]=round_io(-exp2(c_s-c_t)*float(T[t,s]))`; otherwise zero. The store transposes the precomputed-T view [S1865–1896].
- QK raw: `W[s,t]=dot(Q[s,:],K[t,:])`.
- QK operand: `G[s,t]=round_io(W[s,t]*exp2(c_s-c_t)*scale)` for s≥t, otherwise zero [S1921–1937].
- KS: `KS[v,t]=dot(Sio[v,:],K[t,:])`.
- Residual/VKS: `R[v,t]=V[v,t]-round_io(gamma_t*KS[v,t])`, held in io dtype [S3028–3037].
- QS: `QS[v,t]=dot(Sio[v,:],Q[t,:])`; scale in-place as `QS_scaled=QS*gamma_t*scale` [S3048–3054].
- NV: `NV[v,s]=Σ_t R[v,t]*B_Ainv[s,t]`, FP32 output then io copy [S2590–2598, S3062–3070].
- Delta/decayV: `Delta[v,t]=NV_fp32[v,t]*exp2(c_end-c_t)`, then round to io. Delta uses the FP32 NV fragment before narrowing [S3073–3095].
- Ointra: `Σ_t NVio[v,t]*G[s,t]`; **tcgen05 accumulates it directly onto QS_scaled**, forming final O in the same FP32 QS accumulator [S2605–2617].
- State recurrence: first `Sdecayed=gamma_end*Sprev_fp32`; then `Snext=Sdecayed+Σ_t Delta_io[v,t]*K[t,k]` through the dS accumulating GEMM [S2972–2985, S2624–2633].
- Final O narrows once from the already combined FP32 accumulator and is staged/stored [S3103–3121].

There is no separate SIMD “O=QS+Ointra” add and no separate post-dS state-add stage: these additions are the MMA accumulator contracts. Matrix multiplication FLOPs use the usual 2MNK convention, including accumulating operations.

| GEMM | Logical M×N×K | Declared tcgen05 atom | Atoms/block | FLOP/block | A / B |
|---|---|---|---:|---:|---|
| QK | 64×64×128 | 64×64×16 | 8 | 1,048,576 | SMEM / SMEM |
| KS | 128×64×128 | 128×64×16 | 8 | 2,097,152 | TMEM / SMEM |
| QS | 128×64×128 | 128×64×16 | 8 | 2,097,152 | TMEM / SMEM |
| NV | 128×64×64 | 128×64×16 | 4 | 1,048,576 | TMEM / SMEM |
| Ointra | 128×64×64 | 128×64×16 | 4 | 1,048,576 | TMEM / SMEM |
| dS | 128×128×64 | 128×128×16 | 4 | 2,097,152 | TMEM / SMEM |
| Total | | | **36** | **9,437,184** | |

The atom is built with K=16 [S543–599]; reduction loops are S1770–1779 and S2548–2632. The fixed tilers come from W646–649. **There is no logical-to-declared tensor padding in these six K4 GEMMs.** The pair helper invokes two independent QK calls [S1999–2006] and two independent CG0 epilogues [S2021–2032]; it does not fuse them into a 128-token matrix multiply.

Keep QK shape efficiency eta64 symbolic. Source specifies M=64, not its throughput fraction. The primary common numeric scenario uses eta64=1. A separate sensitivity eta64=0.5 is an assumption motivated by partial datapath utilization, not a source-proven performance ratio. No warp×32 multiplier applies to tensor FLOPs.

## 4. DAG, joins and backpressure

The mathematical per-block DAG is:

```text
G_i ──┬─ T_i→T_sandwich_i ───────────────┐
      ├─ rawQK_i→QK_materialize_i ───────┼──────────┐
      │                                 │          │
S_i→S_io→KS_i→Residual_i→NV_i────────────┘          │
      └→QS_i→QS_scale_i──────────────┐              │
NV_i→NV/Delta_convert_publish_i─────┼→Ointra_i─────┘→O_stage_i→O_store_i
S_i→S_decay_i───────────────────────┴→dS_i→S_(i+1)
```

The real source adds substantial program-order edges:

1. CG0 `T_sandwich_i → QK_materialize_i → T_sandwich_(i+1)`. Ainv becomes available early, but a QK wait blocks the next T transform.
2. CG1 `S_io_publish → S_decay → gate_expansion → Residual → QS_scale → NV/Delta_publish → O_stage`. Even when NV is ready early, its read/cast stage follows QS scale.
3. Warp10 `KS → QS → NV → Ointra → dS` instruction issue order. KS waits both K and Q [S2552–2553]. Async tensor commands have completion barriers; issue order alone is not a host-style synchronous serial GEMM call.
4. CG1 next-block entry joins `O_stage_i` and `dS_i`: it must finish staging the current O and then wait the next recurrent state before publishing S_io.
5. Warp11 warms G0/G1/G2/G3, then at block i publishes G_(i+4) if present **before** waiting/storing O_i. Each O store ends with `cp_async_bulk_wait_group(0)` [S1539–1588, S3196–3200].
6. Full NV and Delta stores are interleaved by subtile, but both ready commits are after the complete loop [S3082–3099]. An ideal model may not invent early-subtile Ointra or dS consumption.

Buffer reuse edges:

| Buffer | Capacity | Producer cannot safely reuse until |
|---|---:|---|
| Q SMEM | 2 blocks | BOTH QK and QS consumers release the old stage |
| K SMEM | 3 blocks | BOTH QK and dS consumers release |
| V SMEM | 3 blocks | CG1 reaches the matching NV-result wait and releases V [S3058–3059] |
| T SMEM | 2 blocks | CG0 completes transformed-T store + named barrier, then releases T |
| Ainv SMEM | 3 blocks | corresponding NV MMA consumer releases |
| QK SMEM | 2 blocks | corresponding Ointra consumer releases |
| CG0 QK TMEM | 2 slots | corresponding CG0 QK epilogue reads/releases |
| CG1 KS/NV TMEM | 1 slot | KS read+multiply releases before NV acquisition; NV read releases before next KS |
| QS/O TMEM | 1 slot | QS-scale release permits Ointra; O_stage release permits next QS |
| FP32 S TMEM | 1 slot | CG1 state decay release permits dS; next read waits dS |
| Gate SMEM | 5 blocks | BOTH CG0 and CG1 release; four prefetched blocks remain in flight |
| O SMEM | 2 blocks | matching TMA store has actually completed |

VKS/NV/Delta fixed-input producers use `current_handle()+advance()`, with no ordinary acquire/empty wait [S3006–3007, S3078–3081]. Safe reuse is proven by preceding accumulator completion and source program order, as documented for state input [S2920–2925]. A model must retain those proof edges even though no explicit empty-stage wait appears.

All tensor work shares one tensor capacity budget despite two issuers. CG0 and CG1 scalar/copy work share SM scalar, conversion, SMEM and TMEM-copy resources. Warp9 and warp11/CG1 global transfers share the global memory capacity. Independent branch maxima are valid only after same-resource workload has been summed.

## 5. Data and non-MMA work ledger

Per64-token block, GMEM bulk payload is Q16 KiB +K16 KiB +V16 KiB +T8 KiB +alpha256 B +O16 KiB = **73,984 B**. Each CTA also reads **65,536 B** of FP32 state once. No beta load or intermediate NV/Delta global traffic occurs in K4.

Explicit TMEM-copy payload is:

```text
S read 64KiB +S_io write32KiB +S_decayed write64KiB
+QK read16KiB +KS read32KiB +VKS write16KiB
+QS read32KiB +QS write32KiB
+NV read32KiB +NV write16KiB +Delta write16KiB
+O read32KiB
=393216B =384KiB/block.
```

Separately, the six GEMMs nominally consume **98,304 B/block of SMEM operand feed** and **114,688 B/block of TMEM operand feed**. These are not new HBM traffic. Whether those feeds contend with the common modeled copy ports must remain an explicit sensitivity; simply adding them to peak-MMA time can double-charge throughput already embodied in the tensor peak. The primary serial stage model charges explicit copies and tensor service separately; the aggregate resource-floor ledger also exposes “include operandfeed” budgets.

Bulk TMA SMEM writes/reads and matrix copies total131,072 B/block. Source lane-request gate accesses add100,352 B/block:
CG0 T16,640 B +CG0 QK16,640 B +CG1 expansion66,560 B +gate-channel stores512 B.
Thus the nominal source-request SMEM ledger is231,424 B/block, or329,728 B if tensor operand feed is included.

**These gate bytes are logical lane requests, not proven physical shared transactions.** Shared broadcasts and compiler common-subexpression elimination may reduce repeated requests. CG1 generates128×64 gamma/decay values from64 distinct scalar pairs; its source has8,192 decay exp2 invocations, but the generated ISA may reuse them. The report never equates the expanded source expressions with measured physical bytes.

Nominal source arithmetic is82,338 ordinary scalar operations/block: **74,146 FP32 arithmetic operations and8,192 io-dtype residual subtractions**. The clean FP16 compiled specialization confirms packed `sub.f16x2`/HADD2 for the residual. These counts are also82,338 scalar arithmetic results because K4's nominal non-tensor source uses only adds/subtracts/multiplies, with no FFMA: KS multiplication is rounded to io before the io residual subtraction, preventing their fusion. R_scalar counts results (a hypothetical FMA would be1 result/2 FLOP), not official FP32 FLOPs. V staysio in its register fragment, so the source requires no additional V io→FP32 conversion. There are61,440 source conversion elements/block,12,480 SFU calls before possible compiler reuse, and11 gate shuffle instructions/block. Ordinary scalar counts exclude predicates, indices, pointer arithmetic, fences and barriers, which remain symbolic. Gate scalar counts are64 epsilon additions +258 active prefix additions +32 crosssegment additions=354. The single gate warp has13 source warp-add instructions and11 shuffles. K4 uses its own gate helper, with two register columns per lane:5 offsets×2 shuffle_sync_up calls=10, plus1 crosscolumn shuffle_sync. It does not import AlphaProcessor or use end_log. It cannot be asserted to sustain the entire SM's scalar peak; eta_gate/warp issue capacity remains symbolic.

The last full block still executes the final-block bounds predicate in CG0 and predicated gate-load path. Its effective tensor shapes and bulk bytes remain identical. Additional branch/predicate issue costs are unknown, not measured0.

## 6. Approved common scenario and per-stage numbers

Use f_ref=1.965 GHz, SM_count=148, dense FP16/BF16 peak2.25 PFLOP/s and HBM7.7 TB/s, as approved by the lead's hardware model:

```text
R16 =2.25e15/(148*1.965e9) =7736.744377965752 FLOP/(SM cycle)
B_HBM_gpu =7.7e12/1.965e9 =3918.5750636132316 B/(GPU cycle)
B_HBM_CTA_scenario =B_HBM_gpu/148 =26.47685853792724 B/cycle
B_SMEM=128 B/cycle; B_TMEM_copy=512 B/cycle
R_convert=64 elements/cycle; R_scalar=128 arithmetic scalar RESULTS/(SM cycle).
```

The equal-share HBM service is an **optional scenario assumption**, not an architecture requirement or coverage-scaling rule. The global HBM floor below uses aggregate bytes/B_HBM_gpu with no forced128/148 reduction.

Each service is ceil(work/rate). For consistency with transfer direction, each G2S input TMA packet receives HBM-read service then SMEM-write service; each S2G output TMA packet receives SMEM-read service then HBM-write service. Both services are summed within the conservative packet; different packets may overlap on distinct modeled resources. Tensor operandfeed is covered by tensor peak service in the primary case and charged to shared SMEM/TMEM only in a separate sensitivity. All unresolved SFU, shuffle, issue, response, fence, barrier and control costs are retained in formulas. The displayed numeric service values set those unresolved terms to0 **only for the ideal scenario**. The per-stage source-request numbers conservatively assume no gate-expression reuse. Matrix widening/narrowing is counted in conversion elements, including T's io→FP32 and FP32→io conversions; actual compiler conversion issue may differ.

| Inventory ID | Stage | Numeric service cycles | Source S lines |
|---:|---|---:|---|
| 1 | FP32 initial or fixed-state load and publish | 2604 | 2180, 2359 |
| 2 | Scalar alpha load | 10 | 1678, 1701 |
| 3 | Log2, two 32-lane inclusive scans, intersegment carry, exp2 | 3 | 1703, 1723 |
| 4 | Write both gate channels and commit | 4 | 1724, 1732 |
| 5 | K TMA | 747 | 2104, 2117 |
| 6 | Q TMA | 747 | 2135, 2146 |
| 7 | V TMA | 747 | 2164, 2175 |
| 8 | precomputed T TMA | 374 | 1604, 1623 |
| 9 | QK = Q K^T | 136 | 1754, 1782 |
| 10 | Signed T gate sandwich and transpose to Ainv | 435 | 1836, 1900 |
| 11 | Read QK, apply causal gamma and scale, publish QK operand | 371 | 1902, 1942 |
| 12 | Read old FP32 state, cast to io, publish state operand | 448 | 2917, 2940 |
| 13 | Scale old FP32 state by end gate and return to state accumulator | 256 | 2972, 2985 |
| 14 | Expand gamma and end/current decay ratios | 584 | 2911, 2915, 2986, 3004 |
| 15 | KS = S_io K^T | 272 | 2552, 2570 |
| 16 | QS = S_io Q^T | 272 | 2572, 2585 |
| 17 | VKS = V - round_io(gamma*KS) in fixed input slot0 | 480 | 3006, 3039 |
| 18 | Scale QS by gamma and scale in-place in FP32 | 256 | 3041, 3056 |
| 19 | NV = VKS Ainv^T | 136 | 2587, 2600 |
| 20 | Read FP32 NV, cast NV and Delta, publish fixed slots0 and1 | 448 | 3058, 3099 |
| 21 | Ointra = NV_io QK^T; accumulate on scaled FP32 QS | 136 | 2602, 2617 |
| 22 | dS = Delta_io K; accumulate on gamma_end*S_fp32 | 272 | 2619, 2634 |
| 23 | Read final O, cast and write O SMEM stage | 320 | 3101, 3121 |
| 24 | TMA O S2G, bulk commit and wait_group0 | 747 | 3163, 3200 |

Important substep sums (the JSON records every component):

```text
Gate load/scan/store =10+3+4 =17 +SFU/shuffle/issue terms
T transform =64 Tread +130 gate requests +49 arithmetic +128 conversion +64 Ainvstore =435
QK epilogue =32 TMEMread +130 gate requests +81 arithmetic +64 conversion +64 SMEMstore =371
S publish =128 read +256 conversion +64 write =448
S decay =128 arithmetic +128 write =256
CG1 gate expansion =520 source gate requests +64 arithmetic =584
Residual =128 Vread +64 KSread +64 mul +128 conversion +64 sub +32 VKSwrite =480
QS scale =64 read +128 arithmetic +64 write =256
NV/Delta publication =64 read +128 NVconversion +64 Delta mul +128 Deltaconversion +32 NVwrite +32 Deltawrite =448
O staging =64 read +128 conversion +128 SMEMstore =320
Initial S =2476 equalshareHBM +128 TMEMwrite =2604
K/Q/V TMA =619 HBM +128 SMEM =747 each; T TMA=310+64=374
O TMA store =128 SMEM read +619 HBM write =747
```

Native QK/NV/Ointra each take136 capacity-service cycles, KS/QS/dS each272; their sum is1,224 cycles/block. Ceil of the combined FLOP budget is1,220; use1,224 for the serial six-call scenario and1,220 as the unrounded aggregate tensor capacity floor.

Aggregate workload capacity floors per block are TMEM-copy768 cycles, source-request SMEM1808 cycles (2576 including tensor operand feed), conversion960 cycles, scalar644 cycles, and equal-share HBM2795 cycles. Integer rounding at each source packet gives645 scalar busy cycles and2796 HBM busy cycles per block in the calendar. These budgets sum work from both groups/issuers and must not be independently assigned to every branch.

## 7. First, steady, last and drain schedules

### Conservative source-order serial scenario

A deliberately conservative legal topological order is:
warm G0..G3; initial S load; for each block publish G_(i+4) if present, issue/complete K,Q,V,T, perform QK, transform T, materialize QK, publish/decay S, expand gates, KS,QS,residual,QS-scale,NV,NV/Delta,Ointra,dS,O-stage,O-store. It respects all source dependencies while disabling independent-warp overlap. It is **a source-compatible serial service estimate**, not the actual issue schedule or a guaranteed hardware upper bound, because unresolved latency costs are omitted numerically.

- Warmup excluding symbolic setup:4×17+2604=2672 cycles.
- First pair after warmup:2×8201=16,402 cycles; entry through first pair=19,074 cycles.
- Ordinary steady pair:16,402 cycles while incoming G_(i+4) exists.
- Last four blocks have no incoming gates:each8184 cycles. Each of the last two pairs=16,368 cycles.
- The last full block's additional predicates remain symbolic.
- O is drained in the same block; there is no extra “pending previous O” stage after the loop.
- Nonlast CP CTA then waits/releases its last state. Last CP CTA takes the source terminal-state helper and may incur128 cycles of syntactic TMEM read; see the optimizer caveat below. Pipeline tails and TMEM free remain symbolic.

```text
C_serial_nonlast
 =tau_setup +2604 +4*17 +(L-4)*8201 +4*8184 +all unresolved stage tails
 =tau_setup +2604 +8201L +all unresolved stage tails.

C_serial_last =C_serial_nonlast +128*theta_terminal_read.
```

eta64=0.5 adds136 cycles/block to this serial estimate; it does not change payload, atom count or algebra. The numerical serial totals for L16/32/64 are133,820 /265,036 /527,468 cycles, or133,948 /265,164 /527,596 with the syntactic terminal read. The half-efficiency sensitivity nonlast totals are135,996 /269,388 /536,172 cycles.

### Ideal conditional DAG floor

The legal overlap opportunities are S publication→KS/QS while CG1 decays S and expands gates, QS while residual is computed, NV while QS is scaled, and dS while O is staged. Warp8/CG0 and input/gate lookahead can progress independently while rings permit.

A max-plus relaxation with the above source-request stage demands gives the following earliest offsets from block-i state-publication start, **before arbitration for shared resources**:

| Event | Offset cycles |
|---|---:|
| S_io ready | 448 |
| S_decayed ready | 704 |
| KS done | 720 |
| QS done | 992 |
| CG1 gate expansion done | 1288 |
| Residual/VKS ready | 1768 |
| NV done | 1904 |
| QS scaled | 2024 |
| NV+Delta ready | 2472 |
| Ointra done | 2608 |
| dS done | 2880 |
| O staging done | 2928 |

The recurrence joins dS and O staging, giving a2928-cycle/block conditional floor. The first state publication cannot precede the2604-cycle initial-state service path. Last O store adds747 conservative packet-service cycles (619 HBM+128 SMEM) after its producer. Thus:

```text
C_relaxed_DAG_floor =2604 +2928L +747
C_ideal_scenario_floor =max(C_relaxed_DAG_floor, aggregate global HBM capacity floor,
                             summed capacity bounds for all shared resources).
```

This yields first-output stage+store lower bound6279 cycles, first-pair lower bound9207 cycles, and2928 cycles per steady block /5856 per pair. These are **conditional relaxed lower bounds**, not a full legal contention schedule. Shared-resource arbitration can only increase the model completion time for the assumed work. Compiler reuse can reduce the assumed work, so this floor is not an unconditional hardware lower bound. The unconditional model capacity floor based on bulk bytes is shown separately.

An actual attained cycle schedule cannot be recovered from source shapes alone: TMA/MMA response latency, queue arbitration, gate broadcast/CSE, issue restrictions and generated copy transaction structure are still missing. No independent maxima are presented as an exact achieved overlap.

## 8. Whole-grid totals and numerical comparison

These are kernel device-cycle scenarios; host launch latency and the other three kernels are outside scope. All128 CTAs may run concurrently on148 SMs with source target occupancy1. Do not multiply a per-CTA recurrence time by128. Whole-grid throughput floors use aggregate data/FLOPs.

| C | L / pairs | Whole-grid tensor FLOP | Bulk GMEM bytes incl initial states | Global HBM capacity floor | Conditional relaxed DAG floor | Primary causal packet calendar | Source-order serial |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 / 8 | 19,327,352,832 | 159,907,840 | 40808 (20.767 µs) | 50199 (25.547 µs) | **57560 (29.293 µs)** | 133820 (68.102 µs) |
| 2048 | 32 / 16 | 38,654,705,664 | 311,427,072 | 79475 (40.445 µs) | 97047 (49.388 µs) | **107032 (54.469 µs)** | 265036 (134.878 µs) |
| 4096 | 64 / 32 | 77,309,411,328 | 614,465,536 | 156809 (79.801 µs) | 190743 (97.070 µs) | **205976 (104.822 µs)** | 527468 (268.432 µs) |

Whole-grid bulk memory is128*(65536+73984L). This excludes tensor-map setup metadata, sequence metadata, cacheline overfetch and other control traffic, whose exact device transfers are not established by this source. Therefore the reported HBM bound is a bulk-payload floor, not an exact measured traffic count.

## 9. Boundary and compiler caveats

Only the16 CTAs with p=P-1 call `_store_final_state` [S1352–1369]. Its source has four terminal state TMEM read subtiles before the disabled GMEM store branch [S2417–2459]. **Clean untimed compilation resolves the target elimination:** the PTX retains the four `tcgen05.ld.32x32b.x32` statements at lines3792/3794/3796/3798, while the corresponding terminal SASS region at PC0x9510–0x95e0 (SASS lines2451–2467) has only barrier wait/release and no LDTM. The four SASS LDTM.x32 occurrences are solely the per-block state publication reads at PC0x7130/0x7150/0x7160/0x7170. Primary theta_terminal_read=0; no target terminal TMEM reads and no K4 final-state GMEM writes are charged. The source ledger retains the syntactic path and the separate standalone-enabled option.

A standalone call to `cp_delta_rule_prefill_dsl_sm100` with an output-state tensor enables a separate optional final-state GMEM write of64 KiB for each of16 last-CP CTAs (1 MiB total). That is not included in the target public pipeline ledger.

No cache-hit assumption, hardware barrier latency, SFU peak, actual instruction issue count after lowering, or measured timing has been invented. Shapes, native atom depth, source program order, ring capacities, data contracts and explicit payloads have high confidence. Port bandwidths, equal-share HBM, effective M64/SIMT efficiency, compiler reuse and zero-latency numbers are clearly labeled assumptions or unresolved terms.

## 10. Full causal packet resource calendar — primary numerical result

The primary calendar is now a complete unrolling of allL blocks, with **1,339 /2,667 /5,323 task nodes** for L16/32/64. The six shared modeled resources are HBM, SMEM, TMEM-copy, conversion, scalar arithmetic and tensor service. Every completed service task and zero-cost issue/control event has a named identifier, source lines, DAG predecessors, a previous same-resource job, integer work/rate service, start/finish times and symbolic unresolved costs.

Files:
[k4_calendar_L16.json](./k4_calendar_L16.json),
[k4_calendar_L32.json](./k4_calendar_L32.json),
[k4_calendar_L64.json](./k4_calendar_L64.json),
[k4_calendar_summary.json](./k4_calendar_summary.json),
and the reproducible builder [k4_calendar.mjs](./k4_calendar.mjs).

The resource calendar uses:

```text
ready_j =MAX(F_p for every source-DAG predecessor p)
start_j =MAX(ready_j,F_previous_job_on_same_resource)
duration_j =CEILING(work_j/rate_resource,1)
finish_j =start_j+duration_j.
```

Jobs are selected by earliest eligible service start, zero-cost events before positive work, then earliest dependency-ready time, then declaration order. This arbitration policy is an explicit scenario choice. It is not inferred from GPU timing, not claimed optimal, and not claimed to reproduce hardware queue arbitration. Once the resource order is chosen, its predecessor edges make every calendar time a transparent max-plus formula suitable for a spreadsheet.

Input TMA issue events carry warp9 program order and exact ring-reuse joins. Their transfer service is a separate HBM task followed by SMEM service. The next issue can happen before the preceding transfer completes, provided its destination stage is free. Output TMA packets correctly use **SMEM-read→HBM-write** service ordering, followed by the source wait_group0 completion and stage release. The preceding O staging makes the shared source ready before the store is issued. The previous v1 reverse packet ordering was not physically causal and has been superseded.

Initial FP32 state is represented as four source subtiles, each16 KiB GMEM→register followed by16 KiB register→TMEM. NV read/cast is represented by two source subtiles; its publication loop has the actual per-subtile NV-store →Delta-cast →Delta-store order and emits both ready commits only after the whole loop. Warp8 pair membership is recorded but does not insert an invented pair barrier or fuse the two QKs.

The calendar preserves source ordering, all finite ring capacities, the one-slot KS/NV accumulator, the one-slot QS/O accumulator and the recurrent FP32 S slot. It also preserves the proof edges behind ready-only fixed-input reuse. All DAG edges and previous-resource edges were checked; no service tasks assigned to the same resource overlap.

### Calendar first, steady and drain behavior

For all three lengths in the primary queue policy:

- G0 is ready at636 cycles; G4 is ready at6896 cycles.
- Initial FP32 S is ready at7545 cycles. Its busy service is only2604 cycles; the remaining delay is competing HBM work and source substep dependencies, not an invented fixed latency.
- The first O is completely stored at11,663 cycles.
- The first pair is completely stored at14,721 cycles.
- The second pair interval is6190 cycles. Subsequent interior complete-pair output intervals are6184 cycles. Interior state-publication and output-store intervals alternate3120 and3064 cycles.
- The penultimate pair interval is5801 cycles as future-gate/input work drains. The final pair interval is6112 cycles.
- The last output packet provides the final drain. The terminal state wait and pipeline tail have zero **numeric service** here, with their real unknown costs still symbolic and terminal state reads statically eliminated.

For the three tested L values, the fully unrolled calendar happens to simplify to:

```text
C_calendar(L)
 =14721 +6190 +(L/2-4)*6184 +5801 +6112
 =8088 +(L/2)*6184
 =57560 /107032 /205976 cycles for L16/32/64.
```

This finite-case simplification is derived from the explicitly unrolled calendars and is valid for these L values and this arbitration/service scenario. It is not an assumed general kernel-runtime law.

### Busy resource totals

| L | Unrolled nodes | HBM busy | SMEM busy | TMEM-copy busy | Conversion busy | Scalar busy | Tensor busy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1339 | 47212 | 28928 | 12416 | 15360 | 10320 | 19584 |
| 32 | 2667 | 91948 | 57856 | 24704 | 30720 | 20640 | 39168 |
| 64 | 5323 | 181420 | 115712 | 49280 | 61440 | 41280 | 78336 |

Busy cycles are sums of all work assigned to a resource. Completion can exceed a resource's busy time because packets can be unavailable, blocked by dependencies or waiting on other resources. The primary values already include same-resource contention between different warp roles; no independent branch receives its own copy of a resource peak.

### Sensitivities

All rate/work sensitivities below **freeze the primary per-resource job order**, then recompute the max-plus timestamps. This isolates the changed assumption and preserves monotonicity for a capacity increase. “Shared operandfeed” adds actual service nodes, so that separate case redispatches the expanded DAG.

| L | Primary source-request gate, HBM/148 | Hypothetical per-warp gate reuse | Same work, HBM/128 | eta64=0.5 | Charge MMA operandfeed to shared copy ports |
|---:|---:|---:|---:|---:|---:|
| 16 | 57560 | 55618 | 56032 | 58453 | 61824 |
| 32 | 107032 | 104018 | 104928 | 108941 | 116032 |
| 64 | 205976 | 200818 | 202720 | 209917 | 224448 |

The hypothetical gate-reuse case charges1024 B per CG0 T/QK gate phase (one set of64 FP32 values in each of4 warps),16 B for the end-gate broadcast, and2064 B for CG1 post-decay gate expansion. Including gate stores, this is4640 gate bytes/block versus100,352 logical lane-request bytes in the primary case. It assumes compiler register reuse and per-warp broadcast; it is **not measured or proved physical SMEM traffic**. Arithmetic service is retained at the source scalar count; SFU remains unknown rather than assuming all repeated expressions were eliminated.

The shared-operandfeed sensitivity adds98,304 SMEM bytes and114,688 TMEM bytes/block to explicit port service before each tensor packet. Its primary-case totals are61,824 /116,032 /224,448 cycles. This is an explicit conservative alternative to the primary peak-tensor convention, not extra HBM traffic and not silently double-charged in the main result.

The summary file also records calendars redispatched under altered gate/throughput assumptions. Such schedules change arbitration, so their differences cannot be interpreted as a guaranteed optimal runtime response to a changed bandwidth.

The clean captured compiled resources are REG168, STACK16, static_SHARED1024, LOCAL0. Static_SHARED excludes dynamic shared memory, and STACK16 alone does not prove register spilling. These untimed static facts corroborate specialization/terminal elimination; no target runtime or performance profile was consulted.


## 11. Direction correction and preserved v1 evidence

The prior v1 calendar and report are preserved in [superseded_k4_v1](./superseded_k4_v1/README.md) solely for audit. Its primary totals were57,274 /106,594 /205,234 cycles. That version placed output-store HBM service before its SMEM read, so it is superseded rather than retained as a feasible physical causal model. The v2 correction follows the S2G transfer direction and the source completion/slot-release dependency; it was made independently of any target timing. No measured timing was read or used to choose the result. Input G2S order, all source operation counts, data volumes and summed resource busy budgets remain unchanged.

The corrected primary calendars and their fixed-order sensitivities are regenerated by [k4_refresh_calendar.mjs](./k4_refresh_calendar.mjs). The source-builder still supports a direct reproducibility check through [k4_calendar.mjs](./k4_calendar.mjs).
