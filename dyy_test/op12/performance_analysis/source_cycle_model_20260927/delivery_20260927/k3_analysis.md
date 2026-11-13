# K3 source dependency and cycle model

Frozen inputs: unmodified pinned FlashInfer commit `e77a4a0d...`, original `gated_delta_net_cp.py` and `gdn_cp_prefill.py` in `ablation_20260908/source_original`. No experiment variant, Event timing, NCU trace, or existing empirical result was consulted. File SHA256 values are recorded in `k3_model.json`. Git full pin and clean-clone verification are held by the lead.

The base scenario is B=1, H=16, D=128, FP32 initial state present, FP32 final state requested, default UTC64 selection. Cases (S,C)=(8192,1024),(16384,2048),(32768,4096) all have P=S/C=8. K3 receives precomputed M/N matrices, so its source work does not depend on C or token count once P is fixed. This statement applies to the three diagonal cases, not all nine Cartesian combinations.

## Source facts and precise recurrence

`CPDeltaRuleFixupUtcmmaSm100` is L1519 onward. The original source has `m_stage=2` for rows64 (L1547–1549), not one. N has one stage. R=64, row_ctas=D/R=2, and grid is B*H*row_ctas=32 (L1542–1549, L1977–1983). There are 256 allocated threads, but payload roles use compute warps0–3, MMA warp4, TMA warp5; warps6–7 only reduce registers (L1550–1559, L2166–2229). Compute uses 120 registers/thread; others use32 as source requests, not measured register allocation.

The true recurrence for each row CTA is

```
A_p = TF32_round(S_p)                  # TMEM accumulator -> registers -> TMEM operand
Acc = N_p                             # SMEM -> registers -> FP32 TMEM accumulator
S_(p+1) = A_p * M_p + Acc              # UTCMMA TF32, FP32 accumulating result
FixedState[p] = S_(p+1)                # TMEM -> registers -> GMEM
```

This is the chained S_next=S*M+N algorithm, with each next partition requiring the preceding FP32 S result. M is provided through the transposed TMA view in host L524–528; the semantic multiplication above uses the kernel operand view, not an assumption that the Python physical row-major transfer buffer is untransposed. Accumulator type Float32 and operand type TFloat32 are explicit L1540–1541. MMA is `MmaTF32Op((64,64,8), CtaGroup.ONE, sourceA=TMEM, A_major=K, B_major=MN)` at L1905–1913. Each mathematical product is [64,128] x [128,128] -> [64,128], 2,097,152 FLOP. The nominal atom implies 2 N tiles and16 K tiles, i.e.32 atom-work equivalents; actual emitted instruction grouping/issue latency remains uncompiled and symbolic.

Host `_get_fixup_kernel` passes `store_initial_state=needs_initial_state` (gdn_cp_prefill L399–407). Therefore initial-enabled base also copies the original state into an FP32 initial-state workspace. Default selection uses16 parallel states and148 SM: SIMT4 threshold9, UTC64 threshold74, so16 selects UTC64 (host L544–556).

## Every source phase, shapes, traffic, and dependence

Let E=R*D*4=32,768 B, Q=D*D*4=65,536 B, and q=8,192 B per register/TMEM subcopy. Bytes below are logical request/service bytes; they are not automatically DRAM misses.

| Phase | Shape per CTA | Traffic | Work | Source dependency / source lines |
|---|---:|---:|---:|---|
| Pipeline setup + TMEM allocation | 256 columns reserved | SMEM barriers,256 TMEM columns | symbolic control | L2039–2073,L2152–2159 |
| Initial load | [64,128] FP32 | GMEM E -> registers -> TMEM E | no type conversion | L1626–1659,L1779–1787 |
| Initial workspace store | [64,128] FP32 | TMEM E -> registers -> GMEM E | none | initial load; L1712–1756,L1788–1797 |
| N issue/load p | [64,128] FP32 | GMEM E -> SMEM E | none | free one-stage N + preceding M issue; L1577–1606,L2206–2216 |
| M issue/load p | [128,128] FP32 | GMEM Q -> SMEM Q | none | N issue p + free two-stage M slot; L1577–1606,L2217–2227 |
| S operand conversion p | [64,128] FP32 -> TF32 | TMEM read E + TMEM write E |8192 conversions | previous fixed-state store finished; L1662–1709,L1821 |
| N handoff p | [64,128] FP32 | SMEM read E + TMEM write E | no arithmetic | conversion + N TMA completion + ready-slot acquire; L1822–1833 |
| Ready commit, N release | no matrix | barrier traffic symbolic | symbolic control | N handoff + TMEM fence; L1659,L1832–1833 |
| GEMM p | A[64,128],B[128,128],C[64,128] | internal tensor traffic separate |2,097,152 FLOP | M wait + ready wait + done-slot acquire; L1871–1884 |
| Done commit, ready/M release | no matrix | barrier traffic symbolic | symbolic control | MMA issued; UMMA pipeline completion/signaling establishes consumer wait; L1884–1886 |
| Fixed-state store p | [64,128] FP32 | TMEM read E + GMEM write E | none | done consumer wait; L1834–1844 |
| Done release | no matrix | barrier traffic symbolic | symbolic control | fixed store source sequence; L1844 |
| Final output + duplicate fixed store | [64,128] FP32 | TMEM read E + GMEM fixed write E + GMEM output write E | FP32->FP32 no numeric conversion cost modeled | final ordinary fixed store; L1845–1854,L1747–1756 |
| TMEM free / return | reserved allocation | control | symbolic latency | final compute role return; L2187–2188 |

Full 325-node, four-subcopy discrete DAG, per-node shape/bytes/FLOP, predecessor list, resource, formula, source location, assumptions, and confidence are in `k3_model.json`; the requested common fields are present on every node. Zero-duration control nodes in the numeric scenario have nonzero unknown symbolic formulas. No exp/log or other SFU work appears in K3; address/control scalar issue costs remain symbolic rather than fabricated.

Explicit TMEM-copy bytes exclude internal matrix-engine S/Acc traffic. Explicit SMEM service accounts for TMA writes and compute N reads; tensor-engine M operand reads are embodied in the tensor-service scenario and are not charged a second time as known SIMT shared loads. Treating these internal physical paths as the same128B/cycle SIMT shared path would require additional hardware evidence.

## Four subcopies, source order, and the272-cycle floor

Rows64 selects `St16x32bx2Op(Repetition(16))` and the corresponding load at L1643,L1677–1680,L1730. NVIDIA [PTX ISA8.7, Table49](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf) maps .16x32bx2 .x16 to16 32-bit registers per lane. Thus128 compute threads*16 registers=2048 FP32 values=8192 B per operation. Full state has8192 values, so four subcopies. Source loops over the remaining sub mode at L1651,L1692–1708,L1747.

Under the lead's SMEM128 B/cycle, TMEM512 B/cycle assumptions:

- Each N subcopy has SMEM service ceil8192/128=64, then TMEM store service ceil8192/512=16.
- Completion-serialized source subcopy schedule is4*(64+16)=320 cycles.
- The separate-engine, zero-issue, streamed packet lower bound is64+3*max(64,16)+16=272 cycles. SMEM subcopy i+1 may overlap an asynchronous TMEM store i. This assumes resource/issue overlap and is not proved by source performance.
- Each S conversion subcopy costs TMEM read16 -> convert ceil2048/64=32 -> TMEM write16, hence4*64=256 cycles completion-serialized.
- An ideal conversion packet floor is64+3*32=160 cycles, only if conversion and TMEM paths overlap across packets and TMEM reads/writes share their512 B/cycle calendar correctly. This is a floor, not a substituted source execution cost.

Same compute warps perform conversion, N copy, and store. Their source phases are sequential: convert ends before N copy, N copy precedes MMA wait, store ends before next conversion. A max(convert,Ncopy,store) is invalid. The floor max above is solely between distinct packet engines with explicit packet predecessors. PTX describes these TMEM accesses as asynchronous; .sync establishes warp participation, not guaranteed immediate transfer completion.

## Pipeline roles, stage ownership, and allowed overlap

Pipeline constructors (L2035–2069):

| Pipeline | Stages | Producer | Consumer | Readiness/release |
|---|---:|---|---|---|
| M PipelineTmaUmma |2| TMA thread1 | MMA thread1 | tx Q bytes; MMA releases after UMMA commit at L1886 |
| N PipelineTmaAsync |1| TMA thread1 |4 compute signaling participants | tx E bytes; compute release at L1833 |
| MMA-ready PipelineAsyncUmma |1|128 compute threads| MMA thread1 | compute commits after N copy/TMEM fence; MMA releases at L1885 |
| MMA-done PipelineUmmaAsync |1| MMA thread1|128 compute threads| commit tracks MMA completion; compute waits then stores and releases L1834–1844 |

TMEM allocation retrieval barrier has160 participants =128 compute threads +32 MMA warp threads (L1559,L2157–2159); reserved TMEM256 columns, accumulator offset0 and operand offset128. There is one accumulator and one operand; these are not P-deep state buffers. SMEM payload is2*Q+E=163,840 B, plus barriers and alignment; exact `SharedStorage.size_in_bytes()` requires the DSL struct layout and is not asserted from payload arithmetic.

For p>=1:

```
issueN[p] >= max(releaseN[p-1], issueM[p-1])
issueM[p] >= max(issueN[p], releaseM[p-2])    # second argument absent for p<2
Ncopy[p]  after convert[p], Nready[p], ready-slot acquisition
MMA[p]    after Ncopy[p], Mready[p], done-slot availability
store[p]  after MMA completion
convert[p+1] after store[p]                   # same compute group program order
```

The max in issue relations is a dependency join, not a replacement for shared-resource service. Producer warp source order is N[p] then M[p], then the next N acquire. Consequently, two M stages alone do not preload M0/M1 at launch: M1 cannot issue until N0 is released. N[p] release happens before the current MMA done wait/store, so N[p+1]/M[p+1] loads may overlap current MMA/store. Acquiring N[p+2] blocks on N[p+1] use; M acquire may also block on the slot released by MMA[p]. This is why a repeated max(TMA,compute) is not the model.

The source's done commit is before ready/M release. PipelineUmma/AsyncUmma hardware completion semantics protect TMEM/SMEM reuse; source-order call locations are not a claim that M is physically free at raw MMA issue time.

## First partition, steady partitions, last partition

Initial-enabled p0: setup/allocate; ordinary GMEM initial state -> TMEM; store initial-state workspace; convert that initial state; wait N0; copy N0 to accumulator; commit ready/release N0; M0 and ready join; MMA0; wait done; fixed-state0 store; release done. M/N prefetch can overlap initial load/workspace subject to actual memory arbitration. P0 runs one complete GEMM in this base case.

Steady p1..p6: previous fixed-state store -> S conversion -> N copy -> MMA -> fixed-state store. N release enables the producer to work ahead during current MMA/store. Next S conversion cannot begin before current store source phase finishes. The complete head comprises two independent row CTAs; there is no cross-row barrier or state handoff.

Last p7: the same conversion/N-copy/MMA/fixed-store chain, then another `store_acc(store_output=True)`. This rereads TMEM, rewrites fixed-state7, and writes output state. It does not replace the ordinary last fixed-state store. Producer has no next partition; then TMEM free, completion, and kernel drain.

No-initial alternative, documented separately: `start=1`; p0 loads N0 to TMEM and stores fixed-state0 with no conversion/M0/GEMM, then N0 release; p1..p7 run7 GEMMs. Initial workspace is omitted. It must not be mixed with the initial-enabled P8 model. If final output is omitted, remove the final repeated TMEM/fixed/output store path, not the ordinary per-partition stores.

## Byte/FLOP totals and grid coverage

Per CTA, base P8:

```
M requests     =8*65,536                 =524,288 B
N requests     =8*32,768                 =262,144 B
fixed stores   =8*32,768                 =262,144 B
initial read + workspace write          = 65,536 B
final duplicate fixed + output write    = 65,536 B
total GMEM requested                    =1,179,648 B
explicit TMEM copies                    =35*32,768=1,146,880 B
explicit SMEM+TMA service                =32*32,768=1,048,576 B
GEMM FLOP                               =8*2,097,152=16,777,216
FP32->TF32 converted elements            =8*8192=65,536
```

Per head:2,359,296 GMEM-request bytes and33,554,432 FLOP. Kernel:37,748,736 GMEM-request bytes and536,870,912 FLOP. Whole M is loaded by both row CTAs, so M request duplication is2x; eliminating only duplicate M misses would change the corresponding distinct-address byte count to29,360,128, but cache hits/writebacks are not assumed.

Grid32 on148 SM permits at most32 simultaneously active SM when one CTA is placed per SM; coverage32/148=21.62%. No numerical latency is derived by multiplying full-chip peak FLOPs with all148 SM while only32 CTAs exist. Independent per-CTA chains set kernel completion (max CTA finish + launch/dispatch/drain); splitting each head across two CTAs halves rows but duplicates full M loads. SM placement, cache sharing, memory distribution, and occupancy require hardware scheduling assumptions. The SMEM payload and TMEM reservation suggest one CTA per SM on the usual B200 capacities, but exact occupancy is not asserted from source alone.

## Approved symbolic rates and numerical schedules

The lead approved f_ref=1.965 GHz, SM=148, TF32 peak1.1e15 FLOP/s, global HBM7.7e12 B/s. These give

```
r_TF32 =1.1e15/(148*1.965e9)=3782.40836256 FLOP/SM/cycle
b_HBM_global=7.7e12/1.965e9=3918.57506361 B/cycle
b_HBM_equal_share=b_HBM_global/148=26.47685854 B/cycle  [scenario only]
r_effective_TF32=eta64*r_TF32
```

UTC64 half-datapath affects efficiency uncertainty. Primary ideal hardware scenario eta64=1 uses555 service cycles per matrix; eta64=0.5 sensitivity uses1110. Half datapath is not independently established to mean exactly half effective throughput. TMEM512, SMEM128 B/cycle, conversion64 elements/cycle, scalar128 ops/cycle are modeling assumptions. Scalar issue work is not enumerated from Python; its tail remains symbolic.

Unresolved latency symbols include TMA response, TMA issue/acquire, MMA response/issue, barrier commit/wait/release, TMEM allocation, launch, address/control instruction overhead, and memory-retirement details. Numeric scenario sets them to zero explicitly. With unknown tails retained, each DAG service is ceil(work/rate)+its relevant latency; dependency and resource calendars propagate the result.

The conservative numeric scheduler uses a separate calendar for HBM, SMEM, TMEM, conversion, and TF32. Each source register subcopy completes its services before the next subcopy; each whole TMA load consumes HBM service and then SMEM service. This intentionally serializes transfer retirement, yielding an admissible conservative schedule, not a TMA microarchitecture claim. Producers may overlap compute on distinct resources. Ready compute memory work has priority0, TMA work priority1; ties use stable node order. The JSON contains node starts/finishes and dependencies so this arbitration is reviewable.

| Numeric item | eta64=1 |
|---|---:|
| Initial copy, isolated |4*(310 HBM+16 TMEM)=1304 cycles |
| Initial workspace store, isolated |1304 cycles |
| N TMA, serialized HBM/SMEM |1238+256=1494 cycles |
| M TMA, serialized HBM/SMEM |2476+512=2988 cycles |
| S conversion |256 cycles |
| N copy |320 cycles |
| GEMM |555 cycles |
| Fixed-state store, isolated |1304 cycles |
| Final repeated fixed/output store |4*(16+310+310)=2544 cycles |

ceil8192/b_HBM=310 for register chunks, whereas ceil32768/b_HBM=1238 for full TMA N; this rounding difference is kept, not silently averaged.

The full admissible DAG scenario finishes at50,203 cycles=25.5486 us. Initial workspace completes at6290 cycles under concurrent memory contention. N0 releases at6866; MMA0 runs6866..7421; fixed-state0 completes11852. Steady fixed completions p1..p6 are17414,22976,28538,34100,39662,45224:5562-cycle intervals. The last fixed state completes47659; final drain is2544, giving50203. Eta64=0.5 yields50757 cycles=25.8305 us under the same arbitration. Contention means an isolated1304-cycle store can take longer on the timeline.

As an even more conservative entirely serialized work scenario, initial2608 +8*(1494+2988+256+320+555+1304)+final2544 =60,488 cycles. This is not a dependency optimum.

Resource lower bounds, distinct from either achieved schedule:

- Per-CTA equal-share HBM work ceil1,179,648/26.47685854=44,554 cycles; elementary-node rounded work44,592. The equal148-way share is a scenario allocation, not an architectural restriction despite grid32.
- Global-request HBM floor ceil37,748,736/3918.57506361=9634 cycles across the chip, only if those requests miss at HBM. This floor is conditional on physical DRAM byte volume; caching changes it.
- Per active CTA explicit SMEM work8192 cycles; explicit TMEM copy work2240; conversion1024; TF32 work4440 at eta64=1. Resource max is only a capacity floor and does not solve the recurrence.
- All source compute phases and store dependencies must also be considered in a critical-path bound. They are summed on the common compute group; only distinct predecessor/resource branches can use a max join.
-272 cycles is a local ideal N-copy packet floor. It is not a full partition latency and it does not prove M prefetch or the complete recurrence can keep pace.

Every numerical result is assumption-labeled source modeling. No cycles/peak calculation substitutes for the source DAG, and no fitted memory latency or empirical correction is used.

