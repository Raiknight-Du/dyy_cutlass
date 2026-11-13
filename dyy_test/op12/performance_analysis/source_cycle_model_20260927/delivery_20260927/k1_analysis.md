# K1 source-driven cycle model

## Scope and source identity

K1 is the one-CTA context-parallel `T` precompute. The pinned `gdn_cp_prefill.py` imports `CPDeltaRuleTPrecomputeSm120` and wraps it as the SM100 specialization; the body and inverse helper analyzed here are the clean pinned source at FlashInfer commit `e77a4a0d276367895c3b50a642fd8f326c03fb72` (`../ablation_20260908/source_original`). No target timing or NCU evidence was used. The shared rate set is `B200_148SM_SOURCE_A_20260927` from the adjacent `hardware_model.md/json`.

Source chain: [gdn_cp_prefill.py](../ablation_20260908/source_original/gdn_cp_prefill.py#L17) imports the base and defines the thin subclass at lines 52–54. The host function forms the 64-token workspace and compiles/launches it at lines 128–180. The work body is [delta_rule_cp_sm120.py](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L80), and its 64×64 in-place inverse is [collective_inverse_hmma.py](../ablation_20260908/source_original/collective_inverse_hmma.py#L14).

## Workload and launch geometry

Assume `B=1`, `num_sab_heads=H=16`, `D=128`, 2-byte K/T elements, FP32 beta, and full sequence lengths divisible by 64. `num_k_heads` remains a separate source parameter: when it is smaller than 16, multiple SAB-head CTAs request the same K tile. This changes logical requests only through the number of CTAs; the HBM cache outcome is not implied by source.

| S | 64-token blocks/sequence | grid CTAs `N=H*S/64` | high-q SMs | high q | low q SMs | low q |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 128 | 2,048 | 124 | 14 | 24 | 13 |
| 16,384 | 256 | 4,096 | 100 | 28 | 48 | 27 |
| 32,768 | 512 | 8,192 | 52 | 56 | 96 | 55 |

The block maps `sab_head_idx=bx%num_sab_heads`, `k_head_idx=sab_head_idx*num_k_heads/num_sab_heads`, and `block_idx_in_seq=bx/num_sab_heads` ([delta_rule_cp_sm120.py:321–335](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L321)). The host grid is `(num_sab_heads*max_t_blocks_per_seq, num_seqs, 1)`, block `(128,1,1)`, with `min_blocks_per_mp=8` ([delta_rule_cp_sm120.py:290–306](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L290)). For these full-length cases, every grid CTA has a valid 64-token tile; the final partial-tile mask is inactive.

## Per-CTA DAG and source accounting

Byte counts below distinguish source logical global requests from explicit SMEM movement. A logical byte count is not automatically a byte count that reaches HBM: the no-cache HBM numbers later are a stated scenario. `FP16/BF16` source operands are both 2 bytes; `inverse_dtype` is explicitly FP16 in the base-class constructor.

| Node | Shape/work per CTA | Source bytes per CTA | Resources and predecessor/overlap | Source |
|---|---|---|---|---|
| `seq_decode` | Read `cu_seqlens[seq_idx:seq_idx+2]`; derive `seq_len`, block offset, tile id and `valid_len`; 0 FLOPs counted | 8 B logical global | Before both producers; shared sequence metadata can cache across CTAs | [delta_rule_cp_sm120.py:321–335](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L321) |
| `Kload` | `(64,128)` K tile, 8,192 elements | 16,384 B global TMA request; 16,384 B SMEM stage | Follows `seq_decode`. TMA producer is warp 1; overlaps the independent beta producer. One stage. TMA descriptor/issue/engine/response and mbarrier completion stay symbolic. | [delta_rule_cp_sm120.py:98–128](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L98), [252–289](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L252) |
| `beta_load` | `(64,)` FP32 beta | 256 B logical global scalar reads; 256 B SMEM stage | Follows `seq_decode`; warp 2 producer overlaps `Kload`. For H=16 each lane's token accesses are 64 B apart; 256 B is logical payload, not a transaction/HBM claim. | [delta_rule_cp_sm120.py:130–152](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L130) |
| `load_join` | Wait both one-stage pipelines, async-SMEM fence, CTA sync | 0 additional data bytes | Both producers must complete before `Wkk`. Preserve `L_TMA_response`, `L_mbarrier`, `L_fence`, and barrier/sync edges. | [delta_rule_cp_sm120.py:375–431](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L375) |
| `Wkk` | `(64,128) × (128,64) → (64,64)` FP32 accumulators; `2*64*64*128 = 1,048,576` FP16/BF16 tensor FLOPs | 32,768 B explicit SMEM operand loads (the K tile is read once as A and once as B); no new global bytes | After `load_join`; A/B `LdMatrix` copies precede `cute.gemm`; K pipeline is released after the GEMM. `R_HMMA=rho_HMMA*R16`, with `rho_HMMA=1` only as the approved optimistic assumption; this warp HMMA path is not evidenced equal to `tcgen05`. | [delta_rule_cp_sm120.py:433–462](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L433) |
| `strictlower_betaweight` | `(64,64)` accumulator; strict lower triangle has `64*63/2=2,016` FP32 multiplies; diagonal and upper values set to zero | 2,016 source-indexed beta references = 8,064 B SMEM-reference demand; only a 64-value/256 B beta vector is unique | After `Wkk`; uses `sBeta[row]`, then multiplies each retained accumulator by beta. Reuse of repeated beta values by generated code is unknown. | [delta_rule_cp_sm120.py:181–194](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L181) |
| `Aconstruct` | `(64,64)` FP16 inverse input; retain 2,016 strict-lower values; zero the other 2,080 positions | 4,096 FP32→FP16 converted elements; 8,192 B SMEM store | After beta weighting. The helper later replaces the diagonal with identity while loading rows. | [delta_rule_cp_sm120.py:154–179](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L154), [collective_inverse_hmma.py:41–55](../ablation_20260908/source_original/collective_inverse_hmma.py#L41) |
| `inverse8_gauss` | Eight independent 8×8 diagonal blocks. 56 conditional row-update terms/block = 448 multiply-add candidates = 896 FP32 FLOPs; 448 row-scale negations; 42 warp shuffle instructions. 512 half→float and 512 float→half conversions. | 1,024 B SMEM row loads + 1,024 B SMEM row stores | After `Aconstruct`; first inverse stage; then a 128-thread barrier. | [collective_inverse_hmma.py:26–73](../ablation_20260908/source_original/collective_inverse_hmma.py#L26), [378–386](../ablation_20260908/source_original/collective_inverse_hmma.py#L378) |
| `inverse16_blocks` | Four 8→16 calls. **Each** performs two M16×N8×K8 MMA operations, 2,048 FLOPs each. Total executed = 16,384 tensor FLOPs. The first operation's 16×8 accumulator is duplicated by a stride-zero broadcast; the second slice is computed even though the store writes only slice 0. 1,024 FP32→FP16 converted outputs, 512 FP32 negations. | 1,536 B operand SMEM reads + 512 B output SMEM writes | After the 8×8 barrier; then a 128-thread barrier. Counting only the mathematically unique 8×8 result would miss executed MMA work. | [collective_inverse_hmma.py:79–172](../ablation_20260908/source_original/collective_inverse_hmma.py#L79), [388–393](../ablation_20260908/source_original/collective_inverse_hmma.py#L388) |
| `inverse32_blocks` | Two 16→32 calls. Each performs two 16×16×16 GEMMs; total executed = 32,768 tensor FLOPs. 1,024 FP32→FP16 conversions and 512 FP32 negations. | 3,072 B operand SMEM reads + 1,024 B output SMEM writes | After the 16×16 barrier; then a 128-thread barrier. | [collective_inverse_hmma.py:177–251](../ablation_20260908/source_original/collective_inverse_hmma.py#L177), [395–401](../ablation_20260908/source_original/collective_inverse_hmma.py#L395) |
| `inverse64_block` | One 32→64 call across four warps. `−inv(D)@C`: 65,536 FLOPs; `@inv(A)`: 65,536 FLOPs; total 131,072 tensor FLOPs. 3,072 FP32→FP16 conversions, 1,024 FP32 negations and 1,024 FP16 cross-warp adds. | Four-warp operand copies: 12,288 B SMEM reads (D, C and A slices are each duplicated across the opposite warp coordinate); x=0 partial write 2,048 B; x=1 read 2,048 B and final write 2,048 B; total 18,432 B. | After 32×32 barrier. The two explicit cross-warp barriers surround partial store/reload/add; unknown barrier and FP16-add service is symbolic (numeric omission is 0 only in Scenario A). | [collective_inverse_hmma.py:257–370](../ablation_20260908/source_original/collective_inverse_hmma.py#L257), [403–404](../ablation_20260908/source_original/collective_inverse_hmma.py#L403) |
| `beta_sign_fold_Tstore` | Read `(64,64)` inverse, compute `T[row,col]=−beta[row]*inverse[row,col]`, write `(64,64)` output. 4,096 FP32 multiplies, 4,096 sign negations and 4,096 FP32→source-dtype conversions. | 8,192 B inverse SMEM read; 4,096 source-indexed beta refs =16,384 B SMEM-reference demand; 8,192 B logical global output stores | After inverse's final barrier; beta pipeline is released after the store. For full tiles, all 4,096 output positions are written. | [delta_rule_cp_sm120.py:196–236](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L196), [464–477](../ablation_20260908/source_original/delta_rule_cp_sm120.py#L464) |

The full source-visible logical global traffic is `D_global_logical=16,384+256+8+8,192=24,840 B/CTA`. The no-cache HBM scenario uses the same amount per CTA, explicitly assuming every source request reaches HBM and no transaction amplification. Actual HBM traffic can differ with K-head reuse/cache residency and beta transaction sectors. There is no T read or extra output initialization for these full tiles.

### Exact inverse work derivation

`CollectiveInverse().run()` makes 8 diagonal 8×8 Gauss calls, 4 calls to `blockwise_8x8_to_16x16`, 2 calls to `blockwise_16x16_to_32x32`, and 1 call to `blockwise_32x32_to_64x64` ([collective_inverse_hmma.py:371–404](../ablation_20260908/source_original/collective_inverse_hmma.py#L371)). Work is derived from the emitted source MMA tile shapes, not from the unique mathematical output alone:

| Level | Calls | Executed MMA work | Reason |
|---|---:|---:|---|
| 8×8 diagonal Gauss | 8 blocks | 448 FP32 multiply-add candidates = 896 FLOPs | `sum_{r=0..6} r*(8-r-1)=56` active updates/block; 8 blocks |
| 8→16 | 4 | 16,384 tensor FLOPs | Per call, two `M16×N8×K8` operations, each 2,048 FLOPs; the stride-zero broadcast duplicates M=16 and both result slices are computed |
| 16→32 | 2 | 32,768 tensor FLOPs | Per call, two 16×16×16 operations, each 8,192 FLOPs |
| 32→64 | 1 | 131,072 tensor FLOPs | Four warps collectively cover two 32×32×32 products, including two K-split partials and their reduction |

Thus blockwise inverse HMMA work is **180,224 FLOPs/CTA**, and K1's total warp-HMMA work is **1,228,800 FLOPs/CTA** (`Wkk` plus all inverse levels). Source scalar FP32 multiply/add FLOPs add 7,008 (2,016 beta multiplies + 896 Gauss FLOPs + 4,096 T-fold multiplies). Track sign operations separately: 6,592 FP32 negations and 1,024 FP16 adds. The 448 Gauss multiply-add expressions are counted as 448 FMA results for the numeric `R_scalar` scenario; if the compiler leaves multiply and add separate, scalar-result demand is 448 higher. These are distinct from tensor FLOPs.

## Residency from source and untimed compiled metadata

Launch is 128 threads = 4 warps/CTA. The supplied untimed clean pinned-source CUBIN metadata is 64 registers/thread, zero stack, 1,024 B static shared memory and zero local memory; it is structural compile evidence, not timing evidence. The static SASS has 76 `HMMA.16816.F32` sites, 2 `HMMA.1688.F32` sites and 48 LDSM sites. These are static sites, not dynamic execution counts; dynamic counts below come from source loop/tile accounting. Register demand is 64×128=8,192 registers/CTA, so the 65,536-register SM cap allows 8 CTAs. Warp capacity allows 16 and CTA capacity allows 32.

The source `SharedStorage` contains two 16-byte barrier arrays, a 64×128 2-byte K tile, a 64×64 2-byte inverse tile, and 64 FP32 beta values. Accounting for the 128-byte K alignment gives 24,960 B requested dynamic shared storage: 32 B barriers +96 B pad +16,384 B K +8,192 B inverse +256 B beta. Adding the supplied 1,024 B static allocation and CUDA's 1,024 B/CTA reservation gives 27,008 B/CTA; eight CTAs account for 216,064 B of the 233,472 B SM budget. Lead-supplied untimed `cudaOccupancyMaxActiveBlocksPerMultiprocessor` evidence at 24,960 B dynamic shared memory reports 8 active CTAs/SM, consistent with the 64-register/thread ceiling. This is structural compile/occupancy evidence only, not launch or timing evidence. The supported model ceiling is `b=8`. The source launch requests `min_blocks_per_mp=8`, consistent with this ceiling.

## Scenario A resource demands and CTA path

Rates are inherited unchanged from `hardware_model.json`: `R16=7,736.744377965752 FLOP/(SM-cycle)`, `rho_HMMA=1` (optimistic legacy-warp-HMMA assumption), `B_smem=128 B/(SM-cycle)`, `R_scalar=128 results/(SM-cycle)`, `R_cvt=64 elements/(SM-cycle)`, `R_warp_issue=4 warp instructions/(SM-cycle)`, `B_gpu=3,918.5750636132316 B/(GPU-cycle)`, and optional `B_share=26.47685853792724 B/(SM-cycle)`. No rate was fitted to target outcomes.

The following source-ordered CTA path keeps data dependencies and called source stages. Within independent per-element loops it allows an ideal resource pipeline; the node's resource demands still contend for one resource. Extra TMA, barrier, shuffle-dependency, MMA-response, SMEM-response, fence and drain tails are symbolic and set to zero only for the numeric Scenario A floor.

| Source-ordered node | Numeric service cycles | Service accounting |
|---|---:|---|
| `seq_decode` | 1 | 8 B / `B_gpu`, rounded as one source node |
| `Kload + beta_load` | 130 | Producers may overlap, but their combined 16,640 B SMEM write demand costs 130 cycles at `B_smem`; global HBM request floor is 5 cycles and does not dominate |
| `Wkk` | 392 | A/B shared loads 32,768/128 =256, then 1,048,576/R16 =136 |
| `strictlower_betaweight` | 63 | max(8,064 B beta references/128, 2,016 scalar multiplies/128); independent elements may pipeline |
| `Aconstruct` | 128 | 4,096 conversions/64=64, followed in source by 8,192 B SMEM write/128=64 |
| `inverse8_gauss` | 81 | 8 SMEM read +8 half→float convert; then 7 source rounds, each with 64 row-scale negates followed by `src_row` repetitions of 2 warp-shuffle instructions then `8*(7-src_row)` FMA results; finally 8 float→half convert +8 SMEM write. Each unrolled service node is separately integer-ceiled. |
| `inverse16_blocks` | 40 | 8 D/C SMEM input +2 MMA +4 sign results +8 conversion +4 A-inverse SMEM +2 MMA +8 conversion +4 output SMEM |
| `inverse32_blocks` | 58 | 24 SMEM input +3 MMA +4 sign results +8 conversion +3 MMA +8 conversion +8 output SMEM |
| `inverse64_block` | 218 | 64 SMEM D/C input +9 MMA +8 sign results +16 conversion +32 A-inverse SMEM +9 MMA +32 conversion +16 partial store +16 reload +16 final store; barrier and FP16-add extra service are symbolic |
| `beta_sign_fold_Tstore` | 195 | 64 inverse SMEM read +max(128 beta references/128, 64 scalar results/128,64 conversions/64) +3 global-store cycles |
| **Source-ordered CTA service path, global-HBM CTA-alone service** | **1,306** | Sum above; the K and beta producers overlap as their source branches permit. Extra response/issue tails remain symbolic; this is not a B200 timing prediction. |

A fully serialized in-node sensitivity, which does not pipeline the independent beta-fold/T-store element work, is 1,457 cycles before symbolic tails. The former `max(42/4,896/128)` Gauss accounting gives a 43-cycle isolated resource bound (and 1,268-cycle whole-CTA sensitivity), but is not a source-causal path because it places the shuffle and its dependent FMA pool in parallel. It remains only a relaxed resource bound; the finite cohort schedule below uses the round-by-round source order. Neither isolated span substitutes for shared resource service at grid scale.

Per CTA, source-indexed SMEM demand is 116,864 B: TMA K stage 16,384; beta stage 256; Wkk operand reads 32,768; strict-lower beta references 8,064; Aconstruct store 8,192; inverse levels 2,048 +2,048 +4,096 +18,432; final inverse read 8,192; final beta references 16,384. The beta vector has only 256 unique bytes, so if generated code reuses each staged beta value, an explicit lower-demand sensitivity is 92,928 B/CTA (726 SMEM cycles). The reference-count scenario is 913 SMEM cycles/CTA. This range is about source indexing/code generation; it is not a bank-conflict bound.

Other CTA aggregate work: 1,228,800 HMMA FLOPs (aggregate floor 159 SM cycles, per source MMA-node rounding 164 cycles); 14,336 converted scalar elements (224 cycles); 13,152 FP32 scalar-result proxy entries (103 cycles, with a +448-result uncontracted Gauss sensitivity); 304 source-dynamic warp HMMA instructions (76 issue cycles at four/cycle); and 42 warp shuffle instructions (11 issue cycles). The 13,152 entries comprise 2,016 beta multiplies +448 contracted Gauss FMA results +4,096 T multiplies +6,592 sign-negate results. Negations are tracked separately from FP32 FLOPs, but charged to this scalar-result proxy; exact lowering to bitwise instructions and any additional issue cost remain unknown.

## K1 multi-CTA floors and feasible ideal schedule

Use the balanced SM assignment listed above and `b=8`. For each SM, let `q_i` be its assigned CTA count. The reference-byte per-SM relaxed resource bound is

`C_i >= max(ceil(q_i/b)*L_CTA, ceil(q_i*D_smem/B_smem), ceil(q_i*F_HMMA/R_HMMA), ceil(q_i*E_cvt/R_cvt), ceil(q_i*E_scalar/R_scalar))`,

with `L_CTA=1,306`, `D_smem=116,864 B`, `F_HMMA=1,228,800`, `E_cvt=14,336`, and `E_scalar=13,152`. The legacy pooled-Gauss `L_CTA=1,268` is retained as a relaxed resource-only sensitivity, not a source-causal latency. The one shared GPU HBM queue contributes separately:

`C_GPU >= max(max_i C_i, ceil(N*D_HBM/B_gpu))`.

This charges one SM service budget across all resident CTAs. It does not multiply a CTA path by a grid wave count while leaving every CTA a full SM peak. With the stated no-cache logical-HBM scenario:

| N | q_max / b | `ceil(q/b)*L_CTA` | max-SM SMEM service | max-SM HMMA service | max-SM conversion service | max-SM scalar service | whole-GPU HBM floor | K1 ideal resource floor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 14 / 8 | 2,612 | 12,782 | 2,224 | 3,136 | 1,442 | 12,983 | **12,983 cycles** |
| 4,096 | 28 / 8 | 5,224 | 25,564 | 4,448 | 6,272 | 2,884 | 25,965 | **25,965 cycles** |
| 8,192 | 56 / 8 | 9,142 | 51,128 | 8,895 | 12,544 | 5,768 | 51,930 | **51,930 cycles** |

The whole-GPU HBM totals are 50,872,320 B, 101,744,640 B and 203,489,280 B respectively (`24,840*N`); they are a stated no-cache/one-request-one-HBM-byte scenario, not mandatory traffic. Under the separate optional equal-share HBM policy, the busiest SM's HBM service is 13,135 / 26,269 / 52,538 cycles and slightly exceeds its SMEM service. Keep that policy beside the universal whole-GPU HBM floor, not in its place.

The primary finite-grid schedule illustration is a synchronized per-SM cohort model, an explicit scheduling assumption also allowed by the shared model. For cohort size `k`, each phase aggregates the `k` CTAs' demand against one SM's single resource budgets; it never gives each CTA its own full rate. Source-sequential subphases are summed. Only independent source work within a phase uses `max` across resources. All unknown tails remain zero only in the numeric ideal scenario.

`C_batch(k)` includes: (1) `seq_decode` HBM service `ceil(k*8/B_share)`; (2) a whole-packet K/beta producer phase that sums HBM service `ceil(k*16,640/B_share)` and subsequent SMEM staging `ceil(k*16,640/B_smem)`; (3) eight sequential Wkk K passes, each `ceil(k*4,096/B_smem)+ceil(k*131,072/R_HMMA)` for LdMatrix then MMA; (4) beta fold; (5) conversion then SMEM store for Aconstruct; (6) all source-ordered inverse stages. For the 8×8 Gauss helper, `src_row=0..6` is expanded: each round charges 64 scalar-result-proxy negates, then each `i<src_row` charges two warp-shuffle instructions followed by `8*(7-src_row)` FMA results. Every such SIMT/ISSUE event is integer-ceiled and ordered before its dependent successor; dependent response tails remain symbolic. This is a conservative serial source schedule across these events, not a claimed hardware issue trace. The 16×16, 32×32 and 64×64 stages include the duplicated 16×8 HMMA slice, cross-warp partial IO and reduction; and (7) inverse read, beta/sign/conversion loop, then T HBM store. The whole-packet producer policy is conservative relative to a streamed HBM/SMEM overlap sensitivity; K and beta are independent producer branches but their combined traffic is charged against one shared HBM share and one SMEM budget, not two full rates. Table entries are cycles for the whole cohort. `C_SM,cohort(q)=floor(q/8)*C_batch(8)+C_batch(q mod 8)` and `C_GPU,cohort=max(max_i C_SM,cohort(q_i),ceil(N*D_HBM/B_gpu))`.

| k | `seq_decode` | `K+beta` | Wkk, 8 K passes | beta fold | Aconstruct | inverse 8 | inverse 16 | inverse 32 | inverse 64 | T fold/store | `C_batch(k)` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 759 | 392 | 63 | 128 | 81 | 40 | 58 | 218 | 502 | 2,242 |
| 2 | 1 | 1,517 | 784 | 126 | 256 | 113 | 78 | 114 | 434 | 1,003 | 4,426 |
| 3 | 1 | 2,276 | 1,176 | 189 | 384 | 174 | 116 | 170 | 652 | 1,505 | 6,643 |
| 4 | 2 | 3,034 | 1,568 | 252 | 512 | 208 | 154 | 226 | 868 | 2,006 | 8,830 |
| 5 | 2 | 3,793 | 1,960 | 315 | 640 | 271 | 192 | 282 | 1,086 | 2,508 | 11,049 |
| 6 | 2 | 4,551 | 2,352 | 378 | 768 | 308 | 230 | 338 | 1,302 | 3,009 | 13,238 |
| 7 | 3 | 5,310 | 2,744 | 441 | 896 | 370 | 268 | 394 | 1,520 | 3,510 | 15,456 |
| 8 | 3 | 6,068 | 3,136 | 504 | 1,024 | 402 | 306 | 450 | 1,736 | 4,012 | 17,641 |

| S | max q | cohort decomposition | high-q SM cohort cycles | low-q SM cohort cycles | whole-GPU HBM floor | cohort-model kernel cycle estimate |
|---:|---:|---|---:|---:|---:|---:|
| 8,192 | 14 | `C_batch(8)+C_batch(6)` / `C_batch(8)+C_batch(5)` | 30,879 | 28,690 | 12,983 | **30,879** |
| 16,384 | 28 | `3*C_batch(8)+C_batch(4)` / `3*C_batch(8)+C_batch(3)` | 61,753 | 59,566 | 25,965 | **61,753** |
| 32,768 | 56 | `7*C_batch(8)` / `6*C_batch(8)+C_batch(7)` | 123,487 | 121,302 | 51,930 | **123,487** |

This cohort schedule is conservative and does not claim to reproduce an unsynchronized GPU's best staggered CTA schedule. The separate relaxed resource floor above is lower; a fluid expression `L+(q−1)*II` based on aggregate work and average occupancy alone is not evidence of an attainable schedule, so it is retained only as a relaxed sensitivity and is not the primary K1 cycle result.

## Unknowns retained

The numeric ideal setting omits unsourced extra cost only; it does not erase edges. Keep descriptor/launch/dispatch; TMA issue, outstanding capacity and response; global HBM and SMEM response; MMA issue/accumulator response; shuffle dependency; conversion/FMA dependency; any sign-negate issue cost beyond the scalar-result proxy; all explicit mbarrier, fence, CTA sync and named barriers; FP16 cross-warp add service; and final store/drain visibility symbolic with no invented finite range. The source model counts 6,592 FP32 sign-negate results in the 13,152 scalar-result proxy, while separately flagging that some may lower as bitwise operations with different issue behavior. The helper contains three level barriers and two cross-warp barriers; the caller adds a barrier before and after `run()`. The launch also has pipeline waits/fences and initial/final release edges.

The K1 figures are source-driven scenario service floors and schedule illustrations only. No performance claim or fitted prediction is made.
