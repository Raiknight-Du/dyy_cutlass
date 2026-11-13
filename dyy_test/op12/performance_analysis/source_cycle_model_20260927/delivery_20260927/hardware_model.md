# Common B200 source-driven resource/cycle model

Model ID: B200_148SM_SOURCE_A_20260927. Approval owner: lead agent. Scenario A constants approved by the lead before timing comparison. No target kernel timing, achieved-throughput, stall, or elapsed-cycle evidence was inspected to establish this model.

This is a source-ordered ideal-resource scenario with symbolic response and synchronization tails. Its integer cycle results are **scenario service floors**, not exact predictive B200 cycles. An assumed bandwidth is not a measured hardware limit.

## Evidence classes and scope

- **SOURCE FACT:** pinned original source, its instruction shapes, launch geometry, branches, loop counts, byte counts, and synchronization dependencies. Source identity supplied by the lead: FlashInfer commit `e77a4a0d276367895c3b50a642fd8f326c03fb72`, clean pinned tree at `../ablation_20260908/source_original`.
- **SOURCE FACT / DEVICE METADATA:** 148 enabled SMs and reference SM clock 1,965 MHz were supplied by the lead from device metadata. They are configuration inputs, not general B200 architectural guarantees and not timing fitting.
- **OFFICIAL HARDWARE FACT:** NVIDIA specifications or documented ISA/capacity semantics, with primary URLs below.
- **MODELING ASSUMPTION:** converting whole-card peaks into uniform per-SM service rates; issue/resource sharing; unknown efficiencies; and every unsourced bandwidth, conversion rate, or zero tail.

Scope is a single full B200 GPU with no co-running workload or MIG partition. A different device configuration requires another scenario; it must not silently reuse these constants.

## Official inputs and reference-cycle conversion

The NVIDIA Blackwell October 2025 datasheet, technical specifications on printed page 8, lists the HGX B200 GPU at 4.5 PFLOP/s FP16/BF16 and 2.2 PFLOP/s TF32 with sparsity. Its footnote makes dense performance one-half those values. It also specifies FP32 75 TFLOP/s and 180 GB HBM3e at 7.7 TB/s. These are peaks; TF32 is rounded. The GB200 columns have different numbers and must not replace the HGX B200 column. [NVIDIA Blackwell datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf)

Let `S=148`, `f=1.965e9 cycles/s`, `P16=2.25e15 FLOP/s`, `PTF32=1.1e15 FLOP/s`, `P32=75e12 FLOP/s`, and `BHBM=7.7e12 byte/s`. SI TB/GB and bytes are used. FMA counts as two FLOPs.

| Quantity | Scenario A value | Unit and attribution |
|---|---:|---|
| Dense FP16/BF16 tensor service R16=P16/(S f) | 7,736.744377965752 | FLOP per SM per reference cycle; derived modeling assumption |
| Dense TF32 tensor service RTF32=PTF32/(S f) | 3,782.408362561034 | FLOP per SM per reference cycle; derived modeling assumption |
| FP32 peak equivalent P32/(S f) | 257.89147926552505 | FLOP per SM per reference cycle; derived modeling assumption |
| FP32 FMA result equivalent P32/(2 S f) | 128.94573963276252 | FMA results per SM per reference cycle; derived modeling assumption |
| Whole-card HBM Bgpu=BHBM/f | 3,918.5750636132316 | byte per GPU per reference cycle; derived peak equivalent |
| Equal-share HBM Bshare=BHBM/(S f) | 26.47685853792724 | byte per SM per reference cycle; optional sharing assumption |

NVIDIA does not thereby document a 7,736.744-cycle-level instruction pipeline. Dividing a marketed whole-card peak by a selected reference clock is the scenario's accounting convention. Actual clocks and sustained rates may differ.

## Other common resource assumptions

| Resource | Numeric ideal setting | Symbolic form / scope |
|---|---:|---|
| Generic shared-memory traffic | 128 byte/SM-cycle | `B_smem`; aggregated demand of all resident CTAs on that SM |
| Explicit tensor-memory load/store traffic | 512 byte/SM-cycle | `B_tmem`; aggregated demand on the SM, not 512 per warp |
| Generic FP32 add/multiply/FMA | 128 results/SM-cycle | `R_scalar`; one FMA result is two FLOPs |
| Generic conversions | 64 scalar elements/SM-cycle | `R_cvt`; assumption, not a documented universal Blackwell converter rate |
| Warp HMMA FP16/BF16 | `rho_HMMA R16`, rho_HMMA=1 | optimistic equality assumption; not proven equal to tcgen05 throughput |
| Aggregate warp issue | 4 warp instructions/SM-cycle | ideal maximum availability of four schedulers; instruction eligibility/issue interval remains symbolic |
| TMA transport | no extra independently sourced rate | constrained by charged HBM/SMEM demand plus symbolic TMA engine/issue/response terms |
| SFU, warp shuffle/reduction, fences, barriers, allocation | no numeric exact rate/latency | symbolic; ideal omitted service/response cost is zero only for the numeric scenario |

The CUDA 12.9.1 programming guide documents 128 FP32 cores, four fifth-generation Tensor Cores, 16 SFUs, and four warp schedulers per compute-capability 10.x SM. Warp assignment is static among schedulers; each selects a ready assigned warp at an issue opportunity. Arithmetic throughput tables count scalar results per SM clock and distinguish conversions by category. Core count is not a proof that every source expression or packed conversion has the same service rate. [CUDA programming guide, architecture and throughput](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#compute-capability-10-x)

The service-floor convention is `s(v,r)=ceil(d(v,r)/R(r))`, applied to a real source node or homogeneous service batch. Do not round every scalar operation separately. A zero-work node has zero service. Avoid applying another per-CTA or per-warp multiplier to an already aggregated SM rate.

## Instruction and byte accounting

For a dense matrix instruction of shape M by N by K, charge `2 M N K` executed FLOPs, including source-required padding/masked tensor lanes that still execute. A tcgen05 instruction has a single issuer thread but operates on its full CTA or CTA-pair tile. A legacy `mma.sync` is warp-collective: one logical warp instruction already counts the complete warp tile; it must not be multiplied by 32 threads.

The PTX ISA documents TMEM as 128 rows by 512 columns of 32-bit cells in the SM100 CTA address view, allocated in power-of-two column counts in units of 32. Allocation may block. tcgen05 MMA/copy/commit initiation is single-thread; TMEM load/store is warp-collective with each warp accessing one quarter of the rows. MMA/load/store are asynchronous, and completion/visibility must follow the documented fences, waits, and barriers. M=64, CTA-group 1, non-ws is labeled half-datapath utilization; converting that label to a 0.5 throughput factor is a modeling inference. [PTX ISA 8.8, fifth-generation Tensor Cores](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions)

Use `eta_shape` explicitly when needed: `s_mma=ceil(F/(eta_shape R_dtype))`. Main optimistic scenario uses `eta_shape=1`; `eta_shape=0.5` is a separately labeled sensitivity for the documented half-datapath shape, not a verified timing law. A two-CTA instruction spanning two SMs must be attributed to its two SM resource domains; counting the full instruction work once on each SM doubles the GPU work incorrectly.

Keep these demands distinct:

1. `D_global_logical`: bytes requested by source global/TMA operations.
2. `D_HBM`: bytes that reach HBM under a stated cache scenario, including transaction/replay overhead if assumed. A mandatory-HBM-read estimate cannot be inferred from logical reads when cache reuse is possible.
3. `D_smem_explicit`: explicit SMEM loads, stores, staging and copies.
4. `D_tmem_explicit`: explicit TMEM load/store/copy bytes, using the instruction shape/container width.
5. Implicit tensor operand/accumulator traffic: keep separate until a bandwidth/interface model is stated; do not automatically charge tensor-internal traffic against the generic SMEM/TMEM port assumptions.

A conversion of E values is `ceil(E/R_cvt)`. A packed instruction producing two values counts two converted elements, while issuing one warp instruction. Generic scalar service uses scalar result counts, not a total FLOP count that makes an FMA cost twice an add by accident. Issued warp instruction demand is a separate resource and may dominate even when few lanes are active.

A bank-conflict/replay multiplier `gamma_smem>=1` may be added only from an explicit source access-layout analysis or a stated sensitivity. Shared-memory capacity says nothing about 128 byte/cycle bandwidth.

## HBM: global floor and optional equal-share scenario

The universal peak-bandwidth floor for a stated HBM traffic scenario is

`C_HBM_global = ceil(D_HBM_total / Bgpu)`.

All CTAs share this one GPU resource. **Do not divide 7.7 TB/s by SM coverage as a mandatory correction.** Fewer active SMs may still generate enough requests to drive multiple memory partitions. How much bandwidth they realize is not determined by SM coverage alone.

A separate equal-share model assigns each SM `Bshare=Bgpu/S`, and makes resident CTAs on that SM share that allowance. Its per-SM demand floor is `ceil(D_HBM_SM/Bshare)`. For m active SMs it assumes an effective aggregate allowance `m Bshare`; this is an explicit allocation policy, not an official hardware constraint. It must be labeled alongside, not substituted for, the whole-card floor.

An optimistic CTA-alone HBM service may use `D_CTA/Bgpu` as an additional relaxed floor. It cannot be granted independently to every concurrent CTA, because their aggregate demand must still respect `Bgpu`.

## Residency from source and compiled resources

CUDA 12.9.1's Blackwell tuning guide lists 64 resident warps/SM, 32 CTAs/SM, 65,536 32-bit registers/SM, and 228 KiB shared memory/SM for compute capability 10.0. A CTA may address 227 KiB because CUDA reserves 1 KiB per CTA. The default static limit is 48 KiB; larger dynamic allocations require opt-in. Cluster launches require cluster-aware occupancy. [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/blackwell-tuning-guide/index.html#occupancy)

For T threads/CTA, W=ceil(T/32), requested shared bytes H, and register allocation G after architecture/compiler allocation rounding, form the resource ceiling

`b_upper=min(32, floor(64/W), floor(H_carveout/H_alloc), floor(65536/G), other_launch_constraints)`.

Use the requested carveout and actual allocated shared bytes, including reservation and allocation granularity; `H+1024` is an explanatory estimate if exact allocation is unavailable. Registers and shared-memory allocations need their own rounding, not just `T*registers_per_thread`.

Only use source launch metadata, ptxas/static cubin resource metadata, and an occupancy query on the exact compiled original kernel to fix b. These are untimed structural checks. If registers or rounded allocation are missing, report `b_resource_upper` and bracket b; do not assert the upper ceiling is actual residency. Achieved occupancy from target timed profiling is excluded until freeze.

TMEM allocations/permits and cluster cooperation can constrain effective concurrency after CTA admission. Do not infer a TMEM residency cap solely by summing the per-CTA address-view capacities. Add a cap only with documented physical resource/permit behavior or exact untimed occupancy evidence. As one useful official software reference, CUTLASS describes the allocation permit as SM-level. [NVIDIA CUTLASS TMEM primitives](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html)

## Source dependency graph and unknown tails

Create nodes for source-visible work and edges for data dependencies, loop-carried state, explicit source ordering, reuse hazards, barrier/fence visibility, and completion waits. A node may claim several resources; parallel resource overlap is allowed only when the source ordering permits it. A `max` over unrelated whole-kernel totals is not the source critical path.

Keep the following response/latency terms symbolic, with domain `[0,+infinity)` and **no fabricated finite range**:

- `L_TMA_issue/engine/response`, descriptor/setup latency and outstanding-request capacity.
- `L_MMA_issue/response`, dependent accumulator response and outstanding tensor capacity.
- `L_SMEM_response`, `L_TMEM_response`, `L_HBM_response`.
- `L_FMA_dep`, `L_CVT_dep`, `L_SFU_dep`, SFU/shuffle/reduction service and response.
- `L_barrier`, `L_mbarrier`, `L_fence`, `L_commit`, `L_alloc/dealloc`.
- `L_CTA_dispatch`, launch overhead, and final store/drain visibility.

The numeric ideal service scenario sets unsourced **extra** tails to zero and ignores unsourced additional service demand. It preserves every dependency and known source operation. Zero tail does not erase its barrier edge, does not mean a physical barrier is free, and does not authorize overlapping a producer with its dependent consumer. Unknowns do not have confidence intervals without independent evidence.

If a node has service `s_v` and extra response tail `L_v`, its dependent consumer may start no earlier than `finish_service(v)+L_v`. Keep response tails separate from service occupancy so latency and throughput are not conflated.

## Warmup, steady-state and drain

For n identical iterations, let W be the source-derived fill, D the final drain, and I the steady-state initiation interval under the chosen resource allocation. A regular pipeline approximation is `C=W+(n-1) I+D`. Determine W and D from the source graph, including first operands and final completion/store edges.

The interval must satisfy both resource and recurrence bounds:

`I >= max(max_r d_iteration,r/R_r, max_recurrence cycle_latency/distance)`.

Buffer/stage count also bounds in-flight work. In a simple periodic issue/completion stream with k usable slots and response L, `I>=L/k`; this is a pipeline model assumption, not a general exact TMA formula. A two-stage source buffer does not prove complete latency hiding. Each claimed overlap must survive actual dependencies and reuse edges. For short n, evaluate the graph directly; steady-state formulas may conceal fill/drain dominance.

## K1: multiple resident CTAs must contend for resources

Let N be homogeneous CTAs, q_i CTAs assigned to SM i, b resident CTAs/SM, L the ideal isolated CTA graph span, and d_r the total per-CTA demand on a shared per-SM resource r. For fixed non-preemptive whole-CTA residency slots, a useful relaxed bound is

`C_i >= max(ceil(q_i/b)*L, max_r ceil(q_i*d_r/R_r))`.

Add the whole-card HBM floor separately:

`C_GPU >= max(max_i C_i, ceil(N*d_HBM/Bgpu))`.

Balanced assignment `q_i in {floor(N/S),ceil(N/S)}` is a scheduling assumption. If assignment is unknown, also report the all-GPU relaxed work floor `ceil(N*d_r/(S*R_r))`; neither this nor the balanced expression gives an exact launch schedule.

The common error is `L*ceil(N/(S*b))`, while every simultaneous CTA retains the full SM rate. That gives b copies of a single physical resource. With b tensor-saturating CTAs, each CTA must share tensor service; occupancy can hide response gaps, not multiply the SM tensor peak.

For a reviewable source-phase cohort scenario, suppose k CTAs enter a matching source phase p together. The phase's aggregate service is `S_p(k)=max_r ceil(k*d_p,r/R_r)` only when its resource demands are allowed to overlap. If they are source-sequential, sum them in graph order. Dependent response tails are then included by the source graph. Compute `C_batch(k)` from those aggregate nodes and the replicated CTA dependencies, not from a solo CTA duration. With q=ab+u per SM, the synchronized cohort scenario is

`C_i,cohort=a*C_batch(b)+[u>0]*C_batch(u)`.

Cohorts are a stated scheduling assumption and may lose opportunities to stagger phases. A more faithful ideal model admits CTAs up to b, releases a slot only at source exit, and schedules all ready nodes against shared per-SM resource queues plus the one GPU HBM queue. The next node starts at the maximum of predecessor readiness, required resource availability, and CTA admission. Replication does not replicate hardware. No target timings are needed to construct this schedule.

## Sensitivities kept out of the main scenario

Old `8192 FP16/BF16` and `4096 TF32 FLOP/(SM-cycle)` are explicit alternate assumptions. At the 1,965 MHz reference they imply 2.38239744 PFLOP/s and 1.19119872 PFLOP/s, respectively: 5.8843% and 8.2908% above the chosen published peaks. Old `32 byte/(SM-cycle)` GMEM implies 9.30624 TB/s across 148 SMs. Old `8 TB/s` HBM is also a separately labeled alternate; none replaces Scenario A.

Sensitivity parameters are `rho_HMMA`, `eta_shape`, `B_smem`, `B_tmem`, `R_cvt`, `gamma_smem`, b, cache/HBM traffic policy, and symbolic tails. Report changes by changing one named assumption. Do not tune them to target outcomes.

Freeze rule: the lead records the model ID/constants, kernel source counts and graph, traffic scenario, residency bracket, and scheduling policy before reading timing evidence. Any later revision must retain the old model and disclose the changed assumption and independent reason.

