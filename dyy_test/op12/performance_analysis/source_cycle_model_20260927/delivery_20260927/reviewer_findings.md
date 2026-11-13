# Independent source-model audit

Scope: pinned original e77a4a0d276367895c3b50a642fd8f326c03fb72; B1/H16/D128; paired (S,C)=(8192,1024),(16384,2048),(32768,4096); P8. This reviewer read source, model/code, and untimed static metadata only. No target timings, achieved throughput, stall profiles or fitted constants were inspected. Findings below refer to the files as inspected before the lead's corrections; the lead is regenerating K2.

## Findings requiring source-graph corrections

### R1 — Major: K2 compute-role phase work starts before its source phase

Evidence: gated_delta_net_cp.py:790–847 initializes M then, in each CG0 loop, materializes X, optionally converts M/Z, reads block_coeff, waits M update and scales M. Lines 874–916 initialize N then, in each CG1 loop, convert N, wait Y, read block_coeff, run processY, scale N and wait N update. k2_calendar.mjs instead gives m_coeff and n_coeff only alpha readiness, process_y V/alpha reads only V/alpha readiness, and no preceding CG0 M-scale edge on next X materialization. These are same-role operations, not independent producers.

Correction: add the missing prior-role/phase edges before list dispatch; preserve independent warp roles and TMA asynchronous issue. At minimum, X materialization follows prior M-scale, m_coeff follows current X materialization/current Z-convert, n_coeff follows Y completion, and processY reads follow the phase entry after Y/coeff readiness. Preserve helper subcopy data dependencies. **Primary numeric results must be regenerated.** A review-only replay with those edges plus R2, unchanged demands/rates/arbitration, changes K2 43405/76429/142477 to 47731/86003/162547 cycles. These are diagnostic intermediate values, not proposed final numbers: they precede the lead's final-store and helper corrections.

Status: notified lead; implementation correction in progress.

### R2 — Major: K2 alpha producer iterates before finishing the preceding alpha block

Evidence: gated_delta_net_cp.py:205–223 calls load_alpha_block, AlphaProcessor.run, mask, fence and commit in one warp before entering its next loop iteration. k2_calendar.mjs chains alphaIssue only between zero-duration issue nodes. Its subsequent raw load/scan/math/store work for multiple blocks may overlap across resources despite one synchronous source warp.

Correction: next alpha issue follows preceding alpha ready/commit (including the retained symbolic tail); retain stage-reuse edges. **Primary numeric results must be regenerated** together with R1; no rate or source-demand change is needed.

Status: notified lead; implementation correction in progress.

### R3 — Major: K1 Gauss pool order does not represent its shuffle/FMA recurrence

Evidence: collective_inverse_hmma.py:59–70 interleaves row_scale negation, shuffle of a fragment, and dependent active-lane multiply-add within src_row=0..6. k1_model.json synchronized_cohort_scenario inverse8_gauss lists the entire 896-scalar-result pool before the entire 42-shuffle-issue pool. The whole-pool order does not preserve shuffle-to-FMA RAW or the fragment recurrence. The isolated max(shuffle service, scalar service) similarly serves only as a relaxed resource bound.

Correction: unroll the seven src_row rounds, and preferably each dependent shuffle/FMA step, with known service demand and symbolic response dependencies. Eight diagonal blocks give FMA demands 8*s*(7-s), 64 source negates per round, and 2*s warp shuffles per round; retain source lane predicates and distinguish sign-lowering assumptions. A serial legal schedule or a documented independent-block overlap schedule is acceptable. **K1 primary numeric results must be regenerated or the existing result must be relabeled as a resource-accounting relaxation rather than a source-causal execution plan.** Numeric coincidence after graph repair does not invalidate the finding.

Status: notified lead; correction pending.

### R6 — Major: K2 whole-helper pools hide source register-tile order

Evidence: gated_delta_net_cp.py:358–372,413–429,469–486,543–570 each iterate register tiles and execute their read/conversion/math/write chain before advancing the source loop. The inspected K2 calendar instead has one whole-matrix read node, one whole-matrix conversion node, and one whole-matrix write node, with processY V/alpha work in independent global pools. This batching changes arbitration/readiness and is not the original helper instruction order. It is distinct from the missing outer-role phase edges in R1.

Correction: derive register tile count from the declared TMEM copy and partitioned shape; preserve per-tile RAW and issue order, with response/service overlap only where explicitly assumed. For these shapes the128x64 path has two register tiles and128x128 conversion path has four. A separately labeled aggregate-helper surrogate can be retained as a sensitivity. Regenerate K2 primary values after tile subdivision; the R1/R2 diagnostic replay is not a full corrected primary result. Status: notified lead; helper correction requested. Primary numeric results must be regenerated.

### R4 — Minor: K2 final output graph invents an M-store-to-N-store barrier

Evidence: gated_delta_net_cp.py:913–917 stores N before done_consumer.wait. k2_calendar.mjs n_output_gmem instead depends on m_done_token, which is downstream of M HBM store. The source's done token is a final CTA join, not a prerequisite to N's global store.

Correction: remove done from N-store prerequisites; retain final CTA join of both stores. The extra edge can be a deliberately serialized scheduling policy, but must not be called a source dependency. Regenerate alongside R1/R2 because arbitration can change completion. Status: lead independently confirmed and requested patch.

## Counts and limits checked without a new arithmetic finding

- K1 Wkk=2*64*64*128=1048576 tensor FLOP; inverse levels 16384+32768+131072=180224; total1228800/CTA. The stride-zero 16x8 inverse executes both accumulator slices; counting only unique mathematical elements is wrong. Logical global total24840/CTA includes8-byte cu_seqlens metadata. 64-register/128-thread compiled metadata and untimed occupancy8 agree with source shared payload and resource ceiling. Cohorts remain an explicit admission/assignment policy; resident CTAs share one SM budget.
- K2 first block runs X/Y/M-update/N-update=7U even though N starts zero. Later add Z=2U, giving(9L-2)U. Final M/N are FP32,65536 bytes each. No extra K-transpose TMA exists. Known fixed false TMA completion edge is removed from primary and serial-packet alternative is disclosed. List dispatch may produce shorter spans with added constraints; that is not a monotone optimized hardware bound.
- K3 P8 initial-enabled model runs8 TF32 GEMMs of2*64*128*128=2097152 FLOP/CTA; grid32 is two row CTAs per head. Initial-state workspace is an extra store. Final source store_acc(True) repeats both final TMEM read and fixed-state store, plus output-state store; per-CTA logical global total1179648 bytes agrees. No-initial and no-final branches must remain separate. The recurrence includes fixed-state store before next conversion. N1-stage/M2-stage prefetch and releases are documented. Conservative subcopy serialization is an admissible explicit schedule policy, not a known transfer latency.
- K4 six GEMMs are U,2U,2U,U,U,2U=9U/block. QS/O and state additions are accumulator operations, with no additional standalone FP32 adds. Public wrapper passes K4 final state=None while K3 owns final output. K4 initial/fixed read65536/CTA and block bulk73984 bytes agree. Source terminal unused reads are distinguished from static DCE evidence; no final-state HBM store is charged. Corrected S2G nodes read SMEM before HBM write. NV/Delta commits occur after both full publication loops; next-state and reuse edges are present. eta sensitivity uses logical_flops in normalization, avoiding double division.
- Explicit TMA destination writes and explicit TMEM copies are separate from tensor FLOP service. Tensor-internal operand feed is not silently added to generic copy ports; K4 provides a separate sensitivity.
- Workbook assumptions and FLOP formulas agree with source counts, including K3 row CTA multiplier and K1 inverse. Work-and-Traffic bulk totals intentionally omit separately disclosed metadata; this does not make them physical HBM byte totals.

## Physical uncertainty, distinct from arithmetic/source bugs

R5 — Major uncertainty retained, no mandatory numeric change: Bgpu/148 per active SM is an allocation assumption. It is not established by a grid of32 or128 CTAs; the global peak bound is separate. The divisor32/128 sensitivities expose substantial effects. Shared/TMEM/conversion rates, legacy HMMA equality, shape eta, cache misses/transactions, eligible-warp issue, barrier/SFU/response and drain tails remain assumptions or symbolic unknowns. No finite physical tail range can be supplied from this audit. Source-causal list schedules are conditional plans, not unique strict hardware lower bounds.

No target timing reader or fitting operation was found in the inspected normalizer, calendar builders, workbook builder or static audit script. Model inputs are source/model JSON and static binary text. This establishes inspected code provenance, not a claim about inaccessible historical activity.

Audit outcome: no Critical count/dtype error found. Four Major source-graph findings require correction before freezing the primary model; one Minor final-store ordering finding should be corrected in the same rebuild. Known fixed issues were not reopened as arithmetic defects.

## First post-correction review

K2 regenerated primary50758/92294/175366 cycles and2065/4129/8257 nodes now contain the requested CG0/CG1 phase edges, synchronous alpha-producer chain, two/four-register-tile helpers and independent final stores. R1/R2/R4 are implemented; R6 tile ordering is implemented, with inferred tile geometry remaining explicitly labeled until shape metadata is corrected. The four tiles for scale/store naturally use128x32 rather than64x64 logical geometry; bytes/service remain16384 per FP32 tile.

Three residual edge details were sent to the lead: CG0 first X materialization must follow M initialization; M-update issue must not claim an invented coefficient-load dependency across warp roles; M-input release follows Z-ready consumption rather than preceding Z-convert. The latter has no currently observed enabled reuse hazard, but the named release event must retain source meaning.

### R7 — Major: K2 AlphaProcessor misses epsilon adds and batches a dependent scan

Evidence: local copied source alpha.py:37–38 has64 `raw+1e-10` FP32 additions before log2. The model's418 Alpha scalar result slots comprise258 scan adds+32 carry adds+64 subtract+64 negate, excluding these64 epsilon additions. Correct Alpha count is482 result slots and418 conventional FP32 add/sub FLOP, excluding unary negates and SFU. This changes K2 total scalar-result/FLOP counts by+64L.

The same helper at alpha.py:40–49 interleaves five shuffle offsets and dependent add rounds, followed by carry; whole11-shuffle-then290-add pools do not preserve scan RAW. alpha.py:60–63 has subtract before exp2 then negate; a combined subtraction/negation node after all exp2 also gives incorrect symbolic response dependencies.

Correction: add epsilon arithmetic before log2; unroll scan rounds with2*(32-off) active add results for off1,2,4,8,16, then32 carry adds. Preserve shuffle-to-add and carry RAW. Split channel2 subtraction→exp2→negate, including source stores or their explicitly justified aggregate phase. Do not infer physical SFU timing. Status: lead notified; correction pending. Primary counts and calendar must be regenerated.

## K1 correction accepted

The repaired inverse8_gauss has53 source-ordered substeps: initial read/conversion,7 rounds of64 negates,21 two-warp shuffle events each followed by its dependent FMA, then final conversion/store. Independent sums recover448 FMA results,448 negates and42 warp-shuffle instructions. Every recorded predecessor is the preceding source substep; no whole scalar-before-shuffle pool remains. Regenerated cohort kernel values30879/61753/123487 are accepted for the stated synchronized-cohort scenario. R3 is resolved, with unary-sign lowering, FMA contraction, warp issue and dependent response rates still explicitly conditional.

K4 gate preprocessing may remain an opaque helper node with bundled scan/SFU/issue response tail: its consumers see only complete gate publication and no exposed intermediate incorrectly ordered readiness. The3-cycle aggregate scalar service is a conditional helper surrogate, not an exact internal single-warp instruction path. Thus no mandatory K4 scalar-scan unroll or numeric revision was requested.

## Final focused review — accepted

The final K2 calendar46619/83643/157691 cycles with2577/5153/10305 nodes passes independent predecessor/resource-exclusion checks. Direct sums of all TC16, TMEM, SMEM, CVT, SIMT and HBM demands match the declared source totals in each case. The Alpha block now has482 scalar results and418 conventional FP32 FLOP; the added64 epsilon operations, dependent10 scan stages plus carry, end-log subtraction, exp2, negation and channel publication match the inspected helper source. The producer cannot enter its next block before publication completes.

CG0 first X materialization now follows initialization, subsequent materialization follows prior M-scale, M coefficient reads follow current X/Z phase but are not forced to wait M-update completion, and M's separate issuer has no coefficient prerequisite. M-input release now follows Z-ready consumption. N final output has no M-completion prerequisite; CTA completion joins both finished outputs. Register-tile ordering and labels are repaired. R1/R2/R3/R4/R6/R7 are resolved; R5 physical uncertainty remains explicitly retained. **No additional primary numeric change is required.** Freeze/export must use these final calendars and corrected counts.

The shared ISSUE=4 setting is a K1 shuffle-service proxy only. This calendar does not enumerate/enforce all frontend issue, active/predicated warp instruction granularity, conversion eligibility or other instruction classes. Gauss has known42 FMA warp instructions and42 shuffle warp instructions, but adding a full global issue calendar requires a complete instruction-demand ledger. Its missing service/response remains symbolic; no exact dependent FMA latency is fabricated. That scope is valid for the stated incomplete physical conditional model and cannot support a claim of a complete hardware instruction-issue model.

ProcessY V conversion and lane-alpha reads may use an explicitly assumed ideal overlap on distinct modeled resources once their source phase is entered; arithmetic joins both. Exact compiler instruction interleave and same-warp eligibility remain unknown. K4's opaque gate-helper abstraction likewise retains internal instruction uncertainty. These limitations do not reopen source count or mandatory primary numerical corrections.

Final accepted primary cycles: K1=30879/61753/123487; K2=46619/83643/157691; K3=50203 for P8; K4=57560/107032/205976. No target timing evidence was read in any review phase.
