# Independent review: original K1, TCGEN Wkk, scalar coalesced store, combined

Reviewed frozen local `evidence/results`, `evidence/ncu`, source, and job logs.
Recomputed all values on CPU with `source/review_evidence.py`; machine-readable
output is `review/independent_review.json`. This reviewer performed no GPU,
JIT, timing, or profiler execution. Upstream cleanliness remains lead-owned.
Downloaded queue-v3 guard telemetry was independently checked per job below.

## Accepted performance conclusion

For FP16 B1/H16/D128/S32768, the scalar coalesced store yields a small consistent
improvement. This TCGEN Wkk implementation and its combination are substantially
slower. The experiment establishes the performance of these implementations;
it does not establish that every TCGEN implementation is intrinsically slower.

Each process has 100 retained Event samples per variant/cache mode. Values below
are the median of the three independently computed process medians, in us.

| Variant | Evicted | Change vs baseline | Repeat without eviction | Change vs baseline |
|---|---:|---:|---:|---:|
| Original | 74.048 | — | 72.144 | — |
| TCGEN | 169.552 | +128.976% | 168.224 | +133.178% |
| Scalar coalesced store | 72.608 | -1.945% | 71.712 | -0.599% |
| Combined | 169.632 | +129.084% | 167.792 | +132.579% |

Evicted process medians are original [74.048,74.016,74.176], TCGEN
[169.552,169.520,169.856], store [72.656,72.608,72.512], and combined
[169.632,169.616,170.112]. Repeat mode is not certified fully L2-hot, and the
768MiB eviction helper is not proof of a perfect hardware cold reset.

## Correctness and instruction gates

All four primary FP16 variants are bitwise equal in all three processes.
Both FP16 ragged and empty-sequence/GQA validations completed with bitwise
equality; combined also equals TCGEN exactly. Supported validation lengths are
[1,63,64,65,127,257], and [0,1,0,65] with K heads4/output heads16.

The physical rule is `gT[c,r] = dtype(-beta[r]*inverse[c,r])`, so the PyTorch
workspace matrix is upper triangular. Static review preserves this transpose,
row beta indexing in physical gT, head mapping, chunk indices, and padded zeros.
The helper poisons used output slots and checks diagonal/triangle/padding when
called by the lead harness; unused workspace holes are intentionally excluded
from poisoning. This report treats the saved FP16 run artifacts as the scope
of empirical correctness, rather than universal input coverage.

Primary SASS confirms the change is real: TCGEN arms contain 8 UTCHMMA sites;
HMMA.16816.F32 sites fall from76 to12, with inverse HMMA retained. Store arms
have two STG.E.U16 sites inside a rolled loop rather than32 unrolled sites.
These are static instruction counts, not dynamic request counts. All four
primary binaries have zero LDL/STL spill sites, STACK0 and LOCAL0. Registers
are original/store64, TCGEN/combined52; dynamic shared memory is24960B for all.

BF16 validation did not pass the acceptance gate: the suite stopped at a layout
spill during compilation, before numerical validation. The BF16 empty/GQA case
was consequently not run. Do not claim all-BF16 correctness, no-spill behavior,
or performance from these FP16 results.

## NCU traffic audit

The exported report contains one profiled kernel per arm. Divide raw `.sum`
counts by8192 launched CTAs. These are measured whole-kernel counts, not stage
costs or theory weights. NCU replay time is not used as performance timing.

| Counter per CTA | Original | TCGEN | Store | Combined |
|---|---:|---:|---:|---:|
| All LSU wavefronts | 1983.591 | 1831.159 | 1928.232 | 1828.277 |
| Global/local read WF | 42.810 | 46.226 | 41.512 | 45.178 |
| Global/local write WF | 591.220 | 615.121 | 160.207 | 168.715 |
| Shared read WF | 1101.869 | 455.671 | 1479.213 | 901.355 |
| Shared write WF | 132.046 | 576.928 | 131.388 | 576.394 |
| Shared misc WF | 116.067 | 133.261 | 115.968 | 132.504 |
| Global store requests | 128 | 128 | 128 | 128 |
| Global store sectors | 512 | 512 | 256 | 256 |

The exact global-store result is4→2 sectors/request and512→256 sectors/CTA.
The earlier source prediction8→2 is not the measured baseline. Measured
global/local write WF decreases sharply with coalescing, but shared reads
increase. TCGEN also adds scalar shared writes while removing much original
shared-read work. Thus neither intervention is a pure change to one hardware
instruction family. Total LSU wavefront reduction does not imply proportional
latency reduction. Category subcounts need not exactly sum to the separately
measured total across replay passes; do not force an accounting equality or
silently substitute sectors for wavefronts.

## Concurrency: wording required

| Observation | Original | TCGEN | Store | Combined |
|---|---:|---:|---:|---:|
| Driver max-active-CTA query | 8 | 1 | 8 | 1 |
| NCU CTA-active / elapsed cycles | 7.2242 | 1.3835 | 7.2258 | 1.5355 |
| NCU active warps / active cycle | 30.0355 | 5.5982 | 30.3009 | 6.1893 |
| NCU eligible warps / scheduler active cycle | 0.9656 | 0.2129 | 1.4078 | 0.2705 |

Safe conclusion: the driver query reports1 for the TCGEN binaries and NCU
observes strongly lower concurrency. A strict one-resident-CTA-per-SM claim
is contradicted by the measured CTA/cycle and active-warp values. Conventional
NCU limits still report shared-memory8 CTAs for every arm, and register8 vs9.
The driver result and conventional resource model must be reported separately.

[NVIDIA's profiling model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)
defines active/resident warps as mapped to an SM subpartition from launch to
completion, including stalled warps. Do not relabel them as unscheduled queue
entries merely to reconcile the driver estimate. Tensor-memory allocation,
release, and deallocation occur within a CTA lifetime; phase-dependent resource
release is a plausible explanation for some overlap, but is not proven here.
[PTX9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-tcgen05-dealloc-tcgen05-relinquish-alloc-permit)
documents potentially blocking dynamic TMEM allocation, not a universal1CTA
hardware limit. The installed CUDA occupancy header also models barriers and
virtual resources beyond registers/shared memory; their exact relationship to
this driver result is not established.

The min8 compile control still has52 registers and driver query1, so merely
attributing the result to `min_blocks_per_mp=1` is unsupported. That parameter
is a compiler occupancy hint, not by itself a measured residency cap.

The matched-one-CTA experiment raises all variants' dynamic shared memory to
130048B and reports driver query1 for all. Three processes were independently
recomputed,100 samples per variant/cache/process. Median-of-process-medians
in original/TCGEN/store/combined order: evicted186.960/238.864/210.704/263.488us,
repeat184.704/237.920/209.312/262.528us. All agree with the final matched CSV.
It also slows TCGEN relative to its
primary result. It changes shared-memory allocation/carveout and execution
conditions; treat it as sensitivity evidence, not exact causal isolation of
residency or a numerical decomposition of the regression.

No further NCU collection is required to conclude the primary performance
comparison. Fully resolving the driver-versus-dynamic-concurrency discrepancy
would be a separate attribution task. No stage timing was inferred from these
whole-kernel counters.

## Additional bulk-store candidate

This candidate is separate from the above frozen primary experiment. Initial
static review found a DSL-specific issuance error: `CopyBulkS2GOp` requires a
converged warp because the backend elects an issuing lane. `if tidx==0` around
`cute.copy` is invalid; use an entire warp with no outer lane election. This
was reported to lead and engineer before GPU validation. The corrected local
file uses converged warp0 and independently verified SHA256
`ae1753e8e6b26bd2dab23fc2917b5f7d11f211f564bbe01855f1bc4354fd2aa8`.
Static review is cleared for compilation: in-place lifetime protection,
physical `(1,64)` order,16-byte alignment,8192-byte length, async-proxy
publication and full completion wait are coherent. Backend transpose-STSM
partitioning, no-spill compilation and bitwise correctness remain runner gates.
The final candidate compiles without spills and includes UBLKCP.G.S, but
full-kernel correctness fails:15927618/33554432 elements differ (47.5%).
The branch is rejected, with no accepted Event or NCU results. This empirical
failure supersedes the provisional static clearance and does not overwrite or
relabel the primary scalar-store results above.

Full-harness memcheck/synccheck failed with a host double-free before K1
compilation. The minimal PyTorch CUDA memcheck control passed with0 errors.
Therefore this is an unresolved full-harness/tool interaction, not evidence
that the whole sanitizer environment is unusable or that K1 passed sanitizer.

## Per-job guard qualification

The enclosing queue-v3 final status is `INVALID_COMPETITION`, not COMPLETED.
The first recorded external process is PID805505 at epoch1794196331.5920703,
2026-11-09 03:52:11.592UTC. Its ownership and activity are not established by
this review. A later external observation does not retroactively establish
competition during earlier job intervals.

| Saved job | Start UTC | End UTC | In-job samples |
|---|---|---|---:|
| primary r0 | 03:41:04.067 | 03:41:09.688 | 5 |
| primary r1 | 03:41:09.688 | 03:41:15.358 | 5 |
| primary r2 | 03:41:15.358 | 03:41:21.029 | 6 |
| matched1 r0 | 03:41:21.029 | 03:41:26.750 | 5 |
| NCU original | 03:41:26.750 | 03:41:57.203 | 29 |
| NCU TCGEN | 03:41:58.205 | 03:42:27.906 | 29 |
| NCU store | 03:42:27.906 | 03:42:57.458 | 28 |
| NCU combined | 03:42:57.458 | 03:43:27.409 | 29 |
| validation suite | 03:43:28.411 | 03:43:43.544 | 15 |
| min8 compile | 03:45:14.653 | 03:45:17.770 | 3 |

All samples within these saved intervals report SM1845MHz, memory3996MHz,
and no external process. Each primary/NCU job returned0. Validation returned1
because of the BF16 spill gate, not a recorded competition event. The primary
r0 preceding sample is120MHz at0.194s before job start; every in-job sample
is1845MHz. Subsequent adjacent boundary samples are also1845MHz. Coarse
telemetry cannot prove every kernel-cycle frequency or exclude an unobserved
transient between polls, but it supports retaining these earlier intervals
with this explicit qualification. Do not label the entire guard session clean.

The JSON recomputation includes exact epochs, preceding/following samples,
clock sets, external observations, and margins to the first external event,
and automatically includes newly downloaded matched-control done files.

Final downloaded matched r1/r2 jobs also qualify against queue-v3 telemetry:
6 and5 in-job samples respectively, all1845/3996MHz with no external process,
both ending before the later competition. These are the additional matched
processes used above.

Queue-v4 ended `COMPLETED_PENDING_REVIEW` with no invalid observations. Its
bulk-compile, bulk-correctness and minimal sanitizer-control job windows have
5/5/2 telemetry samples respectively, all1845/3996MHz and no external process.
The bulk-correctness returncode1 records the numerical failure. There are9
external observations during the guard's earlier idle waiting, before these
target jobs; do not describe its entire telemetry history as external-free.
Automatic-clock restoration command `nvidia-smi -rgc` returned0. The restored
frequency is allowed to vary; no fixed post-release SM frequency is required.

## Final report sign-off

`EXPERIMENT_RESULTS_CN.md` was reviewed after final evidence mirroring and
wording corrections. Primary timings, deltas, whole-kernel traffic, register
and instruction gates, all three matched processes, BF16 limitation, rejected
bulk branch, sanitizer uncertainty and scoped guard qualifications are
consistent with the evidence. PASS for the requested comparison of the two
changes separately and together. This is not a global optimality or complete
causal-attribution claim. No further GPU work is required for that scope.
