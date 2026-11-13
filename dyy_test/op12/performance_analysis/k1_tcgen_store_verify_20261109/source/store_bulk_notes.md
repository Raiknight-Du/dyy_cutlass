# Additional shared-layout plus bulk-store strategy

Export `store_bulk.BulkStoreMixin`; place first in the MRO before the
original full K1 class. This is a separate strategy from
`CoalescedStoreMixin`; existing primary experiment files are unchanged.
No extra shared storage or host descriptor is needed. Original inverse
storage is reused only after every warp has loaded it into registers.
The hook keeps the original inverse LDSM, beta expression, cast, valid
mask and physical coordinate identity:
`gT_CR[c,r] = dtype(-Float32(beta[r])*Float32(sKK_inv[c,r]))`.

Sequence: original LDSM inverse reload; original elementwise computation;
CTA barrier to protect inverse lifetime; STSM transpose=True into a
linear `(1,64)` output view of the same 8192 bytes; async-proxy fence
and CTA barrier; converged warp 0 collectively invokes one 8192-byte
`CopyBulkS2GOp` (the SDK elects its issuing lane internally); commit
and full wait group 0; CTA barrier before leaving the hook. Full wait is
conservative versus wait.read 0, and ensures no pending global write on
return. All warp-0 lanes enter copy, commit and wait; an outer single-lane
guard would violate the SDK collective requirement and is not used.
The upstream shared inverse allocation is 16-byte aligned;
raw 1D bulk copy requires only 16-byte source/destination alignment and a
size multiple of 16, so its alignment is sufficient. Global tiles are
at least 128-byte aligned and each occupies 8192 bytes.

This uses installed Cutlass 4.7 source APIs `CopyBulkS2GOp` with
`num_bits_per_copy=65536`, `recast_ptr(dtype=...)`,
`cp_async_bulk_commit_group`, and `cp_async_bulk_wait_group(0)`.
The frozen upstream already uses
`make_tiled_copy_C(StMatrix8x8x16bOp(transpose=True,num_matrices=4),mma)`
for an output epilogue in `delta_rule_sm120.py`.
The register fragment remains the original MMA-C fragment; the transposed
STSM is partitioned against the changed destination layout. Compile and
GPU correctness must confirm backend partitioning/constraints.

Alignment reference: [NVIDIA PTX cp.async.bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).

## Risks and acceptance

The desired reduction is elimination of scalar LSU global stores and
avoidance of scalar four-way-conflicted inverse reads. Costs are extra
STSM, three CTA barriers and the bulk wait. Linear shared output has
a 128-byte pitch and may itself cause STSM bank conflicts. No promise
of lower total wavefronts or faster full-kernel timing is made. NCU must
measure shared writes as well as reads, and bulk writes are not accounted
as ordinary LSU STG wavefronts. Retaining full tile fragments while
changing the epilogue can affect register allocation. Reject added
LDL/STL/stack spills with the original min_blocks_per_mp=8 launch.

No compute stage is omitted or approximated. No new shared allocation,
intermediate mathematical layout, or launch bound is introduced. Preserve
this as an additional arm, never relabel it as the primary store mapping.

Required runner checks: compile/no-spill; bitwise baseline-versus-bulk
full output for finite FP16 and BF16 data; full/partial chunks and
multi-sequence/head cases; unchanged padding zeros; SASS showing bulk
S2G and no final scalar STG; full K1 events and complete shared/global
traffic. No GPU or JIT was executed by this authoring subtask.
