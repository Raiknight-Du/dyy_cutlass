# Contiguous final-store candidate

Export: `store_layout.CoalescedStoreMixin`. Put it first in the subclass
MRO: `class LayoutOnly(CoalescedStoreMixin, Original): pass` and
`class Both(CoalescedStoreMixin, Tcgen05Only): pass`. It has no constructor
and overrides only the existing `@cute.jit store_t_to_gmem` signature.
Use separate concrete classes/compile cache identities for all four arms.
The mixin assumes the existing BLK=64, CTA=128, FP16 inverse, FP16/BF16
output, and the original shared-inverse tensor coordinate convention.

## Exact semantics

The upstream method partitions the same logical coordinates for its
inverse reload and global destination, then names them `col, row`.
Accordingly the preserved physical-axis rule is:

```
gT_CR[col, row] = dtype(-Float32(sBeta[row]) * Float32(sKK_inv[col, row]))
```

when both coordinates are below `valid_len`; otherwise it writes positive
zero. `gT_CR` strides are `(1,64)`, so its `(col,row)` axes correspond to
the transposed workspace convention. This description deliberately uses
physical tensor axes rather than renaming the inverse mathematical axes.
Do not change the inverse indices to `[row,col]`. No triangle predicate
is introduced: inverse values, including diagonal/triangle semantics, are
consumed exactly as supplied by the original full kernel. No multiply,
inverse, K load, barrier, or preceding stage is removed or modified.

The map is `row=warp+4*i`, `col=lane+32*j`, for `i=0..15`, `j=0..1`.
Revision 2 keeps the outer row loop rolled (`cutlass.range(..., unroll=1)`)
and only unrolls the two column halves. This adds loop/address-control
instructions but limits the compiler's simultaneous live values.
Every `(col,row)` has one writer and all 4096 outputs are written, including
padding. Conversion and negation/multiplication ordering match upstream.
The unconditionally issued inverse read follows upstream's full inverse
reload even for masked coordinates; beta loads remain row-masked.

## Hardware changes and limits

This is an output-mapping experiment, **not a pure STG-only intervention**.
The upstream LDSM inverse reload and MMA-C register partition are replaced
by direct scalar shared reads. No shared-memory transpose, scratch memory,
extra barrier, warp exchange, or modification of the intermediate inverse
layout is added. Register scheduling and allocation can still change.

For a full tile, the source specifies 32 scalar stores/thread (128 warp
store requests/CTA). A warp addresses 64 contiguous bytes, giving an ideal
2 sectors per request, versus the prior helper's 8. Ideal global-store
wavefront geometry is therefore 128/CTA versus the helper's 512/CTA.
These are predictions until SASS/NCU verifies the full K1 compilation;
they are not measurements or guarantees about an optimizer's lowering.

The original shared layout has offset
`8*(col%8)+(row%8)+64*(col//8)+512*(row//8)` in FP16 elements.
At fixed row, the new scalar loads have 4 different words per shared
bank per warp: about 128 inverse read requests and 512 inverse read
wavefronts/CTA if compiled to scalar LDS.U16. This can outweigh the STG
benefit. Beta is hoisted across the two column halves: 16 scalar loads
per thread (64 warp requests/CTA), and each full-tile warp load is a
shared broadcast. The predicted combined shared read wavefront count
for this method is therefore 576/CTA, subject to actual SASS lowering.
The original controlled result of 192 shared read wavefronts/CTA must
not be silently reused for this candidate. Tail predicates reduce beta
activity. Compiler hoisting/packing may change instruction counts.

## Required full-kernel validation by the GPU runner

1. Compile the original, tcgen05-only, layout-only, and both concrete
   classes with isolated compile identities. Inspect generated SASS for
   contiguous final STG, direct LDS, register count, and spills.
2. Compare full K1 output to the frozen original for FP16 and BF16,
   including full and partial chunks (1, 31, 32, 33, 63, 64, 65 tokens),
   multiple sequences and GQA heads. Verify all padded rows/columns zero.
   A same-compute layout-only comparison should be bitwise exact for
   finite inputs; tcgen05 accumulation can need the established tolerance.
3. Run the actual full K1 timing protocol for all four arms. Collect
   global/shared read/write wavefronts and requests and occupancy/spills;
   interpret store savings together with the extra shared traffic.
   A slower layout arm is a valid result and should be reported.

Only CPU/static checks are performed by this subtask. CUDA compilation,
GPU correctness, timing, and NCU are reserved to the serialized GPU runner.

CPU checks completed: Python AST parses; enumeration confirms a bijection
over all 4096 coordinates; valid lengths 0, 1, 31, 32, 33, 63, 64 produce
exactly `valid_len**2` unmasked coordinates; all 128 source-level warp
store groups span exactly two aligned 32-byte sectors.

## Compile feedback and revision

The runner's revision-1 FP16 compile used 64 registers/thread and an
8-byte stack frame. SASS contained one `STL.64 [R1], R6` at 0x08d0 and
one `LDL.LU.64 R24, [R1]` at 0x33e0: a global output address spilled
across the compute/inverse body. Baseline used 64 registers and zero
stack. Revision 2 therefore rolls the outer row loop while preserving
the original launch bound (`min_blocks_per_mp=8`) and all output mapping.
No launch-bound relaxation is part of this candidate. The revised SASS
must pass the same no-spill gate; that result is not yet established.
