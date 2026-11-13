def _mark_state_layout(s_cute, use_state_indices: bool, DK: int) -> None:
    """Mark the recurrent-state tensor layout for compilation.

    Packed mode (``use_state_indices=False``): the caller passes a compact
    ``[num_seqs, H, V, K]`` state, so we keep the original marking that both
    makes the layout dynamic (stride-1 K dim auto-deduced) and attaches a
    ``divisibility=DK`` hint on the K dim for wider vectorized state copies.

    Pool/indexed mode (``use_state_indices=True``): the caller passes its real
    SSM state pool ``[N_pool, H, V, K]`` whose dim-0 (slot) stride is padded by
    the mamba conv+ssm cache packing, i.e. ``stride[0] > H*V*K`` -> the layout is
    NON-COMPACT. ``mark_compact_shape_dynamic`` asserts a compact layout and
    raises ``RuntimeError: The stride_order is not consistent with the layout``.
    ``mark_layout_dynamic()`` alone pins the single stride-1 dim (mode 3 = K) to
    stride 1 and carries every other stride (including the padded dim-0) through
    as a dynamic runtime value. The kernel addresses the state purely via
    ``s.stride[...]`` (reshape at ~475-497 and ``mS_init/mS_out[..., state_row]``),
    so a padded dim-0 stride is handled correctly. cutlass-dsl offers no way to
    attach a divisibility hint without also requiring compactness, so this path
    drops the ``divisibility=DK`` hint; the stride-1 K dim is retained, so the
    128x128 state autovec copy still vectorizes (possibly a narrower vector).
    That copy is a negligible fraction of the kernel, so correctness is kept
    with no meaningful perf cost.
    """
    if use_state_indices:
        s_cute.mark_layout_dynamic()
    else:
        s_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
        )
