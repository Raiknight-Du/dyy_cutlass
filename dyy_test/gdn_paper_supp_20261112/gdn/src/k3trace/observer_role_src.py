# Observer role source.  generate_variants.py embeds this file verbatim into the generated variants module,
# so it must only use names available after `from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import *`
# (cute, cutlass, tcgen05, pipeline, ...).  Everything below runs on warps 6 and 7 only, after the pinned
# kernel's own `setmaxregister_decrease(self.num_regs_other)`; warps 0-5 are untouched.

OBS_MAX_TRIES = 1 << 20   # bounded wait: a timed-out wait writes status=1 instead of hanging the CTA
OBS_NUM_REGS = 64         # register budget of the observer warps. The pinned kernel decreases warps 6-7 to 32; under 32 the
                          # observer loop spilled (STACK 16 -> 48 B). 64 keeps the CTA pool consistent: 4x120 + 2x32 + 2x64 = 672
                          # <= 8x120 = 960 per-thread registers allocated at launch; the compute warps' increase target is 120,
                          # i.e. the launch count, so no warp ever waits for registers released by the observers.


@cute.jit
def _obs_wait(mbar_ptr, parity: cutlass.Int32) -> cutlass.Int32:
    """Wait for phase `parity` of one mbarrier with mbarrier.try_wait (hardware suspend, no spinning).
    Returns the number of try_wait calls; OBS_MAX_TRIES means timeout."""
    tries = cutlass.Int32(0)
    done = cutlass.Int32(0)
    while done == 0:
        ok = cute.arch.mbarrier_try_wait(mbar_ptr, parity)
        tries = tries + 1
        if ok:
            done = cutlass.Int32(1)
        if tries >= OBS_MAX_TRIES:
            done = cutlass.Int32(1)
    return tries


@cute.jit
def _obs_flag(cond: cutlass.Boolean) -> cutlass.Int64:
    v = cutlass.Int64(0)
    if cond:
        v = cutlass.Int64(1)
    return v


@cute.jit
def _stamp_issue(g_obs: cute.Tensor, obs: cutlass.Int32, p: cutlass.Int32, slot: cutlass.Int32):
    """Issue-side stamp for the 32-register warps 4/5 (OBSI variant only); lane 0 writes one clock64."""
    if cute.arch.lane_idx() == 0:
        g_obs[cute.arch.block_idx()[0], obs, p, slot] = cute.arch.clock64()


# ---- method inserted into the generated class (indentation handled by the generator) ----
@cute.jit
def run_observer_role(self, warp_idx, bx, storage, num_chunks, g_obs: cute.Tensor, g_hdr: cute.Tensor):
    if cute.arch.lane_idx() == 0:
        obs = warp_idx - 6
        # barrier layout (cutlass.pipeline._make_sync_object): full[0..stages) then empty[stages..2*stages)
        n_full = storage.n_mbar.data_ptr()          # n_stage = 1
        n_empty = n_full + 1
        m_full0 = storage.m_mbar.data_ptr()         # m_stage = 2 -> full0, full1, empty0, empty1
        ready_full = storage.mma_ready_mbar.data_ptr()
        done_full = storage.mma_done_mbar.data_ptr()
        done_empty = done_full + 1
        g_hdr[bx, obs, 0] = cutlass.Int64(cute.arch.smid())
        g_hdr[bx, obs, 1] = cutlass.Int64(cute.arch.globaltimer())
        g_hdr[bx, obs, 2] = cute.arch.clock64()
        g_hdr[bx, obs, 7] = cutlass.Int64(num_chunks)
        max_tries = cutlass.Int32(0)
        if obs == 0:
            # chain A (strictly causal): READY(p) -> MMA_DONE(p) -> UPDATE_RELEASED(p)
            for p in cutlass.range(num_chunks, unroll=1):
                par = p & 1
                m_ptr = m_full0 + (p & 1)
                m_par = (p >> 1) & 1
                t = _obs_wait(ready_full, par)
                g_obs[bx, obs, p, 0] = cute.arch.clock64()
                g_obs[bx, obs, p, 3] = _obs_flag(cute.arch.mbarrier_test_wait(m_ptr, m_par))
                g_obs[bx, obs, p, 4] = cutlass.Int64(t)
                if t > max_tries:
                    max_tries = t
                t = _obs_wait(done_full, par)
                g_obs[bx, obs, p, 1] = cute.arch.clock64()
                g_obs[bx, obs, p, 5] = cutlass.Int64(t)
                if t > max_tries:
                    max_tries = t
                t = _obs_wait(done_empty, par)
                g_obs[bx, obs, p, 2] = cute.arch.clock64()
                g_obs[bx, obs, p, 6] = cutlass.Int64(t)
                if t > max_tries:
                    max_tries = t
        else:
            # chain B: N_LANDED(p) -> M_LANDED(p) (test first; flagged if already complete) -> N_RELEASED(p)
            for p in cutlass.range(num_chunks, unroll=1):
                par = p & 1
                m_ptr = m_full0 + (p & 1)
                m_par = (p >> 1) & 1
                t = _obs_wait(n_full, par)
                g_obs[bx, obs, p, 0] = cute.arch.clock64()
                g_obs[bx, obs, p, 4] = cutlass.Int64(t)
                if t > max_tries:
                    max_tries = t
                already = _obs_flag(cute.arch.mbarrier_test_wait(m_ptr, m_par))
                t = cutlass.Int32(0)
                if already == 0:
                    t = _obs_wait(m_ptr, m_par)
                g_obs[bx, obs, p, 1] = cute.arch.clock64()
                g_obs[bx, obs, p, 2] = already
                g_obs[bx, obs, p, 5] = cutlass.Int64(t)
                if t > max_tries:
                    max_tries = t
                t = _obs_wait(n_empty, par)
                g_obs[bx, obs, p, 3] = cute.arch.clock64()
                g_obs[bx, obs, p, 6] = cutlass.Int64(t)
                if t > max_tries:
                    max_tries = t
        status = cutlass.Int64(0)
        if max_tries >= OBS_MAX_TRIES:
            status = cutlass.Int64(1)
        g_hdr[bx, obs, 3] = cutlass.Int64(cute.arch.globaltimer())
        g_hdr[bx, obs, 4] = cute.arch.clock64()
        g_hdr[bx, obs, 5] = status
        g_hdr[bx, obs, 6] = cutlass.Int64(max_tries)


@cute.jit
def _obs_wait_to(mbar_ptr, parity: cutlass.Int32) -> cutlass.Int32:
    """Bounded wait; returns 1 on timeout, 0 otherwise (one live counter, no per-event bookkeeping)."""
    tries = cutlass.Int32(0)
    done = cutlass.Int32(0)
    while done == 0:
        ok = cute.arch.mbarrier_try_wait(mbar_ptr, parity)
        tries = tries + 1
        if ok:
            done = cutlass.Int32(1)
        if tries >= OBS_MAX_TRIES:
            done = cutlass.Int32(2)
    timed_out = cutlass.Int32(0)
    if done == 2:
        timed_out = cutlass.Int32(1)
    return timed_out


# ---- lean observer: indexed by %smid (read inside the branch, so no CTA index is live across the dispatch),
#      no per-event try counts; otherwise the same events as run_observer_role ----
@cute.jit
def run_observer_role_lean(self, warp_idx, storage, num_chunks, g_obs: cute.Tensor, g_hdr: cute.Tensor):
    if cute.arch.lane_idx() == 0:
        obs = warp_idx - 6
        sm = cute.arch.smid()
        n_full = storage.n_mbar.data_ptr()
        n_empty = n_full + 1
        m_full0 = storage.m_mbar.data_ptr()
        ready_full = storage.mma_ready_mbar.data_ptr()
        done_full = storage.mma_done_mbar.data_ptr()
        done_empty = done_full + 1
        g_hdr[sm, obs, 0] = cutlass.Int64(sm)
        g_hdr[sm, obs, 1] = cutlass.Int64(cute.arch.globaltimer())
        g_hdr[sm, obs, 2] = cute.arch.clock64()
        g_hdr[sm, obs, 7] = cutlass.Int64(num_chunks)
        timeout = cutlass.Int32(0)
        if obs == 0:
            for p in cutlass.range(num_chunks, unroll=1):
                par = p & 1
                timeout = timeout | _obs_wait_to(ready_full, par)
                g_obs[sm, obs, p, 0] = cute.arch.clock64()
                g_obs[sm, obs, p, 3] = _obs_flag(cute.arch.mbarrier_test_wait(m_full0 + (p & 1), (p >> 1) & 1))
                timeout = timeout | _obs_wait_to(done_full, par)
                g_obs[sm, obs, p, 1] = cute.arch.clock64()
                timeout = timeout | _obs_wait_to(done_empty, par)
                g_obs[sm, obs, p, 2] = cute.arch.clock64()
        else:
            for p in cutlass.range(num_chunks, unroll=1):
                par = p & 1
                m_ptr = m_full0 + (p & 1)
                m_par = (p >> 1) & 1
                timeout = timeout | _obs_wait_to(n_full, par)
                g_obs[sm, obs, p, 0] = cute.arch.clock64()
                already = _obs_flag(cute.arch.mbarrier_test_wait(m_ptr, m_par))
                if already == 0:
                    timeout = timeout | _obs_wait_to(m_ptr, m_par)
                g_obs[sm, obs, p, 1] = cute.arch.clock64()
                g_obs[sm, obs, p, 2] = already
                timeout = timeout | _obs_wait_to(n_empty, par)
                g_obs[sm, obs, p, 3] = cute.arch.clock64()
        g_hdr[sm, obs, 3] = cutlass.Int64(cute.arch.globaltimer())
        g_hdr[sm, obs, 4] = cute.arch.clock64()
        g_hdr[sm, obs, 5] = cutlass.Int64(timeout)


# ---- chain A only: a single observer warp (6); warp 7 behaves exactly as in the pinned kernel ----
@cute.jit
def run_observer_role_a(self, warp_idx, storage, num_chunks, g_obs: cute.Tensor, g_hdr: cute.Tensor):
    if warp_idx == 6:
        if cute.arch.lane_idx() == 0:
            sm = cute.arch.smid()
            m_full0 = storage.m_mbar.data_ptr()
            ready_full = storage.mma_ready_mbar.data_ptr()
            done_full = storage.mma_done_mbar.data_ptr()
            done_empty = done_full + 1
            g_hdr[sm, 0, 0] = cutlass.Int64(sm)
            g_hdr[sm, 0, 1] = cutlass.Int64(cute.arch.globaltimer())
            g_hdr[sm, 0, 2] = cute.arch.clock64()
            g_hdr[sm, 0, 7] = cutlass.Int64(num_chunks)
            timeout = cutlass.Int32(0)
            for p in cutlass.range(num_chunks, unroll=1):
                par = p & 1
                timeout = timeout | _obs_wait_to(ready_full, par)
                g_obs[sm, 0, p, 0] = cute.arch.clock64()
                g_obs[sm, 0, p, 3] = _obs_flag(cute.arch.mbarrier_test_wait(m_full0 + (p & 1), (p >> 1) & 1))
                timeout = timeout | _obs_wait_to(done_full, par)
                g_obs[sm, 0, p, 1] = cute.arch.clock64()
                timeout = timeout | _obs_wait_to(done_empty, par)
                g_obs[sm, 0, p, 2] = cute.arch.clock64()
            g_hdr[sm, 0, 3] = cutlass.Int64(cute.arch.globaltimer())
            g_hdr[sm, 0, 4] = cute.arch.clock64()
            g_hdr[sm, 0, 5] = cutlass.Int64(timeout)


# ---- both chains on ONE observer warp (6), in causal order per partition:
#      N_LANDED -> READY -> N_RELEASED -> MMA_DONE -> UPDATE_RELEASED, with M tested (flag) when N lands and when READY fires.
#      N_RELEASED follows READY by a few instructions of the compute warps, so that stamp may be an observation bound. ----
@cute.jit
def run_observer_role_ab(self, warp_idx, storage, num_chunks, g_obs: cute.Tensor, g_hdr: cute.Tensor):
    if warp_idx == 6:
        if cute.arch.lane_idx() == 0:
            sm = cute.arch.smid()
            n_full = storage.n_mbar.data_ptr()
            n_empty = n_full + 1
            m_full0 = storage.m_mbar.data_ptr()
            ready_full = storage.mma_ready_mbar.data_ptr()
            done_full = storage.mma_done_mbar.data_ptr()
            done_empty = done_full + 1
            g_hdr[sm, 0, 0] = cutlass.Int64(sm)
            g_hdr[sm, 0, 1] = cutlass.Int64(cute.arch.globaltimer())
            g_hdr[sm, 0, 2] = cute.arch.clock64()
            g_hdr[sm, 0, 7] = cutlass.Int64(num_chunks)
            timeout = cutlass.Int32(0)
            for p in cutlass.range(num_chunks, unroll=1):
                par = p & 1
                timeout = timeout | _obs_wait_to(n_full, par)
                g_obs[sm, 1, p, 0] = cute.arch.clock64()
                g_obs[sm, 1, p, 2] = _obs_flag(cute.arch.mbarrier_test_wait(m_full0 + (p & 1), (p >> 1) & 1))
                timeout = timeout | _obs_wait_to(ready_full, par)
                g_obs[sm, 0, p, 0] = cute.arch.clock64()
                g_obs[sm, 0, p, 3] = _obs_flag(cute.arch.mbarrier_test_wait(m_full0 + (p & 1), (p >> 1) & 1))
                timeout = timeout | _obs_wait_to(n_empty, par)
                g_obs[sm, 1, p, 3] = cute.arch.clock64()
                timeout = timeout | _obs_wait_to(done_full, par)
                g_obs[sm, 0, p, 1] = cute.arch.clock64()
                timeout = timeout | _obs_wait_to(done_empty, par)
                g_obs[sm, 0, p, 2] = cute.arch.clock64()
            g_hdr[sm, 0, 3] = cutlass.Int64(cute.arch.globaltimer())
            g_hdr[sm, 0, 4] = cute.arch.clock64()
            g_hdr[sm, 0, 5] = cutlass.Int64(timeout)
            g_hdr[sm, 1, 0] = cutlass.Int64(sm)
            g_hdr[sm, 1, 2] = g_hdr[sm, 0, 2]
            g_hdr[sm, 1, 5] = cutlass.Int64(timeout)
            g_hdr[sm, 1, 7] = cutlass.Int64(num_chunks)
