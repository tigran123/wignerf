"""
SolverWorker: one thread per ticked variant. Owns its own ArrayBackend
(FFT plans and CUDA context are per-thread), integrates with its own
adaptive dt, and lands exactly on every record time tau_k of the session
clock by clamping its final substep. Emits quantized VariantFrames into the
session's FrameHistory and signals the asyncio streamer.

Commands arrive on cmd_q as dicts:
  {"kind": "params", "cp": CompiledPotential|None, "mass": ..., "c": ...,
   "hbar_eff": ..., "tol": ...}   (absent keys = unchanged)
"""

import logging
import queue
import threading
import traceback
from time import monotonic

from . import boundary, frame, initial
from .grid import GridState, embed_window
from .propagator import Propagator
from .protocol import VARIANTS, variant_id
from .xp import ArrayBackend

log = logging.getLogger(__name__)



class SolverWorker(threading.Thread):
    def __init__(self, session, key, slot, device):
        super().__init__(daemon=True, name="wignerf-%s-%s" % (session.id, key))
        self.session = session
        self.key = key
        self.slot = slot
        self.device = device
        self.flavor = VARIANTS[key]
        self.cmd_q = queue.Queue()
        self.stop_evt = threading.Event()
        self._grid_state = None      # live window; regrids replace it
        self._applied_epoch = 0      # newest RegridPlan epoch applied here
        self.force_adjust = True
        self.dt = 0.0
        self.steps_total = 0
        self.steps_per_sec = 0.0
        # exponent slots, each (dts, (expU, expT)) or None — see _exponents
        self._exp_main = None    # the full step self.dt (the hot path)
        self._exp_odd = None     # the clamped substep that lands on tau_k
        self._rate_mark = (0, monotonic())

    def stop(self):
        self.stop_evt.set()
        self.session.clock.kick()

    # -- thread body --------------------------------------------------------

    def run(self):
        try:
            self._run()
        except Exception as e:
            log.exception("worker %s failed", self.name)
            self.session.post_error("variant '%s' solver died: %s" % (self.key, e),
                                    detail=traceback.format_exc())
            # Pause the session: with this variant dead the lockstep
            # frontier can never advance, and the siblings would fill the
            # history with records that never complete (and never evict).
            self.session.clock.set_running(False)
        finally:
            self._release_gpu_pool()
            # The true VRAM-free moment: if this worker was mid-record when the
            # session closed, the join in close() timed out and the thread ran
            # on to here — so this timestamp, not close()'s return, is when the
            # card actually gives the memory back.
            log.info("worker %s released GPU pool", self.name)

    def _release_gpu_pool(self):
        """Return unused CuPy pool blocks to the driver when the session
        closes. CuPy pools freed memory per process, so nvidia-smi keeps
        showing it as 'used' even when idle — releasing here means closed
        sessions visibly give their VRAM back. The pool is per-device, so
        re-enter THIS worker's device; blocks still referenced by other
        live sessions are untouched (free_all_blocks frees only free
        blocks)."""
        # Drop this worker's OWN arrays FIRST — before the GPU guard, because on
        # the CPU backend these are host RAM (1 GiB at 4096^2) held just as long.
        # free_all_blocks() returns only blocks that are FREE in the pool, and
        # _run's locals (W, prop) are gone by now but ATTRIBUTES are not: the two
        # exponent slots still hold 4 complex meshes — at complex128, 256 MiB at
        # 2048^2, 1.0 GiB at 4096^2, 4.0 GiB at 8192^2 (half that in a float32
        # session) — so without this the release leaves exactly that much
        # behind. It then returns to the pool only when the
        # worker is collected (the session<->worker cycle, so: not by
        # refcounting) and to the DRIVER only at some later worker's
        # free_all_blocks() — which is why VRAM used to come back on the SECOND
        # "Restart session" rather than the first.
        self._exp_clear()
        backend = getattr(self, "_backend", None)
        if backend is None or not backend.is_gpu:
            return
        try:
            with backend.device():
                xp = backend.xp
                # this thread's cuFFT plans (work areas are real VRAM) — the
                # plan cache is per thread AND device, and this thread is done
                try:
                    xp.fft.config.get_plan_cache().clear()
                except Exception:
                    log.debug("cuFFT plan cache clear failed", exc_info=True)
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            log.debug("GPU pool release failed", exc_info=True)

    def _run(self):
        cfg = self.session.cfg
        backend = ArrayBackend(device=self.device,
                               fft_threads=self.session.fft_threads,
                               precision=cfg.precision)
        self._backend = backend
        with backend.device():
            # the worker's window mirrors the session's GridState (same
            # dataclass arithmetic), so record geometry and lattice points
            # agree with the scheduler bitwise
            self._grid_state = GridState.from_spec(cfg.grid)
            g, Wnat, _, _ = initial.from_spec(
                cfg.grid, cfg.ic, cfg.hbar_eff, backend,
                grid=self._grid_state.make_grid(backend))
            U, gradU = self.session.compiled_potential.for_backend(backend)
            prop = Propagator(g, mass=cfg.mass, c=cfg.c, hbar_eff=cfg.hbar_eff,
                              tol=cfg.tol, U=U, gradU=gradU, **self.flavor)
            if not self._finite(prop, backend):
                raise ValueError("non-finite propagator exponents "
                                 "(check U, mass, c)")
            # The IC is built in float64 (initial.py is precision-independent);
            # adopt the working dtype HERE so record 0 is measured on the same
            # footing as every record after it. solve_spectral would do it at
            # step 1 anyway, which would leave record 0 — the Cauchy data every
            # later record is compared against — as the one frame with
            # double-precision observables.
            W = g.shift(Wnat).astype(backend.real_dtype, copy=False)
            t = cfg.t1
            self.dt = cfg.record_dt/8.
            self._emit(0, t, W, prop, backend)      # record 0 = the Cauchy data
            frontier = 0
            while not self.stop_evt.is_set():
                self._drain_commands(prop, backend)
                tgt = self.session.clock.next_target(
                    frontier, self.session.history.latest_complete())
                if tgt is None:
                    self.session.clock.wait_work(0.1)
                    continue
                k, t_tgt = tgt
                # a scheduled regrid applies to every record >= its k_star
                # (k_star is past all in-flight records, so the switch is
                # lockstep-uniform); the epoch guard makes stale plans inert
                plan = self.session.current_regrid()
                if plan is not None and k >= plan.k_star \
                   and self._applied_epoch < plan.epoch:
                    # hand W over in a box and drop this frame's own reference,
                    # so the old state is unreachable while the new (at ndim=2,
                    # multi-GiB) one is allocated — see _apply_regrid
                    box = [W]
                    W = None
                    W = self._apply_regrid(plan, prop, backend, box)
                    self._applied_epoch = plan.epoch
                W, t = self._advance(prop, W, t, t_tgt)
                self._emit(k, t_tgt, W, prop, backend)
                frontier = k

    # -- stepping -----------------------------------------------------------

    @staticmethod
    def _finite(prop, backend):
        xp = backend.xp
        return bool(xp.isfinite(prop.dU_im).all()) and bool(xp.isfinite(prop.dT_im).all())

    def _exp_clear(self):
        self._exp_main = self._exp_odd = None

    def _exponents(self, prop, dts):
        """Two purposeful slots, not a general cache.

        Within a record almost every substep uses the full self.dt; only the
        LAST one is clamped to land exactly on tau_k, and that value is unique
        per record and never asked for again. So pin the full step and keep a
        single slot for the clamped straggler — the hot path hits ~always and
        exactly one exp() rebuild happens per record.

        The 8-entry dict this replaces kept seven dead complex128 pairs alive:
        256 MiB per worker at 1024^2 and 1 GiB at 2048^2, two thirds of the
        entire device working set, to avoid a rebuild costing ~25% of one
        Strang step once per record (a record is >= 8 substeps)."""
        for slot in (self._exp_main, self._exp_odd):
            if slot is not None and slot[0] == dts:
                return slot[1]
        pair = prop.exponents(dts)
        if dts == self.dt:
            self._exp_main = (dts, pair)
        else:
            self._exp_odd = (dts, pair)
        return pair

    def _advance(self, prop, W, t, t_tgt):
        eps = 1e-12*max(1.0, abs(t_tgt))
        while abs(t_tgt - t) > eps and not self.stop_evt.is_set():
            direction = 1.0 if t_tgt > t else -1.0
            if self.dt == 0.0 or (self.dt > 0) != (direction > 0):
                self.dt = direction*(abs(self.dt) or self.session.cfg.record_dt/8.)
                self._exp_clear()
                self.force_adjust = True
            rem = t_tgt - t
            adjust_due = self.force_adjust or self.steps_total % 20 == 0
            if adjust_due and abs(self.dt) <= abs(rem):
                # adjust_step only ever shrinks (as in solve.py): give dt a
                # chance to climb back after a transient (a stiff U applied
                # live, then reverted) by trying a 1/0.7 larger step — the
                # controller shrinks it right back if the accuracy is not
                # there. |rem| <= record_dt caps growth at one record.
                dt_try = self.dt/0.7
                if self.force_adjust or abs(dt_try) > abs(rem):
                    dt_try = self.dt
                W, self.dt, eU, eT = prop.adjust_step(dt_try, W)
                self._exp_main, self._exp_odd = (self.dt, (eU, eT)), None
                self.force_adjust = False
                t += self.dt
            else:
                dts = direction*min(abs(self.dt), abs(rem))
                W = prop.solve_spectral(W, *self._exponents(prop, dts))
                t += dts
            self.steps_total += 1
        return W, t_tgt    # land exactly on the record time (no drift)

    def _emit(self, k, t, W, prop, backend):
        vf, obs = frame.build(W, prop.grid, prop.hbar_eff, prop=prop,
                              dt=self.dt, vid=variant_id(**self.flavor))
        self.session.history.put(k, t, self.slot, vf, self._grid_state.geom())
        # boundary watch every record: O(sum N) host sums on the marginals
        # observables already brought over — no extra device sync
        self.session.report_edge(
            self.slot, k,
            boundary.edge_report(obs.marg, prop.grid.d, backend.precision,
                                 labels=prop.grid.labels))
        self.session.notify_frame()
        # a landing record may open the skew gate for waiting siblings
        self.session.clock.kick()
        n, mark = self._rate_mark
        now = monotonic()
        if now - mark > 1.0:
            self.steps_per_sec = (self.steps_total - n)/(now - mark)
            self._rate_mark = (self.steps_total, now)

    # -- regrid -------------------------------------------------------------

    def _apply_regrid(self, plan, prop, backend, box):
        """Exact fixed-lattice regrid of the live state: whole-cell window
        move/double on the frozen lattice — W values are COPIED to their
        identical lattice points, entering cells are zero, nothing is ever
        interpolated. The transform runs in natural order (the window overlap
        is contiguous there; fftshifted order would split it).

        `box` is a one-element list holding W, so this can DROP the caller's
        reference to the old state before the new grid is built. That is not
        fastidiousness: at ndim=2 a doubling allocates a multi-GiB array while
        the old one is still live, and _run's own local would otherwise pin it
        for the whole switch. The exponent slots go first for the same reason —
        they are 64 B/cell, the largest single item in the working set, and a
        grid change invalidates them anyway (force_adjust below rebuilds).
        """
        old, new = self._grid_state, plan.state
        # 1. the slots, before anything new is allocated
        self._exp_clear()
        # 2. the state: unshift, embed, and drop every old reference as we go
        Wnat = prop.grid.unshift(box[0])
        box[0] = None
        Wnew = embed_window(Wnat, old, new, backend.xp)
        del Wnat
        # 3. the propagator, which releases its own old meshes and the pool
        #    before rebuilding at the new shape (Propagator.set_grid)
        g = new.make_grid(backend)
        prop.set_grid(g)
        if not self._finite(prop, backend):
            # the session pre-validated U on the new window, so this is a
            # genuine invariant break -> the worker-death path pauses the run
            raise ValueError("non-finite propagator exponents after regrid "
                             "to %s" % new.describe())
        self._grid_state = new
        self.force_adjust = True
        log.info("%s: regrid epoch %d applied at k>=%d: %s",
                 self.name, plan.epoch, plan.k_star, new.describe())
        return g.shift(Wnew)

    # -- commands -----------------------------------------------------------

    def _drain_commands(self, prop, backend):
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            if cmd.get("kind") != "params":
                continue
            prev = dict(U=prop.U, gradU=prop.gradU, mass=prop.mass, c=prop.c,
                        hbar_eff=prop.hbar_eff, tol=prop.tol)
            try:
                kwargs = {}
                if cmd.get("cp") is not None:
                    U, gradU = cmd["cp"].for_backend(backend)
                    kwargs.update(U=U, gradU=gradU)
                for f in ("mass", "c", "hbar_eff", "tol"):
                    if cmd.get(f) is not None:
                        kwargs[f] = cmd[f]
                prop.set_physics(**kwargs)
                if not self._finite(prop, backend):
                    raise ValueError("non-finite propagator exponents")
            except Exception as e:
                prop.set_physics(**prev)   # roll back, keep evolving
                self.session.post_error(
                    "variant '%s': parameter change rejected (%s)" % (self.key, e))
            else:
                self._exp_clear()
                self.force_adjust = True
