"""
Preview endpoints:
- POST /api/preview/potential — compile a U(x) expression, report per-family
  validity + warnings and return plot samples (debounced client-side).
- POST /api/preview/wigner — build the initial Wigner function (mixture or
  cat) and return it as the SAME binary frame bundle the WebSocket streams,
  so the frontend reuses its decoder and WebGL panel for IC editing.
  Diagnostics travel in X-Wignerf-* headers.

The Wigner preview runs on a GPU when one has room, and hands the VRAM
straight back (it used to be CPU-only "to keep the GPU free for sessions",
which was the right instinct and the wrong trade: the CPU build is 52x slower
and the preview is not small — it is built at the SESSION's grid, so at 8192^2
it was 25.9 s of CPU against 0.50 s on an RTX 3090, on every reload AND every
IC edit, while the main W panel — same array, same size, built by a GPU worker
— appeared in 1.4 s).

The transient peak is what matters, not the steady state: 5.50 GiB at 8192^2
(measured; 88 bytes per cell, plateauing there from three cat components up).
So a device is chosen only if it currently has that much free with headroom,
GPU previews are serialized so two peaks cannot stack, the build's device
arrays all die with its frame, and the pool is released immediately after.
Anything unexpected — OOM above all — falls back to the CPU, which is slow but
always correct.

The CPU fallback is bounded too (_cpu_fit_error), and that is not belt and
braces: routers/sessions._fit_error asks the driver whether a SESSION fits, but
it runs at creation, while this endpoint builds the full state on every
keystroke in the setup form. WIGNERF_MAX_CELLS_2D is only a rail — deliberately
far past any card, precisely because _fit_error is the real check there — so
without a fit check of its own the fallback is the one path with nothing
bounding it, on the one host (CPU-only) where _pick_device never even runs.

The pool the preview allocates from is its OWN (see _pool), not the process
default one the solver workers share. That is what makes "release immediately
after" honest: free_all_blocks() acts on whatever pool it is handed, so
releasing the DEFAULT pool also returned the running workers' cached blocks to
the driver — on every IC keystroke — which is the exact opposite of what the
free-VRAM check above is for. cupy.cuda.using_allocator is thread-local and
previews run in starlette's threadpool while workers own their own threads, so
neither can see the other's pool. The isolation is free: a cold 1 GiB
allocation measured 3.1 ms against 2.6 ms pool-warm, and since the release
empties the pool after every preview, every preview was already cold.
"""

import logging
import threading
import traceback
from dataclasses import replace
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import config
from core import frame, initial, protocol, xp
from core.potential import (PotentialError, compile_potential,
                            sample_potential, sample_potential_grid)
from core.protocol import GridSpec, ICSpec, grid_limit_error
from core.xp import ArrayBackend, resolve_devices

log = logging.getLogger(__name__)
router = APIRouter()

# Measured transient peak of a cat IC build, per grid cell, per ndim.
#
# ndim=1 (RTX 3090, 2026-07-25): 88 B/cell — 0.34 GiB at 2048^2, 1.38 at
# 4096^2, 5.50 at 8192^2 — plateauing there (64 for a single component, 88 from
# three up, unchanged at eight) because cat_wigner reuses its temporaries per
# pair rather than accumulating them.
#
# ndim=2 (same card, 2026-07-26): 56 B/cell for a cat and 32 for a mixture,
# again flat in the component count from two up. LOWER than 1D, which is not a
# mistake: the 2D cross-Wigner is the outer product of two per-dimension cores,
# and those cores are 2D arrays — only the product is ever full size, so the
# closed form's own temporaries cost nothing at 4D scale.
# The two EXPRESSION kinds are bounded BELOW these, at every size, and that is
# by construction rather than by luck: their builds are blocked against fixed
# byte budgets (initial.IC_BLOCK_BYTES, initial.PHI_KERNEL_BYTES), so their peak
# is the float64 output plus a constant and their per-cell figure is largest at
# the SMALLEST grid. Measured on the 3090, 2026-08-04, in a fresh process each
# (a shared pool retains blocks and reads as 0):
#
#   psi     1D 1024^2 36.4   2048^2 17.2   4096^2 10.3   2D 32^4 20.1   64^4 9.0
#   wexpr   1D 1024^2 12.6                                       64^4 9.8
#
# against the cat figures above, which the same instrument reproduces as 88.1
# and 56.0 — which is what says it is measuring the right thing. An arbitrary
# expression's UNBLOCKED transient is not a constant at all (it grows with the
# expression tree's width, and MAX_EXPR_LEN is 500), so the blocking is what
# lets this table stay one number per ndim.
PREVIEW_BYTES_PER_CELL = {1: 88, 2: 56}
PREVIEW_HEADROOM = 1.4     # never take the last of a card a session is using
# ...and the same idea for host RAM, which MemAvailable already reports net of
# reclaimable page cache — so this only has to leave the box some slack.
CPU_FIT_MARGIN = 0.9

_backends = {}                    # device spec -> ArrayBackend (contexts are
_pools = {}                       # device spec -> private cupy MemoryPool
_backends_lock = threading.Lock()  # expensive; make each one once)
# One GPU preview at a time. The client debounces, but two concurrent 8192^2
# builds would want 11 GiB between them and the second would OOM a card that
# comfortably fits either alone.
_gpu_lock = threading.Lock()


def _backend(spec="cpu"):
    with _backends_lock:
        b = _backends.get(spec)
        if b is None:
            b = _backends[spec] = ArrayBackend(device=spec)
        return b


def _pool(spec, b):
    """This device's private CuPy memory pool — see the module docstring for
    why the preview must not allocate from (and above all must not release)
    the pool the solver workers share. Module-level rather than per-request so
    the release is observable from a test; it is emptied after every preview
    either way, so a fresh pool would behave identically."""
    with _backends_lock:
        p = _pools.get(spec)
        if p is None:
            p = _pools[spec] = b.xp.cuda.MemoryPool()
        return p


def _pick_device(cells, ndim=1):
    """The CUDA device with the most free VRAM, if this build comfortably fits
    in it; otherwise None, meaning use the CPU. Asking the driver for free
    memory (rather than tracking our own sessions) is what keeps a preview from
    evicting a running solver: whatever else is on the card, including another
    process entirely, is already reflected."""
    need = PREVIEW_BYTES_PER_CELL[ndim]*cells*PREVIEW_HEADROOM
    try:
        import cupy
        specs = [s for s in resolve_devices(config.DEVICE) if s.startswith("cuda")]
    except Exception:
        return None
    best, best_free = None, 0
    for spec in specs:
        try:
            with cupy.cuda.Device(int(spec.split(":")[1])) as d:
                free = d.mem_info[0]
        except Exception:
            continue
        if free > need and free > best_free:
            best, best_free = spec, free
    return best


def _cpu_fit_error(cells, ndim):
    """Whether a CPU build of this preview fits in host RAM, as a message — or
    None when it does, or when it cannot be told.

    The GPU path has _pick_device; this is the FALLBACK's equivalent, and the
    fallback is the one that can take a box down. WIGNERF_MAX_CELLS_2D is only a
    rail — deliberately far past any single card, on the grounds that
    routers.sessions._fit_error asks the driver for the real answer — but
    _fit_error runs at session creation and this endpoint fires on every
    keystroke in the setup form, at the full session grid, long before anyone
    presses Restart. At the rail a 2D cat build is ~7.5 GiB; on a CPU-only host
    (no cupy, so no _pick_device to decline) that is the OOM the ceiling at the
    door exists to prevent, just 4x further out than the reasoning that put it
    there. Same question, same source of truth: xp.device_free_bytes, which
    reads MemAvailable for "cpu".

    Unknown free memory does NOT refuse, exactly as in _fit_error: there the
    rail is the only guard, and guessing would be worse.
    """
    need = PREVIEW_BYTES_PER_CELL[ndim]*cells*PREVIEW_HEADROOM
    free = xp.device_free_bytes("cpu")
    if free is None or need <= free*CPU_FIT_MARGIN:
        return None
    return ("this grid needs ~%.1f GiB of host memory to preview and only "
            "%.1f GiB is available. Reduce an axis."
            % (need/1024**3, free/1024**3))


def _release(b, pool):
    """Give the build's VRAM back to the driver. Call this only once the
    arrays are unreachable — free_all_blocks() frees blocks that are already
    FREE, so one surviving reference keeps the whole 5.5 GiB resident (the
    lesson _release_gpu_pool records in core/worker.py). On the FAILURE path
    that reference is the in-flight exception's traceback, which pins
    _build_frame's frame; hence preview_wigner calls this a second time once
    the handler has exited.

    A MemoryPool is per-device internally, so re-enter the build's device or
    free_all_blocks() empties the wrong arena. The per-thread cuFFT plan cache
    goes first: under a private allocator a plan's work area comes from `pool`,
    so the cache would hold it against the free. Unlike
    worker._release_gpu_pool this does NOT touch the pinned pool — that is host
    RAM shared with every worker's device->host staging, and a preview has no
    business reclaiming it."""
    if not b.is_gpu:
        return
    try:
        with b.device():
            try:
                b.xp.fft.config.get_plan_cache().clear()
            except Exception:
                log.debug("preview: plan cache clear failed", exc_info=True)
            pool.free_all_blocks()
    except Exception:
        log.debug("preview: pool release failed", exc_info=True)


class PotentialPreviewIn(BaseModel):
    expr: str
    x1: float
    x2: float
    n: int = Field(default=400, ge=16, le=4096)
    grid: Optional[GridSpec] = None   # enables the extended-range quantum probe
    hbar_eff: float = Field(default=1.0, gt=0)
    # 2D only: the y window to sample the heatmap over. The editor's y-cut chart
    # zooms this exactly as x1/x2 zoom the x one; absent (the unzoomed case) it
    # defaults to the grid's own y extent. Neither window restricts the validity
    # probes, which read req.grid — see the comment in preview_potential.
    y1: Optional[float] = None
    y2: Optional[float] = None
    # 2D heatmap resolution (the client reads its axis cuts out of the same
    # array, so one sample grid serves both)
    n2: int = Field(default=128, ge=16, le=512)


@router.post("/preview/potential")
def preview_potential(req: PotentialPreviewIn):
    ndim = req.grid.ndim if req.grid is not None else 1
    ext = req.grid.spatial_extended(req.hbar_eff) if req.grid is not None else None
    # What to PLOT: the request's own x window (the editor zooms it), and for
    # the second axis whatever it asked for, else the grid's extent.
    sample = [(req.x1, req.x2)]
    if ndim > 1:
        gy = req.grid.axes[1]
        sample.append((req.y1 if req.y1 is not None else gy.lo,
                       req.y2 if req.y2 is not None else gy.hi))
    sample = tuple(sample)
    # What to VALIDATE on: the SIMULATION box, never the zoom. These are two
    # different questions and conflating them makes the panel's verdict differ
    # from the one routers.sessions.compile_for will give — zoom past a pole
    # (1/x on [-6,6], zoomed to [1,6]) and the classical badge reads ✓, the
    # Solve gate opens, and POST /sessions 422s on a potential the editor just
    # approved. Only the QUANTUM probe box (`extended`) was ever grid-derived.
    ranges = req.grid.spatial_ranges() if req.grid is not None else sample
    try:
        cp = compile_potential(req.expr, ndim=ndim, ranges=ranges,
                               extended=ext)
    except PotentialError as e:
        return {"ok": False, "error": str(e)}
    out = {
        "ok": True,
        "ndim": ndim,
        "validity": {"quantum": cp.quantum_valid, "classical": cp.classical_valid},
        "reasons": cp.reasons,
        "warnings": cp.warnings,
        "latex": cp.latex,
        # the FULL gradient, one entry per spatial axis. The old `dudx_latex`
        # scalar is gone: at ndim=2 it was grad_latex[0], i.e. dU/dx presented as
        # the gradient of a two-variable potential, and nothing has ever read it.
        "grad_latex": list(cp.grad_latex),
        "extended_range": [list(r) for r in ext] if ext else None,
    }
    if ndim == 1:
        xs, us = sample_potential(cp, req.x1, req.x2, req.n)
        out["samples"] = {"x": xs, "U": us}
    else:
        xs, ys, rows = sample_potential_grid(cp, sample, req.n2)
        out["samples"] = {"x": xs, "y": ys, "U": rows}
    return out


class WignerPreviewIn(ICSpec):
    grid: GridSpec
    hbar_eff: float = Field(default=1.0, gt=0)
    # NOT what this endpoint computes in — the preview always builds in float64
    # on its own backend, and PREVIEW_BYTES_PER_CELL is right to ignore this.
    # It is the precision of the SESSION the form would create, and it is here
    # solely so grid_limit_error's refusal can quote that session's footprint.
    # Without it the message quoted float64 at a form set to float32 and
    # over-reported by 1.9x — 52.0 GiB against the 28.0 the Setup panel showed
    # two columns away, for the same grid.
    precision: Optional[Literal["float64", "float32"]] = None


def _build_frame(b, req, compiled=None):
    """Build the preview bundle on `b`. EVERY device array is a local of this
    function, so all of them die when it returns — which is precisely what lets
    the caller's _release() hand the VRAM back. Keeping the build in its own
    frame is the point, not a style choice: the same structural reason
    session._sweep_idle exists (a name still bound to a device array outlives
    the work and pins gigabytes).

    Returns only host data: `wq` comes back through backend.asnumpy inside
    quantize, and rho/phi likewise out of observables."""
    with b.device():
        g, W, warns, edge, norm = initial.from_spec(
            req.grid, req, req.hbar_eff, b, compiled=compiled)
        # The measured deficit for the analytically-normalized Gaussian kinds;
        # for the expression kinds the state carries its own answer (psi knows
        # what fraction of its mass the box holds, and a wexpr was normalized BY
        # this very sum, so measuring it here would report a structural zero).
        if norm.method == "analytic":
            norm = replace(norm, deficit=abs(1.0 - float(W.sum())*g.dmu))
        vf, _obs = frame.build(g.shift(W), g, req.hbar_eff)
        payload = protocol.pack_frame(0, 0.0, g.geom(), [vf],
                                      flags=protocol.FLAG_LIVE_PREVIEW)
    return payload, norm, warns, edge


def _respond(payload, norm, warns, edge=()):
    # HTTP headers are latin-1 only: percent-encode so warnings can carry
    # Unicode (sigma, hbar, rho...); the client decodeURIComponent()s it.
    #
    # The edge finding rides in its OWN header, as `axis:mass` pairs, rather
    # than as prose in Warnings: the session's boundary watch reports the same
    # fact about the same axes from record 0 on, so the client has to be able to
    # drop the axes already covered and word only what is left. A string it had
    # to pattern-match would make that a guess.
    #
    # The deficit header is EMPTY when the number is not a diagnostic, rather
    # than a structural zero: an expression W is normalized by its own grid sum,
    # so "norm deficit on grid: 0.000e+00" would be a health check that always
    # passes, which reads as reassurance and is worse than no line at all. The
    # client renders that line under `v-if`, so an empty value removes it.
    # X-Wignerf-IC-Norm carries what IS worth saying in its place — for a wexpr
    # the raw integral it was divided by, which pre-answers "why does
    # multiplying my expression by 5 change nothing?".
    return Response(content=payload, media_type="application/octet-stream",
                    headers={"X-Wignerf-Norm-Deficit":
                             "%.3e" % norm.deficit if norm.method != "grid-sum"
                             else "",
                             "X-Wignerf-IC-Norm": "%s:%.6g:%.6g"
                             % (norm.method, norm.raw, norm.momentum_mass),
                             "X-Wignerf-Warnings": quote(" | ".join(warns)),
                             "X-Wignerf-Edge": ",".join(
                                 "%s:%.3e" % (a, m) for a, m in edge)})


@router.post("/preview/wigner")
def preview_wigner(req: WignerPreviewIn):
    # The SAME ceilings session creation enforces, and for a stronger reason:
    # this endpoint builds the FULL state at the requested grid, and it fires on
    # every keystroke in the setup form — long before anyone presses Restart. It
    # used to have no bound at all, so a form grid of 256^4 (4.3e9 cells, one
    # dims switch away from the 1D default) went to the CPU fallback below and
    # allocated 34 GiB arrays until the kernel OOM-killed the server. The GPU
    # path was never the risk: _pick_device simply declines and falls through.
    # An omitted precision resolves to the host default here for the same reason
    # SessionCreate._check resolves it that way: this message describes the
    # session the grid WOULD create, so it has to answer the same question the
    # same way or the two disagree about the same grid.
    bad = grid_limit_error(req.grid, precision=req.precision or config.PRECISION)
    if bad:
        raise HTTPException(422, bad)
    # A malformed IC is a CLIENT error and is decided here, before any backend
    # is touched: it costs a handful of tuples and no arrays. Deciding it inside
    # the build instead is what forced the GPU branch below to translate every
    # ValueError into a 422 — including cupy's, which turned a device problem
    # into "your IC is wrong" and skipped the CPU fallback that exists for it.
    #
    # compile_ic covers both shapes: components_of for the Gaussian kinds, and a
    # sympy parse for the expression ones. The parse allocates nothing either,
    # which is what keeps it on this side of the device choice — and the compiled
    # result is then handed to the build rather than recomputed inside it.
    try:
        compiled = initial.compile_ic(req, req.hbar_eff, req.grid.ndim,
                                      ranges=initial.probe_ranges(req.type, req.grid,
                                                 req.hbar_eff))
    except ValueError as e:
        raise HTTPException(422, str(e))
    spec = _pick_device(req.grid.cells, req.grid.ndim)
    # What to release once the failed build's frame is unreachable. The
    # `finally` below covers the success path, but it CANNOT free a failed one:
    # while the exception propagates its traceback still references
    # _build_frame's frame and every device array in it, and free_all_blocks()
    # frees only blocks that are already free. Nor would a release inside the
    # `except` help — the exception is live for the whole handler. Measured at
    # 128 MiB: `finally` alone left all of it reserved, `finally` plus this
    # left none. Without it a preview that OOMs at 8192^2 parks multiple GiB on
    # the card until the next SUCCESSFUL preview, starving the very solver the
    # fallback exists to protect.
    failed = None
    ic_error = None          # a CLIENT error caught on the GPU path, re-raised
                             # after the release below rather than from inside
                             # the handler that cannot free anything
    if spec is not None:
        b = pool = None      # _backend() itself can raise (a vanished device)
        try:
            b = _backend(spec)
            pool = _pool(spec, b)
            with _gpu_lock, b.xp.cuda.using_allocator(pool.malloc):
                try:
                    return _respond(*_build_frame(b, req, compiled))
                finally:
                    _release(b, pool)
        except initial.ICError as e:
            # NOT a device failure, so NOT a fallback: an IC that cannot be
            # built (a W integrating to zero, a psi that is not finite) will
            # fail identically on the CPU and then surface as a 500. Most of
            # these are only knowable once W exists, which is why they cannot
            # join the pre-flight above — but they are still the client's.
            #
            # THE RELEASE IS DEFERRED THE SAME WAY THE OOM PATH'S IS, and this
            # is the whole point of `failed` existing. Calling _release here
            # would free nothing at all, for the reason stated above the `failed
            # = None` line: the exception is live for the duration of this
            # handler, so its traceback still references _build_frame's frame
            # and every device array in it. A 422 would then park the whole
            # build on the card until some LATER preview succeeded, which is
            # exactly the starvation the structure exists to prevent — and it
            # is worse here than on the OOM path, because a client typing an
            # expression hits this repeatedly and by design.
            if pool is not None:
                failed = (b, pool)
            ic_error = e
        except Exception:
            # OOM above all (another session can claim the card between the
            # free-memory check and the build), but anything device-shaped
            # lands here: the preview must still come back, just slower.
            #
            # format_exc(), NOT exc_info=True: a LogRecord built with exc_info
            # stores the (type, value, TRACEBACK) tuple, and any handler that
            # keeps records — pytest's log capture does, and so does anything
            # buffering for a report — then holds _build_frame's frame and its
            # device arrays alive past the release below, which is the whole
            # bug this structure exists to fix. Rendering the frames to text
            # here keeps every line of the diagnostic and none of the
            # references.
            log.warning("preview: GPU build on %s failed, falling back to "
                        "CPU\n%s", spec, traceback.format_exc())
            if pool is not None:
                failed = (b, pool)
    if failed is not None:
        _release(*failed)
    # Now that the handler has exited and the frame is unreachable, the client's
    # own error can be answered — and it must NOT fall through to the CPU, which
    # would rebuild the identical bad IC just to fail identically.
    if ic_error is not None:
        raise HTTPException(422, str(ic_error))
    # The CPU fallback's own fit check — the last guard before an allocation
    # nothing else bounds. After the release above, so a failed GPU build hands
    # its VRAM back either way.
    cpu_bad = _cpu_fit_error(req.grid.cells, req.grid.ndim)
    if cpu_bad:
        # 503 when a GPU was picked and merely failed: the grid may be perfectly
        # fine and the shortage transient, so this is "not right now", not "your
        # request is wrong". 422 when the CPU was always the target.
        raise HTTPException(503 if spec is not None else 422, cpu_bad)
    try:
        return _respond(*_build_frame(_backend("cpu"), req, compiled))
    except initial.ICError as e:
        raise HTTPException(422, str(e))


class WavefunctionPreviewIn(BaseModel):
    """ψ(x) / ψ(x, y) for the IC editor's two line charts. A BARE expression
    string, exactly as PotentialPreviewIn takes one — this endpoint answers a
    syntax error without building any state, which is the whole reason it is not
    folded into /preview/wigner: that one returns a binary frame and can cost a
    multi-GiB transient, and moving the cut selector must not pay for one."""
    expr: str
    grid: GridSpec
    hbar_eff: float = Field(default=1.0, gt=0)
    # Which SPATIAL axis to cut along at ndim=2. The momentum axis follows from
    # axes.conjugate rather than being chosen separately, so the pair can never
    # be mismatched — the classic multi-D error, and a silent one.
    cut_axis: int = Field(default=0, ge=0, le=1)
    n: int = Field(default=512, ge=16, le=4096)


@router.post("/preview/wavefunction")
def preview_wavefunction(req: WavefunctionPreviewIn):
    """ψ on the spatial lattice and φ on the grid's OWN momentum lattice, each
    as re/im (the client squares them for |·|²).

    φ IS COMPUTED BY QUADRATURE ON THE FORM'S MOMENTUM AXIS, never by an FFT of
    ψ. The FFT's dual lattice is 2πℏ/(N·dq) wide and in general is NOT the
    momentum axis the user chose — and this chart sits directly under a W
    preview whose vertical axis IS that axis. Two adjacent plots of one state on
    two different momentum axes is exactly the class of error lib/potentialCuts
    exists to record.

    An {ok, error} union rather than an HTTP error, matching /preview/potential:
    a half-typed expression is the normal state of an input someone is typing
    into, not an exceptional one.
    """
    nd = req.grid.ndim
    # The same rail /preview/wigner is behind. This endpoint allocates too — psi
    # on the spatial lattice and, per axis, a chunked (N_k x N_q) quadrature —
    # and it fires on every keystroke, so a grid a session would refuse must not
    # be sampled here either.
    bad = grid_limit_error(req.grid)
    if bad:
        return {"ok": False, "error": bad}
    # A blank box is the normal state of an input someone is typing into, and it
    # is answered HERE rather than by constructing an ICSpec and letting its
    # validator fire: pydantic's ValidationError is a ValueError, so it would be
    # caught below and rendered verbatim — the multi-line "1 validation error
    # for ICSpec … [type=value_error, input_value=…]" blob, echoing the input,
    # straight into the strip under the charts. That blob is what ICSpec's own
    # hand-written sentences exist to avoid, and apiErrorText never sees it
    # because this endpoint answers 200.
    if not (req.expr or "").strip():
        return {"ok": False, "error": "ψ: an expression is required"}
    try:
        cic = initial.compile_ic(
            ICSpec(type="psi", expr=req.expr), req.hbar_eff, nd,
            ranges=req.grid.spatial_extended(req.hbar_eff))
        psi, phi = initial.wavefunction_cuts(
            req.grid, cic, req.hbar_eff, _backend("cpu"),
            cut_axis=min(req.cut_axis, nd - 1), n=req.n)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except MemoryError:
        # Not a ValueError, so it used to escape as a 500. The docstring
        # promises this endpoint answers in the union, and a host under memory
        # pressure is exactly when a half-typed expression should not take the
        # request down.
        return {"ok": False, "error": "not enough memory to sample ψ on this "
                                      "grid — reduce the axis sizes"}
    # `axis` is an INDEX, never a label: labels live in core/axes.py,
    # lib/axes.ts and render_mpl.py, and a fourth copy on the wire is how that
    # three-way mirror breaks.
    return {"ok": True, "ndim": nd, "psi": psi, "phi": phi}
