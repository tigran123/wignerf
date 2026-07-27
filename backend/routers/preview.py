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
from typing import Optional
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
    # 2D only: the y window to sample the heatmap over. Defaults to the grid's
    # own y extent; x1/x2 stay the (possibly zoomed) x window, exactly as in 1D.
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


def _build_frame(b, req):
    """Build the preview bundle on `b`. EVERY device array is a local of this
    function, so all of them die when it returns — which is precisely what lets
    the caller's _release() hand the VRAM back. Keeping the build in its own
    frame is the point, not a style choice: the same structural reason
    session._sweep_idle exists (a name still bound to a device array outlives
    the work and pins gigabytes).

    Returns only host data: `wq` comes back through backend.asnumpy inside
    quantize, and rho/phi likewise out of observables."""
    with b.device():
        g, W, warns, edge = initial.from_spec(req.grid, req, req.hbar_eff, b)
        deficit = abs(1.0 - float(W.sum())*g.dV)
        vf, _obs = frame.build(g.shift(W), g, req.hbar_eff)
        payload = protocol.pack_frame(0, 0.0, g.geom(), [vf],
                                      flags=protocol.FLAG_LIVE_PREVIEW)
    return payload, deficit, warns, edge


def _respond(payload, deficit, warns, edge=()):
    # HTTP headers are latin-1 only: percent-encode so warnings can carry
    # Unicode (sigma, hbar, rho...); the client decodeURIComponent()s it.
    #
    # The edge finding rides in its OWN header, as `axis:mass` pairs, rather
    # than as prose in Warnings: the session's boundary watch reports the same
    # fact about the same axes from record 0 on, so the client has to be able to
    # drop the axes already covered and word only what is left. A string it had
    # to pattern-match would make that a guess.
    return Response(content=payload, media_type="application/octet-stream",
                    headers={"X-Wignerf-Norm-Deficit": "%.3e" % deficit,
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
    bad = grid_limit_error(req.grid)
    if bad:
        raise HTTPException(422, bad)
    # A malformed IC is a CLIENT error and is decided here, before any backend
    # is touched: it costs a handful of tuples and no arrays. Deciding it inside
    # the build instead is what forced the GPU branch below to translate every
    # ValueError into a 422 — including cupy's, which turned a device problem
    # into "your IC is wrong" and skipped the CPU fallback that exists for it.
    try:
        initial.components_of(req, req.hbar_eff, req.grid.ndim)
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
    if spec is not None:
        b = pool = None      # _backend() itself can raise (a vanished device)
        try:
            b = _backend(spec)
            pool = _pool(spec, b)
            with _gpu_lock, b.xp.cuda.using_allocator(pool.malloc):
                try:
                    return _respond(*_build_frame(b, req))
                finally:
                    _release(b, pool)
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
    # The CPU fallback's own fit check — the last guard before an allocation
    # nothing else bounds. After the release above, so a failed GPU build hands
    # its VRAM back either way.
    cpu_bad = _cpu_fit_error(req.grid.cells, req.grid.ndim)
    if cpu_bad:
        # 503 when a GPU was picked and merely failed: the grid may be perfectly
        # fine and the shortage transient, so this is "not right now", not "your
        # request is wrong". 422 when the CPU was always the target.
        raise HTTPException(503 if spec is not None else 422, cpu_bad)
    return _respond(*_build_frame(_backend("cpu"), req))
