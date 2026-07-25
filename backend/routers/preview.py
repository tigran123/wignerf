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
from core import initial, observables, protocol
from core.potential import PotentialError, compile_potential, sample_potential
from core.protocol import GridSpec, ICSpec
from core.quantize import quantize
from core.xp import ArrayBackend, resolve_devices

log = logging.getLogger(__name__)
router = APIRouter()

# Measured on an RTX 3090, 2026-07-25: the transient peak of a cat IC build is
# 88 bytes per grid cell — 0.34 GiB at 2048^2, 1.38 at 4096^2, 5.50 at 8192^2 —
# and it plateaus there (64 for a single component, 88 from three up, unchanged
# at eight), because cat_wigner reuses its temporaries per pair rather than
# accumulating them.
PREVIEW_BYTES_PER_CELL = 88
PREVIEW_HEADROOM = 1.4     # never take the last of a card a session is using

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


def _pick_device(cells):
    """The CUDA device with the most free VRAM, if this build comfortably fits
    in it; otherwise None, meaning use the CPU. Asking the driver for free
    memory (rather than tracking our own sessions) is what keeps a preview from
    evicting a running solver: whatever else is on the card, including another
    process entirely, is already reflected."""
    need = PREVIEW_BYTES_PER_CELL*cells*PREVIEW_HEADROOM
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


@router.post("/preview/potential")
def preview_potential(req: PotentialPreviewIn):
    x_ext = None
    x_range = (req.x1, req.x2)
    if req.grid is not None:
        x_ext = req.grid.x_extended(req.hbar_eff)
        x_range = (req.grid.x1, req.grid.x2)
    try:
        cp = compile_potential(req.expr, x_range=x_range, x_extended=x_ext)
    except PotentialError as e:
        return {"ok": False, "error": str(e)}
    xs, us = sample_potential(cp, req.x1, req.x2, req.n)
    return {
        "ok": True,
        "validity": {"quantum": cp.quantum_valid, "classical": cp.classical_valid},
        "reasons": cp.reasons,
        "warnings": cp.warnings,
        "latex": cp.latex,
        "dudx_latex": cp.dUdx_latex,
        "samples": {"x": xs, "U": us},
        "extended_range": list(x_ext) if x_ext else None,
    }


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
        g, W, warns = initial.from_spec(req.grid, req, req.hbar_eff, b)
        deficit = abs(1.0 - float(W.sum())*g.dx*g.dp)
        Ws = g.shift2d(W)
        wq, wmin, wmax = quantize(Ws, b)
        obs = observables.compute_basic(Ws, g, req.hbar_eff)
        vf = protocol.VariantFrame(
            vid=0, wq=wq, wmin=wmin, wmax=wmax, E=0.0,
            x_mean=obs.x_mean, x_std=obs.x_std,
            p_mean=obs.p_mean, p_std=obs.p_std,
            purity=obs.purity, dt=0.0, rho=obs.rho, phi=obs.phi)
        payload = protocol.pack_frame(0, 0.0, g.geom(), [vf],
                                      flags=protocol.FLAG_LIVE_PREVIEW)
    return payload, deficit, warns


def _respond(payload, deficit, warns):
    # HTTP headers are latin-1 only: percent-encode so warnings can carry
    # Unicode (sigma, hbar, rho...); the client decodeURIComponent()s it.
    return Response(content=payload, media_type="application/octet-stream",
                    headers={"X-Wignerf-Norm-Deficit": "%.3e" % deficit,
                             "X-Wignerf-Warnings": quote(" | ".join(warns))})


@router.post("/preview/wigner")
def preview_wigner(req: WignerPreviewIn):
    spec = _pick_device(req.grid.Nx*req.grid.Np)
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
        except ValueError as e:
            # a bad IC spec, not a device problem — the CPU would reject it too
            raise HTTPException(422, str(e))
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
    try:
        return _respond(*_build_frame(_backend("cpu"), req))
    except ValueError as e:
        raise HTTPException(422, str(e))
