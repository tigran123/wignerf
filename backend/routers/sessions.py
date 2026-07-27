"""
Session lifecycle: create (validates + compiles the potential, spawns the
solver workers), inspect, delete, and the scalar time-series backfill.
"""

import logging
import os
import time
from functools import partial

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import config
from core import describe
from core import session as sessions
from core.potential import PotentialError, compile_potential
from core.protocol import VARIANTS, SessionCreate, grid_limit_error
from core import xp
from core.xp import devices_allowed, resolve_devices

log = logging.getLogger(__name__)
router = APIRouter()


def _fft_threads(n_variants):
    if config.FFT_THREADS > 0:
        return config.FFT_THREADS
    return max(1, min(4, (os.cpu_count() or 4)//(2*n_variants)))


async def compile_for(cfg_grid, expr, hbar_eff, variants):
    """Compile U off the event loop and enforce per-family validity for
    the requested variants. Raises HTTPException(422) on failure."""
    try:
        cp = await to_thread.run_sync(partial(
            compile_potential, expr, ndim=cfg_grid.ndim,
            ranges=cfg_grid.spatial_ranges(),
            extended=cfg_grid.spatial_extended(hbar_eff)))
    except PotentialError as e:
        raise HTTPException(422, "potential: %s" % e)
    needs_q = any(VARIANTS[v]["quantum"] for v in variants)
    needs_c = any(not VARIANTS[v]["quantum"] for v in variants)
    if needs_q and not cp.quantum_valid:
        raise HTTPException(422, "potential is not quantum-valid: "
                            + "; ".join(cp.reasons))
    if needs_c and not cp.classical_valid:
        raise HTTPException(422, "potential is not classical-valid: "
                            + "; ".join(cp.reasons))
    return cp


# One CUDA context + cuFFT plan cache per process per device, on top of the
# per-worker arrays (CLAUDE.md's GPU section measures it at ~300 MiB).
CONTEXT_BYTES = 300*1024**2
# Leave a tenth of the card. Free memory is a moving target — another process
# can claim some between this check and the first allocation — and unlike the
# IC preview a session has no CPU fallback to drop to.
FIT_MARGIN = 0.9


def _fit_error(cfg, devices):
    """Whether the devices this session's workers land on actually have room,
    or None when they do / when it cannot be told.

    This is the guard that MEANS something in 2D. WIGNERF_MAX_CELLS_2D is a
    fixed cell count, i.e. a proxy for "will it fit", and a proxy is wrong in
    both directions: it refused 128×128×64×64 (13.0 GiB for one worker) on a
    24 GiB card, and it would have waved through 6.5 GiB × 2 workers onto an
    11 GiB one. Ask the driver instead — the same question
    routers/preview.py's _pick_device asks — and let the rail be only a rail.

    Skipped at ndim=1: WIGNERF_MAX_GRID already bounds a 2D array to 4096² =
    16.8M cells (~2.7 GiB/worker), so 1D cannot reach the sizes that need this.
    N⁴ can, which is the whole point.
    """
    if cfg.grid.ndim < 2:
        return None
    per = cfg.grid.cells*config.BYTES_PER_CELL_2D
    assignment = sessions.assign_devices(cfg.variants, devices)
    counts = {}
    for dev in assignment.values():
        counts[dev] = counts.get(dev, 0) + 1
    for dev, n in sorted(counts.items()):
        free = xp.device_free_bytes(dev)
        if free is None:
            continue                       # cannot tell: the rail is the guard
        need = n*per + (CONTEXT_BYTES if dev != "cpu" else 0)
        if need > free*FIT_MARGIN:
            return ("%s has %.1f GiB free and this session would put %.1f GiB "
                    "on it (%d worker%s × %.1f GiB%s). Reduce an axis, drop a "
                    "variant, or pick a device with more room."
                    % (dev, free/1024**3, need/1024**3, n,
                       "" if n == 1 else "s", per/1024**3,
                       "" if dev == "cpu" else " + 0.3 GiB CUDA context"))
    return None


@router.post("/sessions")
async def create_session(cfg: SessionCreate, request: Request):
    if cfg.ic.type == "mixture" and any(c.sigma_k is None
                                        for c in cfg.ic.components):
        raise HTTPException(422, "sigma_k is required for mixture components")
    bad = grid_limit_error(cfg.grid, len(cfg.variants))
    if bad:
        raise HTTPException(422, bad)
    nd = cfg.grid.ndim
    # the ceilings this session must report and (for auto-expand) plan against
    cap, cells = config.max_grid(nd), config.max_cells(nd)
    cp = await compile_for(cfg.grid, cfg.potential, cfg.hbar_eff, cfg.variants)
    # Per-session overrides NARROW the host's policy, never widen it: the
    # history cap is clamped to WIGNERF_HISTORY_MB (a client must not be able
    # to ask for more RAM than the box has), and an unknown/absent device is a
    # 422 rather than a worker that dies on start.
    history_mb = min(cfg.history_mb or config.HISTORY_MB, config.HISTORY_MB)
    device = cfg.device or config.DEVICE
    if cfg.device:
        # Resolve HERE, not inside SimSession, so a bad spec is a clean 422
        # naming the device instead of an exception from somewhere in session
        # construction — and so this except cannot swallow an unrelated
        # ValueError raised later by the grid or the IC.
        try:
            want = resolve_devices(cfg.device)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(422, "device %r: %s" % (cfg.device, e))
        # Existing-and-resolvable is not the same as allowed. Without this a
        # host pinned to WIGNERF_DEVICE=cuda:1 (or to cpu, to keep its cards
        # free for something else) could be overridden by any client that asked
        # for another device, which makes the env var a suggestion rather than
        # a policy. `allowed` is the exact list /api/device advertises as
        # `choices`, so the Setup panel can never offer a device this refuses.
        allowed = devices_allowed(config.DEVICE)
        outside = [d for d in want if d not in allowed]
        if outside:
            raise HTTPException(
                422, "device %r: %s not available to sessions on this host — "
                "WIGNERF_DEVICE=%s allows %s"
                % (cfg.device, ", ".join(outside), config.DEVICE,
                   ", ".join(allowed)))
    # The devices this session's workers will actually land on — and whether
    # they have room. Checked here, after the device policy above has settled
    # which pool applies, and before anything is allocated.
    fit = _fit_error(cfg, resolve_devices(device))
    if fit:
        raise HTTPException(422, fit)
    s = sessions.create_session(
        cfg, cp, device=device,
        fft_threads=_fft_threads(len(cfg.variants)),
        history_bytes=history_mb*1024*1024,
        max_grid=cap, history_mb_max=config.HISTORY_MB, max_cells=cells)
    # Prefix the WS path with the app's root_path so it inherits the nginx
    # prefix (uvicorn --root-path /wignerf, from APP_ROOT_PATH). Empty in dev.
    root_path = request.scope.get("root_path", "").rstrip("/")
    return {"session_id": s.id, "ws_url": "%s/api/ws/%s" % (root_path, s.id),
            "variants": cfg.variants, "record_dt": cfg.record_dt,
            "warnings": cp.warnings}


@router.get("/sessions/{sid}")
def get_session(sid: str):
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    return s.status()


@router.get("/sessions/{sid}/setup")
def get_setup(sid: str):
    """The session's ORIGINAL config as a downloadable document — what the
    run started from, which the SPA can import back into the setup form (the
    same blob rides in every exported mp4's `comment` tag). Descriptive
    filename, like the video export."""
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    doc = describe.setup_document(s.cfg, s.param_log)
    g = doc["config"]["grid"]
    name = "wignerf-setup-%s-%s-%s.json" % (
        "-".join(v.upper() for v in doc["config"]["variants"]),
        "x".join(str(a["N"]) for a in g["axes"]),
        time.strftime("%Y%m%d-%H%M"))
    return JSONResponse(doc, headers={
        "Content-Disposition": 'attachment; filename="%s"' % name})


@router.delete("/sessions/{sid}")
def delete_session(sid: str):
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    # Arrival timestamp: compared against uvicorn's "connection closed" line it
    # measures how long the browser sat on the keepalive DELETE before flushing
    # it (worst on a full-quit vs a reload/tab-close).
    log.info("DELETE %s received — closing", sid)
    t0 = time.monotonic()
    s.close()
    t1 = time.monotonic()
    # Drop our OWN reference before collecting: `s` is a live frame local, so
    # leaving it bound roots the session↔worker cycle and gc.collect() below
    # would free nothing. With it gone, the only remaining root is a streamer
    # coroutine still unwinding on the event loop (if any) — and
    # _collect_closed keeps the sweep armed for exactly that case, so the 5 s
    # sweeper retries once it releases. This makes RSS return here in the
    # common case and bounds it at one sweep otherwise.
    del s
    # Reap the just-closed session's cyclic FrameHistory NOW rather than
    # leaving it to the next TTL sweep. This is the prompt-departure path — the
    # frontend's pagehide beacon DELETEs here — so returning RSS to the OS
    # promptly is the whole point. close() armed _closed_since_sweep, so this
    # runs the gc.collect()+malloc_trim. Safe on the event loop: delete_session
    # is a sync endpoint, so FastAPI runs it in a threadpool thread.
    sessions._collect_closed()
    # close() = worker join (VRAM; but a join that times out on a mid-record
    # worker frees the card LATER — see "worker … released GPU pool");
    # collect = gc + malloc_trim (RSS).
    log.info("DELETE %s: close() %.2fs, collect %.2fs",
             sid, t1 - t0, time.monotonic() - t1)
    return {"ok": True}


def _series_variant(v):
    """Per-record scalars of one variant. The generic mean/std lists plus, at
    ndim=1, the flat x/p spelling the SPA has always read — the same
    compatibility bargain session.grid_payload makes."""
    d = {"vid": v.vid, "E": v.E, "purity": v.purity, "dt": v.dt,
         "mean": list(v.mean), "std": list(v.std), "lz": v.lz}
    if v.ndim == 1:
        d.update(x_mean=v.x_mean, x_std=v.x_std,
                 p_mean=v.p_mean, p_std=v.p_std)
    return d


@router.get("/sessions/{sid}/series")
def series(sid: str, start: int = 0, end: int = 1 << 62):
    """Per-record scalars (gapless even when live streaming skipped frames)."""
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    first, last = s.history.extent()
    if last < 0:
        return {"records": [], "extent": [first, last]}
    lo, hi = max(start, first), min(end, last)
    out = []
    for k in range(lo, min(hi, lo + 2000) + 1):
        rec = s.history.get(k)
        if rec is None:
            continue
        t, _geom, variants = rec
        out.append({"n": k, "t": t,
                    "variants": [_series_variant(v) for v in variants]})
    return {"records": out, "extent": [first, last]}
