"""Health and device-info endpoints."""

from functools import lru_cache

from fastapi import APIRouter

import config
from core import axes as axes_mod
from core.xp import (ArrayBackend, device_total_bytes, devices_allowed,
                     resolve_devices)

router = APIRouter()


@lru_cache(maxsize=1)
def _probe_backend():
    """Probe the whole device pool; top-level device/is_gpu/fft_provider
    describe the fastest (first) device for backward compatibility.

    Each entry carries its bare `spec` ("cuda:1") as well as the human name
    ("cuda:1 (NVIDIA GeForce RTX 3090)"); only the bare form round-trips
    through SessionCreate.device.

    `devices` is the POOL — what a default session spreads its workers over.
    `choices` is what the Setup panel offers and what POST /api/sessions
    accepts: `xp.devices_allowed`, i.e. the pool plus cpu. Both come from that
    one helper deliberately — an offered device the API then refuses (or the
    reverse) is a form that lies about what it can do."""
    def _probe(d):
        b = ArrayBackend(device=d)
        # total_bytes, NOT free: this dict is lru_cached, and free memory moves.
        # Total is static, and it is what the Setup panel needs to say "this
        # grid can never fit here" without re-polling. Whether it fits RIGHT NOW
        # stays with routers/sessions._fit_error, which asks the driver at
        # create time and quotes the exact numbers.
        return {"spec": d, "device": b.name, "is_gpu": b.is_gpu,
                "fft_provider": b.fft_provider,
                "total_bytes": device_total_bytes(d)}
    try:
        pool = resolve_devices(config.DEVICE)
        allowed = devices_allowed(config.DEVICE)
        # probe each device ONCE: _probe builds a real ArrayBackend, and every
        # pool device appears in both lists
        probed = {d: _probe(d) for d in allowed}
        infos = [probed[d] for d in pool]
        return {**infos[0], "devices": infos,
                "choices": [probed[d] for d in allowed],
                "pool": config.DEVICE, "precision": config.PRECISION}
    except Exception as e:
        return {"device": "unavailable", "is_gpu": False,
                "devices": [], "choices": [], "pool": config.DEVICE,
                "precision": config.PRECISION, "error": str(e)}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/device")
def device():
    """Host facts the setup form needs BEFORE it can create anything.

    The per-ndim ceilings live here rather than on `status` because `status`
    reports them for the ndim of the session that is RUNNING — `config.max_grid`
    is resolved once, at creation — while the form has to describe the ndim it is
    SHOWING, and `dims` is restart-only, so the two disagree for as long as a
    switch sits ahead of its restart. Reading them off `status` made the Setup
    panel offer N up to 4096 for a 2D grid the API refuses past 128, hide the 2D
    footprint estimate entirely (its `bytes_per_cell` is 1D-null) and, in the
    other direction, collapse the 1D N select to a single option.

    Deliberately OUTSIDE `_probe_backend`'s lru_cache: these are cheap env
    reads, they must follow a monkeypatched `config` in the tests, and the
    probe's own error path must carry them too. Spread rather than mutate — that
    dict IS the cache.
    """
    return {**_probe_backend(),
            "max_grid": {str(n): config.max_grid(n) for n in axes_mod.NDIMS},
            "max_cells": {str(n): config.max_cells(n) for n in axes_mod.NDIMS},
            "bytes_per_cell_2d": config.BYTES_PER_CELL_2D}
