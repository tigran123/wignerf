"""Health and device-info endpoints."""

from functools import lru_cache

from fastapi import APIRouter

import config
from core.xp import ArrayBackend, devices_allowed, resolve_devices

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
        return {"spec": d, "device": b.name, "is_gpu": b.is_gpu,
                "fft_provider": b.fft_provider}
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
    return _probe_backend()
