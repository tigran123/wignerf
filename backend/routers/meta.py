"""Health and device-info endpoints."""

from functools import lru_cache

from fastapi import APIRouter

import config
from core.xp import ArrayBackend, resolve_devices

router = APIRouter()


@lru_cache(maxsize=1)
def _probe_backend():
    """Probe the whole device pool; top-level device/is_gpu/fft_provider
    describe the fastest (first) device for backward compatibility.

    Each entry carries its bare `spec` ("cuda:1") as well as the human name
    ("cuda:1 (NVIDIA GeForce RTX 3090)"); only the bare form round-trips
    through SessionCreate.device.

    `devices` is the POOL — what a default session spreads its workers over.
    `choices` is what the Setup panel offers, which is the pool PLUS cpu: the
    CPU backend is always a legal target (and a useful one — a float64 sanity
    run, or keeping a session off a card you need elsewhere), but on a CUDA
    host `resolve_devices("auto")` returns GPUs only, so it would never appear
    in a list built from the pool alone."""
    def _probe(d):
        b = ArrayBackend(device=d)
        return {"spec": d, "device": b.name, "is_gpu": b.is_gpu,
                "fft_provider": b.fft_provider}
    try:
        infos = [_probe(d) for d in resolve_devices(config.DEVICE)]
        choices = list(infos)
        if not any(c["spec"] == "cpu" for c in choices):
            choices.append(_probe("cpu"))
        return {**infos[0], "devices": infos, "choices": choices,
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
