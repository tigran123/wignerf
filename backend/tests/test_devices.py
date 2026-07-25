"""
Device-pool resolution (core.xp.resolve_devices) and variant->device
assignment (core.session.assign_devices). Pure logic — CUDA is faked, so
these run everywhere including the CPU-only VPS.
"""

from types import SimpleNamespace

import pytest

from core import xp
from core.session import assign_devices


def fake_cupy(props):
    """A stand-in exposing just what resolve_devices touches."""
    rt = SimpleNamespace(getDeviceCount=lambda: len(props),
                         getDeviceProperties=lambda i: props[i])
    return SimpleNamespace(cuda=SimpleNamespace(runtime=rt))


# device 0 = the display card (2080 Ti-ish), device 1 = the big compute
# card (3090-ish): auto must put device 1 first despite PCI order.
TWO_GPUS = [
    {"multiProcessorCount": 68, "totalGlobalMem": 11 << 30, "name": b"slow"},
    {"multiProcessorCount": 82, "totalGlobalMem": 24 << 30, "name": b"fast"},
]


# -- resolve_devices ---------------------------------------------------------

def test_auto_no_cuda_is_cpu(monkeypatch):
    monkeypatch.setattr(xp, "_import_cupy", lambda: None)
    assert xp.resolve_devices("auto") == ["cpu"]


def test_auto_orders_fastest_first(monkeypatch):
    monkeypatch.setattr(xp, "_import_cupy", lambda: fake_cupy(TWO_GPUS))
    assert xp.resolve_devices("auto") == ["cuda:1", "cuda:0"]


def test_explicit_list_kept_as_written(monkeypatch):
    monkeypatch.setattr(xp, "_import_cupy", lambda: fake_cupy(TWO_GPUS))
    assert xp.resolve_devices("cuda:0,cuda:1") == ["cuda:0", "cuda:1"]
    assert xp.resolve_devices(" cuda:1 , cuda:0 ") == ["cuda:1", "cuda:0"]
    assert xp.resolve_devices("cpu") == ["cpu"]


def test_bad_specs_rejected(monkeypatch):
    monkeypatch.setattr(xp, "_import_cupy", lambda: fake_cupy(TWO_GPUS))
    with pytest.raises(ValueError):
        xp.resolve_devices("cuda:0,cuda:0")          # duplicate
    with pytest.raises(ValueError):
        xp.resolve_devices("auto,cuda:1")            # auto inside a list
    with pytest.raises(ValueError):
        xp.resolve_devices("tpu:0")                  # unknown kind
    with pytest.raises(RuntimeError):
        xp.resolve_devices("cuda:7")                 # no such device
    monkeypatch.setattr(xp, "_import_cupy", lambda: None)
    with pytest.raises(RuntimeError):
        xp.resolve_devices("cuda:0")                 # cuda without cupy


def test_bare_cuda_is_canonicalized(monkeypatch):
    """A bare "cuda" means device 0 — ArrayBackend has always defaulted it that
    way. It must come back CANONICAL, or devices_allowed's set membership
    rejects a device the host actually offers."""
    monkeypatch.setattr(xp, "_import_cupy", lambda: fake_cupy(TWO_GPUS))
    assert xp.resolve_devices("cuda") == ["cuda:0"]
    assert xp.resolve_devices("cuda:1,cuda") == ["cuda:1", "cuda:0"]
    with pytest.raises(ValueError):
        xp.resolve_devices("cuda,cuda:0")            # one device, named twice


# -- devices_allowed ---------------------------------------------------------

def test_allowed_is_the_pool_plus_cpu(monkeypatch):
    """cpu is always a legal session target but never appears in an "auto" pool
    on a CUDA host, so it is appended — and never duplicated when the pool
    already names it."""
    monkeypatch.setattr(xp, "_import_cupy", lambda: fake_cupy(TWO_GPUS))
    assert xp.devices_allowed("auto") == ["cuda:1", "cuda:0", "cpu"]
    assert xp.devices_allowed("cuda:0") == ["cuda:0", "cpu"]
    assert xp.devices_allowed("cuda:1,cpu") == ["cuda:1", "cpu"]
    assert xp.devices_allowed("cpu") == ["cpu"]


def test_allowed_excludes_a_card_outside_the_pool(monkeypatch):
    """The point of the whole helper: a host pinned to one card must not hand a
    session the other one."""
    monkeypatch.setattr(xp, "_import_cupy", lambda: fake_cupy(TWO_GPUS))
    assert "cuda:0" not in xp.devices_allowed("cuda:1")


def test_allowed_on_a_cpu_only_host(monkeypatch):
    monkeypatch.setattr(xp, "_import_cupy", lambda: None)
    assert xp.devices_allowed("auto") == ["cpu"]


# -- assign_devices ----------------------------------------------------------

def test_four_variants_two_gpus_rel_on_fast():
    a = assign_devices(["qn", "qr", "cn", "cr"], ["cuda:1", "cuda:0"])
    assert a == {"qr": "cuda:1", "cr": "cuda:1",
                 "qn": "cuda:0", "cn": "cuda:0"}


def test_two_variants_two_gpus_one_each():
    a = assign_devices(["qn", "cn"], ["cuda:1", "cuda:0"])
    assert a == {"qn": "cuda:1", "cn": "cuda:0"}


def test_three_variants_two_gpus_fast_takes_two():
    a = assign_devices(["qn", "qr", "cn"], ["cuda:1", "cuda:0"])
    assert a == {"qr": "cuda:1", "qn": "cuda:1", "cn": "cuda:0"}


def test_single_device_takes_all():
    a = assign_devices(["qn", "qr", "cn", "cr"], ["cpu"])
    assert set(a.values()) == {"cpu"}


def test_more_devices_than_variants():
    a = assign_devices(["qr"], ["cuda:1", "cuda:0"])
    assert a == {"qr": "cuda:1"}
