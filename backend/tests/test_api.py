"""REST API tests for the preview/meta endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import protocol
from core.quantize import dequantize
from main import app

client = TestClient(app)

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_device():
    r = client.get("/api/device")
    assert r.status_code == 200
    assert "device" in r.json()


def test_potential_preview_valid():
    r = client.post("/api/preview/potential",
                    json={"expr": "x^2/2", "x1": -6, "x2": 6, "grid": GRID})
    d = r.json()
    assert d["ok"] and d["validity"]["quantum"] and d["validity"]["classical"]
    assert len(d["samples"]["x"]) == 400
    assert d["extended_range"][0][0] < -6


def test_potential_preview_heaviside():
    r = client.post("/api/preview/potential",
                    json={"expr": "Heaviside(x)", "x1": -6, "x2": 6, "grid": GRID})
    d = r.json()
    assert d["ok"] and d["validity"]["quantum"] and not d["validity"]["classical"]


def test_the_validity_probe_follows_the_GRID_not_the_zoom():
    """Two different questions live in this request and must not be conflated.

    x1/x2 is the editor's PLOT window — it zooms, and zooming out past the
    domain is how the interesting part of U is found. The validity boxes are
    the SIMULATION grid, because that is what routers.sessions.compile_for will
    probe at create time. Tie the classical gradient probe to the zoom instead
    and the panel's verdict stops predicting the API's: zoom past a pole and the
    badge reads ✓, the Solve gate opens, and POST /sessions 422s on a potential
    the editor had just approved.
    """
    # 1/x is singular at 0, which is INSIDE the grid but outside the plot window
    r = client.post("/api/preview/potential",
                    json={"expr": "1/x", "x1": 1, "x2": 6, "grid": GRID})
    d = r.json()
    assert d["ok"] and not d["validity"]["classical"], d
    # ...while the samples still follow the zoom, which is what they are for
    assert d["samples"]["x"][0] == pytest.approx(1.0)

    # and the converse: a pole far OUTSIDE the domain must not block Solve,
    # however far out the plot is zoomed. 30 is clear of the extended Bopp
    # range at this grid too, so this isolates the range plumbing.
    r = client.post("/api/preview/potential",
                    json={"expr": "1/(x - 30)", "x1": -50, "x2": 50,
                          "grid": GRID})
    d = r.json()
    assert d["ok"] and d["validity"]["classical"] and d["validity"]["quantum"], d
    assert d["samples"]["x"][0] == pytest.approx(-50.0)


def test_potential_preview_rejected():
    r = client.post("/api/preview/potential",
                    json={"expr": "__import__('os')", "x1": -6, "x2": 6})
    d = r.json()
    assert not d["ok"] and "error" in d


def _decode(resp):
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/octet-stream")
    return protocol.unpack_frame(resp.content)


def test_wigner_preview_mixture_roundtrip():
    r = client.post("/api/preview/wigner", json={
        "type": "mixture", "grid": GRID,
        "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707, "sigma_p": 0.707}],
    })
    f = _decode(r)
    assert (f.geom.Nx, f.geom.Np) == (64, 64)
    assert (f.geom.x1, f.geom.x2, f.geom.p1, f.geom.p2) == (-6.0, 6.0, -7.0, 7.0)
    assert f.flags & protocol.FLAG_LIVE_PREVIEW
    v = f.variants[0]
    W = dequantize(v.wq, v.wmin, v.wmax)
    # dequantized norm: 16-bit quantization of a 64x64 frame keeps the grid
    # integral within ~1e-3
    assert abs(W.sum()*(12./64)*(14./64) - 1.0) < 1e-3
    assert v.x_mean == pytest.approx(2.0, abs=1e-3)
    assert float(r.headers["X-Wignerf-Norm-Deficit"]) < 1e-4
    # marginals are float32 natural-order arrays of the right size
    assert v.rho.shape == (64,) and v.phi.shape == (64,)
    assert np.argmax(v.rho) == np.abs(np.linspace(-6, 6, 64, endpoint=False) - 2.0).argmin()


def test_wigner_preview_cat_has_negativity():
    r = client.post("/api/preview/wigner", json={
        "type": "cat", "grid": GRID, "hbar_eff": 1.0,
        "components": [{"x0": -2.0, "p0": 0.0, "sigma_x": 0.5},
                       {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5}],
    })
    assert _decode(r).variants[0].wmin < 0


def test_wigner_preview_mixture_requires_sigma_p():
    r = client.post("/api/preview/wigner", json={
        "type": "mixture", "grid": GRID,
        "components": [{"x0": 0.0, "p0": 0.0, "sigma_x": 0.5}],
    })
    assert r.status_code == 422


# -- Wigner preview device selection -------------------------------------
#
# The preview is built at the SESSION's grid, so at 8192^2 it is the same
# 67M-cell array the solver evolves — 25.9 s on the CPU against 0.50 s on an
# RTX 3090, on every reload AND every IC edit, while the main W panel showed
# the identical array in 1.4 s because a GPU worker built it. It now runs on a
# GPU when one has room and gives the VRAM straight back. These tests pin the
# parts that must hold with or without CUDA.

CAT2 = {"type": "cat", "grid": GRID,
        "components": [{"x0": -2.0, "p0": 0.0, "sigma_x": 0.5},
                       {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5}]}


def test_preview_falls_back_to_cpu_when_the_device_path_fails():
    """A device that vanishes between the free-memory check and the build (a
    session claiming the card, an OOM, a driver hiccup) must still produce a
    preview — slower, never a 500. Naming a device that cannot exist drives
    the same path without needing CUDA."""
    import routers.preview as pv
    saved = pv._pick_device
    pv._pick_device = lambda cells, ndim=1: "cuda:99"
    try:
        r = client.post("/api/preview/wigner", json=CAT2)
        assert r.status_code == 200
        assert _decode(r).variants[0].wmin < 0        # a real cat frame
    finally:
        pv._pick_device = saved


def test_preview_bad_ic_is_422_before_anything_is_built():
    """A malformed IC is not a device problem: it must 422 without being built
    at all, let alone built twice.

    It used to be decided INSIDE the build, which is why the GPU branch had to
    translate every ValueError into a 422 — cupy's included, so a transient
    device failure was reported as "your IC is wrong" and skipped the CPU
    fallback that exists for it. `initial.components_of` answers the question up
    front for a handful of tuples and no arrays."""
    import routers.preview as pv
    calls = []
    saved_build, saved_pick = pv._build_frame, pv._pick_device
    pv._pick_device = lambda cells, ndim=1: "cpu"       # _backend("cpu") always works

    def counting(b, req):
        calls.append(b)
        return saved_build(b, req)

    pv._build_frame = counting
    try:
        r = client.post("/api/preview/wigner", json={
            "type": "mixture", "grid": GRID,
            "components": [{"x0": 0.0, "p0": 0.0, "sigma_x": 0.5}]})
        assert r.status_code == 422
        assert "sigma_k" in r.text, r.text
        assert calls == [], "a bad IC reached a backend"
    finally:
        pv._build_frame, pv._pick_device = saved_build, saved_pick


def test_pick_device_refuses_a_build_that_would_not_fit():
    """The guard is free VRAM, so a grid too large for any card falls to the
    CPU instead of OOMing a running solver. 2^20 cells per axis is beyond any
    GPU made; on a CPU-only host this returns None for every size anyway."""
    import routers.preview as pv
    assert pv._pick_device(1 << 40) is None


def _gpu_preview_spec():
    """The device a preview of CAT2 would pick, or a skip. Every test below
    needs the same three guards."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")
    import routers.preview as pv
    spec = pv._pick_device(GRID["Nx"]*GRID["Np"])
    if spec is None:
        pytest.skip("no CUDA device with room")
    return cupy, pv, spec


def test_preview_releases_all_device_memory():
    """The peak is 5.5 GiB at 8192^2, so it has to come back — and it only
    does because _build_frame's device arrays die with its frame before
    _release() runs (free_all_blocks frees only what is already FREE).

    Also: it comes back out of the preview's OWN pool. The default pool is
    shared with every solver worker in the process, so releasing THAT one
    handed their cached blocks back to the driver on every IC keystroke — the
    opposite of what _pick_device's free-VRAM check is for. The seeded block
    below stands in for a running worker's cache and must survive untouched."""
    cupy, pv, spec = _gpu_preview_spec()
    default = cupy.get_default_memory_pool()
    with cupy.cuda.Device(int(spec.split(":")[1])):
        cached = cupy.zeros((1 << 20,), dtype=cupy.float64)   # 8 MiB
        del cached                       # now a FREE block in the DEFAULT pool
        seeded = default.total_bytes()
        assert seeded > 0, "the fixture failed to seed a cached block"
        assert client.post("/api/preview/wigner", json=CAT2).status_code == 200
        assert pv._pools[spec].used_bytes() == 0
        assert pv._pools[spec].total_bytes() == 0
        assert default.total_bytes() == seeded, \
            "the preview evicted a worker's cached blocks"
        default.free_all_blocks()


def test_preview_releases_device_memory_when_the_build_fails():
    """The path the CPU fallback exists for. While the exception propagates its
    traceback still references _build_frame's frame — hence every device array
    in it — so the release in the `finally` frees nothing, and a release inside
    the `except` would not either (the exception is live for the whole
    handler). Measured at 128 MiB: `finally` alone left all of it reserved.
    Without the second release a preview that OOMs at 8192^2 parks GiB on the
    card until the next SUCCESSFUL preview."""
    cupy, pv, spec = _gpu_preview_spec()
    saved = pv._build_frame

    def allocate_then_fail(b, req):
        if not b.is_gpu:
            return saved(b, req)
        # enter the device exactly as the real _build_frame does, or the block
        # lands in another device's arena and the release looks like a pass
        with b.device():
            hold = b.xp.zeros((1 << 21,), dtype=b.xp.float64)   # 16 MiB
            raise RuntimeError("simulated mid-build OOM, holding %d bytes"
                               % hold.nbytes)

    pv._build_frame = allocate_then_fail
    try:
        # the preview still comes back, on the CPU
        assert client.post("/api/preview/wigner", json=CAT2).status_code == 200
    finally:
        pv._build_frame = saved
    with cupy.cuda.Device(int(spec.split(":")[1])):
        assert pv._pools[spec].used_bytes() == 0
        assert pv._pools[spec].total_bytes() == 0


def test_preview_gpu_and_cpu_agree():
    """Same frame and same diagnostics either way — the device is a speed
    choice, never a physics one."""
    _cupy, pv, _spec = _gpu_preview_spec()
    gpu = client.post("/api/preview/wigner", json=CAT2)
    saved = pv._pick_device
    pv._pick_device = lambda cells, ndim=1: None
    try:
        cpu = client.post("/api/preview/wigner", json=CAT2)
    finally:
        pv._pick_device = saved
    assert gpu.status_code == cpu.status_code == 200
    # The QUANTIZED frame is bitwise identical — that is the strong statement,
    # and it is what the panel draws.
    g, c = _decode(gpu).variants[0], _decode(cpu).variants[0]
    assert np.array_equal(g.wq, c.wq)
    assert g.purity == pytest.approx(c.purity, rel=1e-9)
    # The same warnings fire (qualitative, so exact).
    assert gpu.headers["X-Wignerf-Warnings"] == cpu.headers["X-Wignerf-Warnings"]
    # The norm deficit is a global sum, so cuFFT's reduction order and numpy's
    # pairwise summation disagree in the last digits — measured 4.463e-13 vs
    # 4.461e-13. Both are machine noise; asserting the printed STRING would be
    # pinning the summation order, not the physics.
    dg = float(gpu.headers["X-Wignerf-Norm-Deficit"])
    dc = float(cpu.headers["X-Wignerf-Norm-Deficit"])
    assert dg < 1e-9 and dc < 1e-9
    assert dg == pytest.approx(dc, rel=1e-3)
