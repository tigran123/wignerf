"""REST API tests for the preview/meta endpoints."""

import time

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


def test_device_reports_each_pool_device_s_total_memory():
    """The Setup panel's 2D footprint line turns red on "cannot fit this host",
    and TOTAL is what lets it say that without re-polling: free memory moves,
    total does not. Missing, the line stayed plain grey at 26.00 GiB/device on a
    host whose largest card is 24 — and WIGNERF_MAX_CELLS_2D did not cover it
    either, because 128×128×128×64 is EXACTLY 2**27 and the rail is a `>`.

    None is allowed (no cupy, an unreadable /proc): the panel then simply does
    not warn, exactly as _fit_error declines to refuse on unknown free memory.
    """
    d = client.get("/api/device").json()
    for dev in d["devices"]:
        assert "total_bytes" in dev, dev
        tb = dev["total_bytes"]
        assert tb is None or (isinstance(tb, int) and tb > 0), dev


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

    def counting(b, req, compiled=None):
        calls.append(b)
        return saved_build(b, req, compiled)

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

    def allocate_then_fail(b, req, compiled=None):
        if not b.is_gpu:
            return saved(b, req, compiled)
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


# -- the expression initial conditions ----------------------------------------

WEXPR = {"type": "wexpr", "expr": "3.5*exp(-x^2-p^2)", "grid": GRID}
PSI = {"type": "psi", "expr": "hermite(3,x)*exp(-x^2/2)", "grid": GRID}


def test_the_expression_ics_preview_like_any_other():
    """Both kinds come back as a normal frame — and NORMALISED, which is the
    only interesting thing about the response. `purity > 0` (all this used to
    assert) is true of any non-zero W of any kind, so it could not tell the two
    apart, could not see an un-normalised state, and could not see a wexpr and a
    psi returning the same thing."""
    seen = []
    for body in (WEXPR, PSI):
        r = client.post("/api/preview/wigner", json=body)
        assert r.status_code == 200, r.text
        f = _decode(r)
        v = f.variants[0]
        assert f.geom.ndim == 1 and len(v.planes) == 1
        # int W dmu = 1: the whole point of "auto-normalised". The marginal is
        # already the reduction over p, so summing it with dx is that integral.
        m = v.marg[0]
        dx = (f.geom.hi[0] - f.geom.lo[0])/f.geom.N[0]
        assert float(np.asarray(m).sum())*dx == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < v.purity
        seen.append(np.asarray(m).copy())
    # a Gaussian W and a Hermite-3 psi are not the same state — an assertion
    # this loose would otherwise pass with one kind wired to the other
    assert np.abs(seen[0] - seen[1]).max() > 1e-3


def test_the_norm_header_says_how_the_state_was_normalised():
    """A wexpr is normalised BY its own grid sum, so reporting a deficit there
    would be a health check that always passes — worse than no line. The raw
    integral goes in its place, which pre-answers "why does multiplying my
    expression by 5 change nothing?"."""
    r = client.post("/api/preview/wigner", json=WEXPR)
    assert r.headers["X-Wignerf-Norm-Deficit"] == ""
    method, raw, _mass = r.headers["X-Wignerf-IC-Norm"].split(":")
    assert method == "grid-sum"
    assert float(raw) == pytest.approx(3.5*np.pi, rel=1e-4)

    r = client.post("/api/preview/wigner", json=PSI)
    assert r.headers["X-Wignerf-Norm-Deficit"] != ""     # psi keeps a real one
    assert r.headers["X-Wignerf-IC-Norm"].startswith("psi-extended:")


def test_a_bad_expression_is_422_before_any_device_is_touched():
    """The property the components_of pre-flight protects, for the kinds that
    replaced it: a client error must never be mistaken for a device error, and
    must never reach the CPU fallback that exists for the latter."""
    import routers.preview as pv
    saved = pv._pick_device
    pv._pick_device = lambda *a, **k: pytest.fail("a bad expression picked a device")
    try:
        for expr, kind in (("exp(-x^2/", "psi"), ("exp(-x^2-p^2)", "psi"),
                           ("exp(-x^2-p^2)*I", "wexpr")):
            r = client.post("/api/preview/wigner",
                            json={"type": kind, "expr": expr, "grid": GRID})
            assert r.status_code == 422, (expr, r.text)
    finally:
        pv._pick_device = saved


def test_an_unnormalisable_w_is_a_422_and_not_a_cpu_retry():
    """This one can only be known once W exists, so it cannot join the
    pre-flight — but it is still the client's problem, and retrying it on the
    CPU would fail identically and then surface as a 500 (initial.ICError).

    THE BUILD COUNT IS THE ASSERTION, not the status code. Both paths answer 422
    with the same sentence — the GPU branch's and the CPU branch's — so a status
    check alone passes with the whole ICError branch deleted, and the property
    the name claims is "not a CPU RETRY". Counting is what test_preview_bad_ic_
    is_422_before_anything_is_built already does, for the same reason.
    """
    import routers.preview as pv
    calls = []
    saved_build, saved_pick = pv._build_frame, pv._pick_device
    # Pretend a device was picked, so the GPU branch is the one that runs and
    # the fall-through to the CPU below it is reachable at all.
    pv._pick_device = lambda cells, ndim=1: "cpu"

    def counting(b, req, compiled=None):
        calls.append(b)
        return saved_build(b, req, compiled)

    pv._build_frame = counting
    try:
        for expr in ("x*p*exp(-x^2-p^2)", "-exp(-x^2-p^2)"):
            del calls[:]
            r = client.post("/api/preview/wigner",
                            json={"type": "wexpr", "expr": expr, "grid": GRID})
            assert r.status_code == 422, r.text
            assert "cannot be normalized" in r.text
            assert len(calls) == 1, "the bad IC was rebuilt on the CPU"
    finally:
        pv._build_frame, pv._pick_device = saved_build, saved_pick


def test_a_grid_over_the_rail_is_refused_before_the_expression_is_parsed():
    """Both entry points answer the GRID first.

    /preview/wigner always did; POST /sessions compiled the IC before reaching
    grid_limit_error, so one body got two different errors depending on which
    endpoint it reached — the disagreement grid_limit_error's docstring is
    about — and paid for a sympy parse plus a 33^4 finiteness probe on a request
    that could never start a session."""
    huge = {"ndim": 1, "axes": [{"lo": -6, "hi": 6, "N": 16384},
                                {"lo": -7, "hi": 7, "N": 16384}]}
    body = {"type": "wexpr", "expr": "1/(x^2+p^2)", "grid": huge}
    a = client.post("/api/preview/wigner", json=body)
    b = client.post("/api/sessions", json=dict(grid=huge, potential="x^2/2",
                                               variants=["qn"],
                                               ic={"type": "wexpr",
                                                   "expr": "1/(x^2+p^2)"}))
    assert a.status_code == 422 and b.status_code == 422
    # the same complaint, about the grid, from both — not one about the pole
    for r in (a, b):
        assert "not finite" not in r.text, r.text


def test_the_schema_refuses_a_kind_without_its_own_shape():
    """Hand-written sentences, not pydantic's "List should have at least 1
    item": lib/apierror.apiErrorText renders these to a person."""
    r = client.post("/api/preview/wigner", json={"type": "psi", "grid": GRID})
    assert r.status_code == 422 and "expr" in r.text
    r = client.post("/api/preview/wigner", json={"type": "mixture",
                                                 "components": [], "grid": GRID})
    assert r.status_code == 422 and "Gaussian component" in r.text


def test_the_wavefunction_endpoint_pairs_each_cut_with_its_conjugate_axis():
    """The momentum axis FOLLOWS from axes.conjugate rather than being chosen,
    so the pair cannot be mismatched — the classic multi-D error, and silent."""
    r = client.post("/api/preview/wavefunction",
                    json={"expr": "hermite(1,x)*exp(-x^2/2)", "grid": GRID})
    d = r.json()
    assert d["ok"] and d["psi"]["axis"] == 0 and d["phi"]["axis"] == 1
    assert d["psi"]["at"] is None                    # no cut to make at ndim=1
    # a real psi has a real wavefunction and a complex Fourier image
    assert all(v == 0.0 for v in d["psi"]["im"])
    assert max(abs(v) for v in d["phi"]["im"]) > 1e-6


def test_the_wavefunction_cut_is_transform_then_cut_at_ndim_2():
    """φ(px, py0) is a SLICE OF THE FULL 2D TRANSFORM of ψ, never the 1D
    transform of the slice ψ(x, y0).

    THE ψ MUST BE NON-SEPARABLE OR THIS PROVES NOTHING. The two orders agree
    exactly at ndim=1 and exactly for any separable ψ — i.e. for everything
    anyone types first — which is precisely what would let the wrong one ship.
    exp(-(x^2+y^2)/2 + x*y/2) couples them, so the reference below is computed
    the honest way (a full 2D quadrature, then the cut) and disagrees with the
    cheap way by far more than the tolerance.
    """
    import numpy as np
    g2 = {"ndim": 2,
          "axes": [{"lo": -6, "hi": 6, "N": 48}, {"lo": -6, "hi": 6, "N": 40},
                   {"lo": -5, "hi": 5, "N": 32}, {"lo": -5, "hi": 5, "N": 36}]}
    src = "exp(-(x^2+y^2)/2 + x*y/2)"
    d = client.post("/api/preview/wavefunction",
                    json={"expr": src, "grid": g2, "cut_axis": 0}).json()
    assert d["ok"], d
    # conjugate(2, 0) == 2, i.e. px — NOT 1 (y) and NOT 3 (py)
    assert d["psi"]["axis"] == 0 and d["phi"]["axis"] == 2

    def vec(a):
        lo, hi, n = g2["axes"][a]["lo"], g2["axes"][a]["hi"], g2["axes"][a]["N"]
        return lo + (hi - lo)/n*np.arange(n)

    x, y, px = vec(0), vec(1), vec(2)
    psi = np.exp(-(x[:, None]**2 + y[None, :]**2)/2 + x[:, None]*y[None, :]/2)
    psi = psi/np.sqrt((np.abs(psi)**2).sum()*(x[1] - x[0])*(y[1] - y[0]))
    # transform BOTH axes, then cut — the correct order
    kx = np.exp(-1j*px[:, None]*x[None, :])*((x[1] - x[0])/np.sqrt(2*np.pi))
    py = vec(3)
    ky = np.exp(-1j*py[:, None]*y[None, :])*((y[1] - y[0])/np.sqrt(2*np.pi))
    phi2 = kx @ psi @ ky.T
    m = int(np.argmin(np.abs(py)))
    want = phi2[:, m]
    got = np.array(d["phi"]["re"]) + 1j*np.array(d["phi"]["im"])
    assert d["phi"]["at"] == pytest.approx(float(py[m]))
    assert np.abs(got - want).max() < 1e-10*np.abs(want).max() + 1e-12
    # ...and the WRONG order (cut psi first, then transform one axis) is a
    # genuinely different answer, so the check above has something to catch
    n = int(np.argmin(np.abs(y)))
    wrong = kx @ psi[:, n]
    assert np.abs(wrong - want).max() > 1e-3*np.abs(want).max()


def test_the_wavefunction_endpoint_is_bounded_and_answers_in_band():
    """It allocates (psi on the lattice, plus a chunked quadrature per axis) and
    it fires on every keystroke, so a grid a session would refuse must not be
    sampled here either — and the refusal stays inside the {ok, error} union the
    docstring promises rather than becoming an HTTP error."""
    huge = {"ndim": 1, "axes": [{"lo": -6, "hi": 6, "N": 16384},
                                {"lo": -7, "hi": 7, "N": 16384}]}
    r = client.post("/api/preview/wavefunction",
                    json={"expr": "exp(-x^2/2)", "grid": huge})
    assert r.status_code == 200
    assert r.json()["ok"] is False and r.json()["error"]


def test_a_blank_wavefunction_box_is_a_sentence_not_a_pydantic_blob():
    """Clearing the psi input is the normal state of an input someone is typing
    into. Constructing an ICSpec and letting its validator fire would render
    pydantic's "1 validation error for ICSpec ... [type=value_error,
    input_value=...]" dump straight into the strip under the charts — the blob
    apiErrorText exists to avoid, and it never sees this one, because this
    endpoint answers 200."""
    for blank in ("", "   "):
        d = client.post("/api/preview/wavefunction",
                        json={"expr": blank, "grid": GRID}).json()
        assert d["ok"] is False
        assert "validation error" not in d["error"]
        assert "input_value" not in d["error"]
        assert "expression is required" in d["error"]


def test_an_ic_carrying_BOTH_shapes_is_normalised_not_refused():
    """Carrying both shapes is legal and must stay legal.

    The editor holds a default for every tab at once, so a form on the wexpr tab
    genuinely has Gaussian components behind it — and so does every stored config
    and every setup document written before `expr` existed. Refusing the foreign
    field made all of those a 422: on a cold start with cleared local data,
    selecting the W(x,p) tab answered "an IC of type 'wexpr' is one expression
    and carries no Gaussian components (got 1)", i.e. the schema telling a client
    its own defaults were invalid. It is DROPPED instead, so nothing downstream
    can act on it or publish it.
    """
    comp = {"q0": [0.0], "k0": [0.0], "sigma_q": [0.5], "sigma_k": [0.5]}
    # the exact cold-start body the SPA sends from the wexpr tab
    r = client.post("/api/preview/wigner",
                    json={"type": "wexpr", "expr": "exp(-x^2-p^2)", "grid": GRID,
                          "components": [comp]})
    assert r.status_code == 200, r.text
    # ...and a session too, which is where the user actually saw it
    r = client.post("/api/sessions",
                    json={"grid": GRID, "potential": "x^2/2", "variants": ["qn"],
                          "ic": {"type": "psi", "expr": "exp(-x^2/2)",
                                 "components": [comp]}})
    assert r.status_code == 200, r.text
    client.delete("/api/sessions/%s" % r.json()["session_id"])

    # the foreign field is gone by the time anything downstream sees the model,
    # which is what keeps it out of the exported setup document
    from core.protocol import ICSpec
    assert ICSpec(type="psi", expr="exp(-x^2/2)",
                  components=[comp]).components == []
    assert ICSpec(type="mixture", components=[comp], expr="x"*4000).expr is None

    # ...but a kind still cannot arrive WITHOUT its own shape
    r = client.post("/api/preview/wigner",
                    json={"type": "wexpr", "grid": GRID, "components": [comp]})
    assert r.status_code == 422 and "needs an 'expr'" in r.text, r.text


def test_the_wavefunction_endpoint_reports_errors_in_band():
    """An {ok, error} union like /preview/potential: a half-typed expression is
    the normal state of an input someone is typing into."""
    for expr in ("exp(-x^2/", "exp(-x^2-p^2)"):
        d = client.post("/api/preview/wavefunction",
                        json={"expr": expr, "grid": GRID}).json()
        assert d["ok"] is False and d["error"]


def test_a_bad_ic_expression_is_refused_before_any_worker_starts():
    """The IC is otherwise built inside each SolverWorker, so without the
    create-time compile a bad expression would answer 200 and then kill four
    worker threads — reachable by a direct API call or an imported config, which
    the SPA's preview gate does not cover."""
    base = dict(grid=GRID, potential="x^2/2", variants=["qn"])
    with TestClient(app) as c:
        r = c.post("/api/sessions", json=dict(
            base, ic={"type": "psi", "expr": "exp(-x^2-p^2)"}))
        assert r.status_code == 422 and "may appear as a variable" in r.text

        r = c.post("/api/sessions", json=dict(
            base, ic={"type": "psi", "expr": "exp(-(x-2)^2/2)"}))
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        try:
            # record 0 is the Cauchy data, emitted by each worker
            # before its loop — so seeing it is what proves the IC
            # really built on the worker thread and not only here.
            for _ in range(100):
                st = c.get("/api/sessions/%s" % sid).json()
                if st["record_extent"][1] >= 0:
                    break
                time.sleep(0.05)
            assert st["record_extent"][1] >= 0
        finally:
            c.delete("/api/sessions/%s" % sid)
