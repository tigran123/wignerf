"""
The opt-in float32 ("preview") solver mode.

The rest of the suite pins float64: every other fixture builds an
ArrayBackend at its default precision, so the correctness anchors in
test_propagator.py — above all test_relativistic_matches_nonrel_at_large_c
and test_relativistic_uncertainty_shear, whose claims this mode is documented
as unable to support — keep asserting their 1e-12/1e-10 bounds against double
arithmetic and are untouched by the feature. This module is the other half:
it pins that float32 does what it promises and nothing more.

Two of these tests assert DTYPES rather than numbers, which looks fussy until
you notice both of the mode's failure modes are invisible in results:

  - hand a complex64 array to a complex128 pyFFTW plan and auto_align_input
    copies it up, returning a correct complex128 answer at complex128 speed;
  - `B *= expT` with B complex64 and expT complex128 is legal in numpy AND
    cupy, and yields a correct complex64 B through a full complex128
    temporary.

Both produce right answers, so no physics assertion can catch either. Only
the dtype can.
"""

from math import pi, sqrt

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import boundary, describe, observables
from core.grid import Grid, embed_window
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.protocol import TOL_MIN_F32
from core.xp import ArrayBackend
from main import app

SIG = 1.0/sqrt(2.0)
HARMONIC = dict(U=lambda x: x**2/2., dUdx=lambda x: x)


def _grid(precision, n=64):
    return Grid(-6.0, 6.0, n, -7.0, 7.0, n,
                ArrayBackend(device="cpu", precision=precision))


def _coherent(grid):
    return grid.shift2d(mixture_wigner(grid, [GaussianComponent(2.0, 0.0, SIG, SIG)]))


def _evolve(prop, W, dt, nsteps):
    expU, expT = prop.exponents(dt)
    for _ in range(nsteps):
        W = prop.solve_spectral(W, expU, expT)
    return W


# -- what must NOT change in float32 -------------------------------------


@pytest.mark.parametrize("quantum,relativistic",
                         [(True, False), (True, True), (False, False), (False, True)])
def test_exponent_construction_stays_double(quantum, relativistic):
    """The rate meshes and H are built in float64 in BOTH modes, and are
    bitwise identical between them.

    This is the load-bearing property of the whole design. Relativistic dT
    built in float32 has max abs error 455 against max |dT| = 228 — 200% —
    because mc^2 cancels inside a difference of ~1.9e4-magnitude terms, so a
    float32 session that let single precision reach _rate_mesh would not be a
    less accurate simulation, it would be a different one. The parametrization
    covers the relativistic variants precisely because they are the ones that
    cannot survive it."""
    props = {}
    for precision in ("float64", "float32"):
        g = _grid(precision)
        props[precision] = Propagator(g, quantum=quantum,
                                      relativistic=relativistic, **HARMONIC)
    a, b = props["float64"], props["float32"]
    for name in ("dU_im", "dT_im", "H"):
        assert getattr(b, name).dtype == np.float64, name
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_observables_stay_float64():
    """Reductions accumulate and leave in float64 whatever the solver does, so
    rho/phi keep the dtype history.py's byte accounting and the <f4 wire codec
    were written against."""
    g = _grid("float32")
    prop = Propagator(g, quantum=True, **HARMONIC)
    obs = observables.compute(_coherent(g), prop)
    assert obs.rho.dtype == np.float64 and obs.phi.dtype == np.float64
    assert isinstance(obs.norm, float) and isinstance(obs.purity, float)


def test_regrid_keeps_the_working_dtype():
    """embed_window must not upcast a float32 state back to float64 — that
    would silently double the working set at exactly the moment (auto-expand)
    the domain just got bigger."""
    from dataclasses import replace
    from core.grid import GridState
    old = GridState(x0=-6.0, p0=-7.0, dx=12.0/64, dp=14.0/64,
                    ox=0, op=0, Nx=64, Np=64)
    new = replace(old, ox=-32, Nx=128)          # doubled along x, support centred
    for dtype in (np.float64, np.float32):
        Wn = np.ones((64, 64), dtype=dtype)
        out = embed_window(Wn, old, new, np)
        assert out.dtype == dtype
        # and the copy is still bitwise exact, which is the whole point of
        # the fixed-lattice regrid
        assert np.array_equal(out[32:96, :], Wn)


# -- what float32 must actually DO ---------------------------------------


def test_float32_really_is_single_precision():
    """The spectral path is complex64 end to end. Asserting the pyFFTW plan
    dtype is the point: a complex128 plan silently upcasts its input and
    returns a correct answer at double speed and double memory."""
    g = _grid("float32")
    assert g.backend.complex_dtype is np.complex64
    prop = Propagator(g, quantum=True, **HARMONIC)
    expU, expT = prop.exponents(0.01)
    assert expU.dtype == np.complex64 and expT.dtype == np.complex64
    W = prop.solve_spectral(_coherent(g), expU, expT)
    assert W.dtype == np.float32
    if g.backend.fft_provider == "pyfftw":
        probe = np.zeros((64, 64), dtype=np.complex64)
        assert prop._fft0(probe).dtype == np.complex64
        assert prop._ifft1(probe).dtype == np.complex64


def test_float64_is_the_default():
    b = ArrayBackend(device="cpu")
    assert b.precision == "float64" and b.complex_dtype is np.complex128


def test_host_precision_is_the_schema_default(monkeypatch):
    """WIGNERF_PRECISION must actually DRIVE the default, not merely be
    advertised. It was decorative once: `/api/device` reported float32 while
    every session that omitted the field was built with a hard-coded float64,
    so the setting silently did nothing."""
    import config
    from core.protocol import SessionCreate
    assert SessionCreate(**_cfg()).precision == "float64"
    monkeypatch.setattr(config, "PRECISION", "float32")
    assert SessionCreate(**_cfg()).precision == "float32"
    # an explicit choice still wins over the host's
    assert SessionCreate(**_cfg(precision="float64")).precision == "float64"


def test_bad_host_precision_falls_back_to_float64(monkeypatch):
    """A typo must not become a precision. Falling back to the SAFE value is
    never dangerous; passing `flaot32` through would have advertised it to
    every client and handed it to SessionCreate's default."""
    import importlib
    import config
    monkeypatch.setenv("WIGNERF_PRECISION", "flaot32")
    assert importlib.reload(config).PRECISION == "float64"
    monkeypatch.setenv("WIGNERF_PRECISION", "float32")
    assert importlib.reload(config).PRECISION == "float32"
    monkeypatch.delenv("WIGNERF_PRECISION")
    assert importlib.reload(config).PRECISION == "float64"


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="unknown precision"):
        ArrayBackend(device="cpu", precision="float16")


# -- how far the accuracy actually falls ---------------------------------


def test_coherent_state_revival_survives_float32():
    """The coarse physics still holds: after one period the coherent state
    returns to itself within the same 2% the float64 test demands. What
    float32 costs is the 1e-12 diagnostics, not the qualitative dynamics —
    which is exactly what makes it usable as a preview."""
    g = _grid("float32")
    prop = Propagator(g, quantum=True, **HARMONIC)
    W0 = _coherent(g)
    n = 2000
    W = _evolve(prop, W0, 2.*pi/n, n)
    rel_l1 = float(np.sum(np.abs(W - W0), dtype=np.float64)) \
        / float(np.sum(np.abs(W0), dtype=np.float64))
    assert rel_l1 < 0.02


def test_float32_drift_is_bounded_and_worse_than_float64():
    """Upper bounds, so a regression to something worse than measured fails —
    and a lower bound against float64, so this test also fails if float32 is
    silently not being used at all (the pyFFTW/exponent upcast traps)."""
    drift = {}
    for precision in ("float64", "float32"):
        g = _grid(precision)
        prop = Propagator(g, quantum=True, **HARMONIC)
        W0 = _coherent(g)
        o0 = observables.compute(W0, prop)
        o1 = observables.compute(_evolve(prop, W0, 0.01, 200), prop)
        drift[precision] = (abs(o1.norm - o0.norm), abs(o1.purity - o0.purity))
    # measured 2026-07-25 at 64^2/200 steps: float32 (7.5e-8, 3.4e-5),
    # float64 (3.3e-16, 4.0e-14)
    assert drift["float32"][0] < 1e-5, "norm drift"
    assert drift["float32"][1] < 1e-3, "purity drift"
    assert drift["float64"][0] < 1e-12 and drift["float64"][1] < 1e-10
    assert drift["float32"][1] > 100*drift["float64"][1]


def _centered(g):
    return g.shift2d(mixture_wigner(g, [GaussianComponent(0.0, 0.0, SIG, SIG)]))


def test_float32_noise_reaches_the_float64_edge_trigger():
    """The measurement behind the precision-keyed threshold and behind the
    auto_expand refusal — pinned, because it is the one place where float32
    does not merely lose accuracy but tells an active lie.

    A coherent state parked at the ORIGIN of a 256² domain is ~8 sigma from
    the edge band; its true band mass is ~1e-15, and float64 reports exactly
    that, flat, forever. float32's own spectral noise climbs past the float64
    trigger within a few hundred steps."""
    masses = {}
    for precision in ("float64", "float32"):
        g = _grid(precision, n=256)
        prop = Propagator(g, quantum=True, **HARMONIC)
        obs = observables.compute(_evolve(prop, _centered(g), 0.01, 600), prop)
        es = boundary.edge_report(obs.rho, obs.phi, g.dx, g.dp, precision)
        masses[precision] = max(es.x_mass, es.p_mass)
        # whatever the noise, the SESSION's own threshold must stay quiet —
        # a permanent boundary alarm on a contained state is a broken warning
        assert not es.triggered, (precision, es)
    # float64 stays at the ringing floor the boundary docstring claims
    assert masses["float64"] < 1e-12
    # ...and float32 would have tripped the float64 trigger. If this ever
    # stops being true, the raised threshold and the auto_expand refusal can
    # both be revisited — until then they are load-bearing.
    assert masses["float32"] > boundary.EDGE_THRESHOLD


def test_float32_refuses_auto_expand():
    """Detection still runs and still warns; only the automatic REGRID is
    refused, because it sizes the new domain from a support scan whose 1e-8
    threshold is two orders below the same noise."""
    from pydantic import ValidationError
    from core.protocol import SessionCreate
    with pytest.raises(ValidationError, match="auto-expand is not available"):
        SessionCreate(**_cfg(precision="float32", auto_expand=True))
    # both halves of the pair are individually fine
    SessionCreate(**_cfg(precision="float32"))
    SessionCreate(**_cfg(auto_expand=True))
    with TestClient(app) as client:
        r = client.post("/api/sessions",
                        json=_cfg(precision="float32", auto_expand=True))
        assert r.status_code == 422 and "auto-expand" in r.text


def test_live_auto_expand_toggle_is_refused_in_float32():
    """The schema rule is two clicks away otherwise — auto_expand is a live
    ParamChange field, so the live path needs the same guard as create."""
    import json
    with TestClient(app) as client:
        info = client.post("/api/sessions",
                           json=_cfg(precision="float32")).json()
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            for _ in range(200):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "error" and d.get("code") == "bad_auto_expand":
                    break
                assert d["type"] != "params_applied", d
            else:
                raise AssertionError("the refusal never arrived")
        assert client.get("/api/sessions/%s" % sid).json()["auto_expand"] is False
        client.delete("/api/sessions/%s" % sid)


# -- the API surface ------------------------------------------------------


GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
IC = {"type": "mixture",
      "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.70711,
                      "sigma_p": 0.70711}]}


def _cfg(**over):
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": IC, "variants": ["qn"],
           "record_dt": 0.05, "delay": 0.0}
    cfg.update(over)
    return cfg


def test_precision_defaults_to_float64_and_rides_in_status():
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json=_cfg()).json()["session_id"]
        assert client.get("/api/sessions/%s" % sid).json()["precision"] == "float64"
        sid = client.post("/api/sessions",
                          json=_cfg(precision="float32")).json()["session_id"]
        assert client.get("/api/sessions/%s" % sid).json()["precision"] == "float32"


def test_tol_below_the_float32_floor_is_refused():
    """The combination is unreachable, not merely unwise: adjust_step compares
    a full step against two half steps, which agree only to ~1e-7 in single
    precision, so a smaller tol makes the controller shrink dt forever."""
    with TestClient(app) as client:
        r = client.post("/api/sessions",
                        json=_cfg(precision="float32", tol=1e-8))
        assert r.status_code == 422
        assert "float32 roundoff floor" in r.text
        # the same tol is perfectly fine in float64
        assert client.post("/api/sessions", json=_cfg(tol=1e-8)).status_code == 200
        # and float32 at or above the floor is fine
        assert client.post("/api/sessions",
                           json=_cfg(precision="float32",
                                     tol=TOL_MIN_F32)).status_code == 200


def test_unknown_device_is_a_clean_422():
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=_cfg(device="cuda:99"))
        assert r.status_code == 422 and "device" in r.text
        r = client.post("/api/sessions", json=_cfg(device="quantum-annealer"))
        assert r.status_code == 422 and "device" in r.text
        # cpu is always available
        assert client.post("/api/sessions",
                           json=_cfg(device="cpu")).status_code == 200


def test_device_outside_the_host_pool_is_refused(monkeypatch):
    """WIGNERF_DEVICE has to be a POLICY, not a suggestion. It only checked that
    the spec parsed and the card existed, so a host pinned to one device — or to
    cpu, to keep its cards free for something else — could be overridden by any
    client that asked for another. The refusal names the pool, because "device
    not available" without saying what IS available is a dead end."""
    import config
    monkeypatch.setattr(config, "DEVICE", "cpu")
    with TestClient(app) as client:
        # in the pool
        assert client.post("/api/sessions",
                           json=_cfg(device="cpu")).status_code == 200
        # outside it — refused whether or not this host physically has the card
        r = client.post("/api/sessions", json=_cfg(device="cuda:0"))
        assert r.status_code == 422
        assert "WIGNERF_DEVICE=cpu" in r.text and "cpu" in r.text
        # and every member of a list is checked, not just the first
        r = client.post("/api/sessions", json=_cfg(device="cpu,cuda:0"))
        assert r.status_code == 422 and "cuda:0" in r.text


def test_history_cap_is_clamped_to_the_host_ceiling():
    """A session may narrow the host's RAM policy, never widen it."""
    import config
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json=_cfg(history_mb=64)) \
            .json()["session_id"]
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["history_cap_bytes"] == 64*1024*1024
        assert st["history_mb_max"] == config.HISTORY_MB
        sid = client.post("/api/sessions",
                          json=_cfg(history_mb=config.HISTORY_MB*100)) \
            .json()["session_id"]
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["history_cap_bytes"] == config.HISTORY_MB*1024*1024


def test_setup_document_and_export_block_carry_the_precision():
    """A preview run's video outlives the session that could have explained
    it, so the metadata must say so — and a float64 run's block must not grow
    a line of noise."""
    from core.protocol import SessionCreate
    f32 = SessionCreate(**_cfg(precision="float32"))
    f64 = SessionCreate(**_cfg())
    assert describe.setup_document(f32, [])["config"]["precision"] == "float32"
    lines32 = describe.param_lines(f32, [], 0, 10)
    lines64 = describe.param_lines(f64, [], 0, 10)
    assert any("PREVIEW" in ln and "float32" in ln for ln in lines32)
    assert not any("precision" in ln for ln in lines64)
    assert len(lines32) == len(lines64) + 1


def test_device_endpoint_reports_bare_specs():
    """The Setup panel's device select needs the round-trippable spec, not
    just the human name."""
    with TestClient(app) as client:
        d = client.get("/api/device").json()
        assert d["precision"] and d["pool"]
        for entry in d["devices"]:
            assert entry["spec"] in ("cpu",) or entry["spec"].startswith("cuda:")
