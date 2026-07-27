"""Boundary watch (Phase: detection + events): edge-band unit checks, the
'boundary' WS event, status fields, the live auto_expand toggle, and the
IC-preview measure-based edge warning."""

import json
from urllib.parse import unquote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import boundary
from core import session as sessions
from main import app

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
EDGE_IC = {"type": "mixture",
           "components": [{"x0": 5.2, "p0": 0.0, "sigma_x": 0.5,
                           "sigma_p": 0.707}]}


def _mk(client, **over):
    # precision is PINNED, not left to WIGNERF_PRECISION: this whole module is
    # precision-dependent (EDGE_THRESHOLD_BY_PRECISION raises the trigger to
    # 1e-4 in float32, and auto_expand is refused there outright), so a suite
    # run ON a float32 host — which this project supports — would fail these
    # for a reason that has nothing to do with what they check.
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": EDGE_IC,
           "variants": ["qn"], "record_dt": 0.05, "delay": 0.0,
           "precision": "float64"}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def _gauss(v, x0, s):
    g = np.exp(-((v - x0)**2)/(2.*s*s))
    return g/(g.sum()*(v[1] - v[0]))


def test_edge_report_degenerate_axes_stay_quiet():
    """On axes shorter than 8 bands (N < 32) the band pair covers over a
    quarter of the axis — at N <= 8 the slices overlap and a uniform
    marginal would read as edge mass 2.0. Such axes must never trigger
    (they would otherwise warn always and auto-expand-storm to the cap)."""
    for n in (4, 8, 16):
        m = np.full(n, 1.0/(n*0.1))      # normalized uniform density, d=0.1
        es = boundary.edge_report([m, m], (0.1, 0.1))
        assert es.mass == (0.0, 0.0) and not es.triggered
    # 32 cells is the smallest meaningful axis: centered stays clear...
    xv = np.linspace(-6, 6, 32, endpoint=False)
    dx = xv[1] - xv[0]
    centered = _gauss(xv, 0.0, 0.7)
    assert not boundary.edge_report([centered, centered], (dx, dx)).triggered
    # ...and a genuine edge state still trips it
    assert boundary.edge_report([_gauss(xv, 5.5, 0.7), centered],
                                (dx, dx)).axes == ["x"]


def test_edge_report_unit():
    xv = np.linspace(-6, 6, 256, endpoint=False)
    pv = np.linspace(-7, 7, 256, endpoint=False)
    dx, dp = xv[1] - xv[0], pv[1] - pv[0]
    rho_c, phi_c = _gauss(xv, 0.0, 0.7), _gauss(pv, 0.0, 0.7)
    assert not boundary.edge_report([rho_c, phi_c], (dx, dp)).triggered
    es = boundary.edge_report([_gauss(xv, 5.5, 0.7), phi_c], (dx, dp))
    assert es.axes == ["x"] and es.x_mass > boundary.EDGE_THRESHOLD
    # a diverged run (non-finite marginals) must never trigger
    rho_n = rho_c.copy()
    rho_n[0] = np.nan
    assert not boundary.edge_report([rho_n, phi_c], (dx, dp)).triggered


def test_a_band_mass_inside_its_own_noise_does_not_trip():
    """A density cannot be negative, so a marginal's negative part MEASURES the
    numerical floor of that reading — and on a coarse grid the floor is above
    EDGE_THRESHOLD (measured 5.35e-5 at 32 cells per axis against the 1e-6
    trigger, matching exp(-(pi*sigma_q/dx)^2/2) spectral truncation to two
    digits). A band mass buried in that is noise, not edge mass, and claiming it
    strobes the UI warning every record or two."""
    d = 0.1
    n = 64
    band = boundary.edge_band(n)                 # 4 cells per side at n=64

    def marginal(band_mass, neg):
        """A density with exactly `band_mass` in the outer band and a negative
        dip of magnitude `neg` in the middle (which leaves the band alone)."""
        m = np.zeros(n)
        m[0] = m[-1] = band_mass/(2.*d)
        m[n//2] = -neg/d
        m[n//2 - 1] = 1.0/d                      # the actual state, well inside
        return m

    clean = marginal(1e-5, 1e-12)                # 10x the trigger, no noise
    assert boundary.edge_report([clean, clean], (d, d)).axes == ["x", "p"]
    # Same band mass, but the reading now carries ringing of comparable size:
    # 8 * 1e-5 > 1e-5, so it can no longer support the claim.
    noisy = marginal(1e-5, 1e-5)
    es = boundary.edge_report([noisy, noisy], (d, d))
    assert es.mass[0] == pytest.approx(1e-5) and not es.triggered
    assert es.noise[0] == pytest.approx(1e-5)
    assert es.floor(0) == pytest.approx(boundary.EDGE_NOISE_MARGIN*1e-5)
    # ...and real edge mass above the floor still trips, noise or not
    assert boundary.edge_report([marginal(1e-3, 1e-5), clean],
                                (d, d)).axes == ["x", "p"]
    assert band == 4                             # the band the dip avoided


def test_a_flapping_edge_reading_is_never_announced():
    """The detector must not post a state change on a reading whose sign flips.
    Ungated this produced 79 changes in a 201-record 32^4 run and 243 WS events
    in 25 s in a browser — each one rewriting the header warning, which wraps
    the header and moves the W panels by a line. EDGE_CONFIRM consecutive
    records are required in BOTH directions. The very first reading is exempt —
    see _confirm_edge."""
    with TestClient(app) as client:
        s = sessions.get_session(_mk(client)["session_id"])
        labels = s.axis_labels
        hot = boundary.EdgeState((1.0, 0.0), labels, 1e-6, (0.0, 0.0))
        cold = boundary.EdgeState((0.0, 0.0), labels, 1e-6, (0.0, 0.0))
        # the session starts PAUSED, so no worker is reporting alongside us.
        # The first reading is taken as measured (an IC at the edge must warn at
        # once); everything after it needs confirming.
        s.report_edge(0, 0, cold)
        assert s.boundary_state["axes"] == []
        for k in range(1, 21):
            s.report_edge(0, k, hot if k % 2 else cold)
        assert s.boundary_state["axes"] == []
        # a sustained reading is announced, on the EDGE_CONFIRM'th record
        for i in range(boundary.EDGE_CONFIRM - 1):
            s.report_edge(0, 100 + i, hot)
            assert s.boundary_state["axes"] == []
        s.report_edge(0, 200, hot)
        assert s.boundary_state["axes"] == ["x"]
        # and it clears the same way, not on the first quiet record
        for i in range(boundary.EDGE_CONFIRM - 1):
            s.report_edge(0, 300 + i, cold)
            assert s.boundary_state["axes"] == ["x"]
        s.report_edge(0, 400, cold)
        assert s.boundary_state["axes"] == []


def test_boundary_event_and_status():
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            saw = None
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "boundary":
                        saw = d
                        break
            assert saw is not None, "boundary event never arrived"
            assert saw["axes"] == ["x"] and saw["action"] == "warn"
            assert saw["x_mass"] > boundary.EDGE_THRESHOLD
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["grid"]["ndim"] == 1
            assert r["grid"]["lo"] == [-6.0, -7.0]
            assert r["grid"]["hi"] == [6.0, 7.0]
            assert r["grid"]["N"] == [64, 64]
            # the flat 1D spelling rides along for the SPA
            assert (r["grid"]["x1"], r["grid"]["x2"], r["grid"]["Nx"]) == (-6.0, 6.0, 64)
            assert (r["grid"]["p1"], r["grid"]["p2"], r["grid"]["Np"]) == (-7.0, 7.0, 64)
            assert r["auto_expand"] is False
            assert r["max_grid"] >= 64
            assert r["boundary"]["axes"] == ["x"]
        client.delete("/api/sessions/%s" % sid)


def test_auto_expand_live_toggle():
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            for _ in range(200):
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d["type"] == "params_applied":
                        assert d["applied"]["auto_expand"] is True
                        break
            else:
                raise AssertionError("params_applied never arrived")
            r = client.get("/api/sessions/%s" % sid).json()
            assert r["auto_expand"] is True
        client.delete("/api/sessions/%s" % sid)


def test_wigner_preview_edge_warning():
    client = TestClient(app)
    r = client.post("/api/preview/wigner",
                    json={"grid": GRID, **EDGE_IC})
    assert r.status_code == 200
    # The measure-based total-W check rides in its OWN header as axis:mass
    # pairs, NOT as prose in Warnings — the session's boundary watch reports the
    # same fact about the same axes from record 0, so the client has to be able
    # to drop the axes already covered rather than pattern-match a sentence.
    edge = r.headers["X-Wignerf-Edge"]
    assert edge.startswith("x:"), edge
    mass = float(edge.split(":")[1])
    assert mass > boundary.EDGE_THRESHOLD, edge
    # and it is NOT also duplicated into the prose warnings
    warns = unquote(r.headers["X-Wignerf-Warnings"])
    assert "of the total probability lies within" not in warns, warns
    # a centered state stays clean
    r = client.post("/api/preview/wigner", json={
        "grid": GRID, "type": "mixture",
        "components": [{"x0": 0.0, "p0": 0.0, "sigma_x": 0.5,
                        "sigma_p": 0.707}]})
    assert r.headers["X-Wignerf-Edge"] == ""


def test_an_ic_at_the_edge_warns_on_its_first_record():
    """The confirmation delay must not silence the case the watch exists for.
    A session whose IC already sits at the edge may compute exactly one record
    (it starts paused), so its first reading is announced as measured."""
    with TestClient(app) as client:
        s = sessions.get_session(_mk(client)["session_id"])
        hot = boundary.EdgeState((1.0, 0.0), s.axis_labels, 1e-6, (0.0, 0.0))
        s.report_edge(0, 0, hot)
        assert s.boundary_state["axes"] == ["x"]
