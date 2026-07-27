"""
2D sessions through the API: the record path end to end, and the four explicit
refusals that stand in for deferred work (milestones M1-M4 in CLAUDE.md).

The refusal tests are not box-ticking. Each gate replaces a feature that would
otherwise half-work — a float32 2D run whose mixed-precision rules were never
re-verified, a relativistic 2D variant whose mc^2 cancellation was never
checked — and a gate that silently stopped firing is exactly how a half-feature
ships. They also pin that the messages NAME the milestone, so whoever hits one
learns what is missing rather than that "2D is broken".
"""

import json

import pytest
from fastapi.testclient import TestClient

import config

from core import protocol
from main import app

G2 = {"ndim": 2,
      "axes": [{"lo": -6.0, "hi": 6.0, "N": 16},
               {"lo": -6.0, "hi": 6.0, "N": 16},
               {"lo": -7.0, "hi": 7.0, "N": 16},
               {"lo": -7.0, "hi": 7.0, "N": 16}]}
IC2 = {"type": "mixture", "components": [
    {"q0": [1.0, 0.0], "k0": [0.0, 0.5],
     "sigma_q": [0.7, 0.7], "sigma_k": [0.7, 0.7]}]}


def cfg2(**over):
    c = {"grid": G2, "potential": "(x^2+y^2)/2", "ic": IC2,
         "variants": ["qn", "cn"], "record_dt": 0.05, "delay": 0.0}
    c.update(over)
    return c


def test_a_2d_session_streams_planes_not_the_state():
    """The whole point of the 2D record: six 2D projections and four
    marginals, never the 4D array. At 16^4 that is 7 KiB for two variants
    against 131 KiB for the raw state; at 64^4 it is 50 KiB against 33 MiB per
    variant, which is what takes the browser-receive ceiling out of the
    picture."""
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            f = payload = None
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes"):
                    payload, f = m["bytes"], protocol.unpack_frame(m["bytes"])
                    break
            assert f is not None, "no frame arrived"
            assert f.geom.ndim == 2 and f.geom.N == (16, 16, 16, 16)
            assert len(f.variants) == 2
            v = f.variants[0]
            assert len(v.planes) == 6 and len(v.marg) == 4
            assert [(p.a, p.b) for p in v.planes] == [
                (0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
            for p in v.planes:
                assert p.wq.shape == (f.geom.N[p.a], f.geom.N[p.b])
            # the raw state would be this many bytes at uint16, per variant
            raw = 2*16**4
            assert len(payload) < raw//4, len(payload)
            # <Lz> = x*py - y*px = 1*0.5 - 0*0
            assert v.lz == pytest.approx(0.5, abs=1e-4)
            assert v.mean[0] == pytest.approx(1.0, abs=1e-4)
            assert v.mean[3] == pytest.approx(0.5, abs=1e-4)
            # the 1D spellings must REFUSE on a 2D frame rather than silently
            # return axis 0/1 — a missed call site has to fail loudly
            with pytest.raises(AttributeError):
                v.rho

        st = client.get("/api/sessions/%s" % sid).json()
        assert st["ndim"] == 2
        assert st["grid"]["labels"] == ["x", "y", "px", "py"]
        assert st["grid"]["N"] == [16, 16, 16, 16]
        assert "x1" not in st["grid"]        # the flat spelling is 1D-only
        rec = client.get("/api/sessions/%s/series" % sid).json()["records"][0]
        sv = rec["variants"][0]
        assert len(sv["mean"]) == 4 and len(sv["std"]) == 4
        assert "x_std" not in sv
        client.delete("/api/sessions/%s" % sid)


def test_2d_setup_document_round_trips():
    """The setup document (and therefore mp4 import) works for 2D from the
    first cut — only the VIDEO render is gated (M4)."""
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        r = client.get("/api/sessions/%s/setup" % sid)
        assert r.status_code == 200
        doc = r.json()
        assert doc["config"]["grid"]["ndim"] == 2
        assert [a["N"] for a in doc["config"]["grid"]["axes"]] == [16]*4
        assert "16x16x16x16" in r.headers["content-disposition"]
        # and it is accepted back verbatim
        again = client.post("/api/sessions", json=doc["config"])
        assert again.status_code == 200, again.text
        client.delete("/api/sessions/%s" % again.json()["session_id"])
        client.delete("/api/sessions/%s" % sid)


# ---------------------------------------------------------------------------
# the deferred-work gates
# ---------------------------------------------------------------------------

def test_a_float32_host_default_does_not_block_2d(monkeypatch):
    """A gate must refuse what was ASKED FOR. float32 is refused in 2D (M1),
    but WIGNERF_PRECISION is a host DEFAULT, not a request — and the SPA, curl
    and scripts/ws_smoke.py --ndim 2 all omit the field. Resolving that default
    through the M1 gate made every 2D session on a float32 host 422 over a value
    nobody sent (it took this whole test module down with it). An omitted
    precision therefore resolves to float64 at ndim=2, and only an EXPLICIT
    float32 hits the gate below."""
    monkeypatch.setattr(config, "PRECISION", "float32")
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2())
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        assert client.get("/api/sessions/%s" % sid).json()["precision"] \
            == "float64"
        # ...while a 1D session on the same host still gets what it configured
        one_d = {"grid": dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64),
                 "potential": "x^2/2", "variants": ["qn"], "record_dt": 0.05,
                 "ic": {"type": "mixture", "components": [
                     {"x0": 0.0, "p0": 0.0, "sigma_x": 0.7, "sigma_p": 0.7}]}}
        r1 = client.post("/api/sessions", json=one_d)
        assert r1.status_code == 200, r1.text
        sid1 = r1.json()["session_id"]
        assert client.get("/api/sessions/%s" % sid1).json()["precision"] \
            == "float32"
        client.delete("/api/sessions/%s" % sid1)
        client.delete("/api/sessions/%s" % sid)


@pytest.mark.parametrize("over,needle", [
    (dict(precision="float32"), "M1"),
    (dict(variants=["qr"]), "M2"),
    (dict(variants=["qn", "cr"]), "M2"),
    (dict(auto_expand=True), "M3"),
])
def test_deferred_features_are_refused_with_their_milestone(over, needle):
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2(**over))
        assert r.status_code == 422, r.text
        assert needle in r.text, r.text


def test_auto_expand_is_also_refused_live():
    """auto_expand is live-toggleable, so the create-time refusal is otherwise
    two clicks away."""
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            for _ in range(60):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                assert d["type"] != "params_applied", d
                if d["type"] == "error":
                    assert "M3" in d["message"]
                    break
            else:
                raise AssertionError("live auto_expand was not refused")
        assert client.get("/api/sessions/%s" % sid).json()["auto_expand"] is False
        client.delete("/api/sessions/%s" % sid)


def test_mp4_export_is_refused_for_2d():
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        r = client.post("/api/sessions/%s/export" % sid, json={"fps": 30})
        assert r.status_code == 422, r.text
        assert "M4" in r.text
        client.delete("/api/sessions/%s" % sid)


def test_the_cell_ceiling_is_what_bounds_2d_memory(monkeypatch):
    """A per-axis cap is no guard in 4D: 128^4 is 268M cells (~40 GiB per
    worker) while every axis is inside a 128 rail. The refusal must name the
    estimate, because "reduce an axis" is otherwise a guess."""
    monkeypatch.setattr(config, "MAX_CELLS_2D", 16*16*16*16)
    big = {"ndim": 2, "axes": [{"lo": -6.0, "hi": 6.0, "N": 32}]
           + [dict(a) for a in G2["axes"][1:]]}
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2(grid=big))
        assert r.status_code == 422, r.text
        assert "MAX_CELLS_2D" in r.text and "GiB per variant worker" in r.text


def test_the_device_fit_check_is_the_operative_2d_guard(monkeypatch):
    """WIGNERF_MAX_CELLS_2D is a RAIL; the guard that means something asks the
    driver how much is free on the devices this session's workers land on.

    A fixed cell count cannot do that job — it is wrong in both directions. It
    refused 128x128x64x64 (67M cells, 13.0 GiB for ONE worker) on a 24 GiB card,
    and it would wave through two 6.5 GiB workers onto an 11 GiB one. This test
    stubs the free-memory probe so it runs anywhere, GPU or not.
    """
    import routers.sessions as rs
    free = {}
    monkeypatch.setattr(rs.xp, "device_free_bytes", lambda d: free.get(d))
    monkeypatch.setattr(rs, "resolve_devices", lambda spec: ["cuda:9", "cuda:8"])
    monkeypatch.setattr(config, "MAX_CELLS_2D", 1 << 40)   # rail out of the way

    big = {"ndim": 2, "axes": [{"lo": -8.0, "hi": 8.0, "N": 32},
                               {"lo": -8.0, "hi": 8.0, "N": 32},
                               {"lo": -7.0, "hi": 7.0, "N": 16},
                               {"lo": -7.0, "hi": 7.0, "N": 16}]}
    per = 32*32*16*16*config.BYTES_PER_CELL_2D
    with TestClient(app) as client:
        # one worker, a card with room to spare
        free.clear()
        free["cuda:9"] = int(per*4) + rs.CONTEXT_BYTES
        r = client.post("/api/sessions", json=cfg2(grid=big, variants=["qn"]))
        assert r.status_code == 200, r.text
        client.delete("/api/sessions/%s" % r.json()["session_id"])

        # the SAME grid on a card that cannot hold it: refused, with numbers
        free["cuda:9"] = int(per*0.5)
        r = client.post("/api/sessions", json=cfg2(grid=big, variants=["qn"]))
        assert r.status_code == 422, r.text
        assert "cuda:9" in r.text and "GiB free" in r.text

        # two workers spread over the pair: the SMALLER card is what binds,
        # which no per-session cell count could ever express
        free["cuda:9"] = int(per*10)
        free["cuda:8"] = int(per*0.5)
        r = client.post("/api/sessions",
                        json=cfg2(grid=big, variants=["qn", "cn"]))
        assert r.status_code == 422 and "cuda:8" in r.text, r.text

        # unknown free memory (no cupy, unreadable /proc) must not refuse:
        # there the rail is the only guard and guessing would be worse
        free.clear()
        r = client.post("/api/sessions", json=cfg2(grid=big, variants=["qn"]))
        assert r.status_code == 200, r.text
        client.delete("/api/sessions/%s" % r.json()["session_id"])


def test_the_preview_refuses_what_the_session_would(monkeypatch):
    """The IC preview builds the FULL state at the requested grid and fires on
    every form change, LONG before anyone presses Restart — so it must be bound
    by the same ceilings session creation is. It was not, and a form grid of
    256⁴ (4.3e9 cells, one dims switch away from the 1D default) went to the CPU
    fallback and allocated 34 GiB arrays until the kernel OOM-killed the server
    (2026-07-26, a 125 GiB host). The create-time refusal came far too late.
    """
    monkeypatch.setattr(config, "MAX_CELLS_2D", 16*16*16*16)
    big = dict(G2)
    big["axes"] = [{"lo": -6.0, "hi": 6.0, "N": 32}] + list(G2["axes"][1:])
    with TestClient(app) as client:
        body = dict(IC2, grid=big, hbar_eff=1.0)
        r = client.post("/api/preview/wigner", json=body)
        assert r.status_code == 422, r.status_code
        assert "MAX_CELLS_2D" in r.text
        # ...and the SAME grid is refused at create, by the same message
        s = client.post("/api/sessions", json=cfg2(grid=big))
        assert s.status_code == 422
        assert "MAX_CELLS_2D" in s.text
        # the per-axis rail is enforced there too
        over = dict(G2)
        over["axes"] = [{"lo": -6.0, "hi": 6.0, "N": 256}] + list(G2["axes"][1:])
        r = client.post("/api/preview/wigner", json=dict(IC2, grid=over,
                                                         hbar_eff=1.0))
        assert r.status_code == 422 and "MAX_GRID_2D" in r.text
        # a grid inside the ceilings still previews
        ok = client.post("/api/preview/wigner", json=dict(IC2, grid=G2,
                                                          hbar_eff=1.0))
        assert ok.status_code == 200, ok.text


def test_the_preview_cpu_fallback_is_bounded_by_free_host_memory(monkeypatch):
    """The rail is deliberately far past any card because _fit_error asks the
    driver for the real answer — but _fit_error runs at session CREATION and
    this endpoint fires on every keystroke, at the full session grid. The CPU
    fallback is where an over-large grid lands (on a CPU-only host there is no
    _pick_device to decline at all), and it had no fit check, so the rail's 4x
    loosening moved an unbounded allocation 4x further out."""
    import routers.preview as rp
    free = {}
    monkeypatch.setattr(rp.xp, "device_free_bytes", lambda d: free.get(d))
    monkeypatch.setattr(rp, "_pick_device", lambda cells, ndim=1: None)
    body = dict(IC2, grid=G2, hbar_eff=1.0)
    need = 16**4*rp.PREVIEW_BYTES_PER_CELL[2]*rp.PREVIEW_HEADROOM
    with TestClient(app) as client:
        free["cpu"] = int(need*0.5)
        r = client.post("/api/preview/wigner", json=body)
        assert r.status_code == 422, r.text
        assert "GiB of host memory" in r.text and "available" in r.text
        # room to spare: builds as usual
        free["cpu"] = int(need*4)
        assert client.post("/api/preview/wigner", json=body).status_code == 200
        # unknown free memory must NOT refuse — there the rail is the only
        # guard, exactly as in routers.sessions._fit_error
        free.clear()
        assert client.post("/api/preview/wigner", json=body).status_code == 200


def test_a_bad_ic_is_refused_before_any_device_is_touched(monkeypatch):
    """A malformed IC is a CLIENT error and costs no arrays to detect, so it is
    decided before a backend is chosen. Deciding it inside the build made a bad
    spec and a device failure arrive as the same exception from the same call,
    and the GPU branch translated BOTH into 422 — turning a transient device
    problem into "your IC is wrong" and skipping the CPU fallback."""
    import routers.preview as rp
    touched = []
    monkeypatch.setattr(rp, "_pick_device",
                        lambda cells, ndim=1: touched.append(ndim))
    monkeypatch.setattr(rp, "_backend",
                        lambda spec="cpu": touched.append(spec))
    with TestClient(app) as client:
        # a mixture component with no sigma_k cannot be built
        ic = {"type": "mixture", "components": [
            {"q0": [1.0, 0.0], "k0": [0.0, 0.0], "sigma_q": [0.7, 0.7]}]}
        r = client.post("/api/preview/wigner",
                        json=dict(ic, grid=G2, hbar_eff=1.0))
        assert r.status_code == 422 and "sigma_k" in r.text, r.text
        assert touched == [], touched


def test_the_device_endpoint_carries_the_per_ndim_ceilings(monkeypatch):
    """The setup form's grid ceilings come from HERE, not from `status`.

    `status.max_grid` / `max_cells` / `bytes_per_cell` are resolved once, for the
    ndim of the session that is RUNNING, while the form has to describe the ndim
    it is SHOWING — and `dims` is restart-only, so the two disagree for exactly as
    long as a switch waits for its restart. Reading them off `status` measurably
    broke the panel in both directions: over a live 1D session a 2D form offered N
    up to 4096 against an API ceiling of 128 and rendered NO footprint estimate
    (bytes_per_cell is null at ndim=1 — the one number that says whether a 2D
    session can start, missing precisely before the first 2D restart), and over a
    live 2D session a 1D form's N select collapsed to a single option.

    The monkeypatch is the load-bearing half: these keys sit OUTSIDE
    _probe_backend's lru_cache, and a cached copy would freeze the host's limits
    at whatever the first caller saw.
    """
    with TestClient(app) as client:
        d = client.get("/api/device").json()
        assert d["max_grid"]["1"] == config.MAX_GRID
        assert d["max_grid"]["2"] == config.MAX_GRID_2D
        # unbounded at ndim=1: the per-axis cap already bounds a 2D array
        assert d["max_cells"]["1"] is None
        assert d["max_cells"]["2"] == config.MAX_CELLS_2D
        assert d["bytes_per_cell_2d"] == config.BYTES_PER_CELL_2D
        # ...and the probe's cache does not freeze them
        monkeypatch.setattr(config, "MAX_GRID_2D", 64)
        monkeypatch.setattr(config, "MAX_CELLS_2D", 1 << 20)
        d = client.get("/api/device").json()
        assert d["max_grid"] == {"1": config.MAX_GRID, "2": 64}
        assert d["max_cells"]["2"] == 1 << 20
        # the cached probe's own fields still ride along
        assert d["pool"] and d["precision"]


def test_ic_and_grid_dimensionality_must_agree():
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2(
            ic={"type": "mixture", "components": [
                {"x0": 1.0, "p0": 0.0, "sigma_x": 0.7, "sigma_p": 0.7}]}))
        assert r.status_code == 422
        assert "2D" in r.text and "coordinate" in r.text


def test_a_2d_potential_may_use_y_and_a_1d_one_may_not():
    with TestClient(app) as client:
        assert client.post("/api/sessions",
                           json=cfg2(potential="x^2/2 + y^2/2 + 0.1*x*y")
                           ).status_code == 200
        r = client.post("/api/preview/potential",
                        json={"expr": "x*y", "x1": -6, "x2": 6,
                              "grid": {"x1": -6, "x2": 6, "Nx": 64,
                                       "p1": -7, "p2": 7, "Np": 64}})
        d = r.json()
        assert d["ok"] is False and "only 'x'" in d["error"]
