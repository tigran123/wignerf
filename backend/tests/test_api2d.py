"""
2D sessions through the API: the record path end to end, and the one explicit
refusal that still stands in for deferred work (milestone M3 in CLAUDE.md).

The refusal test is not box-ticking. The gate replaces a feature that would
otherwise half-work — an auto-expanding 4D grid with no memory guard on the
doubling — and a gate that silently stopped firing is exactly how a half-feature
ships. It also pins that the message NAMES the milestone, so whoever hits it
learns what is missing rather than that "2D is broken".

THREE of the four have landed, and each left the opposite kind of test behind:
the thing the gate forbade must now be ACCEPTED and work. M2 (relativistic
qr/cr) and M1 (float32) on 2026-07-27, M4 (mp4 export) on 2026-07-28. See
tests/test_propagator2d.py, tests/test_precision.py and tests/test_export2d.py.
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
    """The setup document (and therefore mp4 import) worked for 2D from the
    first cut, before the video render did (M4, landed 2026-07-28)."""
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

def test_the_host_default_precision_now_applies_at_either_ndim(monkeypatch):
    """ONE resolution rule at every ndim, which is what M1 bought here.

    While float32 was refused in 2D, an omitted precision had to resolve to
    float64 there — a gate must refuse what was ASKED FOR, and WIGNERF_PRECISION
    is a host DEFAULT, not a request. Without that special case every 2D session
    on a float32 host 422'd over a value nobody sent (the SPA, curl and
    scripts/ws_smoke.py --ndim 2 all omit the field), and it took this whole test
    module down with it.

    With the gate gone the special case went too, so this now pins the OPPOSITE:
    a float32 host gives a 2D session float32, exactly as it gives a 1D one."""
    monkeypatch.setattr(config, "PRECISION", "float32")
    one_d = {"grid": dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64),
             "potential": "x^2/2", "variants": ["qn"], "record_dt": 0.05,
             "ic": {"type": "mixture", "components": [
                 {"x0": 0.0, "p0": 0.0, "sigma_x": 0.7, "sigma_p": 0.7}]}}
    with TestClient(app) as client:
        for body in (cfg2(), one_d):
            r = client.post("/api/sessions", json=body)
            assert r.status_code == 200, r.text
            sid = r.json()["session_id"]
            assert client.get("/api/sessions/%s" % sid).json()["precision"] \
                == "float32"
            client.delete("/api/sessions/%s" % sid)


def test_float32_2d_sessions_are_accepted_and_stream():
    """M1's acceptance test, on the pattern M2's retirement left behind: a
    retired gate is replaced by a test that the thing it forbade now WORKS.

    float32 reaches 2D through no new code — the mixed-precision split is
    ndim-blind (see the Propagator docstring) — so what needs pinning at THIS
    level is the wiring: the session runs, reports the precision it was given,
    and streams finite observables rather than the NaN a wrong dtype pairing
    produces. The physics is in tests/test_precision.py."""
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2(precision="float32"))
        assert r.status_code == 200, r.text
        info = r.json()
        sid = info["session_id"]
        assert client.get("/api/sessions/%s" % sid).json()["precision"] \
            == "float32"
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            f = None
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    break
            assert f is not None, "no frame arrived"
            assert f.geom.ndim == 2
            for v in f.variants:
                assert v.E == v.E and abs(v.E) < 1e3, v.E
                assert v.purity == v.purity
        client.delete("/api/sessions/%s" % sid)


@pytest.mark.parametrize("over,needle", [
    (dict(auto_expand=True), "M3"),
])
def test_deferred_features_are_refused_with_their_milestone(over, needle):
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2(**over))
        assert r.status_code == 422, r.text
        assert needle in r.text, r.text


@pytest.mark.parametrize("variants", [["qr"], ["cr"], ["qn", "qr", "cn", "cr"]])
def test_relativistic_2d_sessions_are_accepted_and_stream(variants):
    """The inverse of a gate test, and the reason M2's row left the table above:
    qr/cr must now CREATE in 2D and produce finite records. A relativistic
    variant that 422'd here for two months is exactly the thing a stale gate
    would keep doing after the physics landed."""
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=cfg2(variants=variants, c=10.0))
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["variants"] == variants
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            f = None
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    break
            assert f is not None, "no frame arrived"
            assert f.geom.ndim == 2
            assert len(f.variants) == len(variants), "lockstep bundle incomplete"
            for v in f.variants:
                # the mc^2 that cancels inside dT dominates <H> and is
                # subtracted by observables — a NaN or a stray 100.0 here is
                # exactly what a mishandled rest energy looks like
                assert v.E == v.E and abs(v.E) < 1e3, v.E
                assert v.purity == v.purity
        client.delete("/api/sessions/%s" % sid)


def test_massless_needs_relativistic_variants_and_now_works_in_2d():
    """mass = 0 became reachable in 2D only with M2, because the schema requires
    exclusively relativistic variants there (non-relativistic T = p^2/2m
    diverges). The gradient c*k_i/|k| is 0/0 at the origin, which IS a lattice
    point, so this is the API-level guard that it does not stream NaN."""
    with TestClient(app) as client:
        bad = client.post("/api/sessions",
                          json=cfg2(mass=0.0, variants=["qn", "cr"]))
        assert bad.status_code == 422, bad.text

        r = client.post("/api/sessions",
                        json=cfg2(mass=0.0, variants=["qr", "cr"], c=1.0))
        assert r.status_code == 200, r.text
        info = r.json()
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            f = None
            for _ in range(200):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    break
            assert f is not None, "no frame arrived"
            for v in f.variants:
                assert v.E == v.E, "massless <H> is NaN — the origin of the "\
                                   "momentum lattice reached the 0/0 gradient"
                assert v.purity == v.purity
        client.delete("/api/sessions/%s" % sid)


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


def test_the_2d_export_gate_is_gone():
    """M4 landed on 2026-07-28: a 2D export must get PAST the ndim check and
    be refused only by the ordinary paused-only/empty-history rules, which are
    what a fresh session trips. The frame itself is pinned in
    tests/test_export2d.py."""
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        r = client.post("/api/sessions/%s/export" % sid, json={"fps": 30})
        assert r.status_code == 422, r.text
        assert "M4" not in r.text and "not available for 2D" not in r.text
        assert "no computed records" in r.text
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
    # float64 explicitly: these sessions omit `precision`, so they take the
    # host default, and the fit check scales with it since M1 (112 B/cell in
    # float32 against 208). A bare constant here would silently stop matching
    # the server on a float32 host.
    per = 32*32*16*16*config.bytes_per_cell(2, "float64")
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


def test_the_fit_refusal_describes_the_POOL_not_the_first_device(monkeypatch):
    """Two refusals that need different advice, and the old message gave the
    wrong one for the worse case.

    It reported whichever assigned device sorted FIRST and always closed with
    "Reduce an axis, drop a variant, or pick a device with more room". On a real
    host (RTX 3090 + 2080 Ti) a 128x128x128x64 grid needs 26.0 GiB per worker,
    and it said "cuda:0 has 8.9 GiB free ... pick a device with more room" —
    naming the small card and implying a roomier one existed, when the 3090's
    23.6 GiB could not hold one worker either. The reasonable reading is "so
    what, I have cuda:1", and it is wrong.

    So the roomiest device decides which story is told: whether ANY device can
    hold a single worker is what separates "this grid is too big for this host"
    from "these variants do not distribute onto it".
    """
    import routers.sessions as rs
    free = {}
    monkeypatch.setattr(rs.xp, "device_free_bytes", lambda d: free.get(d))
    monkeypatch.setattr(rs, "resolve_devices", lambda spec: ["cuda:9", "cuda:8"])
    monkeypatch.setattr(config, "MAX_CELLS_2D", 1 << 40)

    big = {"ndim": 2, "axes": [{"lo": -8.0, "hi": 8.0, "N": 32},
                               {"lo": -8.0, "hi": 8.0, "N": 32},
                               {"lo": -7.0, "hi": 7.0, "N": 16},
                               {"lo": -7.0, "hi": 7.0, "N": 16}]}
    # float64 explicitly: these sessions omit `precision`, so they take the
    # host default, and the fit check scales with it since M1 (112 B/cell in
    # float32 against 208). A bare constant here would silently stop matching
    # the server on a float32 host.
    per = 32*32*16*16*config.bytes_per_cell(2, "float64")
    with TestClient(app) as client:
        # (A) NOTHING can hold one worker. Advice must not send the user after
        # another device or a shorter variant list — neither exists to be had.
        free.clear()
        free["cuda:9"] = int(per*0.8)
        free["cuda:8"] = int(per*0.4)
        r = client.post("/api/sessions",
                        json=cfg2(grid=big, variants=["qn", "cn"]))
        assert r.status_code == 422, r.text
        msg = r.json()["detail"]
        assert "no device in the pool can hold even one" in msg, msg
        assert "cuda:9" in msg, "must name the ROOMIEST device: %s" % msg
        assert "will not help" in msg, msg
        assert "pick a device with more room" not in msg, msg

        # (B) a worker DOES fit on cuda:9 — now it is a distribution problem,
        # the over-subscribed device is the one to name, and moving there is
        # real advice with a real number attached
        free.clear()
        free["cuda:9"] = int(per*10)
        free["cuda:8"] = int(per*0.5)
        r = client.post("/api/sessions",
                        json=cfg2(grid=big, variants=["qn", "cn"]))
        assert r.status_code == 422, r.text
        msg = r.json()["detail"]
        assert "cuda:8" in msg and "would put" in msg, msg
        assert "set device to cuda:9" in msg, msg
        assert "no device in the pool" not in msg, msg


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


def test_the_preview_refusal_quotes_the_FORM_precision(monkeypatch):
    """The two refusals a user sees at once must quote the SAME footprint.

    Over the cell cap the Setup panel renders its own footprint line in the GRID
    section and the preview endpoint's 422 lands under the IC plot — the same
    quantity, side by side. The preview took no precision and so always quoted
    float64: at 128⁴ in a float32 form that read "~52.0 GiB per variant worker"
    under the plot against "≈ 28.00 GiB per worker" two columns away, and the
    bigger number was the wrong one.

    Note what the field does NOT mean: the preview still BUILDS in float64 on its
    own backend, and PREVIEW_BYTES_PER_CELL stays precision-blind. This is the
    precision of the session the grid would create, quoted in a message whose own
    words are "per variant worker"."""
    monkeypatch.setattr(config, "MAX_CELLS_2D", 16*16*16*16)
    big = dict(G2)
    big["axes"] = [{"lo": -6.0, "hi": 6.0, "N": 32}] + list(G2["axes"][1:])
    cells = 32*16*16*16
    with TestClient(app) as client:
        want = {}
        for p in ("float64", "float32"):
            r = client.post("/api/preview/wigner",
                            json=dict(IC2, grid=big, hbar_eff=1.0, precision=p))
            assert r.status_code == 422, r.text
            gib = cells*config.bytes_per_cell(2, p)/1024**3
            assert "%.1f GiB" % gib in r.text, (p, r.text)
            want[p] = gib
            # ...and it is the same number the CREATE path would quote, which is
            # the property that actually matters — one of these lands under the
            # IC plot and the other in the header, together
            s = client.post("/api/sessions", json=cfg2(grid=big, precision=p))
            assert s.status_code == 422
            assert "%.1f GiB" % gib in s.text, (p, s.text)
        assert want["float64"] > 1.8*want["float32"], want
        # an omitted precision resolves to the host default, exactly as
        # SessionCreate._check resolves it — not to a hard-coded float64
        monkeypatch.setattr(config, "PRECISION", "float32")
        r = client.post("/api/preview/wigner",
                        json=dict(IC2, grid=big, hbar_eff=1.0))
        assert "%.1f GiB" % want["float32"] in r.text, r.text


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
        # per PRECISION since M1, for the same reason the two above are per
        # ndim: precision is restart-only too, so the form must be able to
        # estimate for the one it is SHOWING. float32 is the smaller figure and
        # the panel would over-report by 1.9x without it.
        assert d["bytes_per_cell_2d"]["float64"] == 208
        assert d["bytes_per_cell_2d"]["float32"] == 112
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
