"""mp4 export: the analytic description, the frame renderer and the job."""

import json
import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import describe, protocol, render_mpl, videoexport
from core.render_mpl import FrameFigure, RangeStats, meta_columns
from main import app

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
IC = {"type": "mixture",
      "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707, "sigma_p": 0.707}]}

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg is not installed")


# ---------------------------------------------------------------------------
# describe.py — the "how to reproduce this" text
# ---------------------------------------------------------------------------

def test_ic_expression_mixture_substitutes_numbers():
    ic = protocol.ICSpec(type="mixture", components=[
        {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0},
        {"x0": -2.0, "p0": 1.0, "sigma_x": 0.5, "sigma_p": 1.0, "weight": 3.0},
    ])
    text = " ".join(describe.ic_expression(ic, 1.0))
    assert "W(x,p,0)" in text
    assert "(x − 2)" in text and "(x + 2)" in text   # never "x−−2"
    assert "(p − 1)" in text
    # amplitudes carry the normalized weights: 1/4 and 3/4 over 2*pi*sx*sp
    assert "%.6g" % (0.25/(2*np.pi*0.5*1.0)) in text
    assert "%.6g" % (0.75/(2*np.pi*0.5*1.0)) in text


def test_ic_expression_cat_gives_psi_and_derived_sigma_p():
    ic = protocol.ICSpec(type="cat", components=[
        {"x0": -2.0, "p0": 0.0, "sigma_x": 0.5},
        {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5, "phase": 3.14159},
    ])
    lines = describe.ic_expression(ic, 2.0)
    text = " ".join(lines)
    assert "ψ(x,0)" in text and "Wigner[ψ]" in text
    assert "e^(i3.14159)" in text
    # sigma_p is DERIVED for cat states: hbar/(2 sigma_x) = 2/(2*0.5) = 2
    assert "σp = 2" in text


def test_param_lines_report_live_changes_in_range():
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    log = [{"at_record": 5, "applied": {"U": "x^4/4"}},
           {"at_record": 900, "applied": {"mass": 2.0}}]
    text = " ".join(describe.param_lines(cfg, log, 0, 100))
    assert "U(x) = x^2/2" in text
    assert "live change at record 5" in text and "x^4/4" in text
    assert "record 900" not in text          # outside the exported range
    blob = json.loads(describe.config_json(cfg, log, export={"frames": 3}))
    assert blob["config"]["potential"] == "x^2/2"
    assert blob["param_log"][1]["applied"]["mass"] == 2.0
    assert blob["export"]["frames"] == 3


def test_param_block_describes_the_first_exported_record():
    """The physics line must describe the frames it sits on, not the values
    the session happened to END with: a run whose ℏ went 1 → 2 → 100 and is
    exported from record 0 says ℏ = 1, and the changes read "before → after"."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"], hbar_eff=100.0, mass=2.0)
    log = [{"at_record": 10, "applied": {"mass": 2.0}, "before": {"mass": 1.0}},
           {"at_record": 20, "applied": {"hbar_eff": 2.0},
            "before": {"hbar_eff": 1.0}},
           {"at_record": 40, "applied": {"hbar_eff": 100.0},
            "before": {"hbar_eff": 2.0}}]
    text = " ".join(describe.param_lines(cfg, log, 0, 50))
    assert "m = 1" in text and "ℏ = 1 " in text
    assert "ℏ 1 → 2" in text and "ℏ 2 → 100" in text and "m 1 → 2" in text
    # exporting from the middle: ℏ was already 2 there
    text = " ".join(describe.param_lines(cfg, log, 30, 50))
    assert "ℏ = 2" in text and "m = 2" in text
    assert "record 20" not in text
    blob = json.loads(describe.config_json(cfg, log, at_record=0))
    assert blob["config"]["hbar_eff"] == 1.0 and blob["config"]["mass"] == 1.0


# ---------------------------------------------------------------------------
# render_mpl.py — the figure
# ---------------------------------------------------------------------------

def _vframe(seed, Nx=32, Np=32):
    rng = np.random.default_rng(seed)
    wq = (rng.random((Nx, Np))*65535).astype(np.uint16)
    # a 1D record: exactly one plane, and that plane IS W
    plane = protocol.PlaneFrame(a=0, b=1, mode=0, wq=wq, wmin=-0.1, wmax=0.3)
    return protocol.VariantFrame(
        vid=protocol.variant_id(True, False), dt=1e-3, E=1.0 + seed,
        purity=1.0, lz=0.0, mean=(0.1, 0.0), std=(0.7, 0.7),
        planes=(plane,),
        marg=(rng.random(Nx).astype("f4"), rng.random(Np).astype("f4")))


def _stats(n=3):
    st = RangeStats(ndim=1, t=[0.0, 0.05, 0.1][:n])
    st.marg_max = [1.0, 1.0]
    st.lo, st.hi = (-6.0, -7.0), (6.0, 7.0)
    st.E["qn"] = [1.0, 1.1, 1.2][:n]
    st.uncert[("qn", 0)] = [0.5]*n
    st.purity["qn"] = [1.0]*n
    st.lz["qn"] = [0.0]*n
    st.scale[("qn", (0, 1))] = 0.3
    return st


def test_video_labels_match_the_ui():
    """The video must NAME things as the SPA does: SeriesPlot.vue's titles
    verbatim (γ keeps 2πℏ∬W²dxdp, not the equivalent Tr ρ²), the Setup
    panel's ℏ, and the mode select's label (the wire value "batch" IS its
    display label)."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"], mode="batch", t2=5.0)
    text = " ".join(describe.param_lines(cfg))
    assert "ℏ = 1" in text and "ℏ_eff" not in text
    assert "mode = batch" in text

    geom = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    fig = FrameFigure(["qn"], stats, meta_columns(cfg, geom, stats, ["qn"],
                                                  0, 2, 3, 30),
                      width=640, height=360)
    try:
        # the series plots carry their title at loc='right' (the axis
        # offset label owns the top-left corner)
        titles = [ax.get_title(loc=where) for ax in fig.fig.axes
                  for where in ("center", "right")]
    finally:
        fig.close()
    assert "purity γ(t) = 2πℏ∬W²dxdp" in titles
    assert "E(t)" in titles and "ΔX·ΔP(t)" in titles
    assert "ρ(x) = ∫W dp" in titles and "ρ(p) = ∫W dx" in titles


def test_series_ylim_matches_uplot_rule():
    """SeriesPlot.vue: pad = max(15% of span, 1e-4 of |max|, 1e-12). A
    purity series that drifts 2e-5 must therefore keep the UI's ±1e-4
    window (a flat-looking line), not matplotlib's tight autoscale."""
    from core.render_mpl import series_ylim
    lo, hi = series_ylim([1.0 - 2e-5*i/100 for i in range(101)])
    assert hi == pytest.approx(1.0 + 1e-4, rel=1e-9)
    assert lo == pytest.approx(1.0 - 2e-5 - 1e-4, rel=1e-9)
    # a genuinely large span falls back to the 15% padding
    lo, hi = series_ylim([0.0, 10.0])
    assert (lo, hi) == pytest.approx((-1.5, 11.5))
    assert series_ylim([]) == (0.0, 1.0)


def test_show_grid_covers_charts_and_w_panels():
    """One setting, every plot: the SPA's "grid lines on plots" toggle must
    reach the W heatmaps too — matplotlib draws the axes grid UNDER the
    image, so the panels need their own lines on top (they had none)."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    out = {}
    for flag in (True, False):
        fig = FrameFigure(["qn"], stats, meta, width=320, height=240,
                          show_grid=flag)
        try:
            panel_ax = fig.images[0][0]
            chart_ax = [a for a in fig.fig.axes
                        if a.get_title(loc="right") == "E(t)"][0]
            # the panel grid is built per record geometry, so it exists only
            # after the first update() (see FrameFigure._apply_geom)
            frame = bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2))
            out[flag] = (len(panel_ax.lines),
                         any(g.get_visible() for g in chart_ax.get_xgridlines()),
                         frame)
        finally:
            fig.close()
    on, off = out[True], out[False]
    assert on[0] > 0 and off[0] == 0, "W panel grid lines ignored the toggle"
    assert on[1] and not off[1], "chart grid ignored the toggle"
    assert on[2] != off[2], "the rendered frames are identical"


def test_theme_repaints_the_frame_but_not_the_heatmap():
    """ExportSpec.theme must reach every piece of chrome — figure face, axes
    face, titles, ticks and the variant curve colours — while leaving the
    heatmap alone: "bwr" with white at W = 0 is the physics convention and is
    identical in both themes on screen too."""
    # the wire contract: light by default (as in the SPA), both names
    # accepted, nothing else
    assert protocol.ExportSpec().theme == "light"
    assert protocol.ExportSpec(theme="dark").theme == "dark"
    with pytest.raises(ValueError):
        protocol.ExportSpec(theme="solarized")

    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    seen = {}
    for th in ("dark", "light"):
        fig = FrameFigure(["qn"], stats, meta, width=320, height=240,
                          theme=th)
        try:
            panel_ax, im, _cell = fig.images[0]
            chart_ax = [a for a in fig.fig.axes
                        if a.get_title(loc="right") == "E(t)"][0]
            frame = bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2))
            seen[th] = dict(
                bg=fig.fig.get_facecolor(),
                axbg=chart_ax.get_facecolor(),
                # NB not chart_ax.title — matplotlib keeps a separate artist
                # per title loc, and the series titles are loc="right", so
                # the default centre one stays untouched (and black)
                spine=chart_ax.spines["bottom"].get_edgecolor(),
                tick=chart_ax.xaxis.get_ticklabels()[0].get_color(),
                marg_title=fig.marg_axes[0].title.get_color(),
                panel_title=panel_ax.title.get_color(),
                rho=fig.marg_lines[(0, "qn")].get_color(),
                cmap=im.get_cmap().name,
                clim=im.get_clim(),
                frame=frame,
            )
        finally:
            fig.close()
    d, l = seen["dark"], seen["light"]
    for k in ("bg", "axbg", "spine", "tick", "marg_title", "panel_title", "rho"):
        assert d[k] != l[k], f"{k} did not follow the theme"
    # the data layer is theme-independent, and so is the rendered size
    assert d["cmap"] == l["cmap"] == "bwr"
    assert d["clim"] == l["clim"]
    assert len(d["frame"]) == len(l["frame"])
    assert d["frame"] != l["frame"], "the two themes rendered identical pixels"


def test_unknown_theme_falls_back_to_the_light_default():
    """The schema only admits dark|light, but FrameFigure is also constructed
    directly (tests, the render pool) — an unknown name must not KeyError
    halfway through building a figure. It lands on light, the SPA's own
    default and ExportSpec.theme's."""
    stats = _stats()
    fig = FrameFigure(["qn"], stats, ([], []), width=320, height=240,
                      theme="chartreuse")
    try:
        assert fig.theme == "light"
    finally:
        fig.close()


def test_axes_follow_the_record_geometry():
    """Auto-expand makes the domain a PER-RECORD fact, and the video follows
    it exactly as the SPA does: freezing the axes at the range union rendered
    every pre-expansion frame as a postage stamp in the corner of its panel.
    Only the VALUE scales (colour, marginal amplitude) are export-wide."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    small = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    big = protocol.RecordGeom.from_1d(64, 64, -12.0, 12.0, -14.0, 14.0)
    stats = _stats(2)
    stats.lo, stats.hi = (-12.0, -14.0), (12.0, 14.0)          # union
    fig = FrameFigure(["qn"], stats,
                      meta_columns(cfg, small, stats, ["qn"], 0, 1, 2, 30),
                      width=640, height=360)
    try:
        panel = fig.images[0][0]
        clim = fig.images[0][1].get_clim()
        rho_ylim = fig.marg_axes[0].get_ylim()
        fig.update(0, 0.0, small, [_vframe(1)], 0, 1)
        assert panel.get_xlim() == (small.x1, small.x2)
        assert panel.get_ylim() == (small.p1, small.p2)
        assert fig.marg_axes[0].get_xlim() == (small.x1, small.x2)
        assert fig.marg_axes[1].get_xlim() == (small.p1, small.p2)
        fig.update(1, 0.05, big, [_vframe(2, 64, 64)], 0, 1)
        assert panel.get_xlim() == (big.x1, big.x2)
        assert panel.get_ylim() == (big.p1, big.p2)
        assert fig.marg_axes[0].get_xlim() == (big.x1, big.x2)
        # value scales are export-wide: no brightness or height pumping
        assert fig.images[0][1].get_clim() == clim
        assert fig.marg_axes[0].get_ylim() == rho_ylim
    finally:
        fig.close()
    # the metadata quotes the first record's window AND the widest one
    left, _right = meta_columns(cfg, small, stats, ["qn"], 0, 1, 2, 30)
    text = " ".join(left)
    assert "grid at record 0: 32×32" in text and "[-6, 6]" in text
    assert "widest" in text and "[-12, 12]" in text


def test_frame_figure_renders_distinct_frames():
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    fig = FrameFigure(["qn"], stats, meta, width=640, height=360)
    try:
        # update() hands back a view of the Agg buffer (RGBA, fed straight
        # to ffmpeg): copy before comparing, the next update overwrites it
        a = bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2))
        b = bytes(fig.update(1, 0.05, geom, [_vframe(2)], 0, 2))
        assert len(a) == 640*360*4 and len(b) == len(a)
        assert a != b, "the frame did not change between records"
        # a regrid mid-video must be accepted (per-record geometry)
        big = protocol.RecordGeom.from_1d(64, 32, -12.0, 12.0, -7.0, 7.0)
        c = bytes(fig.update(2, 0.1, big, [_vframe(3, Nx=64)], 0, 2))
        assert len(c) == len(a)
    finally:
        fig.close()


# ---------------------------------------------------------------------------
# the job, end to end
# ---------------------------------------------------------------------------

def _mk(client, **over):
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": IC,
           "variants": ["qn", "cn"], "record_dt": 0.05, "delay": 0.0}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def _solve_a_few(client, ws, sid, n=6):
    import json as _json
    ws.send_text(_json.dumps({"type": "play"}))
    for _ in range(200):
        time.sleep(0.05)
        if client.get("/api/sessions/%s" % sid).json()["record_extent"][1] >= n:
            break
    ws.send_text(_json.dumps({"type": "pause"}))
    time.sleep(0.3)                     # let in-flight records land
    return client.get("/api/sessions/%s" % sid).json()["record_extent"]


def test_setup_document_is_what_the_run_started_from():
    """The exchangeable "initial conditions": whatever a run did to itself
    (live ℏ/U changes, an auto-expand toggle), the document must still be the
    config POST /api/sessions was given — and must be re-postable."""
    from core.protocol import ParamChange, SessionCreate
    from core.session import SESSIONS
    with TestClient(app) as client:
        info = _mk(client, potential="x^2/2", hbar_eff=1.0)
        sid = info["session_id"]
        s = SESSIONS[sid]
        s.apply_params(ParamChange(hbar_eff=0.25, U="x^4/4", mass=3.0,
                                   auto_expand=True))
        assert s.cfg.hbar_eff == 0.25 and s.cfg.potential == "x^4/4"

        doc = client.get("/api/sessions/%s/setup" % sid).json()
        assert doc["format"] == "wignerf-setup" and doc["version"] == 1
        cfg = doc["config"]
        assert cfg["hbar_eff"] == 1.0 and cfg["potential"] == "x^2/2"
        assert cfg["mass"] == 1.0 and cfg["auto_expand"] is False
        # the document IS a session request
        SessionCreate.model_validate(cfg)
        r = client.post("/api/sessions", json=cfg)
        assert r.status_code == 200, r.text
        client.delete("/api/sessions/%s" % r.json()["session_id"])

        r = client.get("/api/sessions/%s/setup" % sid)
        assert 'filename="wignerf-setup-QN-CN-' in r.headers["content-disposition"]
        client.delete("/api/sessions/%s" % sid)
        assert client.get("/api/sessions/%s/setup" % sid).status_code == 404


def test_choose_encoder_honors_config(monkeypatch):
    """auto uses the GPU h264_nvenc if the runtime probe passes, else libx264;
    cpu/nvenc force the choice. No GPU needed — the probe is monkeypatched."""
    import config as appconfig
    monkeypatch.setattr(videoexport, "_nvenc_ok", lambda: True)
    monkeypatch.setattr(appconfig, "EXPORT_ENCODER", "auto")
    assert videoexport.choose_encoder()[:2] == ["-c:v", "h264_nvenc"]
    monkeypatch.setattr(videoexport, "_nvenc_ok", lambda: False)
    assert videoexport.choose_encoder()[:2] == ["-c:v", "libx264"]
    assert "veryfast" in videoexport.choose_encoder()   # not the old 'medium'
    # cpu forces libx264 even where nvenc works; nvenc forces the GPU encoder
    monkeypatch.setattr(videoexport, "_nvenc_ok", lambda: True)
    monkeypatch.setattr(appconfig, "EXPORT_ENCODER", "cpu")
    assert videoexport.choose_encoder()[:2] == ["-c:v", "libx264"]
    monkeypatch.setattr(videoexport, "_nvenc_ok", lambda: False)
    monkeypatch.setattr(appconfig, "EXPORT_ENCODER", "nvenc")
    assert videoexport.choose_encoder()[:2] == ["-c:v", "h264_nvenc"]
    # the explicit-argument form overrides the config
    assert videoexport.choose_encoder("cpu")[:2] == ["-c:v", "libx264"]


@needs_ffmpeg
def test_export_end_to_end(tmp_path, monkeypatch):
    import config as appconfig
    monkeypatch.setattr(appconfig, "EXPORT_DIR", str(tmp_path))
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            first, last = _solve_a_few(client, ws, sid)
            k1 = min(last, first + 5)
            r = client.post("/api/sessions/%s/export" % sid,
                            json={"k0": first, "k1": k1, "fps": 10,
                                  "width": 640, "height": 360})
            assert r.status_code == 202, r.text
            job = r.json()
            assert job["total"] == k1 - first + 1
            jid = job["job_id"]
            for _ in range(600):
                time.sleep(0.1)
                st = client.get("/api/exports/%s" % jid).json()
                if st["state"] in ("done", "error", "cancelled"):
                    break
            assert st["state"] == "done", st
            assert st["done"] == st["total"] and st["bytes"] > 0

            f = client.get("/api/exports/%s/file" % jid)
            assert f.status_code == 200
            assert f.headers["content-type"] == "video/mp4"
            assert len(f.content) == st["bytes"]

            path = videoexport.get(jid).path
            info_ = videoexport.probe_json(path)
            if info_ is not None:                 # ffprobe available
                v = info_["streams"][0]
                assert v["codec_name"] == "h264"
                assert int(v["nb_frames"]) == st["total"]

            assert client.delete("/api/exports/%s" % jid).json()["ok"]
            assert client.get("/api/exports/%s" % jid).status_code == 404
            assert not (tmp_path / "").exists() or not list(tmp_path.glob("*.mp4"))
        client.delete("/api/sessions/%s" % sid)


@needs_ffmpeg
def test_export_uses_the_parallel_pool(tmp_path, monkeypatch):
    """The multi-process render path (not just the serial fast path) must
    produce a valid h264 mp4 with the right frame count. Forced on a small
    job by lowering the pool threshold and worker count."""
    import config as appconfig
    monkeypatch.setattr(appconfig, "EXPORT_DIR", str(tmp_path))
    monkeypatch.setattr(videoexport, "POOL_MIN_FRAMES", 4)
    monkeypatch.setattr(videoexport, "export_workers", lambda: 2)
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            first, last = _solve_a_few(client, ws, sid, n=8)
            k1 = min(last, first + 7)
            assert k1 - first + 1 >= 2*2      # enough frames to hit the pool
            r = client.post("/api/sessions/%s/export" % sid,
                            json={"k0": first, "k1": k1, "fps": 10,
                                  "width": 320, "height": 240})
            assert r.status_code == 202, r.text
            jid = r.json()["job_id"]
            total = r.json()["total"]
            for _ in range(600):
                time.sleep(0.1)
                st = client.get("/api/exports/%s" % jid).json()
                if st["state"] in ("done", "error", "cancelled"):
                    break
            assert st["state"] == "done", st
            assert st["done"] == st["total"] == total and st["bytes"] > 0
            info_ = videoexport.probe_json(videoexport.get(jid).path)
            if info_ is not None:
                v = info_["streams"][0]
                assert v["codec_name"] == "h264"
                assert int(v["nb_frames"]) == total
            client.delete("/api/exports/%s" % jid)
        client.delete("/api/sessions/%s" % sid)


@needs_ffmpeg
def test_export_rejected_while_running_and_on_bad_range(tmp_path, monkeypatch):
    import json as _json
    import config as appconfig
    monkeypatch.setattr(appconfig, "EXPORT_DIR", str(tmp_path))
    with TestClient(app) as client:
        info = _mk(client, variants=["qn"])
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(_json.dumps({"type": "play"}))
            time.sleep(0.3)
            r = client.post("/api/sessions/%s/export" % sid, json={})
            assert r.status_code == 409 and "pause" in r.text
            ws.send_text(_json.dumps({"type": "pause"}))
            time.sleep(0.3)
            # k1 < k0 is a schema error; a range past the frontier clamps
            assert client.post("/api/sessions/%s/export" % sid,
                               json={"k0": 5, "k1": 2}).status_code == 422
            assert client.post("/api/sessions/%s/export" % sid,
                               json={"variants": ["cr"]}).status_code == 422
            r = client.post("/api/sessions/%s/export" % sid,
                            json={"k0": 0, "k1": 10**6, "fps": 10,
                                  "width": 320, "height": 240})
            assert r.status_code == 202, r.text
            jid = r.json()["job_id"]
            # one export at a time per session
            assert client.post("/api/sessions/%s/export" % sid,
                               json={}).status_code in (409, 202)
            for _ in range(600):
                time.sleep(0.1)
                if client.get("/api/exports/%s" % jid).json()["state"] != "running":
                    break
            client.delete("/api/exports/%s" % jid)
        client.delete("/api/sessions/%s" % sid)
        # closing the session drops its jobs and their files
        assert not list(tmp_path.glob("*.mp4"))


def test_download_name_is_descriptive():
    """The browser must save something readable — variants, records, size,
    time — while the on-disk path keeps its collision-proof ids (two
    exports of the same range in one minute must not overwrite each other,
    least of all while one is being downloaded)."""
    class _Sess:      # duck-typed: ExportJob only needs .id and .cfg here
        id = "0123456789ab"
        cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                     variants=["qn", "cn"])
    spec = protocol.ExportSpec(fps=25, width=3840, height=2160)
    job = videoexport.ExportJob(_Sess(), spec, 0, 99, "/tmp")
    assert job.download_name.startswith("wignerf-QN-CN-100rec-3840x2160-")
    assert job.download_name.endswith(".mp4")
    assert "0123456789ab" not in job.download_name
    assert job.id in job.path            # on disk: still unique per job
    assert job.status()["filename"] == job.download_name
    # a strided export says so, and counts the frames it actually renders
    spec = protocol.ExportSpec(stride=4, width=1920, height=1080)
    job = videoexport.ExportJob(_Sess(), spec, 0, 99, "/tmp")
    assert "-25rec-every4-1920x1080-" in job.download_name


def test_layout_is_resolution_independent():
    """Fonts are in POINTS: a fixed dpi would render every label at half
    its relative size at 4K, so the figure keeps a constant inch size and
    the dpi carries the resolution."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    sizes = {}
    for w, h in ((1920, 1080), (3840, 2160)):
        fig = FrameFigure(["qn"], stats, meta, width=w, height=h)
        try:
            sizes[w] = (fig.fig.get_size_inches().tolist(),
                        len(bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2))))
        finally:
            fig.close()
    assert sizes[1920][0] == pytest.approx(sizes[3840][0])   # same layout
    assert sizes[1920][1] == 1920*1080*4
    assert sizes[3840][1] == 3840*2160*4


def test_metadata_block_never_runs_off_the_figure():
    """The block is anchored with va="top" and grows DOWNWARD, so it used to
    walk off the bottom edge in silence. Measured at 16:9 (the figure is always
    19.2x10.8 in — dpi carries the resolution): 11 lines fit, and a 4-variant
    cat run with 4 live parameter changes is already 10 in float64 and 11 with
    the float32 PREVIEW line. So a realistic export sat exactly at the edge and
    one more live change clipped."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    stats = _stats()
    meta = meta_columns(cfg, protocol.RecordGeom.from_1d(32, 32, -6., 6., -7., 7.),
                        stats, ["qn"], 0, 2, 3, 30)
    fig = FrameFigure(["qn"], stats, meta)
    try:
        F = FrameFigure
        # a block that already fits is untouched — no cosmetic change to any
        # export that was fine before
        assert fig._meta_fontsize(11) == F.META_FONTSIZE
        assert fig._meta_fit(["x"]*11, F.META_FONTSIZE) == ["x"]*11
        # one line past the budget shrinks instead of clipping
        assert F.META_MIN_FONTSIZE <= fig._meta_fontsize(12) < F.META_FONTSIZE
        # and the invariant that matters, across the whole range: the last
        # line's baseline stays inside the figure
        for n in (1, 11, 12, 20, 60, 200):
            fs = fig._meta_fontsize(n)
            kept = fig._meta_fit(["line %d" % i for i in range(n)], fs)
            assert fs >= F.META_MIN_FONTSIZE
            advance = fs*F.META_LINESPACING/72.0/fig.fig.get_figheight()
            assert len(kept)*advance <= F.META_TOP + 1e-9, (n, fs, len(kept))
            # nothing is dropped silently
            if len(kept) < n:
                assert "more lines" in kept[-1] and "comment tag" in kept[-1]
    finally:
        fig.close()


def test_export_unknown_session_and_job():
    with TestClient(app) as client:
        assert client.post("/api/sessions/nope/export", json={}).status_code == 404
        assert client.get("/api/exports/nope").status_code == 404
        assert client.get("/api/exports/nope/file").status_code == 404
        assert client.delete("/api/exports/nope").status_code == 404


# -- the expression initial conditions -----------------------------------------

def _expr_ic(kind, expr):
    from types import SimpleNamespace
    return SimpleNamespace(type=kind, components=[], expr=expr)


@pytest.mark.parametrize("nd", [1, 2])
@pytest.mark.parametrize("kind,expr", [("wexpr", "3.5*exp(-x^2-p^2)"),
                                       ("psi", "hermite(3,x)*exp(-x^2/2)")])
def test_an_expression_ic_describes_itself_without_components(kind, expr, nd):
    """ic_expression used to derive ndim from comps[0].ndim, which is an
    IndexError for a component-less IC — i.e. on every export of one. Both
    render_mpl call sites pass ndim= for exactly this reason."""
    lines = describe.ic_expression(_expr_ic(kind, expr), 1.0, ndim=nd)
    assert lines and expr in lines[0]
    assert "normalized" in lines[1]


@pytest.mark.parametrize("kind", ["wexpr", "psi"])
def test_an_expression_ic_is_never_mathified_even_when_it_wraps(kind):
    """The source is the user's own string — the text you paste back into the IC
    box — and must reach the figure verbatim.

    ASSERTED THROUGH _emit, WHICH IS WHERE IT HAPPENS. The predecessor of this
    test compared ic_expression's plain and math outputs, which are equal by
    construction on that branch, so it passed while the bug was live: _emit only
    skips substitution when a logical line fits in ONE fragment, and the moment
    it wraps it runs ax.sub_math_text on every fragment. A literal `px` then came
    out as `$p_x$` — five characters drawing as two glyphs, which also makes the
    character-count wrapping a guess. ndim=1 cannot see any of this, because `x`
    and `p` are already single letters, so the case has to be 2D AND long enough
    to wrap.
    """
    src = ("exp(-(x-1)^2 - (y+2)^2 - (px-0.5)^2 - (py+0.5)^2) + "
           "0.3*exp(-(x+1)^2 - (y-2)^2 - (px+1.5)^2 - (py-1.5)^2)"
           "*cos(2*px*x + 2*py*y)")
    assert len(src) <= describe.IC_SRC_MAX      # not truncation doing the work
    ic = _expr_ic(kind, src)
    plain = describe.ic_expression(ic, 1.0, ndim=2)
    math = describe.ic_expression(ic, 1.0, ndim=2, math=True)
    # None, not a copy of the plain line: it is the signal _emit already has for
    # "there is no typeset twin", and the only one its wrapping branch honours.
    assert math == [None]*len(plain)
    out = render_mpl._emit(plain, math, 150, 2)
    assert len(out) > 1, "the case must actually wrap or it proves nothing"
    joined = " ".join(out)
    assert "$" not in joined
    assert "p_x" not in joined and "p_y" not in joined
    # every fragment of the source survives, in order
    assert src.replace(" ", "") in joined.replace(" ", "").replace("\n", "")


@pytest.mark.parametrize("nd", [1, 2])
@pytest.mark.parametrize("kind", ["wexpr", "psi"])
def test_a_long_expression_is_truncated_with_a_pointer_not_clipped(kind, nd):
    long = "exp(-x^2/2)*(" + "+".join("%d*x^%d" % (i, i) for i in range(60)) + ")"
    assert len(long) > describe.IC_SRC_MAX
    ic = _expr_ic(kind, long)
    lines = describe.ic_expression(ic, 1.0, ndim=nd)
    assert "…" in lines[0]
    assert "comment tag" in lines[-1]
    # THE BUDGET IS IN PHYSICAL LINES, which is what the block actually spends —
    # the predecessor asserted len(lines) == 3 on the PRE-WRAP list, so it could
    # not see the number its own comment named (the real figure was 5). A
    # truncated expression costs 4 because the pointer is itself a line; see
    # describe.IC_SRC_MAX for the measurement this cap comes from.
    phys = render_mpl._emit(lines, describe.ic_expression(ic, 1.0, ndim=nd,
                                                          math=True), 150, nd)
    assert len(phys) == 4
    # ...and an expression that just fits costs one less, with no pointer
    short = _expr_ic(kind, "a"*describe.IC_SRC_MAX)
    assert len(render_mpl._emit(
        describe.ic_expression(short, 1.0, ndim=nd),
        describe.ic_expression(short, 1.0, ndim=nd, math=True), 150, nd)) == 3
