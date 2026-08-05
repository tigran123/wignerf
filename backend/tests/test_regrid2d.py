"""Auto-expand regrid at ndim=2 (milestone M3, landed 2026-08-01).

A SIBLING of test_regrid.py rather than a parametrization of it, because that
file is written in the 1D-only spellings — `gs.dx`, `g.xv`, `geom.Nx` — which
RAISE at ndim>1 by design. Sharing the fixtures would have meant softening those
raisers, and they are the thing that made the 2D migration tractable.

What is genuinely new here, and what is only new at four axes:

- `embed_window`, `GridState` and `plan_axis` were already generic, so the
  exactness tests are the 1D ones asked at four axes. They still earn their
  place: a wrong axis pairing in the window arithmetic is silent, and it is the
  classic multi-D error.
- The MEMORY GUARD is entirely new (core/fit.regrid_shortfall): at ndim=1 a
  doubling is a rounding error next to VRAM and nothing re-checked the fit after
  session creation. In 4D one axis doubling doubles a multi-GiB working set.
- So is the greedy degradation that gives up doublings it cannot afford.

The end-to-end test runs on the CPU deliberately (see its docstring): a 2D
regrid that only ever ran on a GPU would be a test that silently skips on the
VPS, and this is the milestone where "it fits" is the whole point.
"""

import json
from dataclasses import replace
from math import prod

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config as _cfgmod
from core import fit, protocol
from core.grid import GridState, embed_window
from core.xp import ArrayBackend
from main import app

# Spatial axes 32 cells, momentum axes 16. The 16-cell axes are SILENT by
# construction — boundary._band_mass returns 0.0 below 8 bands, and
# edge_band(16) is 4 — so only x and y can ever trip. That keeps the e2e run
# cheap (262144 cells, ~10 ms/step on a CPU here) and its plan predictable.
AX = [{"lo": -8.0, "hi": 8.0, "N": 32}, {"lo": -8.0, "hi": 8.0, "N": 32},
      {"lo": -7.0, "hi": 7.0, "N": 16}, {"lo": -7.0, "hi": 7.0, "N": 16}]


def _state(N=(32, 32, 16, 16), lo=(-8., -8., -7., -7.),
           hi=(8., 8., 7., 7.)):
    return GridState(anchor=tuple(lo),
                     d=tuple((hi[a] - lo[a])/N[a] for a in range(4)),
                     offset=(0,)*4, N=tuple(N))


def _blob(gs, b):
    """A contained 4D Gaussian in natural order, on this window's lattice."""
    g = gs.make_grid(b)
    v = [b.asnumpy(g.v[a]) for a in range(4)]
    c, s = (1.0, -0.5, 0.5, 0.0), (0.8, 0.9, 0.7, 0.75)
    W = np.ones(gs.N)
    for a in range(4):
        sh = [1]*4
        sh[a] = gs.N[a]
        W = W*np.exp(-((v[a] - c[a])**2/(2.*s[a]**2))).reshape(sh)
    return W/(W.sum()*prod(gs.d))


def test_embed_window_is_exact_at_four_axes():
    """The 1D exactness contract, asked of all four axes at once.

    A move, a single-axis double and a two-axis double. What this is really
    guarding is the AXIS PAIRING inside the window arithmetic: `embed_window`
    slices per axis by global cell index, and a transposed pair would still
    produce a plausible array of the right shape and the right total mass.
    Hence the per-axis assertions on the lattice VECTORS as well as on W.
    """
    b = ArrayBackend(device="cpu")
    gs = _state()
    g = gs.make_grid(b)
    W = _blob(gs, b)
    dmu = prod(gs.d)

    # (a) a whole-cell move, different and non-zero on every axis, both signs
    shift = (+5, -3, +2, -1)
    mv = replace(gs, offset=tuple(gs.offset[a] + shift[a] for a in range(4)))
    gm = mv.make_grid(b)
    assert gm.d == g.d                                   # frozen, bitwise
    for a in range(4):
        s = shift[a]
        src = slice(s, None) if s > 0 else slice(None, s)
        dst = slice(None, -s) if s > 0 else slice(-s, None)
        assert np.array_equal(b.asnumpy(gm.v[a])[dst],
                              b.asnumpy(g.v[a])[src]), "axis %d lattice" % a
    Wm = embed_window(W, gs, mv, np)
    assert np.array_equal(Wm[:-5, 3:, :-2, 1:], W[5:, :-3, 2:, :-1])
    assert not Wm[-5:].any() and not Wm[:, :3].any()     # entering cells zero
    assert not Wm[:, :, -2:].any() and not Wm[:, :, :, :1].any()
    assert abs(Wm.sum() - W.sum())*dmu < 1e-10
    assert abs((Wm*Wm).sum() - (W*W).sum())*dmu < 1e-12

    # (b) ONE axis doubled — the plan a single tripped axis produces, and the
    #     one that doubles the working set
    d1 = replace(gs, offset=(gs.offset[0] - 16,) + gs.offset[1:],
                 N=(64,) + gs.N[1:])
    gd = d1.make_grid(b)
    assert gd.d == g.d
    assert np.array_equal(b.asnumpy(gd.v[0])[16:48], b.asnumpy(g.v[0]))
    W1 = embed_window(W, gs, d1, np)
    assert np.array_equal(W1[16:48], W)
    W1[16:48] = 0.0
    assert not W1.any()
    W1 = embed_window(W, gs, d1, np)
    assert W1.sum()*prod(d1.d) == pytest.approx(W.sum()*dmu, rel=1e-13)
    assert (W1*W1).sum()*prod(d1.d) == pytest.approx((W*W).sum()*dmu, rel=1e-13)

    # (c) TWO axes doubled at once (a radially spreading state trips both), and
    #     the momentum axes left alone — a 4x working set, not 16x
    d2 = replace(gs, offset=(gs.offset[0] - 16, gs.offset[1] - 16)
                 + gs.offset[2:], N=(64, 64) + gs.N[2:])
    W2 = embed_window(W, gs, d2, np)
    assert np.array_equal(W2[16:48, 16:48], W)
    assert W2.shape == (64, 64, 16, 16)
    W2[16:48, 16:48] = 0.0
    assert not W2.any()


def test_the_planner_scans_every_axis_independently():
    """One tripped axis must move only that axis. `GridState.moved` is per-axis
    and `_schedule_regrid` skips untripped ones, so this pins the composition:
    four independent plans must not leak into each other's offsets."""
    gs = _state()
    new = gs
    # x doubles, py moves, y and px untouched
    new = new.moved(0, gs.offset[0] - 16, 64)
    new = new.moved(3, gs.offset[3] + 4, 16)
    assert new.N == (64, 32, 16, 16)
    assert new.offset == (-16, 0, 0, 4)
    assert new.d == gs.d                       # frozen on every axis
    # and the untouched axes keep their extents to the bit
    assert new.lo[1] == gs.lo[1] and new.hi[1] == gs.hi[1]
    assert new.lo[2] == gs.lo[2] and new.hi[2] == gs.hi[2]


# --------------------------------------------------------------------------
# the memory guard (M3's actual new machinery)
# --------------------------------------------------------------------------

def _counts(n, dev="cuda:9"):
    return {dev: n}


def test_the_regrid_guard_asks_the_driver_not_a_cell_count(monkeypatch):
    """WIGNERF_MAX_CELLS_2D is a rail; the guard that means something asks how
    much the assigned devices actually have free — the same argument
    routers/sessions._fit_error rests on, now asked again at REGRID time.

    Stubs the free-memory probe so it runs anywhere, GPU or not.
    """
    free = {}
    monkeypatch.setattr(fit.xp, "device_free_bytes", lambda d: free.get(d))
    old_cells, new_cells = 32**4, 2*32**4
    per_old = old_cells*_cfgmod.bytes_per_cell(2, "float64")
    per_new = 2*per_old

    # room to spare: free memory alone covers the doubling
    free["cuda:9"] = int(per_new*2)
    assert fit.regrid_shortfall(old_cells, new_cells, 2, "float64",
                                _counts(1)) == []

    # nothing free at all: refused, and the shortfall names the device
    free["cuda:9"] = 0
    short = fit.regrid_shortfall(old_cells, new_cells, 2, "float64", _counts(1))
    assert short and short[0][0] == "cuda:9"

    # THE TERM THAT MATTERS: what our own workers already hold is available,
    # because the regrid releases before it allocates. With free memory just
    # under the new footprint this fits only if per_old is counted in.
    free["cuda:9"] = int(per_new*0.95)
    assert fit.regrid_shortfall(old_cells, new_cells, 2, "float64",
                                _counts(1)) == []

    # two workers on one device need twice as much
    assert fit.regrid_shortfall(old_cells, new_cells, 2, "float64",
                                _counts(2)) != []

    # unknown free memory must NOT refuse: there the rail is the only guard and
    # guessing would be worse (the standing rule, shared with _fit_error)
    free.clear()
    assert fit.regrid_shortfall(old_cells, new_cells, 2, "float64",
                                _counts(1)) == []


def test_the_guard_is_precision_aware_and_applies_at_1d(monkeypatch):
    """float32 cuts the footprint to 55% (96 B/cell against 176), so it must
    widen what a doubling can reach — that is most of what M1 bought in 2D.

    And it applies at ndim=1, where it used to decline to have an opinion. That
    exemption rested on WIGNERF_MAX_GRID bounding a 2D array to 4096² — the
    default, not the setting: at the 8192 this repo's wignerf.env sets, a 1D
    doubling to 8192² is the same 4× jump a 2D one is, and a worker that cannot
    afford it dies on a cupy OOM rather than the plan being denied.
    """
    # Free memory chosen INSIDE the window where the two answers differ, in
    # units of one float64 worker (32^4 x 176 B), for 2 workers on one device:
    #   float64 is refused below F = 2.889   (4.40 needed against 0.9F + 1.80)
    #   float32 fits from     F = 1.576      (2.40 needed against 0.9F + 0.98)
    # so anything in (1.58, 2.89) separates them; 2.2 sits clear of both ends.
    # The float64 threshold is precision-INDEPENDENT in its own units; what M7
    # moved is the float32 one, since 96/176 is a shade above 112/208.
    free = {"cuda:9": int(2.2*32**4*176)}
    monkeypatch.setattr(fit.xp, "device_free_bytes", lambda d: free.get(d))
    args = (32**4, 2*32**4)
    assert fit.regrid_shortfall(*args, 2, "float64", _counts(2)) != []
    assert fit.regrid_shortfall(*args, 2, "float32", _counts(2)) == []
    # ndim=1 is checked too, and the same free memory answers it both ways:
    # 4 workers doubling 4096^2 -> 2*4096^2 need 4*2*16.8M*192*1.10 = 26.5 GiB
    # against 0.9*(F + 4*3.0 GiB) with F = 2.2*32^4*176 = 0.38 GiB, so refused;
    # one worker at 512^2 wants 0.2 GiB and fits.
    assert fit.regrid_shortfall(4096**2, 2*4096**2, 1, "float64",
                                _counts(4)) != []
    assert fit.regrid_shortfall(512**2, 2*512**2, 1, "float64",
                                _counts(1)) == []


def test_a_pure_move_is_never_refused_for_memory(monkeypatch):
    """A whole-cell window SHIFT allocates nothing, so no memory reading may
    refuse it — at any precision, on a device reporting nothing free at all.

    Not a corner case: the doubling inequality applied to per_new == per_old
    reduces to F >= (REGRID_PEAK/FIT_MARGIN - 1)*n*per_old = 0.222*n*per_old,
    i.e. it wanted 1.44 GiB free at 64^4 float64 with 2 workers just to slide a
    window. Refusing there also latches the planner's warning, which is what
    used to take ballistic relief away for the rest of a run — the property the
    max_grid cap has always kept ("pure moves still work at the cap").
    """
    monkeypatch.setattr(fit.xp, "device_free_bytes", lambda d: 0)
    for p in ("float64", "float32"):
        for n in (1, 4):
            assert fit.regrid_shortfall(32**4, 32**4, 2, p, _counts(n)) == [], p
    # ...and a SHRINK is not this check's business either
    assert fit.regrid_shortfall(2*32**4, 32**4, 2, "float64", _counts(2)) == []
    # while the doubling on the same starved device is still refused, or this
    # test would pass against a guard that had simply stopped working
    assert fit.regrid_shortfall(32**4, 2*32**4, 2, "float64", _counts(1)) != []


def test_one_driver_reading_serves_a_whole_attempt(monkeypatch):
    """The greedy walk-back asks about several candidate windows. It must
    compare them against ONE reading: a card re-read between candidates could
    accept a plan that neither reading on its own allows."""
    reads = []

    def probe(dev):
        reads.append(dev)
        return 0

    monkeypatch.setattr(fit.xp, "device_free_bytes", probe)
    budget = fit.budgets(_counts(2), context=False)
    assert len(reads) == 1
    reads.clear()
    for _ in range(5):
        assert fit.regrid_shortfall(32**4, 2*32**4, 2, "float64", _counts(2),
                                    budget) != []
    assert reads == [], "regrid_shortfall re-read the driver behind the budget"


def test_a_move_rebuilds_without_releasing_its_plans_or_the_pool(monkeypatch):
    """`Propagator.set_grid` has two branches and NOTHING called it directly
    until now — it was covered only end to end, where neither branch's ordering
    is observable.

    A DOUBLING releases before it allocates: M3 measured the switch at 1.269x the
    new footprint with the old arrays merely dropped at the end, and 1.045 with
    them dropped first, which is what lets the guard count them as available. A
    MOVE deliberately does neither, and that is measured too rather than reasoned
    — `scripts/bench.py --ndim 2 --regrid move` reports peak/steady 1.000 and +0
    MiB from the driver in float64, the only precision a move ever happens in
    (auto-expand is float64-only), with or without the drop. Clearing the cuFFT
    plan cache there would be actively worse: on an unchanged shape that plan is
    still valid and its work area is real VRAM, so it would be thrown away at the
    instant the pool is most loaded.

    What must hold on BOTH branches is that allocation order is the only
    difference: the meshes after a regrid are bitwise what a propagator built on
    that grid from scratch holds. That is the claim "ndim=1 is unaffected" rests
    on.
    """
    from core.propagator import Propagator

    b = ArrayBackend(device="cpu", precision="float64")
    U, gradU = (lambda x, y: (x**2 + y**2)/2.), (lambda x, y: x,
                                                 lambda x, y: y)
    gs = _state()
    prop = Propagator(gs.make_grid(b), quantum=True, U=U, gradU=gradU)
    plans = (prop._fft_sp, prop._ifft_sp, prop._fft_mo, prop._ifft_mo)
    released = []
    monkeypatch.setattr(prop, "_release_pool", lambda: released.append(1))

    moved = gs.moved(0, 4, gs.N[0])                  # whole-cell shift, same N
    prop.set_grid(moved.make_grid(b))
    assert released == [], "a move released the pool (and its live cuFFT plan)"
    assert (prop._fft_sp, prop._ifft_sp, prop._fft_mo, prop._ifft_mo) == plans

    fresh = Propagator(moved.make_grid(b), quantum=True, U=U, gradU=gradU)
    for name in ("dU_im", "dT_im", "U_mesh", "T_mesh"):
        np.testing.assert_array_equal(getattr(prop, name), getattr(fresh, name),
                                      err_msg=name)

    doubled = gs.moved(0, -(gs.N[0]//2), 2*gs.N[0])
    prop.set_grid(doubled.make_grid(b))
    assert released == [1], "a doubling must release before it allocates"
    assert prop._fft_sp is not plans[0], "the old shape's FFT plans survived"
    assert prop.dU_im.shape == doubled.N


def test_the_transient_peak_is_budgeted_for():
    """A doubling is not simply the new footprint: the switch holds the old
    state while it builds the new one. Measured on a 3090 with
    `scripts/bench.py --ndim 2 --regrid` — 1.045 (float64) / 1.083 (float32)
    after the release-before-allocate ordering, against 1.269 before it.

    REGRID_PEAK must stay above the measured value with margin, and must not
    silently become 1.0 (which would budget a doubling that then OOMs at
    k_star, taking the worker down mid-record)."""
    assert 1.08 <= fit.REGRID_PEAK <= 1.5


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def _ic(q0, k0=(0.0, 0.0)):
    return {"type": "mixture", "components": [
        {"q0": list(q0), "k0": list(k0),
         "sigma_q": [0.707, 0.707], "sigma_k": [0.707, 0.707]}]}


def _mk(client, **over):
    cfg = {"grid": {"ndim": 2, "axes": [dict(a) for a in AX]},
           "potential": "0", "ic": _ic((0.0, 0.0), (2.0, 0.0)),
           "variants": ["qn", "cn"], "record_dt": 0.05, "delay": 0.0,
           "device": "cpu", "auto_expand": True}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def _await_regrid(ws, limit=20000):
    for _ in range(limit):
        m = ws.receive()
        if m.get("text"):
            d = json.loads(m["text"])
            if d["type"] == "regrid":
                return d
            assert d["type"] != "error", d
    raise AssertionError("no regrid scheduled in a 2D session")


def test_e2e_2d_free_particle_expands():
    """A 4D free packet drifting at the +x edge must regrid at a lockstep
    k_star, keep every variant on ONE geometry per record, and serve
    mixed-geometry history on both sides of the switch.

    ON THE CPU AND AT 32x32x16x16 ON PURPOSE. A GPU-only 2D regrid test skips
    silently on a CPU host, and this is the milestone whose whole subject is
    whether a doubling fits — the one test that must not be the first thing to
    stop running. The packet starts at the origin with px0=2 and both drifts and
    SPREADS, so the support outgrows half the axis and the plan is a double
    rather than a move; measured at record 26, ~2 s of CPU.
    """
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            regrid = _await_regrid(ws)

            ng = regrid["grid"]
            assert ng["ndim"] == 2
            assert ng["labels"] == ["x", "y", "px", "py"]
            # A DOUBLING, which is the case M3 is actually about — a move costs
            # no extra cells and needs no memory guard. Measured: it lands at
            # record ~26, x alone, in ~2 s of CPU.
            assert regrid["kind"] == {"x": "double"}, regrid["kind"]
            assert ng["N"] == [64, 32, 16, 16], ng["N"]
            # WIDTH and LATTICE ALIGNMENT, not the literal extents. Measured, the
            # switch lands on [-16, 16] at record ~26 — but the planner scans
            # whichever record is `latest_complete()` when the edge trips, and
            # under load (the full suite, a busy host) that can be a record or
            # two later, whose wider support centres the new window somewhere
            # else. Asserting -16.0 made a correct plan fail on machine timing;
            # what must hold is that the doubling is exact on the frozen lattice.
            dx = (AX[0]["hi"] - AX[0]["lo"])/AX[0]["N"]
            assert ng["hi"][0] - ng["lo"][0] == pytest.approx(2*32*dx, rel=1e-12)
            off = (ng["lo"][0] - AX[0]["lo"])/dx
            assert off == pytest.approx(round(off), abs=1e-9), ng["lo"]
            # the frozen lattice, per axis, across four axes
            for a in range(4):
                assert (ng["hi"][a] - ng["lo"][a])/ng["N"][a] == \
                    pytest.approx((AX[a]["hi"] - AX[a]["lo"])/AX[a]["N"],
                                  rel=1e-12), "axis %d spacing" % a
            # every OTHER axis is untouched, extents to the bit: a doubling on
            # one axis must not perturb the other three
            for a in (1, 2, 3):
                assert ng["lo"][a] == AX[a]["lo"] and ng["hi"][a] == AX[a]["hi"]
            k_star = regrid["at_record"]

            post = None
            for _ in range(8000):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    if f.record >= k_star:
                        post = f
                        break
                elif m.get("text"):
                    assert json.loads(m["text"])["type"] != "error", m["text"]
            assert post is not None, "no post-regrid frame arrived"
            assert post.geom.ndim == 2
            assert list(post.geom.N) == list(ng["N"])
            assert len(post.variants) == 2      # lockstep held across 4 axes
            assert len(post.variants[0].planes) == 6
            assert len(post.variants[0].marg) == 4
            for v in post.variants:
                assert v.purity == v.purity and abs(v.E) < 1e4

            # replay a record from BEFORE the switch: it must decode at the old
            # geometry, not the session's current one
            ws.send_text(json.dumps({"type": "seek", "record": 0}))
            ws.send_text(json.dumps({"type": "play"}))
            pre = None
            for _ in range(8000):
                m = ws.receive()
                if m.get("bytes"):
                    f = protocol.unpack_frame(m["bytes"])
                    if f.record < k_star:
                        pre = f
                        break
            assert pre is not None, "no pre-regrid record replayed"
            assert list(pre.geom.N) == [a["N"] for a in AX]
        client.delete("/api/sessions/%s" % sid)


def test_a_2d_window_slides_without_growing():
    """The other half of the planner, and the cheap common case: a packet
    already AT the edge but narrow enough to fit gets a whole-cell window SHIFT,
    not a doubling.

    Worth its own test because it is the one outcome that costs no memory at
    all, so it must stay reachable no matter how tight the guard becomes — the
    same property the max_grid cap has always had ("pure moves still work at the
    cap"). Measured: [-8, 8] slides to [-4, 12], N unchanged, dx frozen.
    """
    with TestClient(app) as client:
        info = _mk(client, ic=_ic((4.0, 0.0), (4.0, 0.0)))
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            regrid = _await_regrid(ws)
        assert regrid["kind"] == {"x": "move"}, regrid["kind"]
        ng = regrid["grid"]
        assert ng["N"] == [a["N"] for a in AX]          # no extra cells
        assert ng["lo"][0] > AX[0]["lo"] and ng["hi"][0] > AX[0]["hi"]
        assert (ng["hi"][0] - ng["lo"][0])/ng["N"][0] == \
            pytest.approx((AX[0]["hi"] - AX[0]["lo"])/AX[0]["N"], rel=1e-12)
        client.delete("/api/sessions/%s" % sid)


def test_a_doubling_the_device_cannot_hold_is_refused_and_the_run_continues(
        monkeypatch):
    """The guard, end to end. A 2D session whose device has no room must not
    expand — and must not die either: it posts `no_room` once, naming the
    device and the numbers, and keeps computing on the old grid.

    'no_room' is a DIFFERENT action from 'capped' on purpose: capped means the
    domain reached the per-axis cell ceiling, this means the hardware cannot
    hold the result, and the two need opposite advice.
    """
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        # Starve the device AFTER creation — which is also the realistic order:
        # the session started when there was room, and something else took it.
        # Patching before creation would just re-test the create-time refusal.
        monkeypatch.setattr(fit.xp, "device_free_bytes", lambda d: 0)
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            seen, saw_regrid = None, False
            for _ in range(20000):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "regrid":
                    saw_regrid = True
                if d["type"] == "boundary" and d["action"] == "no_room":
                    seen = d
                    break
        assert seen is not None, "no 'no_room' warning was posted"
        assert not saw_regrid, "expanded anyway on a device with no room"
        assert "cpu" in seen["message"] and "GiB" in seen["message"], seen
        # WHICH ceiling ran out. The frontend's remedy turns on it, and for a
        # device it is the hardware one (free the card, fewer variants, a roomier
        # device) — see the cell-rail test for the opposite advice.
        assert seen["limit"] == "device", seen
        # the payload the frontend words itself from: WHICH doubling was given
        # up, and that nothing was committed in its place (here the fallback
        # comes back "capped", which is the usual outcome — see the M3 notes)
        assert seen["denied"] == ["x"], seen
        assert seen["applied"] == {}, seen
        assert seen["axes"] == ["x"], seen        # non-empty, or the client drops it
        # the session is still alive and still computing
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["grid"]["N"] == [a["N"] for a in AX]   # never expanded
        client.delete("/api/sessions/%s" % sid)


def test_a_window_slides_even_when_the_device_is_full(monkeypatch):
    """The move of test_a_2d_window_slides_without_growing, on a device
    reporting nothing free. It must still be applied, and must raise no warning
    at all: a shift allocates nothing, so there is nothing for a memory guard to
    have an opinion about.

    This is the regression test for the guard budgeting a move like a doubling.
    With that bug the shift was refused, `no_room` was posted, and the latch it
    set then stopped the planner scheduling anything else for the rest of the
    run — so a drifting packet, whose relief is exactly this shift repeated,
    silently lost auto-expand altogether.
    """
    with TestClient(app) as client:
        info = _mk(client, ic=_ic((4.0, 0.0), (4.0, 0.0)))
        sid = info["session_id"]
        monkeypatch.setattr(fit.xp, "device_free_bytes", lambda d: 0)
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            regrid, warned = None, None
            for _ in range(20000):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "boundary" and d["action"] == "no_room":
                    warned = d
                    break
                if d["type"] == "regrid":
                    regrid = d
                    break
        assert warned is None, warned
        assert regrid is not None, "the window never slid on a full device"
        assert regrid["kind"] == {"x": "move"}, regrid["kind"]
        assert regrid["grid"]["N"] == [a["N"] for a in AX]     # no extra cells
        client.delete("/api/sessions/%s" % sid)


def test_a_denial_does_not_disable_auto_expand_for_the_rest_of_the_run(
        monkeypatch):
    """`no_room` is a MESSAGE latch, not a kill switch. Free the card after the
    refusal and the very next attempt must expand.

    It gated scheduling for one revision, on the argument that it kept a driver
    query off the per-record path. The query is a cudaMemGetInfo and is not even
    taken unless the plan grows, while the gate cost every later regrid — and
    the tripped axes never clear on their own, because a state at the edge is
    what tripped them. `_invalid_posted` still gates, and that one is earned: a
    ~ms sympy probe under the edge lock, every record.
    """
    with TestClient(app) as client:
        info = _mk(client)          # created while there was room, as ever
        sid = info["session_id"]
        free = {"v": 0}
        monkeypatch.setattr(fit.xp, "device_free_bytes", lambda d: free["v"])
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            denied_at = None
            for _ in range(20000):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "boundary" and d["action"] == "no_room":
                    denied_at = d["record"]
                    free["v"] = 1 << 40      # ...and now the card frees up
                if d["type"] == "regrid":
                    assert denied_at is not None, "expanded before the denial"
                    assert d["kind"] == {"x": "double"}, d["kind"]
                    break
            else:
                raise AssertionError("auto-expand never recovered from a denial")
        client.delete("/api/sessions/%s" % sid)


def test_a_cell_rail_denial_does_not_blame_the_card(monkeypatch):
    """The OTHER guard in `_afford_regrid`, and the reason a denial carries a
    `limit`. WIGNERF_MAX_CELLS_2D is a HOST SETTING: freeing VRAM cannot help,
    and neither can float32, because a cell count does not depend on precision.

    Reported as a bare sentence under one action, a rail denial arrived with the
    device remedy attached — 'free the card ... or restart in float32' — over a
    card with 100+ GiB free. The message names the rail; `limit` is what lets the
    frontend pick the matching advice without pattern-matching prose.
    """
    # 262,144 cells at the fixture grid; x doubling would be 524,288. The rail
    # sits between, so the SESSION starts and only the expansion is refused.
    monkeypatch.setattr(_cfgmod, "MAX_CELLS_2D", 300_000)
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            seen, saw_regrid = None, False
            for _ in range(20000):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "regrid":
                    saw_regrid = True
                if d["type"] == "boundary" and d["action"] == "no_room":
                    seen = d
                    break
        assert seen is not None, "no denial was posted for the cell rail"
        assert not saw_regrid, "expanded past WIGNERF_MAX_CELLS_2D"
        assert seen["limit"] == "cells", seen
        assert "WIGNERF_MAX_CELLS_2D" in seen["message"], seen
        assert "cpu" not in seen["message"], seen   # nothing to do with a device
        assert seen["denied"] == ["x"], seen
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["grid"]["N"] == [a["N"] for a in AX]
        client.delete("/api/sessions/%s" % sid)


def test_a_failed_regrid_stops_the_run_instead_of_going_quiet(monkeypatch):
    """The premise the whole memory story rests on, pinned rather than assumed.

    `core/fit.py` accepts two exposures it cannot model — another process (or the
    IC preview) taking the card between the reading and the allocation, and a
    sibling session's committed-but-unlanded plan — on the grounds that LOSING
    that race is loud: the allocation raises inside `worker._apply_regrid`, which
    is fatal by design (a per-worker rollback would desync the lockstep
    geometry), so `run()` posts the error and pauses the session. That is what
    makes it preferable to any mechanism whose failure mode is a quiet refusal.

    If this ever became silent — a swallowed exception, a per-worker retry — the
    accepted limitation would stop being acceptable and the guard would have to
    grow teeth. So assert the error reaches the client and the run stops.
    """
    import core.worker as worker_mod

    def boom(*a, **kw):
        raise MemoryError("Out of memory allocating 1234567890 bytes")

    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        monkeypatch.setattr(worker_mod, "embed_window", boom)
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            err = None
            for _ in range(20000):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "error":
                    err = d
                    break
        assert err is not None, "a failed regrid was silent"
        assert "solver died" in err["message"], err
        assert "Out of memory" in err["message"], err
        st = client.get("/api/sessions/%s" % sid).json()
        assert not st["running"], "the session kept running with a dead worker"
        client.delete("/api/sessions/%s" % sid)


def test_a_2d_session_records_its_expansion_in_status():
    """`grid_payload` carries the generic per-axis tuples at ndim=2 (the flat
    x/p spelling is 1D-only), and status must follow the LIVE window so the
    Setup panel's 'adopt' offers the expanded one."""
    with TestClient(app) as client:
        info = _mk(client, auto_expand=False)
        sid = info["session_id"]
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["grid"]["ndim"] == 2
        assert st["grid"]["N"] == [a["N"] for a in AX]
        assert st["grid"]["labels"] == ["x", "y", "px", "py"]
        assert "Nx" not in st["grid"]          # 1D-only spelling stays 1D-only
        assert st["auto_expand"] is False
        client.delete("/api/sessions/%s" % sid)
