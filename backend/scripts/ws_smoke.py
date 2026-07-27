"""
Headless smoke test against a LIVE server (no browser needed):

    .venv/bin/uvicorn main:app --port 8010 &
    .venv/bin/python scripts/ws_smoke.py [http://127.0.0.1:8010] [--ndim 2]

Creates a harmonic session, plays it, and asserts streaming invariants:
monotone record indices, exact record-time spacing, unit norm after
dequantizing the SPATIAL plane, flat energy, lockstep bundles, exact seek.

--ndim 2 runs the same checks over a 4D phase space at 32^4 (two variants —
the relativistic ones are deferred, milestone M2), which is the one place the
2D record path is exercised against a real uvicorn rather than a TestClient.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import websockets

from core import axes as axes_mod
from core import protocol
from core.quantize import dequantize

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("base", nargs="?", default="http://127.0.0.1:8010",
                 help="server base URL")
_ap.add_argument("--ndim", type=int, default=1, choices=[1, 2],
                 help="spatial dimensions (2 = 4D phase space)")
_opts = _ap.parse_args()
BASE, NDIM = _opts.base, _opts.ndim

if NDIM == 1:
    CFG = {
        "grid": dict(x1=-6.0, x2=6.0, Nx=256, p1=-7.0, p2=7.0, Np=256),
        "potential": "x^2/2",
        "ic": {"type": "mixture",
               "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707,
                               "sigma_p": 0.707}]},
        "variants": ["qn", "qr", "cn", "cr"],
        "c": 10.0,      # low c so the relativistic variants visibly differ
        "record_dt": 0.05,
        "delay": 0.0,   # seconds between played-back frames (0 = max speed)
    }
else:
    CFG = {
        "grid": {"ndim": 2, "axes": [
            {"lo": -6.0, "hi": 6.0, "N": 32}, {"lo": -6.0, "hi": 6.0, "N": 32},
            {"lo": -7.0, "hi": 7.0, "N": 32}, {"lo": -7.0, "hi": 7.0, "N": 32}]},
        "potential": "(x^2 + y^2)/2",
        "ic": {"type": "mixture", "components": [
            {"q0": [2.0, 0.0], "k0": [0.0, 0.5],
             "sigma_q": [0.707, 0.707], "sigma_k": [0.707, 0.707]}]},
        "variants": ["qn", "cn"],
        "record_dt": 0.05,
        "delay": 0.0,
    }
NVAR = len(CFG["variants"])


def _axes():
    """The configured axes, in either grid spelling."""
    g = CFG["grid"]
    if "axes" in g:
        return g["axes"]
    return [{"lo": g["x1"], "hi": g["x2"], "N": g["Nx"]},
            {"lo": g["p1"], "hi": g["p2"], "N": g["Np"]}]


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        r = await http.post("/api/sessions", json=CFG)
        r.raise_for_status()
        info = r.json()
        sid = info["session_id"]
        print("session", sid, "variants", info["variants"])

        ws_url = BASE.replace("http", "ws") + info["ws_url"]
        # compression=None: never negotiate permessage-deflate (12x slower
        # for the multi-MiB frame bundles; the server disables it too)
        async with websockets.connect(ws_url, max_size=64*1024*1024,
                                      compression=None) as ws:
            await ws.send(json.dumps({"type": "play"}))
            frames, by_rec = [], {}
            while len(frames) < 20:
                m = await asyncio.wait_for(ws.recv(), timeout=30)
                if isinstance(m, (bytes, bytearray)):
                    f = protocol.unpack_frame(m)
                    frames.append(f)
                    by_rec[f.record] = f
                else:
                    d = json.loads(m)
                    if d["type"] == "error":
                        raise SystemExit("server error: %s" % d)

            recs = [f.record for f in frames]
            assert recs == sorted(set(recs)), "records not strictly increasing"
            for f in frames:
                assert abs(f.t - f.record*CFG["record_dt"]) < 1e-9, "t spacing broken"
                assert len(f.variants) == NVAR, "lockstep bundle incomplete"
                g = f.geom
                assert g.ndim == NDIM, "header ndim mismatch"
                assert g.lo == tuple(a["lo"] for a in _axes()), \
                    "header geometry mismatch"
                for v in f.variants:
                    assert len(v.planes) == len(axes_mod.planes(NDIM))
                    assert len(v.marg) == 2*NDIM
                    # the SPATIAL plane integrates to 1: at ndim=1 it IS W, at
                    # ndim=2 it is rho(x,y) and the momentum measure is already
                    # folded into the reduction
                    sp = v.planes[0]
                    W = dequantize(sp.wq, sp.wmin, sp.wmax)
                    d = [(a["hi"] - a["lo"])/a["N"] for a in _axes()]
                    norm = W.sum()*d[sp.a]*d[sp.b]
                    assert abs(norm - 1.0) < 1e-2, "norm drifted: %g" % norm
            E0 = {v.vid: v.E for v in frames[0].variants}
            for v in frames[-1].variants:
                assert abs(v.E - E0[v.vid]) < 5e-3*max(1.0, abs(E0[v.vid])), \
                    "energy drift on vid %d" % v.vid
            print("streamed %d lockstep bundles up to record %d, invariants OK"
                  % (len(frames), recs[-1]))

            # exact seek while paused
            await ws.send(json.dumps({"type": "pause"}))
            target = recs[len(recs)//2]
            await ws.send(json.dumps({"type": "seek", "record": target}))
            while True:
                m = await asyncio.wait_for(ws.recv(), timeout=10)
                if isinstance(m, (bytes, bytearray)):
                    f = protocol.unpack_frame(m)
                    if f.record == target:
                        ref = by_rec[target]
                        assert f.t == ref.t
                        assert all(
                            (pa.wq == pb.wq).all()
                            for a, b in zip(f.variants, ref.variants)
                            for pa, pb in zip(a.planes, b.planes)), \
                            "seek returned different bytes"
                        break
            print("seek(%d) returned the identical record" % target)

        r = await http.delete("/api/sessions/%s" % sid)
        r.raise_for_status()
        print("OK")


if __name__ == "__main__":
    asyncio.run(main())
