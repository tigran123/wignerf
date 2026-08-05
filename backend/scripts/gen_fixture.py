"""
Write golden binary frame bundles + JSON metadata for the frontend decoder's
vitest (frontend/src/lib/__fixtures__/). Run from backend/:

    .venv/bin/python scripts/gen_fixture.py

TWO fixtures, because the wire format is generic over ndim and a 1D-only golden
would let a 2D decode bug ship: `frame` is a 1D record (one plane = W, two
marginals) and `frame2d` a 2D one (six planes, four marginals). Both use
deliberately ANISOTROPIC axis counts so a transposed index cannot pass.

The 1D one also carries a CROPPED, DECIMATED plane (protocol v5) with a
different off/step on each axis, so a decoder that read the window fields in
the wrong order, or fell back to the header's N for the plane size, cannot
pass either. The 2D one stays whole — both paths have to be golden, and the
whole-plane path is the one every 2D record and every IC preview takes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import axes as ax
from core import planeview
from core import protocol

OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "__fixtures__"

VIDS = (protocol.variant_id(True, False), protocol.variant_id(False, True))


def build(ndim, N, rng, window=None):
    """(bytes, json-able dict) for one golden record.

    `window` is an optional {(a, b): (Window, Window)} applied to every variant,
    i.e. the display-downsampling path.
    """
    nax = 2*ndim
    assert len(N) == nax
    geom = protocol.RecordGeom(
        ndim, tuple(N),
        tuple(-6.0 - a for a in range(nax)),      # distinct per axis, so a
        tuple(6.0 + a for a in range(nax)))       # swapped lo/hi is visible
    variants, meta = [], []
    for i, vid in enumerate(VIDS):
        planes, pmeta = [], []
        for j, (a, b) in enumerate(ax.planes(ndim)):
            wq = rng.integers(0, 65536, size=(N[a], N[b]), dtype=np.uint16)
            wmin, wmax = -0.25 - j, 1.5 + j
            planes.append(protocol.PlaneFrame(a=a, b=b,
                                             mode=ax.MODE_PROJECTION,
                                             wq=wq, wmin=wmin, wmax=wmax))
            wa, wb = (window or {}).get((a, b),
                                        (planeview.full(N[a]),
                                         planeview.full(N[b])))
            sent = wq[wa.off:wa.off + wa.n*wa.step:wa.step,
                      wb.off:wb.off + wb.n*wb.step:wb.step]
            pmeta.append({"a": a, "b": b, "mode": ax.MODE_PROJECTION,
                          "wmin": wmin, "wmax": wmax,
                          "na": wa.n, "nb": wb.n,
                          "off": [wa.off, wb.off], "step": [wa.step, wb.step],
                          "wq": sent.flatten().tolist()})
        marg = tuple(rng.random(N[a]).astype(np.float32) for a in range(nax))
        mean = tuple(1.0 + a for a in range(nax))
        std = tuple(0.5 + 0.25*a for a in range(nax))
        vf = protocol.VariantFrame(vid=vid, dt=0.01, E=2.5, purity=0.875,
                                   lz=(0.0 if ndim == 1 else -1.25),
                                   mean=mean, std=std, planes=tuple(planes),
                                   marg=marg)
        variants.append(vf)
        meta.append({"vid": vid, "dt": 0.01, "E": 2.5, "purity": 0.875,
                     "lz": vf.lz, "mean": list(mean), "std": list(std),
                     "planes": pmeta,
                     "marg": [m.tolist() for m in marg]})

    views = None if window is None else {
        (v.vid, a, b): w for v in variants for (a, b), w in window.items()}
    buf = protocol.pack_frame(7, 0.35, geom, variants,
                              flags=protocol.FLAG_REPLAY, views=views)
    f = protocol.unpack_frame(buf)              # round-trip self-check
    assert (f.record, f.t) == (7, 0.35) and f.geom == geom, f.geom
    assert [p.a for p in f.variants[0].planes] == [a for a, _ in ax.planes(ndim)]
    doc = {"record": 7, "t": 0.35, "ndim": ndim, "N": list(geom.N),
           "lo": list(geom.lo), "hi": list(geom.hi),
           "flags": protocol.FLAG_REPLAY, "variants": meta}
    return buf, doc


def main():
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    # 1D: a window with a DIFFERENT size, offset AND step on each axis — 3
    # samples of 2 cells from row 4 (rows 4..10 of 16), against 2 samples of 1
    # cell from column 1 (columns 1..3 of 4). Every pair differs, so a decoder
    # that swapped na/nb, off_a/off_b or step_a/step_b produces the wrong shape
    # or the wrong numbers rather than a transpose that might still look right.
    crop = {(0, 1): (planeview.Window(n=3, off=4, step=2),
                     planeview.Window(n=2, off=1, step=1))}
    for name, ndim, N, win in (("frame", 1, (16, 4), crop),
                               ("frame2d", 2, (4, 6, 2, 8), None)):
        buf, doc = build(ndim, N, rng, win)
        (OUT / ("%s.bin" % name)).write_bytes(buf)
        (OUT / ("%s.json" % name)).write_text(json.dumps(doc, indent=1))
        print("wrote", OUT / ("%s.bin" % name), len(buf), "bytes")


if __name__ == "__main__":
    main()
