"""
mp4 export of 2D runs (milestone M4, landed 2026-07-28).

The physics core needed nothing: a 2D record already carries six quantized
planes, four marginals and every scalar the series need. What M4 is about is
that a frame cannot hold all of it — six planes x four variants is 24 panels,
and the diagnostics column has nine plots against 1D's five — so the job
SELECTS, and the figure lays out what it was given.

The gate this replaced named "a plane-set panel grid, four marginals, <Lz> and
the (2pi h)^2 purity scale". All four are checked below. So is the constant the
gate was hiding, which was none of them: RangeStats used to keep ONE colour
scale per variant, and at ndim=2 that would have rendered five of every six
panels blank without erroring — see test_each_plane_keeps_its_own_colour_scale,
which is the load-bearing test in this file.
"""

import json
import shutil
import time

import numpy
import pytest
from fastapi.testclient import TestClient

from core import axes as ax
from core import protocol, render_mpl, videoexport
from core.render_mpl import FrameFigure, RangeStats, meta_columns
from main import app

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg is not installed")

G2 = {"ndim": 2, "axes": [{"lo": -6.0, "hi": 6.0, "N": 16} for _ in range(4)]}
IC2 = {"type": "mixture",
       "components": [{"q0": [1.0, 0.0], "k0": [0.0, 0.0],
                       "sigma_q": [0.707, 0.707], "sigma_k": [0.707, 0.707]}]}


def cfg2(**over):
    d = {"grid": G2, "potential": "(x^2 + y^2)/2", "ic": IC2,
         "variants": ["qn", "cn"], "record_dt": 0.05, "delay": 0.0}
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# fixtures: a 2D record, built the way core/frame.py builds one
# ---------------------------------------------------------------------------

def _geom(N=16, ndim=2, lo=-6.0, hi=6.0):
    return protocol.RecordGeom(ndim, tuple([N]*(2*ndim)),
                               tuple([lo]*(2*ndim)), tuple([hi]*(2*ndim)))


def _plane(a, b, N, amp, seed):
    """One quantized plane whose magnitude is `amp` — the whole point being
    that the six reductions of one state are orders of magnitude apart."""
    rng = numpy.random.default_rng(seed)
    f = amp*(rng.random((N, N)) - 0.25)
    lo, hi = float(f.min()), float(f.max())
    q = numpy.clip(numpy.rint((f - lo)*(65535.0/(hi - lo))), 0,
                   65535).astype(numpy.uint16)
    return protocol.PlaneFrame(a=a, b=b, mode=ax.MODE_PROJECTION, wq=q,
                               wmin=lo, wmax=hi)


def _vframe2d(seed=1, N=16, key="qn", decade=True):
    """A 2D VariantFrame. `decade` spaces the six plane magnitudes a decade
    apart, which is what a real record does (a spatial density against a
    signed reduced Wigner function)."""
    rng = numpy.random.default_rng(seed)
    planes = tuple(_plane(a, b, N, 10.0**(-i) if decade else 1.0, seed + i)
                   for i, (a, b) in enumerate(ax.planes(2)))
    return protocol.VariantFrame(
        vid=protocol.variant_id(key[0] == "q", key[1] == "r"),
        dt=1e-3, E=1.0 + seed, purity=1.0, lz=0.25,
        mean=(0.1, 0.2, 0.0, 0.0), std=(0.7, 0.8, 0.9, 1.0),
        planes=planes,
        marg=tuple(rng.random(N) for _ in range(4)))


def _stats2d(keys=("qn",), n=3, N=16):
    st = RangeStats(ndim=2, t=[0.05*i for i in range(n)])
    st.marg_max = [1.0]*4
    st.lo, st.hi = (-6.0,)*4, (6.0,)*4
    for key in keys:
        st.E[key] = [1.0]*n
        st.purity[key] = [1.0]*n
        st.lz[key] = [0.25]*n
        for d in range(2):
            st.uncert[(key, d)] = [0.5]*n
        for pf in _vframe2d(1, N, key).planes:
            st.scale[(key, (pf.a, pf.b))] = max(pf.wmax, -pf.wmin)
    return st


def _stats1d(n=3):
    st = RangeStats(ndim=1, t=[0.05*i for i in range(n)])
    st.marg_max = [1.0, 1.0]
    st.lo, st.hi = (-6.0, -7.0), (6.0, 7.0)
    st.E["qn"] = [1.0]*n
    st.purity["qn"] = [1.0]*n
    st.lz["qn"] = [0.0]*n
    st.uncert[("qn", 0)] = [0.5]*n
    st.scale[("qn", (0, 1))] = 0.3
    return st


def _vframe1d(seed=1, Nx=16, Np=16):
    rng = numpy.random.default_rng(seed)
    wq = (rng.random((Nx, Np))*65535).astype(numpy.uint16)
    plane = protocol.PlaneFrame(a=0, b=1, mode=ax.MODE_PROJECTION, wq=wq,
                               wmin=-0.1, wmax=0.3)
    return protocol.VariantFrame(
        vid=protocol.variant_id(True, False), dt=1e-3, E=1.0, purity=1.0,
        lz=0.0, mean=(0.1, 0.0), std=(0.7, 0.7), planes=(plane,),
        marg=(rng.random(Nx), rng.random(Np)))


def _fig(keys=("qn",), planes=None, diagnostics=None, n=3, N=16,
         width=640, height=360, **kw):
    keys = list(keys)
    st = _stats2d(tuple(keys), n, N)
    meta = meta_columns(protocol.SessionCreate(**cfg2(variants=keys)),
                        _geom(N), st, keys, 0, n - 1, n, 30,
                        planes=planes, diagnostics=diagnostics)
    return FrameFigure(keys, st, meta, width=width, height=height,
                       planes=planes, diagnostics=diagnostics, **kw), st


# ---------------------------------------------------------------------------
# the load-bearing one
# ---------------------------------------------------------------------------

def test_each_plane_keeps_its_own_colour_scale():
    """THE constant the M4 gate was hiding. RangeStats kept one symmetric W
    scale per VARIANT, which is right at ndim=1 (one plane) and silently wrong
    at ndim=2: the six reductions of one state differ by orders of magnitude —
    a spatial density against a signed reduced Wigner function — so a shared
    scale renders the small ones as uniform white and nothing errors.

    Measured here on a record whose planes are spaced a decade apart: the
    per-plane scales span 1e5, and under one shared scale five of the six
    panels would carry less than 1/10 of one colour step out of 256."""
    fig, st = _fig(planes=list(ax.planes(2)))
    try:
        clims = [im.get_clim() for _axp, im, _c in fig.images]
    finally:
        fig.close()
    assert len(clims) == 6
    tops = [hi for _lo, hi in clims]
    assert min(tops) > 0.0
    assert max(tops)/min(tops) > 1e4, tops
    # every panel is centred on W = 0 (the bwr convention) and symmetric
    for lo, hi in clims:
        assert lo == pytest.approx(-hi)
    # and the alternative really would be blank: against ONE shared scale the
    # faintest plane spans well under a single 8-bit colour step
    shared = max(tops)
    assert min(tops)/shared < 1.0/(255*10)


def test_a_dense_grid_states_its_scales_in_the_titles_not_over_the_heatmap():
    """Past CBAR_MAX_CELLS panels a colorbar is a ~9 px strip with 6.5 pt
    ticks, so each panel's fixed scale moves into its own title instead.

    It has to be the TITLE and not a corner annotation on the heatmap, which
    is where it started: `update()` restores the static background and then
    re-draws the images on top, so anything static INSIDE the axes box is
    painted over — the same reason the panel grid lines are dynamic. It
    rendered as nothing at all, and no test that only checked the artist
    existed would have noticed. Making it dynamic instead would cost 24 text
    renders per frame for a caption that never changes."""
    fig, st = _fig(keys=("qn", "qr", "cn", "cr"), planes=list(ax.planes(2)),
                   width=1920, height=1080)
    try:
        assert len(fig.images) == 24
        # no colorbars, and nothing drawn inside a panel that the image would
        # cover on the next frame
        assert not [a for a in fig.fig.axes if a.get_label() == "<colorbar>"]
        for axp, _im, _c in fig.images:
            assert not axp.texts, axp.texts
        notes = {}
        for axp, _im, cell in fig.images:
            right = axp.get_title(loc="right")
            assert ax.plane_label(2, cell.plane, math=True) in right
            assert "±" in right, right
            notes[cell.plane] = right.split("±")[1]
        # per PLANE, so the six differ — the same fact the colorbars carry
        # when there is room for them
        assert len(set(notes.values())) == 6, notes
    finally:
        fig.close()

    # ...and while there IS room, the colorbar is what says it
    fig, _ = _fig(keys=("qn",), planes=list(ax.planes(2)))
    try:
        assert len([a for a in fig.fig.axes
                    if a.get_label() == "<colorbar>"]) == 6
        for axp, _im, _c in fig.images:
            assert "±" not in axp.get_title(loc="right")
    finally:
        fig.close()


def test_marginal_amplitudes_are_per_axis():
    """The other collapsed field: rho_max/phi_max became marg_max[axis]. Four
    marginals with different amplitudes must not share one y window."""
    st = _stats2d()
    st.marg_max = [1.0, 2.0, 3.0, 4.0]
    meta = meta_columns(protocol.SessionCreate(**cfg2(variants=["qn"])),
                        _geom(), st, ["qn"], 0, 2, 3, 30,
                        diagnostics=render_mpl.diagnostics_available(2))
    fig = FrameFigure(["qn"], st, meta, width=640, height=360,
                      diagnostics=render_mpl.diagnostics_available(2))
    try:
        tops = [fig.marg_axes[a].get_ylim()[1] for a in range(4)]
    finally:
        fig.close()
    assert tops == pytest.approx([1.08, 2.16, 3.24, 4.32])


# ---------------------------------------------------------------------------
# the panel grid
# ---------------------------------------------------------------------------

def test_panels_are_planes_by_variants():
    """Panels are the cartesian product, so PanelGrid.vue's two readings are
    the two edges of one control rather than modes of their own."""
    assert render_mpl.panel_grid(1, 4) == (2, 2)      # compare variants
    assert render_mpl.panel_grid(6, 1) == (2, 3)      # phase portrait, as
    assert render_mpl.panel_grid(1, 2) == (1, 2)      #   PanelGrid.gridClass
    assert render_mpl.panel_grid(1, 1) == (1, 1)
    assert render_mpl.panel_grid(3, 2) == (3, 2)      # matrix: rows = planes
    assert render_mpl.panel_grid(6, 4) == (6, 4)      # the 24-panel maximum

    # "compare variants": one plane, every variant
    fig, _ = _fig(keys=("qn", "cn"), planes=[(0, 1)])
    try:
        cells = [c for _a, _i, c in fig.images]
        assert [c.plane for c in cells] == [(0, 1), (0, 1)]
        assert [c.key for c in cells] == ["qn", "cn"]
    finally:
        fig.close()

    # "phase portrait": every plane, one variant, in axes.PLANES order
    fig, _ = _fig(keys=("qn",))
    try:
        assert [c.plane for _a, _i, c in fig.images] == list(ax.planes(2))
    finally:
        fig.close()

    # and the matrix in between, plane-major
    fig, _ = _fig(keys=("qn", "cn"), planes=[(0, 1), (2, 3)])
    try:
        cells = [(c.plane, c.key) for _a, _i, c in fig.images]
    finally:
        fig.close()
    assert cells == [((0, 1), "qn"), ((0, 1), "cn"),
                     ((2, 3), "qn"), ((2, 3), "cn")]


def test_each_panel_draws_its_own_plane_and_axes():
    """A panel must draw the plane it is labelled with, over that plane's own
    two axes — an index slip here is invisible on a symmetric box, which is
    why the geometry is deliberately anisotropic."""
    N = 16
    geom = protocol.RecordGeom(2, (N, N, N, N), (-6.0, -7.0, -8.0, -9.0),
                               (6.0, 7.0, 8.0, 9.0))
    fig, _ = _fig(keys=("qn",))
    vf = _vframe2d(1, N, "qn")
    try:
        fig.update(0, 0.0, geom, [vf], 0, 2)
        for axp, im, cell in fig.images:
            a, b = cell.plane
            assert axp.get_xlim() == (geom.lo[a], geom.hi[a])
            assert axp.get_ylim() == (geom.lo[b], geom.hi[b])
            assert tuple(im.get_extent()) == (geom.lo[a], geom.hi[a],
                                              geom.lo[b], geom.hi[b])
            # the data really is THIS plane, dequantized and untransposed
            want = render_mpl.dequantize_plane(vf.planes[cell.pi])
            assert numpy.array_equal(im.get_array(), want)
            assert (vf.planes[cell.pi].a, vf.planes[cell.pi].b) == cell.plane
    finally:
        fig.close()


def test_a_regrid_mid_video_is_adopted_per_plane():
    """Auto-expand is gated at 2D (M3), but the per-record geometry path is
    shared with 1D and must not rot in the meantime."""
    fig, _ = _fig(keys=("qn",), planes=[(0, 2)])
    try:
        small, big = _geom(16), _geom(32, lo=-12.0, hi=12.0)
        fig.update(0, 0.0, small, [_vframe2d(1, 16)], 0, 2)
        axp = fig.images[0][0]
        assert axp.get_xlim() == (-6.0, 6.0)
        clim = fig.images[0][1].get_clim()
        fig.update(1, 0.05, big, [_vframe2d(2, 32)], 0, 2)
        assert axp.get_xlim() == (-12.0, 12.0)
        assert fig.images[0][1].get_clim() == clim   # value scales are fixed
    finally:
        fig.close()


# ---------------------------------------------------------------------------
# the diagnostics column
# ---------------------------------------------------------------------------

def test_diagnostics_default_drops_the_marginals_at_2d_only():
    """At ndim=2 the x,y and px,py PANELS already are the spatial and momentum
    densities, so rho(x) is a further reduction of something a panel is
    showing. 1D keeps all five — the frame it always had."""
    assert render_mpl.diagnostics_default(1) == [
        "marg0", "marg1", "E", "uncertainty0", "purity"]
    assert render_mpl.diagnostics_available(2) == [
        "marg0", "marg1", "marg2", "marg3", "E",
        "uncertainty0", "uncertainty1", "purity", "lz"]
    assert render_mpl.diagnostics_default(2) == [
        "E", "uncertainty0", "uncertainty1", "purity", "lz"]
    assert "lz" not in render_mpl.diagnostics_available(1)


def test_the_1d_block_geometry_is_exactly_what_it_always_was():
    """Five plots is one column at [0.675, 0.965] against panels at
    [0.045, 0.60] — the numbers this figure has used since it was written. The
    2D default is also five, so it lands on the same layout."""
    cols, rows, panel_right = render_mpl.diag_layout(5)
    assert cols == [(0.675, 0.965)] and rows == 5
    assert panel_right == 0.60
    # up to DIAG_ROWS_MAX the panels keep their full width
    assert render_mpl.diag_layout(7)[2] == 0.60
    # past it the column splits and the panels pay for it
    cols, rows, panel_right = render_mpl.diag_layout(9)
    assert len(cols) == 2 and rows == 5
    assert cols[0][1] < cols[1][0]              # left to right, no overlap
    assert panel_right == pytest.approx(0.42)
    assert panel_right < cols[0][0]
    # nothing selected: the panels take the whole width
    assert render_mpl.diag_layout(0) == ([], 0, render_mpl.DIAG_RIGHT)


def test_selected_diagnostics_are_the_ones_built():
    for diags in ([], ["E"], ["marg2", "purity"],
                  render_mpl.diagnostics_available(2)):
        fig, _ = _fig(keys=("qn",), planes=[(0, 1)], diagnostics=diags)
        try:
            built = sorted(list(fig.marg_axes) and
                           ["marg%d" % a for a in fig.marg_axes] or [])
            built += sorted(fig.series_axes)
            assert sorted(built) == sorted(diags)
            # one cursor per SERIES plot, none on the marginals
            assert len(fig.cursors) == len(fig.series_axes)
        finally:
            fig.close()


# ---------------------------------------------------------------------------
# the words
# ---------------------------------------------------------------------------

def test_2d_titles_match_the_ui():
    """Every string comes from core/axes.py, which lib/axes.ts mirrors. The
    plain forms are what the SPA shows; the figure typesets the two-letter
    axis names as real subscripts, which is the ONLY difference."""
    assert ax.marginal_title(2, 0) == "ρ(x) = ∫W dy dpx dpy"
    assert ax.marginal_title(2, 1) == "ρ(y) = ∫W dx dpx dpy"
    assert ax.marginal_title(2, 2) == "φ(px) = ∫W dx dy dpy"
    assert ax.marginal_title(2, 3) == "φ(py) = ∫W dx dy dpx"
    assert ax.uncertainty_title(2, 0) == "ΔX·ΔPx(t)"
    assert ax.uncertainty_title(2, 1) == "ΔY·ΔPy(t)"
    assert ax.purity_title(2) == "purity γ(t) = (2πℏ)²⨌W²dxdydpxdpy"
    assert ax.lz_title() == "⟨Lz⟩(t) = ⟨x·py − y·px⟩"
    # 1D is untouched, subscripts included: single-letter names need no math
    assert ax.purity_title(1) == "purity γ(t) = 2πℏ∬W²dxdp"
    assert ax.purity_title(1, math=True) == ax.purity_title(1)
    assert ax.marginal_title(1, 0, math=True) == "ρ(x) = ∫W dp"

    fig, _ = _fig(keys=("qn",),
                  diagnostics=render_mpl.diagnostics_available(2))
    try:
        titles = [a.get_title(loc=w) for a in fig.fig.axes
                  for w in ("center", "left", "right")]
        for pid in render_mpl.diagnostics_available(2):
            assert render_mpl.diagnostic_title(2, pid, math=True) in titles, pid
        # the plane set, named as the SPA's phase portrait names it
        for pl in ax.planes(2):
            assert ax.plane_label(2, pl, math=True) in titles
            assert ax.plane_title(2, pl, math=True) in titles
        head = [t.get_text() for t in fig.fig.texts
                if t.get_text().startswith("wignerf")][0]
        assert head.startswith("wignerf — W(x, y, $p_x$, $p_y$, t)")
    finally:
        fig.close()


def test_subscripts_are_mathtext_and_only_on_the_axis_names():
    """Real subscripts, measured free (they live in STATIC artists, so the
    per-frame blit never touches them). Everything else — the integrals, γ, ℏ,
    ρ, φ — stays the same Unicode the screen uses, so the two cannot drift."""
    assert ax.sub_math("px") == "$p_x$"
    assert ax.sub_math("x") == "x"          # one letter: no math mode at all
    assert ax.sub_math("p") == "p"
    m = ax.purity_title(2, math=True)
    assert m == "purity γ(t) = (2πℏ)²⨌W²dxdyd$p_x$d$p_y$"
    assert "⨌" in m and "γ" in m and "ℏ" in m
    assert ax.lz_title(math=True) == "⟨$L_z$⟩(t) = ⟨x·$p_y$ − y·$p_x$⟩"


def test_the_header_geometry_line_groups_equal_extents():
    """Four ranges spelled out separately run off the canvas at 1080p; 1D
    keeps the exact wording it always had."""
    assert render_mpl.geom_line(protocol.RecordGeom.from_1d(
        32, 64, -6.0, 6.0, -7.0, 7.0)) == \
        "32×64   x ∈ [-6, 6]  p ∈ [-7, 7]"
    assert render_mpl.geom_line(_geom(16)) == \
        "16×16×16×16   x,y,px,py ∈ [-6, 6]"
    g = protocol.RecordGeom(2, (16, 16, 32, 32), (-6.0, -6.0, -8.0, -8.0),
                            (6.0, 6.0, 8.0, 8.0))
    assert render_mpl.geom_line(g) == \
        "16×16×32×32   x,y ∈ [-6, 6]  px,py ∈ [-8, 8]"


def test_the_metadata_block_says_what_the_frame_shows():
    """A 2D export is routinely a SUBSET of the record, and a viewer must be
    able to tell a subset from the whole thing."""
    cfg = protocol.SessionCreate(**cfg2(variants=["qn", "cn"]))
    st = _stats2d(("qn", "cn"))
    left, _right = meta_columns(cfg, _geom(), st, ["qn", "cn"], 0, 2, 3, 30,
                                planes=[(0, 1), (2, 3)],
                                diagnostics=["E", "purity"])
    text = " ".join(left)
    # the block is TYPESET like the rest of the frame, so the axis names carry
    # their real subscripts here too
    assert "panels: x,y, $p_x$,$p_y$" in text
    assert "2 planes × 2 variants = 4" in text
    # a phase portrait is "6 planes × 1 variant", never "1 variants". Compared
    # with whitespace collapsed: the phrase is long enough to be WRAPPED, which
    # inserts the continuation indent mid-sentence.
    lone, _r = meta_columns(cfg, _geom(), st, ["qn"], 0, 2, 3, 30)
    assert "6 planes × 1 variant = 6" in " ".join(" ".join(lone).split())
    assert "plots omitted:" in text
    for lbl in ("ρ(x)", "ρ(y)", "φ($p_x$)", "φ($p_y$)", "ΔX·Δ$P_x$",
                "⟨$L_z$⟩"):
        assert lbl in text
    assert "grid at record 0: 16×16×16×16" in text
    assert "x ∈ [-6, 6]" in text and "$p_y$ ∈ [-6, 6]" in text
    # ...and every $...$ span in it really renders
    for line in left:
        assert render_mpl.mathtext_ok(line), line
    # nothing omitted, nothing said
    left, _ = meta_columns(cfg, _geom(), st, ["qn"], 0, 2, 3, 30,
                           diagnostics=render_mpl.diagnostics_available(2))
    assert "plots omitted" not in " ".join(left)


def test_the_header_follows_the_painted_record_and_the_block_does_not():
    """Two geometry readouts that must NOT agree once they can differ.

    The header's is a property of the frame you are looking at, so it follows
    the PAINTED record exactly as the SPA follows the painted frame. The
    metadata block's is labelled "at record k0" and stays there — it is part of
    "what this video is", and rewriting it per frame would make the block a
    moving target. When the two can diverge (an auto-expand regrid mid-range)
    the block says so and quotes the widest window separately.

    Pinned at 1D here because that is where the wording lives; the 2D case it
    was WAITING for is the test below, now that M3 (2026-08-01) lets a 2D
    session regrid mid-range."""
    g1 = dict(x1=-6.0, x2=6.0, Nx=32, p1=-7.0, p2=7.0, Np=32)
    ic1 = {"type": "mixture",
           "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707,
                           "sigma_p": 0.707}]}
    cfg = protocol.SessionCreate(grid=g1, potential="x^2/2", ic=ic1,
                                 variants=["qn"])
    small = protocol.RecordGeom.from_1d(32, 32, -6.0, 6.0, -7.0, 7.0)
    big = protocol.RecordGeom.from_1d(64, 64, -12.0, 12.0, -14.0, 14.0)
    st = _stats1d(n=2)
    st.lo, st.hi = (-12.0, -14.0), (12.0, 14.0)          # the union
    left, _r = meta_columns(cfg, small, st, ["qn"], 0, 1, 2, 30)
    text = " ".join(" ".join(left).split())
    # the BLOCK: the first exported record, named as such, plus the union
    assert "grid at record 0: 32×32" in text
    assert "x ∈ [-6, 6], p ∈ [-7, 7]" in text
    assert "axes follow each record (auto-expand); widest:" in text
    assert "x ∈ [-12, 12], p ∈ [-14, 14]" in text

    fig = FrameFigure(["qn"], st, (left, _r), width=640, height=360)
    try:
        # the HEADER: whichever record is on screen
        fig.update(0, 0.0, small, [_vframe1d(1, 32, 32)], 0, 1)
        assert "32×32   x ∈ [-6, 6]  p ∈ [-7, 7]" in fig.geom_text.get_text()
        assert "record 0 ∈ [0, 1]" in fig.geom_text.get_text()
        fig.update(1, 0.05, big, [_vframe1d(2, 64, 64)], 0, 1)
        head = fig.geom_text.get_text()
        assert "64×64   x ∈ [-12, 12]  p ∈ [-14, 14]" in head
        assert "record 1 ∈ [0, 1]" in head
        # ...and it is neither the block's record-0 window nor the union
        assert "[-6, 6]" not in head
    finally:
        fig.close()


def test_the_header_and_block_diverge_in_2d_too():
    """The 2D case the test above was waiting for — M3 (2026-08-01) made a
    mid-range regrid reachable at ndim=2, so this is no longer hypothetical.

    Four axes rather than two is the whole point: the header must re-render
    every one of them from the PAINTED record, while the block stays at record
    k0 and states the union separately. An axis-order slip here shows up as a
    momentum extent in a spatial slot, which is exactly the class of silent
    multi-D error the geometry readouts exist to make visible.

    Note the regrid doubles ONE axis (x, 32 -> 64) and leaves the other three
    alone, which is what a single tripped axis really produces — a uniform
    doubling would hide a per-axis bug by symmetry.
    """
    cfg = protocol.SessionCreate(**cfg2(variants=["qn"]))
    small = protocol.RecordGeom(2, (32, 32, 32, 32), (-6.0,)*4, (6.0,)*4)
    big = protocol.RecordGeom(2, (64, 32, 32, 32),
                              (-12.0, -6.0, -6.0, -6.0), (12.0, 6.0, 6.0, 6.0))
    st = _stats2d(("qn",), n=2, N=16)
    st.lo, st.hi = (-12.0, -6.0, -6.0, -6.0), (12.0, 6.0, 6.0, 6.0)
    left, _r = meta_columns(cfg, small, st, ["qn"], 0, 1, 2, 30)
    text = " ".join(" ".join(left).split())
    assert "grid at record 0: 32×32×32×32" in text
    assert "axes follow each record (auto-expand); widest:" in text
    assert "x ∈ [-12, 12]" in text                      # the union, on x only

    fig = FrameFigure(["qn"], st, (left, _r), width=640, height=360,
                      planes=[(0, 2)])
    try:
        fig.update(0, 0.0, small, [_vframe2d(1, 16)], 0, 1)
        head = fig.geom_text.get_text()
        assert "32×32×32×32" in head, head
        assert "record 0 ∈ [0, 1]" in head
        fig.update(1, 0.05, big, [_vframe2d(2, 16)], 0, 1)
        head = fig.geom_text.get_text()
        assert "64×32×32×32" in head, head
        assert "x ∈ [-12, 12]" in head, head
        # ...and the three axes that did NOT move stay GROUPED at their shared
        # extent, which is the geometry line's own rule (see
        # test_the_header_geometry_line_groups_equal_extents) doing exactly the
        # right thing across a regrid: before the switch all four axes shared
        # one group, and the doubling splits x out of it.
        assert "y,$p_x$,$p_y$ ∈ [-6, 6]" in head, head
        # the header is not showing the block's record-0 x window
        assert "x ∈ [-6, 6]" not in head, head
    finally:
        fig.close()


def test_the_auto_expand_union_line_does_not_overflow_the_2d_block():
    """M3 made a regrid reachable at ndim=2, which adds the "axes follow each
    record (auto-expand); widest: …" line — and at four axes it wraps to two.
    That is the constant M3 had to check: the block grows DOWNWARD from
    META_TOP with nothing below it to stop at.

    Measured two ways, because which column dominates depends on the IC:
    with a short mixture IC the union line takes the LEFT column from 9 lines
    to 11 and it becomes the taller one — landing exactly on the 11 that fit at
    the full 8 pt, so nothing shrinks; with a long 2D cat IC the RIGHT column
    is 12 either way and the union line changes nothing at all. Neither reaches
    the 5 pt floor, past which _meta_fit elides (17 lines fit there).

    So what is pinned is the BOUND — the union line costs two wrapped lines and
    must never push the block into elision — rather than a line count that any
    wording change would invalidate."""
    keys = ["qn", "qr", "cn", "cr"]
    cfg = protocol.SessionCreate(**cfg2(variants=keys))
    log = [{"at_record": i, "t": 0.05*i, "applied": {f: v}, "before": {f: b}}
           for i, (f, v, b) in enumerate(
               [("hbar_eff", 2.0, 1.0), ("mass", 2.0, 1.0),
                ("c", 50.0, 137.035999),
                ("U", "(x^2-y^2)/2 + 0.9*x^4", "(x^2-y^2)/2 + 0.3*x^4")], 1)]
    geom = protocol.RecordGeom(2, (128, 128, 64, 64), (-8.0, -8.0, -7.0, -7.0),
                               (8.0, 8.0, 7.0, 7.0))
    planes = list(ax.planes(2))

    def build(union):
        st = _stats2d(tuple(keys), n=48, N=16)
        st.lo = (-16.0, -8.0, -7.0, -7.0) if union else (-8.0, -8.0, -7.0, -7.0)
        st.hi = (16.0, 8.0, 7.0, 7.0) if union else (8.0, 8.0, 7.0, 7.0)
        cols = meta_columns(cfg, geom, st, keys, 0, 47, 48, 30,
                            planes=planes, param_log=log)
        fig = FrameFigure(keys, st, cols, width=1920, height=1080,
                          planes=planes)
        try:
            n = max(len(cols[0]), len(cols[1]))
            return cols, n, fig._meta_fontsize(n), fig.META_MIN_FONTSIZE
        finally:
            fig.close()

    plain, n_plain, fs_plain, _ = build(False)
    grown, n_grown, fs_grown, fmin = build(True)

    text = " ".join(" ".join(grown[0]).split())
    assert "axes follow each record (auto-expand); widest:" in text
    assert "x ∈ [-16, 16]" in text                # all four axes, union on x
    assert "y ∈ [-8, 8]" in text
    assert "$p_x$ ∈ [-7, 7]" in text
    assert text.count("∈") == 8                   # 4 at record 0 + 4 widest
    # the union line lands in the LEFT column and costs it two wrapped lines —
    # four axes do not fit on one, and that is the whole risk being bounded
    assert len(grown[0]) == len(plain[0]) + 2
    assert len(grown[1]) == len(plain[1])          # the right column is untouched
    # ...and the block still renders at a readable size, well clear of the
    # floor past which _meta_fit starts dropping facts on the ground
    assert fs_grown > fmin, "shrunk to the floor: the next stop is elision"
    assert fs_grown >= fs_plain - 1.0, (fs_plain, fs_grown)
    assert n_grown <= n_plain + 2


def test_the_metadata_block_never_breaks_a_line_mid_fact():
    """"px ∈ [-7, 7]" is ONE fact and was being wrapped between the axis name
    and the ∈. Each such group is atomic now; a line may break between them.

    NB textwrap splits on \\s+ and a Unicode NBSP is \\s too, so the marker has
    to be a character textwrap cannot see as whitespace — and it must never
    survive into the figure."""
    cfg = protocol.SessionCreate(**cfg2(variants=["qn"]))
    st = _stats2d(("qn",))
    # a wide anisotropic box, so the extents line is long enough to wrap
    geom = protocol.RecordGeom(2, (128, 128, 64, 64),
                               (-8.0, -8.0, -7.0, -7.0), (8.0, 8.0, 7.0, 7.0))
    left, right = meta_columns(cfg, geom, st, ["qn"], 0, 46, 47, 20)
    for line in left + right:
        assert render_mpl._NB not in line, line
        # the relation must never be stranded at either end of a line
        assert not line.rstrip().endswith("∈"), line
        assert not line.strip().startswith("∈"), line
    # Checked PER LINE, never against the joined column: joining fragments with
    # a space is exactly what reconstitutes a group that was split, so a
    # whole-column search cannot see this bug at all.
    assert len(left) > 6, "the block did not wrap; the test proves nothing"
    for frag in ("x ∈ [-8, 8]", "y ∈ [-8, 8]", "$p_x$ ∈ [-7, 7]",
                 "$p_y$ ∈ [-7, 7]", "6 planes × 1 variant = 6",
                 "1 a.u. of time = 0.0241888 fs"):
        assert any(frag in l for l in left), (frag, left)


def test_the_potential_is_typeset_from_the_users_own_string():
    """U(x,y) is USER input, so it is typeset by a lexical rewrite of what they
    typed — not by round-tripping through sympy.latex, which canonicalises
    (`10*(1-exp(-0.5*(x-1)))^2` comes back as `10(1 - 1.64872127070013
    e^{-0.5x})^2`, sympy having evaluated exp(0.5) at parse time) and emits
    \\frac, which is TALL in a block with fixed line spacing."""
    from core import describe
    cases = {
        "x^2/2": "x^{2}/2",
        "(x^2 + y^2)/2 + 0.3*x*y": "(x^{2} + y^{2})/2 + 0.3\\,x\\,y",
        "1/x^12 - 1/x^6": "1/x^{12} - 1/x^{6}",
        "10*(1-exp(-0.5*(x-1)))^2":
            "10\\,(1-\\mathrm{exp}(-0.5\\,(x-1)))^{2}",
        "Abs(x)+Abs(y)": "\\mathrm{Abs}(x)+\\mathrm{Abs}(y)",
    }
    for src, want in cases.items():
        got = describe.potential_math(src)
        assert got == want, (src, got)
        # the source survives token for token: nothing reordered, no constant
        # folded, and nothing taller than a superscript introduced
        assert "\\frac" not in got
        bare = got.replace("\\,", "*").replace("\\mathrm{", "") \
                  .replace("{", "").replace("}", "")
        assert bare.replace(" ", "") == src.replace(" ", ""), (src, bare)
        assert render_mpl.mathtext_ok("$%s$" % got), got
    # nothing to gain, nothing returned — the caller keeps the plain string
    assert describe.potential_math("x") is None
    assert describe.potential_math("") is None
    # and anything already carrying markup is refused outright
    assert describe.potential_math("x^2 $x$") is None

    cfg = protocol.SessionCreate(**cfg2(potential="(x^2 + y^2)/2 + 0.3*x*y"))
    st = _stats2d(("qn", "cn"))
    _l, right = meta_columns(cfg, _geom(), st, ["qn", "cn"], 0, 2, 3, 30)
    text = " ".join(right)
    assert "U(x,y) = $(x^{2} + y^{2})/2 + 0.3\\,x\\,y$" in text
    for line in right:
        assert render_mpl.mathtext_ok(line), line


def test_an_untypesettable_line_falls_back_instead_of_crashing():
    """A mathtext parse error raises at DRAW time and would take the whole
    export down, and this column is the one place a typeset line is built from
    user input. So every candidate is checked against matplotlib's own parser
    and the plain form is kept when it fails."""
    assert render_mpl.mathtext_ok("plain text, no maths")
    assert render_mpl.mathtext_ok("x ∈ [-6, 6] and $p_x$ too")
    assert not render_mpl.mathtext_ok("$p_x$ and a stray $")
    assert not render_mpl.mathtext_ok("$\\frac{1}$")        # too few arguments
    assert not render_mpl.mathtext_ok("$\\notacommand{x}$")

    # a line that has to WRAP cannot keep an atomic $...$ span, so it stays
    # plain rather than being broken in half
    plain = ["U(x,y) = " + "a"*200]
    math = ["U(x,y) = $" + "a"*200 + "$"]
    out = render_mpl._emit(plain, math, 60, 2)
    assert len(out) > 1 and not any("$" in l for l in out)
    # ...while one that fits is swapped wholesale
    out = render_mpl._emit(["U(x,y) = x^2"], ["U(x,y) = $x^{2}$"], 60, 2)
    assert out == ["U(x,y) = $x^{2}$"]


def test_1d_metadata_is_byte_identical_to_the_plain_form():
    """Every 1D axis name is one letter, so nothing is typeset and no existing
    export changes by a single character."""
    from core import describe
    g1 = dict(x1=-6.0, x2=6.0, Nx=32, p1=-7.0, p2=7.0, Np=32)
    ic1 = {"type": "mixture",
           "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707,
                           "sigma_p": 0.707}]}
    cfg = protocol.SessionCreate(grid=g1, potential="x^2/2", ic=ic1,
                                 variants=["qn"])
    assert describe.param_lines(cfg, math=True)[1:] == \
        describe.param_lines(cfg)[1:]
    assert describe.ic_expression(cfg.ic, 1.0, math=True) == \
        describe.ic_expression(cfg.ic, 1.0)
    assert ax.sub_math_text("p ∈ [-7, 7]", 1) == "p ∈ [-7, 7]"
    # the ONE 1D line that does change is U(x), and only because a plain
    # "x^2/2" is genuinely nicer set as maths
    assert describe.param_lines(cfg, math=True)[0] == "U(x) = $x^{2}/2$"


def test_the_metadata_block_still_fits_at_2d():
    """The block grows DOWNWARD from a fixed anchor. 2D makes every line
    longer — four axes in the grid line, four in the IC, plus the new panels
    line — so re-measure rather than assume the 11-line 1D budget still
    covers a realistic run."""
    cfg = protocol.SessionCreate(**cfg2(
        variants=["qn", "qr", "cn", "cr"], hbar_eff=2.0, mass=3.0,
        ic={"type": "cat", "components": [
            {"q0": [-2.0, 0.0], "k0": [0.0, 0.0], "sigma_q": [0.5, 0.5]},
            {"q0": [2.0, 0.0], "k0": [0.0, 0.0], "sigma_q": [0.5, 0.5],
             "phase": 3.14159}]},
        precision="float32", tol=1e-2))
    log = [{"at_record": i, "applied": {"hbar_eff": 1.0 + i},
            "before": {"hbar_eff": float(i)}} for i in range(1, 5)]
    st = _stats2d(("qn", "qr", "cn", "cr"))
    left, right = meta_columns(cfg, _geom(), st, ["qn", "qr", "cn", "cr"],
                               0, 2, 3, 30, log, planes=[(0, 1)],
                               diagnostics=render_mpl.diagnostics_default(2))
    fig = FrameFigure(["qn", "qr", "cn", "cr"], st, (left, right),
                      planes=[(0, 1)],
                      diagnostics=render_mpl.diagnostics_default(2))
    try:
        n = max(len(left), len(right))
        fs = fig._meta_fontsize(n)
        kept_l = fig._meta_fit(left, fs)
        kept_r = fig._meta_fit(right, fs)
        # the invariant that matters: the last baseline stays on the figure
        advance = fs*FrameFigure.META_LINESPACING/72.0/fig.fig.get_figheight()
        for kept in (kept_l, kept_r):
            assert len(kept)*advance <= FrameFigure.META_TOP + 1e-9
        # and nothing is dropped in silence
        for kept, col in ((kept_l, left), (kept_r, right)):
            if len(kept) < len(col):
                assert "more lines" in kept[-1]
    finally:
        fig.close()
    # the measured worst case for a realistic 2D run, recorded so a future
    # change that blows the budget is visible as a number moving
    assert n <= 16, n


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------

def test_plane_and_diagnostics_refusals_name_what_is_available():
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        r = client.post("/api/sessions/%s/export" % sid, json={"planes": [6]})
        assert r.status_code == 422, r.text
        assert "x,y" in r.text and "y,px" in r.text     # the six it DOES have
        r = client.post("/api/sessions/%s/export" % sid,
                        json={"diagnostics": ["marg9"]})
        assert r.status_code == 422 and "available" in r.text
        assert "uncertainty1" in r.text and "lz" in r.text
        # duplicates are a schema error, before ndim is even consulted
        assert client.post("/api/sessions/%s/export" % sid,
                           json={"planes": [0, 0]}).status_code == 422
        client.delete("/api/sessions/%s" % sid)

    # ...and at ndim=1 the 2D-only plots are refused by name
    with TestClient(app) as client:
        g1 = {"x1": -6.0, "x2": 6.0, "Nx": 32, "p1": -7.0, "p2": 7.0, "Np": 32}
        ic1 = {"type": "mixture",
               "components": [{"x0": 1.0, "p0": 0.0, "sigma_x": 0.7,
                               "sigma_p": 0.7}]}
        info = client.post("/api/sessions",
                           json={"grid": g1, "potential": "x^2/2", "ic": ic1,
                                 "variants": ["qn"]}).json()
        sid = info["session_id"]
        r = client.post("/api/sessions/%s/export" % sid,
                        json={"diagnostics": ["lz"]})
        assert r.status_code == 422 and "1D" in r.text
        r = client.post("/api/sessions/%s/export" % sid, json={"planes": [1]})
        assert r.status_code == 422 and "1D" in r.text and "x,p" in r.text
        client.delete("/api/sessions/%s" % sid)


def test_download_name_records_the_plane_selection():
    """Two exports of one range with different planes are different videos and
    must not save over each other. 1D grows no marker at all."""
    class _Sess:
        id = "0123456789ab"
        cfg = protocol.SessionCreate(**cfg2(variants=["qn", "cn"]))

    job = videoexport.ExportJob(_Sess(), protocol.ExportSpec(planes=[0]),
                                0, 9, "/tmp")
    assert job.download_name.startswith("wignerf-QN-CN-xy-10rec-")
    job = videoexport.ExportJob(_Sess(), protocol.ExportSpec(planes=[4, 5]),
                                0, 9, "/tmp")
    assert job.download_name.startswith("wignerf-QN-CN-xpy+ypx-10rec-")
    job = videoexport.ExportJob(_Sess(), protocol.ExportSpec(), 0, 9, "/tmp")
    assert job.download_name.startswith("wignerf-QN-CN-6pl-10rec-")


@needs_ffmpeg
def test_2d_export_end_to_end(tmp_path, monkeypatch):
    """The whole path on a real 2D session: compute, pause, render every
    plane of both variants, and read the setup document back out of the mp4's
    own comment tag (which is what lib/mp4meta.ts does on import)."""
    import config as appconfig
    monkeypatch.setattr(appconfig, "EXPORT_DIR", str(tmp_path))
    with TestClient(app) as client:
        info = client.post("/api/sessions", json=cfg2()).json()
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "play"}))
            for _ in range(200):
                time.sleep(0.05)
                if client.get("/api/sessions/%s" % sid
                              ).json()["record_extent"][1] >= 5:
                    break
            ws.send_text(json.dumps({"type": "pause"}))
            time.sleep(0.3)
            first, last = client.get("/api/sessions/%s" % sid
                                     ).json()["record_extent"]
            assert last >= first, "no 2D records were computed"
            r = client.post("/api/sessions/%s/export" % sid,
                            json={"k0": first, "k1": min(last, first + 4),
                                  "fps": 10, "width": 640, "height": 360,
                                  "diagnostics":
                                      render_mpl.diagnostics_available(2)})
            assert r.status_code == 202, r.text
            jid = r.json()["job_id"]
            for _ in range(900):
                time.sleep(0.1)
                st = client.get("/api/exports/%s" % jid).json()
                if st["state"] in ("done", "error", "cancelled"):
                    break
            assert st["state"] == "done", st
            assert st["done"] == st["total"] and st["bytes"] > 0

            probe = videoexport.probe_json(videoexport.get(jid).path)
            if probe is not None:
                assert probe["streams"][0]["codec_name"] == "h264"
                assert int(probe["streams"][0]["nb_frames"]) == st["total"]
                tags = probe.get("format", {}).get("tags", {})
                blob = json.loads(tags["comment"])
                assert blob["config"]["grid"]["ndim"] == 2
                assert blob["export"]["planes"] == [list(p)
                                                    for p in ax.planes(2)]
                assert "lz" in blob["export"]["diagnostics"]
                # the document is re-postable, which is what import relies on
                protocol.SessionCreate.model_validate(blob["config"])
            client.delete("/api/exports/%s" % jid)
        client.delete("/api/sessions/%s" % sid)
