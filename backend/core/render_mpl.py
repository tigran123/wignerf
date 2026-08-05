"""
Video-frame renderer for the mp4 export: one matplotlib (Agg) figure that
reproduces the SPA's whole view — tiled W panels, the marginals, the
E(t)/dX*dP(t)/gamma(t)/<Lz>(t) series with a moving time cursor — plus a
metadata block (see core/describe.py) that makes an exported frame
self-contained.

The figure is built ONCE per export job and only its data is updated per
record (set_data / set_extent / set_text), which is what keeps a 1000-frame
export in the minutes range rather than the tens of minutes a fresh figure
per frame would cost.

WHAT GOES IN THE FRAME IS A CHOICE, at every ndim (milestone M4, 2026-07-28).
A 1D record has one plane per variant, two marginals and three series — it all
fits. A 2D record has SIX planes per variant (up to 24 panels), four marginals
and five series: nine diagnostics against 1D's five, and the SPA answers that
by scrolling its column and offering two panel readings, neither of which a
video frame can do. So `planes` and `diagnostics` select what is rendered:

- panels are the CARTESIAN PRODUCT of the selected planes and variants, which
  makes PanelGrid.vue's two readings the two edges of one control ("compare
  variants" = one plane x every variant; "phase portrait" = every plane x one
  variant) rather than modes this renderer has to know about,
- `diagnostics` is a list of plot ids shared verbatim with
  frontend/src/lib/plotPrefs.ts (marg0..marg3, E, uncertainty0/1, purity, lz).

The 2D DEFAULT drops the four marginals, and that is a physical argument, not
a space-saving one: at ndim=2 the (x,y) and (px,py) PANELS already ARE the
spatial and momentum densities, so rho(x) is a further reduction of something
a panel is showing. 1D defaults to everything, i.e. exactly the frame it always
had.

Conventions taken from the live renderer (frontend/src/render/
WignerRenderer.ts) so the video and the screen read the same:
- W is dequantized as wmin + q*(wmax-wmin)/65535; records arrive in NATURAL
  order (frame.build unshifts on the device), so nothing shifts here,
- the colour scale is the SYMMETRIC diverging one, W=0 at the centre of
  "bwr": vmin = -s, vmax = +s with s = max(Wmax, -Wmin) — here taken over
  the WHOLE exported range PER (variant, plane), so the video does not
  flicker. Per plane and not per variant: the six reductions of one state
  differ in scale by orders of magnitude (a spatial density against a signed
  reduced Wigner function), and one shared range renders most panels blank,
- the SPATIAL axes, in contrast, follow the PER-RECORD geometry (the SPA
  re-derives its grid from the painted frame): freezing them at the range
  union would shrink every frame before an auto-expansion into a corner of
  its panel. Only value scales are export-wide,
- variant colours and dash patterns mirror frontend/src/lib/variants.ts, and
  the frame is rendered in the SPA's light or dark theme (ExportSpec.theme,
  defaulted from the app's own toggle): the palettes below mirror the
  --wf-* custom properties in frontend/src/style.css. Change a colour on one
  side and change it on the other — a video that does not read like the
  screen is the whole failure mode this module exists to avoid.
  What does NOT change with the theme is the heatmap itself ("bwr", white at
  W = 0) or anything drawn ON it: the grid, and the per-panel scale caption.

Every display string comes from core/axes.py, which frontend/src/lib/axes.ts
mirrors — change a title in one and change it in all three. NO MATHTEXT
anywhere (see core/describe.py's docstring for the two reasons), so momentum
axes read as plain "px"/"py" rather than with real subscripts as on screen.
"""

import textwrap
from dataclasses import dataclass, field

import numpy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from . import axes as ax
from .protocol import VARIANTS

THEMES = ("dark", "light")

# Labels and dash patterns mirror frontend/src/lib/variants.ts (VARIANT_META);
# the dashes are the anti-occlusion device (coincident curves stay legible
# through each other's gaps) and so are theme-independent.
VARIANT_LABEL_DASH = {
    "qn": ("Quantum, non-relativistic", (0, ())),
    "qr": ("Quantum, relativistic", (0, (12, 7))),
    "cn": ("Classical, non-relativistic", (0, (6, 6))),
    "cr": ("Classical, relativistic", (0, (2, 6))),
}

# variants.ts VARIANT_COLORS: Tailwind *-400 on dark, *-600 on light — a
# curve colour has to hold its own against the PAGE, and #fbbf24 amber on
# white does not.
VARIANT_COLORS = {
    "dark": {"qn": "#38bdf8", "qr": "#a78bfa", "cn": "#fbbf24", "cr": "#34d399"},
    "light": {"qn": "#0284c7", "qr": "#7c3aed", "cn": "#d97706", "cr": "#059669"},
}

# The SPA's --wf-* tokens (frontend/src/style.css), by role:
#   bg      figure face   (--wf-app)
#   axbg    axes face     (--wf-panel)
#   fg      titles        (--wf-chart-text)
#   muted   ticks/labels  (--wf-chart-axis)
#   grid    chart grid + spines (--wf-chart-grid)
#   clock   the header time readout (the QN hue)
#   cursor  time cursor on the series (--wf-cursor)
PALETTE = {
    "dark": {
        "bg": "#0a0a0a", "axbg": "#171717", "fg": "#d4d4d4",
        "muted": "#a3a3a3", "grid": "#3f3f46",
        "clock": "#38bdf8", "cursor": "#f472b6",
    },
    "light": {
        "bg": "#ffffff", "axbg": "#fafafa", "fg": "#404040",
        "muted": "#525252", "grid": "#d4d4d8",
        "clock": "#0284c7", "cursor": "#db2777",
    },
}

# GridOverlay.vue's phase-space grid: rgba(120,120,120,.28), .55 at zero.
# Theme-independent — it is drawn over the heatmap, which is too.
PANEL_GRIDC = "#787878"
PANEL_GRID_ALPHA = 0.28
PANEL_ZERO_ALPHA = 0.55


def variant_style(key, theme="light"):
    """(label, colour, dash) for a variant in `theme` — the shape
    VARIANT_STYLE used to have before the palette became theme-dependent."""
    label, dash = VARIANT_LABEL_DASH[key]
    return label, VARIANT_COLORS[theme][key], dash

AU_TIME_FS = 2.4188843265857e-2      # lib/units.ts
AU_ENERGY_EV = 27.211386245988


def key_of_vid(vid):
    return ("q" if vid & 1 else "c") + ("r" if vid & 2 else "n")


def dequantize_plane(pf):
    """One PlaneFrame -> float32 (N[b], N[a]) in natural order, for imshow.

    The transpose puts the plane's FIRST axis horizontal and its second
    vertical, which is the same rule WignerRenderer.ts follows on screen — at
    ndim=1 that is (x, p) exactly as before, and it generalizes to any (a, b)
    with no special case.

    No unshift here: `frame.build` hands out natural order (it has to, so a crop
    cannot straddle the fftshift seam), so this used to unshift an already
    unshifted plane and put every export's W back on the torus's far side."""
    W = pf.wmin + pf.wq.astype(numpy.float32)*(
        numpy.float32((pf.wmax - pf.wmin)/65535.0))
    return W.T


def axis_of(a1, a2, n):
    """Cell-centre-free natural axis, as MarginalsPlot.buildAxis does."""
    return a1 + numpy.arange(n)*((a2 - a1)/n)


# ---------------------------------------------------------------------------
# What can be plotted, and what is plotted by default
# ---------------------------------------------------------------------------

def diagnostics_available(ndim):
    """Every diagnostics plot id at this ndim, in PlotsColumn.vue's display
    order. The ids are shared verbatim with frontend/src/lib/plotPrefs.ts —
    one vocabulary for the hidden-series preferences, the export wire and the
    metadata block."""
    ids = ["marg%d" % a for a in range(ax.n_axes(ndim))]
    ids.append("E")
    ids += ["uncertainty%d" % d for d in range(ndim)]
    ids.append("purity")
    if ndim > 1:
        ids.append("lz")
    return ids


def diagnostics_default(ndim):
    """What an export renders when it was not told. 1D: all five, i.e. the
    frame this module always produced. 2D: the SERIES only — see the module
    docstring for why the four marginals come off by default there."""
    ids = diagnostics_available(ndim)
    if ndim == 1:
        return ids
    return [i for i in ids if not i.startswith("marg")]


def _marg_axis(pid):
    """The axis index of a "marg{a}" id, or None for a series id."""
    return int(pid[4:]) if pid.startswith("marg") else None


def diagnostic_title(ndim, pid, math=False):
    """The plot's full title, verbatim from core/axes.py (hence from the SPA).
    "E(t)" is a literal on both sides — there is nothing to derive.

    `math` typesets the two-letter axis names as real subscripts; the figure
    passes True, and the plain form is what a test compares against axes.py."""
    a = _marg_axis(pid)
    if a is not None:
        return ax.marginal_title(ndim, a, math)
    if pid == "E":
        return "E(t)"
    if pid.startswith("uncertainty"):
        return ax.uncertainty_title(ndim, int(pid[11:]), math)
    if pid == "purity":
        return ax.purity_title(ndim, math)
    if pid == "lz":
        return ax.lz_title(math)
    raise ValueError("unknown diagnostics plot id: %r" % pid)


def diagnostic_label(ndim, pid, math=False):
    """A chip-sized name for the same plot — for the metadata block and the
    export dialog's checkboxes, where the full title does not fit."""
    a = _marg_axis(pid)
    if a is not None:
        name = ax.sub_math(ax.label(ndim, a)) if math else ax.label(ndim, a)
        return "ρ(%s)" % name      # every marginal, see ax.marginal_title
    if pid == "E":
        return "E"
    if pid.startswith("uncertainty"):
        # drop the "(t)": this names the quantity, not the series
        return ax.uncertainty_title(ndim, int(pid[11:]), math)[:-3]
    if pid == "purity":
        return "γ"
    if pid == "lz":
        return "⟨%s⟩" % (ax.sub_math("Lz") if math else "Lz")
    raise ValueError("unknown diagnostics plot id: %r" % pid)


def axis_label(ndim, a, compact=False, math=True):
    """"x (a₀)" / "$p_y$ (a.u.)" — spatial axes in Bohr, momentum in a.u.
    `compact` drops the unit, which is what a 24-panel grid has room for."""
    name = ax.sub_math(ax.label(ndim, a)) if math else ax.label(ndim, a)
    if compact:
        return name
    return "%s (%s)" % (name, "a.u." if ax.is_momentum(ndim, a) else "a₀")


# ---------------------------------------------------------------------------
# Block geometry. These are the numbers this figure has always used; a 1D
# export (and the 2D default, which is also five plots) is laid out to the
# digit as before, which is what test_diagnostics_selection_shapes_the_column
# pins.
# ---------------------------------------------------------------------------

PLOT_TOP = 0.935
PLOT_BOTTOM = 0.235
PANEL_LEFT = 0.045
DIAG_RIGHT = 0.965
DIAG_GAP = 0.075          # panel block -> first diagnostics column
DIAG_ROWS_MAX = 7         # past this the column splits in two


def panel_grid(n_planes, n_variants):
    """(rows, cols) for the W-panel block.

    When one of the two dimensions is 1 the cells REFLOW by count, reproducing
    PanelGrid.vue's own rule — so 1D and the "compare variants" reading keep
    the 1x1 / 1x2 / 2x2 tiling this figure always had, and the six planes of
    one variant give the 3-across, 2-down the phase portrait shows on screen.
    When BOTH vary there is a structure worth keeping, so the grid is the
    matrix itself: one row per plane, one column per variant."""
    if n_planes == 1 or n_variants == 1:
        n = n_planes*n_variants
        if n <= 1:
            return 1, 1
        if n == 2:
            return 1, 2
        if n <= 4:
            return 2, 2
        return 2, 3
    return n_planes, n_variants


def diag_layout(n_diag):
    """(columns, rows_per_column, panel_right) in figure coordinates.

    One column is the historic [0.675, 0.965] against panels at [0.045, 0.60].
    A second is reached only by asking for more than DIAG_ROWS_MAX plots — in
    practice all nine of a 2D run at once — and it is paid for out of the panel
    block's width, which is what the export dialog says it costs."""
    if n_diag <= 0:
        return [], 0, DIAG_RIGHT
    ncols = -(-n_diag//DIAG_ROWS_MAX)          # ceil
    rows = -(-n_diag//ncols)                   # balanced across the columns
    if ncols == 1:
        return [(0.675, DIAG_RIGHT)], rows, 0.60
    w, gap = 0.21, 0.05
    cols = [(DIAG_RIGHT - j*(w + gap) - w, DIAG_RIGHT - j*(w + gap))
            for j in range(ncols)]
    cols.reverse()                             # left to right
    return cols, rows, cols[0][0] - DIAG_GAP


def geom_line(geom, math=False):
    """The per-record geometry readout in the header.

    1D keeps the wording it always had. 2D groups axes that share an extent —
    "x,y ∈ [-8, 8]  px,py ∈ [-8, 8]" — because four ranges spelled out
    separately run off the canvas at 1080p.

    `math` subscripts the axis NAMES only. The numbers stay in the monospace
    family that stops the record counter wobbling frame to frame — the axis
    names are constant for a whole export, so typesetting them costs that
    nothing."""
    nd = geom.ndim
    ls = [ax.sub_math(l) for l in ax.labels(nd)] if math else ax.labels(nd)
    sizes = "×".join(str(n) for n in geom.N)
    if nd == 1:
        parts = ["%s ∈ [%.4g, %.4g]" % (ls[a], geom.lo[a], geom.hi[a])
                 for a in range(2)]
    else:
        groups = []
        for a in range(ax.n_axes(nd)):
            ext = (geom.lo[a], geom.hi[a])
            if groups and groups[-1][0] == ext:
                groups[-1][1].append(ls[a])
            else:
                groups.append((ext, [ls[a]]))
        parts = ["%s ∈ [%.4g, %.4g]" % (",".join(names), lo, hi)
                 for (lo, hi), names in groups]
    return "%s   %s" % (sizes, "  ".join(parts))


@dataclass
class RangeStats:
    """What the scan pass over the exported records collects (see
    videoexport.ExportJob): the series to plot, the fixed colour scales and
    the fixed marginal amplitudes. `lo`/`hi` are the WIDEST window any record
    in the range used, per axis — metadata only (the panels follow each
    record).

    `scale` is keyed by (variant key, plane) and `uncert` by (variant key,
    spatial dimension): at ndim=2 there are six independent colour scales per
    variant and two uncertainty products, and collapsing either would be
    silently wrong rather than loudly missing."""
    ndim: int = 1
    t: list = field(default_factory=list)
    E: dict = field(default_factory=dict)         # key -> list
    uncert: dict = field(default_factory=dict)    # (key, dim) -> list
    purity: dict = field(default_factory=dict)    # key -> list
    lz: dict = field(default_factory=dict)        # key -> list
    scale: dict = field(default_factory=dict)     # (key, plane) -> W scale
    marg_max: list = field(default_factory=list)  # per phase-space axis
    lo: tuple = ()
    hi: tuple = ()

    def extent(self, a):
        """The widest window on axis `a`, or a placeholder before the first
        record has been seen (the first update() installs the real one)."""
        if a < len(self.lo) and a < len(self.hi):
            return self.lo[a], self.hi[a]
        return 0.0, 1.0

    def amplitude(self, a):
        return self.marg_max[a] if a < len(self.marg_max) else 0.0

    def series(self, pid, key):
        """The values of one series plot for one variant."""
        if pid == "E":
            return self.E.get(key, [])
        if pid == "purity":
            return self.purity.get(key, [])
        if pid == "lz":
            return self.lz.get(key, [])
        if pid.startswith("uncertainty"):
            return self.uncert.get((key, int(pid[11:])), [])
        raise ValueError("not a series plot id: %r" % pid)


def _num_width(values, decimals=3):
    """Width of the widest "%.<decimals>f" in `values` — the field width that
    keeps a per-frame readout from changing length (see FrameFigure)."""
    return max((len("%.*f" % (decimals, v)) for v in values),
               default=decimals + 2)


def _style_axes(ax_, pal, title=None, title_loc="center", grid=True,
                labelsize=7, titlesize=8.5):
    ax_.set_facecolor(pal["axbg"])
    for s in ax_.spines.values():
        s.set_color(pal["grid"])
    ax_.tick_params(colors=pal["muted"], labelsize=labelsize)
    # NB: grid(False, color=...) *enables* the grid (matplotlib treats the
    # line properties as a request), so the two cases must be separate
    if grid:
        ax_.grid(True, color=pal["grid"], linewidth=0.5, alpha=0.6)
    else:
        ax_.grid(False)
    ax_.yaxis.get_offset_text().set(color=pal["muted"], fontsize=6.5)
    if title:
        ax_.set_title(title, color=pal["fg"], fontsize=titlesize, pad=3,
                      loc=title_loc)


def series_ylim(values):
    """The y-window uPlot uses for these plots in the SPA (SeriesPlot.vue's
    scales.y.range): pad = max(15% of the span, 1e-4 of the magnitude,
    1e-12). Reproduced EXACTLY here — matplotlib's own autoscale zooms onto
    the data span instead, which turned a purity drift the UI renders as a
    flat line at 1.000000 into a dramatic dive with a "×10⁻⁵+1" offset
    label. Same numbers, different picture; the video must read like the
    screen."""
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite:
        return (0.0, 1.0)
    mn, mx = min(finite), max(finite)
    pad = max((mx - mn)*0.15, abs(mx)*1e-4, 1e-12)
    return (mn - pad, mx + pad)


def _tick_decimals(splits):
    """SeriesPlot.vue's tick formatter: enough decimals to tell neighbouring
    ticks apart on a tightly-zoomed axis (default formatting prints '1')."""
    step = abs((splits[1] if len(splits) > 1 else 1) - splits[0]) or 1
    from math import ceil, log10
    return max(0, min(10, int(ceil(-log10(step))) + 1))


@dataclass
class _Cell:
    """One W panel: which plane of which variant it draws."""
    plane: tuple
    pi: int        # index into VariantFrame.planes (axes.PLANES order)
    vi: int        # index into the ordered vframes
    key: str


class FrameFigure:
    """Builds the figure once; `update()` returns the RGB bytes of a frame."""

    # The layout is defined at this width; every other resolution renders
    # the SAME figure at a different dpi (font sizes are in POINTS, so a
    # fixed dpi would shrink all text to half its relative size at 4K).
    REF_WIDTH = 1920.0

    # The metadata block: anchored here with va="top", so it grows DOWNWARD out
    # of the strip the gridspecs leave below the plots (bottom=0.235).
    META_TOP = 0.185
    META_FONTSIZE = 8.0
    META_LINESPACING = 1.6
    META_MIN_FONTSIZE = 5.0    # below this it is decoration, not information

    # Past this many panels a colorbar is a ~9 px strip with 6.5 pt ticks —
    # neither readable nor free — so each panel carries its (export-wide,
    # hence static) symmetric scale as a caption instead.
    CBAR_MAX_CELLS = 8

    def __init__(self, variants, stats, meta_lines, width=1920, height=1080,
                 show_grid=True, theme="light", planes=None, diagnostics=None):
        self.variants = list(variants)
        self.stats = stats
        self.ndim = int(getattr(stats, "ndim", 1) or 1)
        # WHAT to draw. Defaults are "the whole record" for panels and
        # diagnostics_default() for the column — see the module docstring.
        self.planes = [tuple(p) for p in
                       (planes if planes is not None else ax.planes(self.ndim))]
        self.diagnostics = list(diagnostics if diagnostics is not None
                                else diagnostics_default(self.ndim))
        # light or dark, from ExportSpec.theme (defaulted to the SPA's own
        # toggle). Resolved once: every colour below comes from self.pal, so
        # the two palettes cannot drift apart in the body of the renderer.
        self.theme = theme if theme in THEMES else "light"
        self.pal = PALETTE[self.theme]
        # ONE grid setting for the whole frame, mirroring the SPA's "grid
        # lines on plots" checkbox: charts get uPlot's grid, the W panels
        # get GridOverlay.vue's (which is why they used to have none — the
        # heatmap is drawn over the axes grid)
        self.show_grid = bool(show_grid)
        self.width, self.height = int(width), int(height)
        dpi = 100.0*self.width/self.REF_WIDTH
        self.fig = Figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi,
                          facecolor=self.pal["bg"])
        self.canvas = FigureCanvasAgg(self.fig)

        cells = self._cells()
        rows, cols = panel_grid(len(self.planes), len(self.variants))
        diag_cols, diag_rows, panel_right = diag_layout(len(self.diagnostics))

        # ---- header. The title names the phase space this run lives in, and
        # then — when the selection has collapsed one of the two dimensions —
        # names the constant ONCE, exactly as PanelGrid's selector does rather
        # than repeating it on every panel.
        head = "wignerf — W(%s, t)" % ", ".join(
            ax.sub_math(l) for l in ax.labels(self.ndim))
        if self.ndim > 1 and len(self.planes) == 1:
            head += "   ·   %s" % ax.plane_title(self.ndim, self.planes[0],
                                                 math=True)
        elif len(self.planes) > 1 and len(self.variants) == 1:
            head += "   ·   %s" % variant_style(self.variants[0],
                                                self.theme)[0]
        title = self.fig.text(0.012, 0.972, head, color=self.pal["fg"],
                              fontsize=13, weight="bold", va="center")
        # The two header readouts change every frame, so they must not
        # WOBBLE. "%.4g" printed a different number of decimals as the value
        # grew (0.02419 → 0.2419 → 2.419 fs: the trailing digits just
        # vanish), so the text changed length and the "(… fs)" part slid
        # about. Fixed decimals + fields padded to the widest value this
        # export will show + a monospace family (a proportional font pads
        # with narrow spaces) pin every glyph to its own pixel column.
        self._tw = _num_width(stats.t)
        self._fw = _num_width([v*AU_TIME_FS for v in stats.t])
        # ...and it must not be OVERPRINTED either. The title used to be one
        # fixed-length string, so a clock hard-coded at x = 0.30 always
        # cleared it; now it carries the plane or variant the selection has
        # collapsed, and "· Quantum, non-relativistic" runs well past that.
        # Measured once, off the static artist, so the clock still sits at a
        # constant x for the whole export.
        self.time_text = self.fig.text(max(0.30, self._text_right(title) + 0.02),
                                       0.972, "", color=self.pal["clock"],
                                       fontsize=13, va="center",
                                       family="DejaVu Sans Mono")
        # right-anchored: the per-record geometry line is long and would run
        # off the canvas at 720p
        self.geom_text = self.fig.text(0.988, 0.972, "", color=self.pal["muted"],
                                       fontsize=9, va="center", ha="right",
                                       family="DejaVu Sans Mono")

        # ---- W panels (left block). The colour scales are per (variant,
        # plane) and fixed for the whole video, so one shared bar would be
        # wrong and a per-frame redraw unnecessary.
        self.images = []
        n_cells = len(cells)
        compact = n_cells > self.CBAR_MAX_CELLS
        # A colorbar is carved out of its own panel, so the gap between
        # columns has to hold BOTH it and the next panel's y tick labels and
        # ylabel. At two columns 0.30 always did; at three the phase
        # portrait's "py (a.u.)" was drawn straight over its left neighbour's
        # bar. Without bars (compact) there is nothing to clear.
        wspace = 0.30 if (cols <= 2 or compact) else 0.62
        gs = self.fig.add_gridspec(rows, cols, left=PANEL_LEFT,
                                   right=panel_right, bottom=PLOT_BOTTOM,
                                   top=PLOT_TOP, wspace=wspace, hspace=0.34)
        tfs = 9.0 if n_cells <= 4 else 8.0 if n_cells <= 8 else 6.5
        lfs = 8.0 if n_cells <= 8 else 6.0
        for i, cell in enumerate(cells):
            r, c = (i//cols, i % cols) if (len(self.planes) == 1
                                           or len(self.variants) == 1) \
                else (self.planes.index(cell.plane), cell.vi)
            axp = self.fig.add_subplot(gs[r, c])
            label, color, _ = variant_style(cell.key, self.theme)
            # the panel grid is drawn on top
            _style_axes(axp, self.pal, grid=False, labelsize=6.5 if compact
                        else 7)
            s = stats.scale.get((cell.key, cell.plane), 1.0)
            # Past CBAR_MAX_CELLS there is no colorbar, so the panel's own
            # title carries its scale. It goes in the TITLE and not in a
            # corner annotation over the heatmap, which is where it started:
            # the per-frame blit re-draws the image on top of the restored
            # background, so anything static inside the axes box is painted
            # over — the same reason the grid lines have to be dynamic. Making
            # this dynamic instead would cost 24 text renders a frame (~2.2 ms
            # each measured), which is most of a 24-panel frame's budget, for
            # a caption that never changes.
            self._title_panel(axp, cell, label, color, tfs,
                              "±%.2g" % s if compact else None)
            a, b = cell.plane
            axp.set_xlabel(axis_label(self.ndim, a, compact),
                           color=self.pal["muted"], fontsize=lfs)
            axp.set_ylabel(axis_label(self.ndim, b, compact),
                           color=self.pal["muted"], fontsize=lfs)
            # extent/limits are placeholders: the first update() installs the
            # first record's own window (see _apply_geom)
            ax1, ax2 = stats.extent(a)
            bx1, bx2 = stats.extent(b)
            im = axp.imshow(numpy.zeros((2, 2), dtype=numpy.float32),
                            origin="lower", cmap="bwr", vmin=-s, vmax=s,
                            extent=(ax1, ax2, bx1, bx2),
                            aspect="auto", interpolation="antialiased")
            axp.set_xlim(ax1, ax2)
            axp.set_ylim(bx1, bx2)
            if not compact:
                cb = self.fig.colorbar(im, ax=axp, fraction=0.05, pad=0.02)
                cb.ax.tick_params(colors=self.pal["muted"], labelsize=6.5)
                cb.outline.set_edgecolor(self.pal["grid"])
                # "%.2g" instead of matplotlib's default, which factors a
                # common power out into an OFFSET TEXT above the bar — and a
                # 2D plane scale is routinely ~1e-5, so every panel grew a
                # stray "1e-5" that the neighbouring axes then clipped to a
                # bare digit. Two significant figures is all a colour scale
                # can be read to anyway.
                cb.ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda v, _pos: "%.2g" % v))
            self.images.append((axp, im, cell))

        # ---- diagnostics column(s), same order as PlotsColumn
        self.marg_axes, self.marg_lines = {}, {}
        self.series_axes, self.cursors = {}, []
        t = numpy.asarray(stats.t, dtype=float)
        for j, (left, right) in enumerate(diag_cols):
            ids = self.diagnostics[j*diag_rows:(j + 1)*diag_rows]
            if not ids:
                continue
            gsr = self.fig.add_gridspec(diag_rows, 1, left=left, right=right,
                                        bottom=PLOT_BOTTOM, top=PLOT_TOP,
                                        hspace=0.75)
            last_series = None
            for row, pid in enumerate(ids):
                axd = self.fig.add_subplot(gsr[row])
                a = _marg_axis(pid)
                if a is not None:
                    self._build_marginal(axd, a)
                else:
                    self._build_series(axd, pid, t)
                    last_series = axd
            if last_series is not None:
                last_series.set_xlabel("t (a.u.)", color=self.pal["muted"],
                                       fontsize=8)

        # ---- metadata block: everything needed to reproduce the run.
        # It grows downward from a fixed anchor with nothing stopping it, so it
        # used to run off the bottom of the figure in silence. Measured at 16:9
        # (the figure is always 19.2x10.8 in — dpi carries the resolution):
        # 11 lines fit, and a 4-variant cat run with 4 live parameter changes is
        # already 10 in float64, 11 with the float32 PREVIEW line. Fit it
        # instead of clipping it — one size for both columns so they stay
        # visually matched.
        left_col, right_col = meta_lines
        fs = self._meta_fontsize(max(len(left_col), len(right_col)))
        left_col = self._meta_fit(left_col, fs)
        right_col = self._meta_fit(right_col, fs)
        for x, col in ((0.012, left_col), (0.335, right_col)):
            self.fig.text(x, self.META_TOP, "\n".join(col),
                          color=self.pal["muted"], fontsize=fs, va="top",
                          linespacing=self.META_LINESPACING,
                          family="DejaVu Sans")

        self._geom = None
        self._abscissa = {}
        self.panel_grid = []
        # Blitting: everything except the artists collected in _dynamic is
        # STATIC (the series are drawn once, the metadata never changes), so
        # a full redraw per frame would spend ~4/5 of its time on ticks,
        # fonts and curves nobody is animating. The dynamic artists are
        # marked animated, the static background is captured once, and each
        # frame restores it and re-draws only the images, marginals, cursors
        # and the two header texts. _apply_geom() re-captures the background
        # whenever the record geometry (hence the ticks) changes.
        self._rebuild_dynamic()
        self.canvas.draw()      # static background, laid out once
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)

    # ------------------------------------------------------------------
    def _text_right(self, artist):
        """Right edge of a laid-out text artist, in figure coordinates."""
        try:
            bb = artist.get_window_extent(self.canvas.get_renderer())
            return bb.x1/float(self.fig.bbox.width)
        except Exception:      # no renderer yet on some backends — fall back
            return 0.0

    def _cells(self):
        """The panel list: the cartesian product of planes and variants, in
        plane-major order so a matrix reads row by row."""
        out = []
        for plane in self.planes:
            pi = ax.plane_index(self.ndim, plane)
            for vi, key in enumerate(self.variants):
                out.append(_Cell(plane=plane, pi=pi, vi=vi, key=key))
        return out

    def _title_panel(self, axp, cell, label, color, fontsize, scale_note=None):
        """Name the panel by whichever of (variant, plane) actually varies.
        matplotlib keeps a separate artist per title `loc`, which is what lets
        a two-part title carry one part in the variant's own colour.

        `scale_note` is the panel's fixed colour range, present only where
        there was no room for a colorbar — see where it is built."""
        if len(self.planes) == 1:
            # 1D, and the "compare variants" reading: the plane is named once
            # in the header, so the panel is the variant — unchanged.
            axp.set_title(label, color=color, fontsize=fontsize, pad=4)
            return
        if len(self.variants) == 1:
            axp.set_title(ax.plane_label(self.ndim, cell.plane, math=True),
                          color=self.pal["fg"], fontsize=fontsize, pad=4,
                          loc="left")
            right = ax.plane_title(self.ndim, cell.plane, math=True)
        else:
            axp.set_title(cell.key.upper(), color=color, fontsize=fontsize,
                          pad=4, loc="left")
            right = ax.plane_label(self.ndim, cell.plane, math=True)
        if scale_note:
            right = "%s  %s" % (right, scale_note)
        axp.set_title(right, color=self.pal["muted"],
                      fontsize=fontsize - 1.0, pad=4, loc="right")

    def _build_marginal(self, axd, a):
        _style_axes(axd, self.pal, ax.marginal_title(self.ndim, a, math=True),
                    grid=self.show_grid)
        self.marg_axes[a] = axd
        for key in self.variants:
            _, color, dash = variant_style(key, self.theme)
            self.marg_lines[(a, key)] = axd.plot([], [], color=color,
                                                 linestyle=dash, lw=1.2)[0]
        a1, a2 = self.stats.extent(a)
        axd.set_xlim(a1, a2)
        top = self.stats.amplitude(a)
        axd.set_ylim(min(0.0, -0.02*top), 1.08*top or 1)

    def _build_series(self, axd, pid, t):
        """Drawn ONCE for the whole exported range; a cursor moves over it."""
        _style_axes(axd, self.pal, diagnostic_title(self.ndim, pid, math=True),
                    title_loc="right", grid=self.show_grid)
        self.series_axes[pid] = axd
        for key in self.variants:
            _, color, dash = variant_style(key, self.theme)
            axd.plot(t, numpy.asarray(self.stats.series(pid, key),
                                      dtype=float),
                     color=color, linestyle=dash, lw=1.2)
        if t.size:
            axd.set_xlim(min(t[0], t[-1]), max(t[0], t[-1]))
        # same y-window and tick labels as the SPA (see series_ylim)
        axd.set_ylim(*series_ylim([v for key in self.variants
                                   for v in self.stats.series(pid, key)]))
        axd.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _pos, axd=axd: "%.*f"
            % (_tick_decimals(list(axd.get_yticks())), v)))
        self.cursors.append(axd.axvline(t[0] if t.size else 0.0,
                                        color=self.pal["cursor"],
                                        lw=1.0, alpha=0.9))

    # ------------------------------------------------------------------
    def _meta_lines_that_fit(self, fontsize):
        """How many lines of `fontsize` fit between META_TOP and the bottom
        edge. Derived from the figure's own height rather than hard-coded, so a
        non-16:9 export (a taller figure, hence more room) keeps the full
        size instead of being shrunk to a 16:9 budget."""
        advance = fontsize*self.META_LINESPACING/72.0        # inches
        return int(self.META_TOP*self.fig.get_figheight()/advance)

    def _meta_fontsize(self, nlines):
        """The largest size at or below META_FONTSIZE that fits `nlines`,
        floored at META_MIN_FONTSIZE (past which _meta_fit elides instead)."""
        if nlines <= 0:
            return self.META_FONTSIZE
        budget = self.META_TOP*self.fig.get_figheight()*72.0/self.META_LINESPACING
        return max(self.META_MIN_FONTSIZE,
                   min(self.META_FONTSIZE, budget/nlines))

    def _meta_fit(self, lines, fontsize):
        """Truncate a column that does not fit even at the minimum size, and
        SAY so — every one of these facts is also in the mp4's `comment` tag
        (describe.config_json), so the pointer is real and not an apology."""
        room = self._meta_lines_that_fit(fontsize)
        if len(lines) <= room or room < 1:
            return lines
        return list(lines[:room - 1]) + [
            "… +%d more lines — full detail in the mp4 comment tag"
            % (len(lines) - room + 1)]

    def _rebuild_dynamic(self):
        """order IS draw order: the panel grid comes after the images so it
        stays visible over the heatmap, exactly as GridOverlay.vue sits above
        the WebGL canvas in the SPA."""
        self._dynamic = ([im for _ax, im, _c in self.images] + self.panel_grid
                         + list(self.marg_lines.values())
                         + self.cursors + [self.time_text, self.geom_text])
        for a in self._dynamic:
            a.set_animated(True)

    def _apply_geom(self, geom):
        """Adopt one record's window: the domain is a PER-RECORD fact
        (auto-expand regrids) and the video follows it exactly as the SPA
        does — a frame from before an expansion must still fill its panel.

        Axis limits live in the blit BACKGROUND (ticks, labels, spines), so
        this costs one full redraw + re-capture. That happens once per
        regrid, a handful of times per export."""
        self._geom = geom
        self._abscissa = {a: axis_of(geom.lo[a], geom.hi[a], geom.N[a])
                          for a in range(ax.n_axes(geom.ndim))}
        for axp, im, cell in self.images:
            a, b = cell.plane
            im.set_extent((geom.lo[a], geom.hi[a], geom.lo[b], geom.hi[b]))
            axp.set_xlim(geom.lo[a], geom.hi[a])
            axp.set_ylim(geom.lo[b], geom.hi[b])
        # only the abscissae move: the marginal amplitude scale stays
        # export-wide (a per-frame y would make curve heights incomparable),
        # like the per-panel colour scale
        for a, axd in self.marg_axes.items():
            axd.set_xlim(geom.lo[a], geom.hi[a])
        for ln in self.panel_grid:
            ln.remove()
        self.panel_grid = []
        self.canvas.draw()      # lays out the new ticks (read back below)
        if self.show_grid:
            # W-panel grid: matplotlib draws the axes grid UNDER the image,
            # so these are explicit lines re-drawn over it every frame
            for axp, _im, cell in self.images:
                a, b = cell.plane
                for v in axp.get_xticks():
                    if geom.lo[a] <= v <= geom.hi[a]:
                        self.panel_grid.append(axp.axvline(
                            v, color=PANEL_GRIDC, lw=0.8,
                            alpha=(PANEL_ZERO_ALPHA if v == 0
                                   else PANEL_GRID_ALPHA)))
                for v in axp.get_yticks():
                    if geom.lo[b] <= v <= geom.hi[b]:
                        self.panel_grid.append(axp.axhline(
                            v, color=PANEL_GRIDC, lw=0.8,
                            alpha=(PANEL_ZERO_ALPHA if v == 0
                                   else PANEL_GRID_ALPHA)))
        self._rebuild_dynamic()   # animated: absent from the capture below
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)

    # ------------------------------------------------------------------
    def update(self, k, t, geom, vframes, k0, k1):
        """Paint one record; returns its RGBA bytes (a view of the Agg
        buffer — write it to the encoder before the next update)."""
        # 3 decimals on BOTH numbers, as the control bar shows them
        self.time_text.set_text("t = %*.3f a.u.  (%*.3f fs)"
                                % (self._tw, t, self._fw, t*AU_TIME_FS))
        self.geom_text.set_text("record %*d ∈ [%d, %d]   %s"
                                % (len(str(k1)), k, k0, k1,
                                   geom_line(geom, math=True)))
        if geom != self._geom:      # first record, or an auto-expand regrid
            self._apply_geom(geom)
        for _axp, im, cell in self.images:
            im.set_data(dequantize_plane(vframes[cell.vi].planes[cell.pi]))
        for (a, key), line in self.marg_lines.items():
            line.set_data(self._abscissa[a],
                          vframes[self.variants.index(key)].marg[a])
        for c in self.cursors:
            c.set_xdata([t, t])
        self.canvas.restore_region(self._bg)
        for a in self._dynamic:
            a.axes.draw_artist(a) if a.axes is not None \
                else self.fig.draw_artist(a)
        # RGBA straight out of the Agg buffer: ffmpeg is fed rgba rawvideo,
        # so no per-frame RGB repack (6 MiB/frame of pure copying at 1080p)
        return self.canvas.buffer_rgba()

    def close(self):
        self.fig.clf()


def meta_columns(cfg, geom, stats, variants, k0, k1, n_frames, fps,
                 param_log=(), planes=None, diagnostics=None):
    """The two text columns of the metadata block (left: what this video is;
    right: the physics + IC expression, wrapped). `geom` is the geometry of
    the FIRST exported record; auto-expand may move it later — the plots
    follow each record, so the widest window is quoted separately.

    `planes`/`diagnostics` are what the frame actually SHOWS. A 2D export is
    routinely a subset of the record, and a viewer must be able to tell a
    subset from the whole thing."""
    from . import describe
    nd = geom.ndim
    pls = [tuple(p) for p in (planes if planes is not None else ax.planes(nd))]
    diags = list(diagnostics if diagnostics is not None
                 else diagnostics_default(nd))
    # Each line is built TWICE, plain and typeset, rather than the typeset one
    # being derived by substitution from the other. Most of them differ only in
    # the axis names and a blanket pass would do — but "ΔX·ΔPx" carries a
    # CAPITAL Px that is not an axis label, so a substitution pass left it
    # plain beside a subscripted "ρ(px)" three words away. Generating both from
    # the same sources keeps every name in one spelling.
    def both(fmt, *plain_args, math_args=None):
        rows.append((fmt % plain_args,
                     fmt % (math_args if math_args is not None else plain_args)))

    rows = []
    both("variants: %s", ", ".join(k.upper() for k in variants))
    if nd > 1:
        counts = (len(pls), _s(len(pls)), len(variants), _s(len(variants)),
                  len(pls)*len(variants))
        both("panels: %s  " + _nb("(%d plane%s × %d variant%s = %d)"),
             ", ".join(ax.plane_label(nd, p) for p in pls), *counts,
             math_args=(", ".join(ax.plane_label(nd, p, math=True)
                                  for p in pls),) + counts)
    both("records %d … %d  →  %d frames @ %g fps  " + _nb("(%.1f s)"),
         k0, k1, n_frames, fps, n_frames/float(fps))
    sizes = "×".join(str(n) for n in geom.N)
    both("grid at record %d: %s;  %s", k0, sizes,
         _extents(nd, geom.lo, geom.hi),
         math_args=(k0, sizes, _extents(nd, geom.lo, geom.hi, math=True)))
    if tuple(stats.lo) != tuple(geom.lo) or tuple(stats.hi) != tuple(geom.hi):
        both("axes follow each record (auto-expand); widest: %s",
             _extents(nd, stats.lo, stats.hi),
             math_args=(_extents(nd, stats.lo, stats.hi, math=True),))
    omitted = [i for i in diagnostics_available(nd) if i not in diags]
    if omitted:
        both("plots omitted: %s",
             ", ".join(diagnostic_label(nd, i) for i in omitted),
             math_args=(", ".join(diagnostic_label(nd, i, math=True)
                                  for i in omitted),))
    both(_nb("units: Hartree atomic (ℏ = mₑ = e = 1);") + "  "
         + _nb("1 a.u. of time = %g fs"), AU_TIME_FS)
    left = _emit([p for p, _m in rows], [m for _p, m in rows], 62, nd)
    # the IC was built at session creation: its σp (derived for cat states)
    # follows the ORIGINAL ℏ, not whatever a live change left behind
    hbar0 = describe.state_at(cfg, param_log, -1).get("hbar_eff", cfg.hbar_eff)
    right_plain = (describe.param_lines(cfg, param_log, k0, k1)
                   + describe.ic_expression(cfg.ic, hbar0, ndim=nd))
    right_math = (describe.param_lines(cfg, param_log, k0, k1, math=True)
                  + describe.ic_expression(cfg.ic, hbar0, ndim=nd,
                                          math=True))
    right = _emit(right_plain, right_math, 150, nd)
    return left, right


def _s(n):
    """Plural suffix — a phase portrait is "6 planes × 1 variant", not
    "1 variants"."""
    return "" if n == 1 else "s"


# A space `textwrap` must not break at. "px ∈ [-7, 7]" is ONE fact and was
# being split across two lines between the axis name and the ∈; so were
# "(6 planes × 1 variant = 6)" and the units figure. textwrap splits on \s+,
# and a Unicode NBSP is \s too, so it has to be a character textwrap cannot
# see as whitespace at all. _emit turns them back into spaces after wrapping,
# which is the only path metadata lines take to the figure.
_NB = "\x00"


def _nb(text):
    return text.replace(" ", _NB)


def _extents(ndim, lo, hi, math=False):
    """"x ∈ [-6, 6], p ∈ [-7, 7]" — every axis, for the metadata block (which
    has room to spell them all out; the header's geom_line groups them).

    Each axis group is atomic: a line may break BETWEEN groups, never inside
    one."""
    ls = [ax.sub_math(l) for l in ax.labels(ndim)] if math else ax.labels(ndim)
    return ", ".join(_nb("%s ∈ [%.6g, %.6g]" % (ls[a], lo[a], hi[a]))
                     for a in range(ax.n_axes(ndim)))


_MATH_SEGMENT = None


def mathtext_ok(s):
    """Would matplotlib render every $...$ segment of `s`?

    The metadata block is the one place where a typeset line is assembled from
    USER input (the potential), so the formatting must not be the only line of
    defence: a mathtext parse error raises at draw time and would take the
    whole export down. Cheap — the parser caches, and this runs once per line
    per figure, not per frame."""
    global _MATH_SEGMENT
    if "$" not in s:
        return True
    import re

    from matplotlib.mathtext import MathTextParser
    if _MATH_SEGMENT is None:
        _MATH_SEGMENT = (re.compile(r"\$([^$]*)\$"), MathTextParser("agg"))
    pat, parser = _MATH_SEGMENT
    segs = pat.findall(s)
    if not segs or s.count("$") % 2:
        return False           # an unpaired $ would swallow the rest of the line
    try:
        for seg in segs:
            parser.parse("$%s$" % seg)
    except Exception:
        return False
    return True


def _emit(plain_lines, math_lines, width, ndim):
    """Lay out one metadata column: wrap each logical line PLAIN, then take
    its typeset twin where that is safe.

    Wrapping plain is the load-bearing part. "$p_x$" is five characters that
    draw as two glyphs, so letting the typeset form into `textwrap` would make
    every column width a guess — which characters land on which line has to be
    decided by the plain text. A logical line that fit without breaking can
    then be swapped wholesale for its typeset twin; one that had to be broken
    keeps its fragments and gets only the per-token axis-name substitution,
    which is valid on any fragment (unlike U(x,y), whose typeset form is a
    single $...$ span that a line break would split).

    Every candidate is checked against matplotlib's own parser and falls back
    to plain, because this column is the one place a typeset line is built
    from user input."""
    out = []
    for plain, math in zip(plain_lines, math_lines):
        frags = _wrap([plain], width)
        if math is None:
            out.extend(frags)
        elif len(frags) == 1 and mathtext_ok(math):
            out.append(math)
        else:
            for f in frags:
                sub = ax.sub_math_text(f, ndim)
                out.append(sub if mathtext_ok(sub) else f)
    # the no-break markers have done their job; nothing downstream should ever
    # see one, and this is the only path a metadata line takes to the figure
    return [l.replace(_NB, " ") for l in out]


def _wrap(lines, width):
    # break_on_hyphens=False, because a minus sign is not a hyphen and this
    # column is full of them. textwrap's default treats "exp(-x^2/2)" as
    # hyphenated and will break it right after the minus, so an IC expression
    # came out as "∝ exp(-" / "x^2/2)*(0*x^0+…" — three characters on one line,
    # the expression severed at an operator, and the block's whole job (the text
    # you paste back into the IC box) broken. It costs lines too: the same
    # expression took 5 physical lines instead of 4.
    #
    # This is the same class of damage _NB exists to prevent — a break INSIDE
    # one fact — and _NB cannot cover it, since _NB only protects spaces. It
    # applies to the whole block rather than the expression lines alone because
    # every other hyphen here is also a minus or a compound term: "[-7, 7]",
    # "1e-5", "single-precision". None of them should ever be split.
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width, subsequent_indent="    ",
                                 break_on_hyphens=False) or [""])
    return out


def variant_keys(vframes):
    """Bundle order -> variant keys, validated against the known set."""
    keys = [key_of_vid(vf.vid) for vf in vframes]
    for k in keys:
        if k not in VARIANTS:
            raise ValueError("unknown variant id in record: %r" % k)
    return keys
