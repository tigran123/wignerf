"""
Axis bookkeeping for 1D and 2D space — the single source of truth for what the
phase-space axes ARE, how they are ordered in an array, which 2D planes get
reduced out of the state, and how all of that is spelled on screen.

ARRAY AXIS ORDER is (q_0 ... q_{ndim-1}, k_0 ... k_{ndim-1}): all spatial axes
first, then all momentum axes. At ndim=1 that is (x, p) — the existing layout,
unchanged. At ndim=2 it is (x, y, px, py).

CONJUGATION is index-matched: theta_i is conjugate to k_i and lives on array
axis ndim+i; lam_i is conjugate to q_i and lives on array axis i. Getting this
pairing wrong is the classic multi-D error, so it is stated once here and
derived everywhere else from `conjugate()`.

PLANES. A 4D W cannot be displayed or streamed, so the state is reduced on the
device to 2D planes: a plane (a, b) is the sum of W over every OTHER axis,
times their cell measures. At ndim=2 the canonical set is all six pairs; at
ndim=1 the complement of (0, 1) is empty, so the single "plane" IS W itself and
1D falls out of the general case with no special-casing and no copy.

The plane/marginal/purity title strings are mirrored by
frontend/src/lib/axes.ts and by core/render_mpl.py — the video has to read like
the screen (see the mp4-export bullet in CLAUDE.md), so change a string in one
place and change it in all three.
"""

NDIMS = (1, 2)

# Axis labels in array order. The 1D momentum axis is "p", not "px": a 1D run
# has no second dimension to distinguish it from, and every existing label,
# tooltip and exported frame says "p".
LABELS = {
    1: ("x", "p"),
    2: ("x", "y", "px", "py"),
}

# Canonical plane set per ndim, in display order. (x,y) and (px,py) come first
# because they are the real spatial/momentum densities; then the two
# same-dimension reduced Wigner functions; then the two mixed ones (which are
# also what <Lz> is computed from, so they are never optional).
PLANES = {
    1: ((0, 1),),
    2: ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)),
}

# Per-plane reduction mode, carried as a byte on the wire. Only projection is
# defined; CUTS (a slice at fixed values of the complementary pair, which keeps
# the interference fringes a projection averages away) are milestone M5 in
# CLAUDE.md and are purely additive here — a new mode value, no version bump.
MODE_PROJECTION = 0


def check_ndim(ndim):
    if ndim not in NDIMS:
        raise ValueError("ndim must be one of %s (got %r)"
                         % (", ".join(map(str, NDIMS)), ndim))
    return int(ndim)


def n_axes(ndim):
    """Number of phase-space axes = 2*ndim."""
    return 2*ndim


def spatial_axes(ndim):
    return tuple(range(ndim))


def momentum_axes(ndim):
    return tuple(range(ndim, 2*ndim))


def is_momentum(ndim, axis):
    return axis >= ndim


def conjugate(ndim, axis):
    """The array axis whose Fourier dual multiplies this one: theta_i (dual of
    k_i) shifts q_i, and lam_i (dual of q_i) shifts k_i. So the spectral
    half-width of the Bopp range on axis `axis` is set by the cell size of
    `conjugate(ndim, axis)`."""
    return axis - ndim if is_momentum(ndim, axis) else axis + ndim


def labels(ndim):
    return LABELS[ndim]


def label(ndim, axis):
    return LABELS[ndim][axis]


def planes(ndim):
    return PLANES[ndim]


def complement(ndim, plane):
    """The axes a plane is reduced OVER — empty at ndim=1."""
    return tuple(a for a in range(2*ndim) if a not in plane)


def plane_index(ndim, plane):
    return PLANES[ndim].index(tuple(plane))


# ---------------------------------------------------------------------------
# Display strings (mirrored in frontend/src/lib/axes.ts and core/render_mpl.py)
#
# Every title function takes an optional `math` flag, the exact counterpart of
# the `html` flag its frontend mirror takes: it typesets the two-letter axis
# names with a REAL subscript, "px" -> "$p_x$", for matplotlib's mathtext.
# Nothing else changes — the integrals, gamma, hbar, rho, phi and the angle
# brackets stay the Unicode characters they are in the plain form, so the
# exported frame and the screen use identical glyphs and 1D (whose axis names
# are one letter each) is untouched at the byte level.
#
# Measured 2026-07-28, because this file used to say mathtext was too slow to
# use: on the real export figure it costs 1.83x per TEXT ARTIST but those are
# all STATIC artists baked into the blit background, so a 24-panel 2D frame
# pays +147 ms ONCE per figure and 1.00-1.02x per frame — i.e. nothing. See
# the mp4-export gotcha in CLAUDE.md. usetex is 12x and needs a LaTeX install
# plus a third spelling of every string, so it is not used.
# ---------------------------------------------------------------------------

def sub_math(name):
    """"px" -> "$p_x$" for matplotlib mathtext; one-letter names are returned
    unchanged (no math mode, hence no font switch, for "x" or "p").

    The mirror of lib/axes.ts subHtml, which does the same job with <sub>.
    Every axis name is one letter plus an optional one-letter subscript, which
    is the whole rule. Unicode cannot do this: the subscript block has ₓ
    (U+2093) but NO subscript y, so "pₓ" beside "py" would look worse than
    plain text."""
    return "$%s_%s$" % (name[0], name[1]) if len(name) == 2 else name


def _lbl(ndim, axis, math):
    return sub_math(label(ndim, axis)) if math else label(ndim, axis)


def sub_math_text(text, ndim):
    """Typeset every whole-word axis name inside a line of otherwise plain
    text: "px ∈ [-8, 8]" -> "$p_x$ ∈ [-8, 8]".

    For the metadata block and the header readout, which are assembled from
    many sources (including describe.py, which stays matplotlib-free) rather
    than built out of the title functions above. A no-op at ndim=1, where every
    axis name is one letter — so a 1D frame is byte-identical.

    Word boundaries matter both ways: it must catch "px" after a space, a comma
    or a "(", and must not fire inside a longer identifier."""
    import re
    names = [l for l in labels(ndim) if len(l) == 2]
    if not names:
        return text
    return re.sub(r"\b(%s)\b" % "|".join(names),
                  lambda m: sub_math(m.group(1)), text)


def plane_label(ndim, plane, math=False):
    """Short chip for a panel corner: "x,p" / "x,py"."""
    a, b = plane
    return "%s,%s" % (_lbl(ndim, a, math), _lbl(ndim, b, math))


def _measure(ndim, over, math=False):
    return " ".join("d" + _lbl(ndim, a, math) for a in over)


def plane_title(ndim, plane, math=False):
    """What the plane IS. Empty at ndim=1: the single plane is W itself, and
    the panel's title there is the variant name, exactly as today."""
    over = complement(ndim, plane)
    if not over:
        return ""
    sign = "∬" if len(over) == 2 else "∫"   # double / single integral
    return "%sW %s" % (sign, _measure(ndim, over, math))


def marginal_title(ndim, axis, math=False):
    """rho(x) = int W dp   /   phi(py) = int W dx dy dpx"""
    sym = "φ" if is_momentum(ndim, axis) else "ρ"
    over = tuple(a for a in range(2*ndim) if a != axis)
    return "%s(%s) = ∫W %s" % (sym, _lbl(ndim, axis, math),
                               _measure(ndim, over, math))


def purity_expr(ndim, math=False):
    """(2 pi hbar)^ndim * int W^2 over the whole phase space, as read."""
    pre = "2πℏ" if ndim == 1 else "(2πℏ)²"
    sign = "∬" if ndim == 1 else "⨌"        # double / quadruple
    ds = "".join("d" + _lbl(ndim, a, math) for a in range(2*ndim))
    return "%s%sW²%s" % (pre, sign, ds)


def purity_title(ndim, math=False):
    return "purity γ(t) = %s" % purity_expr(ndim, math)


def uncertainty_title(ndim, dim, math=False):
    """DeltaX*DeltaP(t) at ndim=1; DeltaX*DeltaPx(t) / DeltaY*DeltaPy(t) at 2."""
    q, k = label(ndim, dim), label(ndim, ndim + dim)
    kk = k.capitalize()
    return "Δ%s·Δ%s(t)" % (q.upper(), sub_math(kk) if math else kk)


def lz_title(math=False):
    """<Lz>(t) — the 2D-only angular-momentum series. No ndim argument: there
    is no 1D reading of it (observables.lz is identically 0.0 there), and the
    axis names it quotes are the ndim=2 ones by construction."""
    lz = sub_math("Lz") if math else "Lz"
    px = sub_math("px") if math else "px"
    py = sub_math("py") if math else "py"
    return "⟨%s⟩(t) = ⟨x·%s − y·%s⟩" % (lz, py, px)
