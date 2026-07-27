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
# ---------------------------------------------------------------------------

def plane_label(ndim, plane):
    """Short chip for a panel corner: "x,p" / "x,py"."""
    a, b = plane
    return "%s,%s" % (label(ndim, a), label(ndim, b))


def _measure(ndim, over):
    return " ".join("d" + label(ndim, a) for a in over)


def plane_title(ndim, plane):
    """What the plane IS. Empty at ndim=1: the single plane is W itself, and
    the panel's title there is the variant name, exactly as today."""
    over = complement(ndim, plane)
    if not over:
        return ""
    sign = "∬" if len(over) == 2 else "∫"   # double / single integral
    return "%sW %s" % (sign, _measure(ndim, over))


def marginal_title(ndim, axis):
    """rho(x) = int W dp   /   phi(py) = int W dx dy dpx"""
    sym = "φ" if is_momentum(ndim, axis) else "ρ"
    over = tuple(a for a in range(2*ndim) if a != axis)
    return "%s(%s) = ∫W %s" % (sym, label(ndim, axis), _measure(ndim, over))


def purity_expr(ndim):
    """(2 pi hbar)^ndim * int W^2 over the whole phase space, as read."""
    pre = "2πℏ" if ndim == 1 else "(2πℏ)²"
    sign = "∬" if ndim == 1 else "⨌"        # double / quadruple
    return "%s%sW²%s" % (pre, sign, "".join("d" + l for l in labels(ndim)))


def purity_title(ndim):
    return "purity γ(t) = %s" % purity_expr(ndim)


def uncertainty_title(ndim, dim):
    """DeltaX*DeltaP(t) at ndim=1; DeltaX*DeltaPx(t) / DeltaY*DeltaPy(t) at 2."""
    q, k = label(ndim, dim), label(ndim, ndim + dim)
    return "Δ%s·Δ%s(t)" % (q.upper(), k.capitalize())
