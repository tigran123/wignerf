"""
Phase-space grid and Fourier-conjugate meshes, generic over ndim (port of
dynamics/solve.py:82-100, generalized 2026-07-26).

An `Axis` describes one phase-space axis; a `Grid` holds 2*ndim of them in the
array order `core.axes` defines — all spatial axes, then all momentum axes. At
ndim=1 that is (x, p) and every array below has exactly the shape it had
before this module was generalized.

Conventions:
- coordinate vectors exclude the right endpoint (periodic domain).
- `v[a]` is the natural-order 1D device vector of axis a (previews, marginal
  axes, moments).
- `C[a]` is the fftshifted coordinate mesh of axis a, and `D[a]` the fftshifted
  mesh of its Fourier DUAL, both broadcast-shaped (1 everywhere but axis a).
  `Q`/`K` are the spatial/momentum halves of `C`; `Lam`/`Theta` the spatial/
  momentum halves of `D`. Propagator state W is kept fftshifted along ALL
  axes; the unshift for display happens in the frontend shader as a
  half-period texture offset (requires every N even).
- the dual of axis a has amplitude pi/d[a], so `dual_amp[a] == pi/d[a]`. theta_i
  (the dual of k_i, which shifts q_i) therefore has amplitude pi/d[k_i] — the
  index-matched pairing lives in `axes.conjugate`, never spelled out again.
- the quantum propagator evaluates U/T on the extended REAL range
  `extended(hbar_eff)[a]`, whose half-width is hbar_eff*dual_amp[conjugate(a)]/2.
"""

import warnings
from dataclasses import dataclass, replace
from math import prod

from numpy import pi

from . import axes as ax


@dataclass(frozen=True)
class Axis:
    """One phase-space axis. `d` overrides the derived cell size: an
    auto-expand regrid keeps the ORIGINAL lattice spacing bitwise-frozen
    (extents move by whole cells, so re-deriving (hi-lo)/n could differ by an
    ulp) — pass GridState.d there; the default derivation is unchanged for
    session-start grids. `anchor` = (a0, offset_cells): materialize lattice
    points as a0 + d*(offset + i), the same float expression for the same
    GLOBAL cell in every window, so overlap points are bitwise-identical
    across regrids (an offset of 0 reduces to the default lo + d*i exactly)."""
    lo: float
    hi: float
    n: int
    d: float = None
    anchor: tuple = None

    @property
    def cell(self):
        return float(self.d) if self.d is not None else (self.hi - self.lo)/self.n


class Grid:
    def __init__(self, axes, backend):
        axes = tuple(axes)
        if len(axes) % 2:
            raise ValueError("a phase-space grid needs an even number of axes "
                             "(ndim spatial + ndim momentum), got %d" % len(axes))
        self.ndim = ax.check_ndim(len(axes)//2)
        self.labels = ax.labels(self.ndim)
        self.spatial_axes = ax.spatial_axes(self.ndim)
        self.momentum_axes = ax.momentum_axes(self.ndim)
        self.n_axes = len(axes)

        for a, axis in enumerate(axes):
            name = self.labels[a]
            if not axis.hi > axis.lo:
                raise ValueError("axis %s: require hi > lo" % name)
            if axis.n < 2 or axis.n % 2:
                raise ValueError("N%s must be even and >= 2" % name)
            if axis.n & (axis.n - 1):
                warnings.warn("N%s=%d is not a power of 2, FFT may be slowed "
                              "down" % (name, axis.n))

        self.backend = backend
        xp = backend.xp
        self.axes = axes
        self.N = self.shape = tuple(int(a.n) for a in axes)
        self.lo = tuple(float(a.lo) for a in axes)
        self.hi = tuple(float(a.hi) for a in axes)
        self.d = tuple(a.cell for a in axes)
        self.dV = prod(self.d)

        v, dual, amp = [], [], []
        for a, axis in enumerate(axes):
            i = xp.arange(axis.n, dtype=xp.float64)
            if axis.anchor is not None:
                a0, off = axis.anchor
                v.append(float(a0) + self.d[a]*(float(off) + i))
            else:
                v.append(self.lo[a] + self.d[a]*i)
            # spectral span from n*d (== the extent span up to an ulp) when the
            # cell size is FROZEN, so the conjugate amplitude pi/d is invariant
            # across regrids
            span = axis.n*self.d[a] if axis.d is not None else self.hi[a] - self.lo[a]
            dd = 2.*pi/span
            amp.append(dd*axis.n/2.)              # == pi/d[a]
            dual.append(-amp[a] + dd*i)
        self.v = tuple(v)
        self.dual_amp = tuple(amp)

        self.C = tuple(backend.fftshift(v[a]).reshape(self._bshape(a))
                       for a in range(self.n_axes))
        self.D = tuple(backend.fftshift(dual[a]).reshape(self._bshape(a))
                       for a in range(self.n_axes))
        self.Q, self.K = self.C[:self.ndim], self.C[self.ndim:]
        self.Lam, self.Theta = self.D[:self.ndim], self.D[self.ndim:]
        self._v_host = None

    @property
    def v_host(self):
        """`v` on the host — what the per-record moment sums weight the natural-
        order marginals with. Cached because a grid outlives thousands of
        records and this would otherwise be 2*ndim device syncs per record."""
        if self._v_host is None:
            self._v_host = tuple(self.backend.asnumpy(x) for x in self.v)
        return self._v_host

    @classmethod
    def from_1d(cls, x1, x2, Nx, p1, p2, Np, backend, dx=None, dp=None,
                x_anchor=None, p_anchor=None):
        """The pre-ndim signature, kept because it reads better in 1D call
        sites and because it is the shape GridSpec accepts from old clients."""
        return cls((Axis(x1, x2, Nx, dx, x_anchor),
                    Axis(p1, p2, Np, dp, p_anchor)), backend)

    def _bshape(self, a):
        s = [1]*self.n_axes
        s[a] = self.N[a]
        return tuple(s)

    def nat_mesh(self, a):
        """NATURAL-order coordinate mesh of axis a, broadcast-shaped over the
        full state shape. The initial-condition builders work in natural order
        (their caller applies shift() before propagating), so this is their
        counterpart to the shifted `C`."""
        return self.v[a].reshape(self._bshape(a))

    def sub_meshes(self, axs):
        """Shifted coordinate meshes broadcast within the SUBSPACE spanned by
        `axs` (in that order): shape tuple(N[a] for a in axs), 1 except on the
        mesh's own position. This is the space a plane reduction lands in, so
        it is what weights a reduction — U on the spatial subspace, T on the
        momentum one, x and py on the (x,py) plane for <Lz>. Built on demand:
        these are 1D arrays, and a reshape of an fftshift costs nothing next to
        the reduction it weights."""
        axs = tuple(axs)
        out = []
        for j, a in enumerate(axs):
            s = [1]*len(axs)
            s[j] = self.N[a]
            out.append(self.backend.fftshift(self.v[a]).reshape(tuple(s)))
        return tuple(out)

    def geom(self):
        """This grid's geometry as the wire/history record fact."""
        from .protocol import RecordGeom
        return RecordGeom(self.ndim, self.N, self.lo, self.hi)

    def extended(self, hbar_eff):
        """Per-axis real range the quantum propagator evaluates U (spatial
        axes) and T (momentum axes) on. The half-width of axis a depends only
        on the CONJUGATE axis's cell size, which is frozen — regrids move
        these intervals, never widen them."""
        out = []
        for a in range(self.n_axes):
            h = hbar_eff*self.dual_amp[ax.conjugate(self.ndim, a)]/2.
            out.append((self.lo[a] - h, self.hi[a] + h))
        return tuple(out)

    def spatial_ranges(self):
        return tuple((self.lo[a], self.hi[a]) for a in range(self.ndim))

    def spatial_extended(self, hbar_eff):
        return self.extended(hbar_eff)[:self.ndim]

    def shift(self, W):
        """Natural-order W -> propagator (fftshifted) order."""
        return self.backend.fftshift(W, axes=tuple(range(self.n_axes)))

    def unshift(self, W):
        """Propagator order -> natural order (display/diagnostics only; the
        streaming path leaves this to the frontend shader)."""
        return self.backend.ifftshift(W, axes=tuple(range(self.n_axes)))

    # -- 1D-only spellings -------------------------------------------------
    # Cheap aliases that keep 1D call sites (and the tests that are this
    # project's correctness anchor) reading naturally. They RAISE at ndim > 1
    # rather than quietly returning axis 0/1, so a call site that was never
    # generalized fails loudly instead of computing a wrong number.

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError(
                "Grid.%s is a 1D-only spelling; this grid has ndim=%d — use "
                "the generic N/lo/hi/d/v tuples" % (name, self.ndim))

    @property
    def Nx(self):
        self._only1d("Nx"); return self.N[0]

    @property
    def Np(self):
        self._only1d("Np"); return self.N[1]

    @property
    def x1(self):
        self._only1d("x1"); return self.lo[0]

    @property
    def x2(self):
        self._only1d("x2"); return self.hi[0]

    @property
    def p1(self):
        self._only1d("p1"); return self.lo[1]

    @property
    def p2(self):
        self._only1d("p2"); return self.hi[1]

    @property
    def dx(self):
        self._only1d("dx"); return self.d[0]

    @property
    def dp(self):
        self._only1d("dp"); return self.d[1]

    @property
    def xv(self):
        self._only1d("xv"); return self.v[0]

    @property
    def pv(self):
        self._only1d("pv"); return self.v[1]

    @property
    def theta_amp(self):
        self._only1d("theta_amp"); return self.dual_amp[1]

    @property
    def lam_amp(self):
        self._only1d("lam_amp"); return self.dual_amp[0]


@dataclass(frozen=True)
class GridState:
    """The session's live grid window on the FROZEN lattice — the exactness
    linchpin of auto-expand. The cell sizes `d` and the anchor are fixed at
    session creation; every regrid is pure integer arithmetic on the window
    (`offset`, `N`), and extents materialize as anchor + integer*d — the same
    float expression every time, so lattice points are bitwise-reproducible
    across regrids and no state value is ever interpolated."""
    anchor: tuple    # lo of the ORIGINAL grid, per axis
    d: tuple         # frozen cell sizes (never re-derived from extents)
    offset: tuple    # window offset from the anchor, in cells (signed)
    N: tuple

    @property
    def ndim(self):
        return len(self.N)//2

    @property
    def n_axes(self):
        return len(self.N)

    @property
    def lo(self):
        return tuple(self.anchor[a] + self.offset[a]*self.d[a]
                     for a in range(self.n_axes))

    @property
    def hi(self):
        return tuple(self.anchor[a] + (self.offset[a] + self.N[a])*self.d[a]
                     for a in range(self.n_axes))

    @classmethod
    def from_spec(cls, spec):
        """Initial window from a GridSpec (cell sizes derived exactly as
        Axis.cell derives them, so the two agree bitwise)."""
        return cls(anchor=tuple(a.lo for a in spec.axes),
                   d=tuple((a.hi - a.lo)/a.N for a in spec.axes),
                   offset=(0,)*len(spec.axes),
                   N=tuple(a.N for a in spec.axes))

    @classmethod
    def from_1d(cls, x0, p0, dx, dp, ox, op, Nx, Np):
        return cls(anchor=(x0, p0), d=(dx, dp), offset=(ox, op), N=(Nx, Np))

    def geom(self):
        from .protocol import RecordGeom
        return RecordGeom(self.ndim, self.N, self.lo, self.hi)

    def make_grid(self, backend):
        lo, hi = self.lo, self.hi
        return Grid(tuple(Axis(lo[a], hi[a], self.N[a], self.d[a],
                               (self.anchor[a], self.offset[a]))
                          for a in range(self.n_axes)), backend)

    def extended(self, hbar_eff):
        """Same contract as Grid.extended / GridSpec.extended: the range the
        quantum propagator evaluates U (spatial axes) and T (momentum axes)
        on. Half-widths depend only on the frozen cell sizes, so regrids move
        these intervals, never widen them."""
        lo, hi = self.lo, self.hi
        out = []
        for a in range(self.n_axes):
            h = hbar_eff*(pi/self.d[ax.conjugate(self.ndim, a)])/2.
            out.append((lo[a] - h, hi[a] + h))
        return tuple(out)

    def describe(self):
        """This window as one log-line string, at any ndim: the single place
        that formats it, so a worker's regrid line and the session's read the
        same (they used to be two near-identical format strings)."""
        lo, hi = self.lo, self.hi
        return "%s %s (%s)" % (
            " × ".join("[%g, %g]" % (lo[a], hi[a]) for a in range(self.n_axes)),
            "×".join(str(n) for n in self.N),
            ",".join(ax.labels(self.ndim)))

    def spatial_ranges(self):
        """Visible spatial box. Same name and contract as GridSpec's, because
        routers.sessions.compile_for is duck-typed over both: a live-U check
        validates against the live WINDOW, a create-time one against the spec."""
        lo, hi = self.lo, self.hi
        return tuple((lo[a], hi[a]) for a in range(self.ndim))

    def spatial_extended(self, hbar_eff):
        return self.extended(hbar_eff)[:self.ndim]

    def union(self, other):
        """Covering window of two windows on the same lattice (validity
        checks while a regrid is pending; NOT power-of-2, never propagated)."""
        offset = tuple(min(self.offset[a], other.offset[a])
                       for a in range(self.n_axes))
        return replace(self, offset=offset, N=tuple(
            max(self.offset[a] + self.N[a], other.offset[a] + other.N[a])
            - offset[a] for a in range(self.n_axes)))

    def moved(self, a, offset, n):
        """This window with axis `a` replaced (what an AxisPlan produces)."""
        off, N = list(self.offset), list(self.N)
        off[a], N[a] = offset, n
        return replace(self, offset=tuple(off), N=tuple(N))

    # -- 1D-only spellings (see Grid._only1d) ------------------------------

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("GridState.%s is a 1D-only spelling; this "
                                 "state has ndim=%d" % (name, self.ndim))

    @property
    def x0(self):
        self._only1d("x0"); return self.anchor[0]

    @property
    def p0(self):
        self._only1d("p0"); return self.anchor[1]

    @property
    def dx(self):
        self._only1d("dx"); return self.d[0]

    @property
    def dp(self):
        self._only1d("dp"); return self.d[1]

    @property
    def ox(self):
        self._only1d("ox"); return self.offset[0]

    @property
    def op(self):
        self._only1d("op"); return self.offset[1]

    @property
    def Nx(self):
        self._only1d("Nx"); return self.N[0]

    @property
    def Np(self):
        self._only1d("Np"); return self.N[1]

    @property
    def x1(self):
        self._only1d("x1"); return self.lo[0]

    @property
    def x2(self):
        self._only1d("x2"); return self.hi[0]

    @property
    def p1(self):
        self._only1d("p1"); return self.lo[1]

    @property
    def p2(self):
        self._only1d("p2"); return self.hi[1]


def embed_window(Wn, old, new, xp):
    """The exact-regrid primitive: natural-order W on window `old` ->
    natural-order W on window `new` (GridStates on one lattice). Overlap
    cells are COPIED by global cell index (bitwise, no interpolation);
    entering cells are zero; leaving cells are dropped (the trigger
    threshold bounds their mass). The result takes W's own dtype so a
    float32 session's regrid neither upcasts nor loses the copy's bitwise
    exactness."""
    Wnew = xp.zeros(new.N, dtype=Wn.dtype)
    dst, src = [], []
    for a in range(new.n_axes):
        g0 = max(old.offset[a], new.offset[a])
        g1 = min(old.offset[a] + old.N[a], new.offset[a] + new.N[a])
        if g1 <= g0:
            return Wnew          # windows do not overlap on this axis at all
        dst.append(slice(g0 - new.offset[a], g1 - new.offset[a]))
        src.append(slice(g0 - old.offset[a], g1 - old.offset[a]))
    Wnew[tuple(dst)] = Wn[tuple(src)]
    return Wnew
