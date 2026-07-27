"""
Observables of a propagator state: norm, plane reductions, marginals, energy,
phase-space moments, purity and (in 2D) angular momentum. All reductions run on
the device; only the results cross to the host. Input W is in the propagator's
fftshifted order, and every reduction of it stays fftshifted in the axes that
SURVIVE (fftshift is a per-axis permutation), which is exactly what the
frontend shader's half-period offset expects.

Everything but the purity comes out of REDUCTIONS rather than full-array
weighted sums: <H> is int U*n_q + int T*n_k over the two family reductions, the
moments come from the 1D marginals, and <Lz> from the two mixed planes. At
ndim=1 that replaces six full-array passes with three; at ndim=2 it is the
difference between weighting a 4D array 2*ndim + 2 times and weighting a handful
of 2D ones. int W^2 is the only full-array pass left.

Every reduction accumulates in float64 and every result LEAVES in float64,
whatever the solver's working precision. In double mode that is what already
happened; in a float32 session it is what keeps E/purity/moments worth reading
(these are sums over up to 16.7M elements) and what keeps the marginals
identical in dtype to before — history.py's byte accounting and the wire's <f4
codec then behave the same in both modes.
"""

from dataclasses import dataclass
from math import prod

import numpy
from numpy import pi

from . import axes as ax


@dataclass
class Observables:
    norm: float
    E: float          # <H> minus rest energy (subtracted for relativistic variants)
    purity: float     # gamma = (2*pi*hbar_eff)^ndim * int W^2 d^nq d^nk (= Tr rho^2)
    mean: tuple       # <q_a>, one per phase-space axis
    std: tuple        # sqrt(<q_a^2> - <q_a>^2), one per axis
    lz: float         # <x*py> - <y*px>; 0.0 at ndim=1 (no angular momentum)
    marg: tuple       # 1D marginal DENSITY per axis, natural order, host float64

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------

    def _only1d(self, name):
        if len(self.marg) != 2:
            raise AttributeError("Observables.%s is a 1D-only spelling; this "
                                 "state has %d axes — use mean/std/marg"
                                 % (name, len(self.marg)))

    @property
    def x_mean(self):
        self._only1d("x_mean"); return self.mean[0]

    @property
    def x_std(self):
        self._only1d("x_std"); return self.std[0]

    @property
    def p_mean(self):
        self._only1d("p_mean"); return self.mean[1]

    @property
    def p_std(self):
        self._only1d("p_std"); return self.std[1]

    @property
    def rho(self):
        self._only1d("rho"); return self.marg[0]

    @property
    def phi(self):
        self._only1d("phi"); return self.marg[1]


def _measure(grid, axs):
    return prod(grid.d[a] for a in axs)


def reduce_axes(W, grid, keep):
    """Sum W over every axis NOT in `keep`, weighted by their cell measures —
    i.e. the density on the subspace `keep`. Returns W itself (no copy) when
    `keep` is every axis, which is what makes ndim=1's single "plane" the state
    rather than a special case."""
    xp = grid.backend.xp
    keep = tuple(keep)
    over = tuple(a for a in range(grid.n_axes) if a not in keep)
    if not over:
        return W
    return xp.sum(W, axis=over, dtype=xp.float64)*_measure(grid, over)


def reduce_planes(W, grid):
    """The canonical 2D plane set (core.axes.PLANES), on the device, in shifted
    order, keyed by the axis pair. At ndim=1 this is {(0, 1): W}."""
    return {plane: reduce_axes(W, grid, plane)
            for plane in ax.planes(grid.ndim)}


def families(W, grid, planes):
    """(n_q, n_k) — the spatial and momentum densities, int W d^nk and
    int W d^nq. At ndim=2 these ARE planes (x,y) and (px,py), so they come free
    out of the plane set; at ndim=1 they are the two 1D marginals themselves and
    the plane set (which holds only W) cannot supply them."""
    sp, mo = grid.spatial_axes, grid.momentum_axes
    return (planes[sp] if sp in planes else reduce_axes(W, grid, sp),
            planes[mo] if mo in planes else reduce_axes(W, grid, mo))


def _family_marginal(n, grid, axs, keep):
    """1D marginal of global axis `keep` out of a family reduction `n` living
    on the subspace `axs` (so n's own axis j is global axs[j])."""
    xp = grid.backend.xp
    axs = tuple(axs)
    over = tuple(j for j in range(len(axs)) if axs[j] != keep)
    if not over:
        return n
    return xp.sum(n, axis=over, dtype=xp.float64) \
        * prod(grid.d[axs[j]] for j in over)


def _cross(planes, grid, plane):
    """<q_a * q_b> from the (a, b) plane density."""
    xp = grid.backend.xp
    ma, mb = grid.sub_meshes(plane)
    return float(xp.sum(ma*mb*planes[plane], dtype=xp.float64)) \
        * _measure(grid, plane)


def _base(W, grid, hbar_eff, planes, fam):
    b = grid.backend
    xp = b.xp
    sp, mo = grid.spatial_axes, grid.momentum_axes
    n_q, n_k = fam

    marg = [None]*grid.n_axes
    for a in sp:
        marg[a] = b.asnumpy(b.ifftshift(_family_marginal(n_q, grid, sp, a)))
    for a in mo:
        marg[a] = b.asnumpy(b.ifftshift(_family_marginal(n_k, grid, mo, a)))

    # Moments from the natural-order marginals against natural-order host
    # coordinate vectors: a few hundred host multiplies per axis, against a
    # weighted sum over the whole state per moment.
    mean, std = [], []
    for a in range(grid.n_axes):
        v, m = grid.v_host[a], marg[a]
        m1 = float((v*m).sum())*grid.d[a]
        m2 = float((v*v*m).sum())*grid.d[a]
        mean.append(m1)
        std.append(float(numpy.sqrt(abs(m2 - m1*m1))))

    return dict(
        norm=float(marg[0].sum())*grid.d[0],
        purity=(2.*pi*float(hbar_eff))**grid.ndim
        * float(xp.sum(W*W, dtype=xp.float64))*grid.dV,
        # <Lz> = <x*py> - <y*px>, each from the mixed plane that already holds
        # exactly the joint density it needs. Undefined in 1D.
        lz=(_cross(planes, grid, (0, 3)) - _cross(planes, grid, (1, 2))
            if grid.ndim == 2 else 0.0),
        mean=tuple(mean), std=tuple(std), marg=tuple(marg),
    )


def compute(W, prop, planes=None):
    """Full observables for a propagator state (uses its sub-shape energy
    meshes and rest energy). Pass `planes` when the caller has already reduced
    them (core.frame does) so the reductions happen once per record."""
    g = prop.grid
    xp = g.backend.xp
    if planes is None:
        planes = reduce_planes(W, g)
    fam = families(W, g, planes)
    d = _base(W, g, prop.hbar_eff, planes, fam)
    E = float(xp.sum(prop.U_mesh*fam[0], dtype=xp.float64)) \
        * _measure(g, g.spatial_axes) \
        + float(xp.sum(prop.T_mesh*fam[1], dtype=xp.float64)) \
        * _measure(g, g.momentum_axes) \
        - prop.rest_energy*d["norm"]
    return Observables(E=E, **d)


def compute_basic(W, grid, hbar_eff=1.0, planes=None):
    """Moments/marginals only (E = 0) — used by the initial-Wigner preview,
    which has no potential attached."""
    if planes is None:
        planes = reduce_planes(W, grid)
    return Observables(E=0.0, **_base(W, grid, hbar_eff, planes,
                                      families(W, grid, planes)))
