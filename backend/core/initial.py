"""
Initial (Cauchy) data on the phase-space grid: sums of Gaussians, generic over
ndim (2026-07-26).

Two types:
- mixture_wigner: statistical mixture — weighted sum of Gaussian blobs,
  W >= 0 everywhere (the generalization of dynamics/initgauss.py);
  sigma_q and sigma_k are independent per component AND per dimension.
- cat_wigner: coherent superposition psi = sum_j c_j psi_j of minimal
  Gaussian packets (sigma_k = hbar_eff/(2 sigma_q) per dimension, derived),
  built from the analytic pairwise cross-Wigner closed form — exact on the
  grid, normalized analytically by <psi|psi>. Interference cross-terms give
  the oscillatory fringes and negative regions.

BOTH FACTORIZE OVER DIMENSIONS, which is why 2D needs no new closed form. The
packets are separable products psi_j(x,y) = g_j(x) h_j(y), and the Wigner
transform of a tensor product is the product of the transforms — so the ndim=2
cross-Wigner of a pair is the OUTER PRODUCT of the two one-dimensional cores
below, each carrying its own 1/(2 pi hbar) (giving (2 pi hbar)^-ndim overall),
and <psi_k|psi_j> is the product of the per-dimension overlaps. The 1D closed
forms are reused verbatim, per dimension.

Returned arrays are in NATURAL (unshifted) order; the caller applies
grid.shift() before propagation.
"""

import cmath
from dataclasses import dataclass
from math import prod
from math import sqrt as msqrt

from numpy import pi

from . import axes as ax
from . import observables
from .boundary import edge_band, edge_report


def _tup(v):
    """Accept a scalar as the 1D shorthand for a one-element tuple."""
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return (float(v),)


@dataclass
class GaussianComponent:
    """One Gaussian packet. The four geometry fields carry ndim values each;
    a bare scalar is accepted as the 1D shorthand."""
    q0: tuple
    k0: tuple
    sigma_q: tuple
    sigma_k: tuple
    weight: float = 1.0
    phase: float = 0.0   # used only by the cat-state builder

    def __post_init__(self):
        self.q0 = _tup(self.q0)
        self.k0 = _tup(self.k0)
        self.sigma_q = _tup(self.sigma_q)
        self.sigma_k = _tup(self.sigma_k)
        self.weight = float(self.weight)
        self.phase = float(self.phase)

    @property
    def ndim(self):
        return len(self.q0)

    def validate(self, ndim=None):
        n = self.ndim
        for name, f in (("k0", self.k0), ("sigma_q", self.sigma_q),
                        ("sigma_k", self.sigma_k)):
            if len(f) != n:
                raise ValueError("component %s has %d values, q0 has %d"
                                 % (name, len(f), n))
        if ndim is not None and n != ndim:
            raise ValueError("component is %dD, the grid is %dD" % (n, ndim))
        if any(s <= 0 for s in self.sigma_q) or any(s <= 0 for s in self.sigma_k):
            raise ValueError("sigma_q and sigma_k must be positive")
        if self.weight <= 0:
            raise ValueError("weight must be positive")

    @property
    def center(self):
        """Centre in phase-space axis order (q..., k...)."""
        return self.q0 + self.k0

    @property
    def sigma(self):
        """Widths in phase-space axis order (q..., k...)."""
        return self.sigma_q + self.sigma_k

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("GaussianComponent.%s is a 1D-only spelling; "
                                 "this component is %dD" % (name, self.ndim))

    @property
    def x0(self):
        self._only1d("x0"); return self.q0[0]

    @property
    def p0(self):
        self._only1d("p0"); return self.k0[0]

    @property
    def sigma_x(self):
        self._only1d("sigma_x"); return self.sigma_q[0]

    @property
    def sigma_p(self):
        self._only1d("sigma_p"); return self.sigma_k[0]


def mixture_wigner(grid, components):
    """Weighted sum of Gaussian phase-space blobs, normalized so that the
    analytic integral of W over the whole phase space is 1 (each blob carries
    Z = 1/prod_i(2*pi*sigma_q_i*sigma_k_i), weights normalized to unit sum)."""
    if not components:
        raise ValueError("at least one Gaussian component is required")
    xp = grid.backend.xp
    ndim = grid.ndim
    wtot = sum(c.weight for c in components)
    W = xp.zeros(grid.shape, dtype=xp.float64)
    for c in components:
        c.validate(ndim)
        Z = c.weight/(wtot*prod(2.*pi*c.sigma_q[i]*c.sigma_k[i]
                                for i in range(ndim)))
        # One exp() over the whole state, with the exponent accumulated
        # per dimension — bitwise the old single-exp form at ndim=1, and the
        # broadcast only reaches the full shape on the LAST term.
        arg = 0
        for i in range(ndim):
            q, k = grid.nat_mesh(i), grid.nat_mesh(ndim + i)
            arg = arg + (q - c.q0[i])**2/(2.*c.sigma_q[i]**2) \
                + (k - c.k0[i])**2/(2.*c.sigma_k[i]**2)
        W += Z*xp.exp(-arg)
    return W


def _overlap_1d(xj, pj, sj, xk, pk, sk, hbar):
    """<psi_k|psi_j> = integral psi_j psi_k* dx for minimal packets
    psi_j = N_j exp(-a_j (x-x_j)^2 + i p_j (x-x_j)/hbar), closed form via
    Gaussian integral of exp(-A x^2 + B x + C)."""
    aj, ak = 1./(4.*sj**2), 1./(4.*sk**2)
    Nj = (2.*pi*sj**2)**(-0.25)
    Nk = (2.*pi*sk**2)**(-0.25)
    A = aj + ak
    B = 2.*aj*xj + 2.*ak*xk + 1j*(pj - pk)/hbar
    C = -aj*xj**2 - ak*xk**2 - 1j*(pj*xj - pk*xk)/hbar
    return Nj*Nk*msqrt(pi)/cmath.sqrt(A)*cmath.exp(B*B/(4.*A) + C)


def _overlap(cj, ck, hbar):
    """Separable packets => the overlap is the product of the per-dimension
    one-dimensional overlaps."""
    s = 1.0 + 0j
    for i in range(cj.ndim):
        s *= _overlap_1d(cj.q0[i], cj.k0[i], cj.sigma_q[i],
                         ck.q0[i], ck.k0[i], ck.sigma_q[i], hbar)
    return s


def _cross_core(grid, i, cj, ck, hbar):
    """The one-dimensional cross-Wigner core of dimension i, on the natural-
    order (q_i, k_i) meshes and broadcast-shaped over the full state:

      W_jk = (N_j N_k / 2 pi hbar) sqrt(pi/alpha)
             * exp(i (p_j u_j - p_k u_k)/hbar)
             * exp(-a_j u_j^2 - a_k u_k^2 + beta^2/(4 alpha)),

    u_j = q - q_j, alpha = (a_j + a_k)/4,
    beta = -a_j u_j + a_k u_k + i (p_j + p_k - 2 k)/(2 hbar).
    """
    xp = grid.backend.xp
    q, k = grid.nat_mesh(i), grid.nat_mesh(grid.ndim + i)
    sj, sk = cj.sigma_q[i], ck.sigma_q[i]
    aj, ak = 1./(4.*sj**2), 1./(4.*sk**2)
    Nj, Nk = (2.*pi*sj**2)**(-0.25), (2.*pi*sk**2)**(-0.25)
    pj, pk = cj.k0[i], ck.k0[i]
    uj, uk = q - cj.q0[i], q - ck.q0[i]
    alpha = (aj + ak)/4.
    beta = -aj*uj + ak*uk + 1j*(pj + pk - 2.*k)/(2.*hbar)
    return (Nj*Nk/(2.*pi*hbar))*msqrt(pi/alpha) \
        * xp.exp(-aj*uj**2 - ak*uk**2 + beta**2/(4.*alpha)) \
        * xp.exp(1j*(pj*uj - pk*uk)/hbar)


def cat_wigner(grid, components, hbar_eff=1.0):
    """Wigner function of psi = (1/sqrt(S)) sum_j c_j psi_j with
    c_j = sqrt(weight_j) exp(i phase_j) and minimal packets (sigma_k is
    derived, component.sigma_k is ignored). Uses the pairwise closed form of
    _cross_core, multiplied over dimensions:

      W = sum_j |c_j|^2 W_jj + 2 Re sum_{j<k} c_j c_k^* prod_i core_i,

    normalized by S = <psi|psi> from the same closed-form overlaps (never by
    grid sum, which would hide truncation error)."""
    if not components:
        raise ValueError("at least one Gaussian component is required")
    ndim = grid.ndim
    for c in components:
        c.validate(ndim)
    xp = grid.backend.xp
    hbar = float(hbar_eff)
    amp = [msqrt(c.weight)*cmath.exp(1j*c.phase) for c in components]

    W = xp.zeros(grid.shape, dtype=xp.float64)
    for j, cj in enumerate(components):
        for k in range(j, len(components)):
            ck = components[k]
            core = None
            for i in range(ndim):
                ci = _cross_core(grid, i, cj, ck, hbar)
                core = ci if core is None else core*ci
            term = (amp[j]*amp[k].conjugate()*core).real
            W += term if j == k else 2.*term

    S = sum(amp[j]*amp[k].conjugate()*_overlap(components[j], components[k], hbar)
            for j in range(len(components)) for k in range(len(components)))
    return W/S.real


def minimal_sigma_p(sigma_x, hbar_eff=1.0):
    """sigma of the conjugate axis for a minimal (pure-state) packet; the UI
    shows this as the derived, read-only sigma_k for cat components. Per
    DIMENSION — a 2D cat derives sigma_px from sigma_x and sigma_py from
    sigma_y independently."""
    return hbar_eff/(2.*sigma_x)


def components_of(ic, hbar_eff, ndim=None):
    """The validated GaussianComponents of an ICSpec — sigma_k derived for a
    cat state, taken from the component for a mixture. Raises ValueError for a
    spec that cannot be built.

    Split out of from_spec so a caller can decide "is this IC well formed?"
    WITHOUT allocating anything: it is a handful of tuples and no arrays, so
    routers/preview.py answers that question before it picks a device. Deciding
    it inside the build instead meant a bad spec and a device failure arrived
    as the same exception type from the same call."""
    comps = []
    for c in ic.components:
        if ic.type == "cat":
            sk = tuple(minimal_sigma_p(s, hbar_eff) for s in c.sigma_q)
        else:
            sk = c.sigma_k
            if sk is None:
                raise ValueError("sigma_k is required for mixture components")
        comp = GaussianComponent(c.q0, c.k0, c.sigma_q, sk, c.weight, c.phase)
        comp.validate(ndim)
        comps.append(comp)
    if not comps:
        raise ValueError("at least one Gaussian component is required")
    return comps


def from_spec(grid_spec, ic, hbar_eff, backend, grid=None):
    """Build (Grid, W_natural, warnings, edge_axes) from protocol.GridSpec/
    ICSpec — `edge_axes` being the (axis_label, band_mass) pairs kept out of
    `warnings` so a caller can render them itself (see preview_warnings)
    (duck-typed: anything with the same attributes works). Shared by the
    IC preview endpoint and by each SolverWorker at session start; the
    worker passes its GridState-built `grid` so the lattice materialization
    matches the session's window bitwise."""
    from .grid import Axis, Grid
    g = grid if grid is not None else Grid(
        tuple(Axis(a.lo, a.hi, a.N) for a in grid_spec.axes), backend)
    comps = components_of(ic, hbar_eff, g.ndim)
    if ic.type == "cat":
        W = cat_wigner(g, comps, hbar_eff)
    else:
        W = mixture_wigner(g, comps)
    edge = []
    warns = preview_warnings(g, comps, ic.type, hbar_eff, W, edge_axes=edge)
    return g, W, warns, edge


def preview_warnings(grid, components, kind, hbar_eff=1.0, W=None,
                     edge_axes=None):
    """Quality diagnostics for an initial condition (warnings, not blocks):
    - fringe Nyquist (cat): packets separated by dq_jk along dimension i
      interfere with k_i-period 2*pi*hbar/|dq_jk|; require d[k_i] <
      pi*hbar/|dq_jk| with a safety factor of 2 (dually d[q_i] vs dk_jk);
    - packet mass within 4 sigma of a domain edge, on every axis (tails wrap
      under the periodic spectral propagator);
    - quantum validity of the TOTAL W via the purity bound: any density
      operator satisfies Tr rho^2 = (2*pi*hbar)^ndim * integral W^2 <= 1.
      This is a necessary condition on the complete state — individual
      components are only a decomposition and carry no physical meaning
      (a valid W may well be a sum of sub-Heisenberg blobs). Violation
      proves W is not a quantum state; it remains a perfectly good
      classical phase-space density.

    Messages may contain any Unicode; the preview endpoint percent-encodes
    them for HTTP-header transport."""
    warns = []
    hbar = float(hbar_eff)
    ndim = grid.ndim
    labels, lo, hi, d = grid.labels, grid.lo, grid.hi, grid.d
    for n, c in enumerate(components):
        centre, sigma = c.center, c.sigma
        near = [labels[a] for a in range(grid.n_axes)
                if centre[a] - 4.*sigma[a] < lo[a]
                or centre[a] + 4.*sigma[a] > hi[a]]
        if near:
            warns.append("component %d is within 4σ of a domain edge in %s "
                         "(tails will wrap around)" % (n + 1, ", ".join(near)))
    if W is not None:
        purity = (2.*pi*hbar)**ndim*float((W*W).sum())*grid.dV
        if purity > 1. + 1e-6:
            warns.append("W is not a valid quantum state: Tr ρ² = %s = %.8g > 1 "
                         "(fine for the classical variants)"
                         % (ax.purity_expr(ndim), purity))
        # measure-based edge check on the sampled TOTAL W: catches what the
        # per-component 4σ boxes cannot (interference cross-terms of a cat
        # state carry their own mass). W is in natural order here.
        margs = [observables.reduce_axes(W, grid, (a,))
                 for a in range(grid.n_axes)]
        es = edge_report(margs, d, labels=labels)
        # es.tripped, not a bare threshold compare: the runtime watch and this
        # check must agree on what "too close" means, and on a coarse grid that
        # is the measured noise floor rather than EDGE_THRESHOLD (boundary.py).
        for a in es.tripped:
            if edge_axes is not None:
                # Structured, for a caller that renders this itself: the runtime
                # boundary watch says the same thing about the same axes the
                # moment record 0 exists, so the two were showing one fact
                # twice. Handing over (axis, mass) lets the client drop exactly
                # the axes already covered and word what is left once. Note the
                # sentence below is NOT the same string with the tail cut off —
                # a caller that wants prose still gets the full explanation,
                # because it has nowhere else to put it.
                edge_axes.append((labels[a], es.mass[a]))
                continue
            warns.append(
                "%.2g of the total probability lies within %.3g of the %s "
                "edge (the outer %d cells of the %s marginal) — the "
                "spectral domain is periodic, so whatever reaches an edge "
                "re-enters from the opposite side"
                % (es.mass[a], edge_band(grid.N[a])*d[a], labels[a],
                   edge_band(grid.N[a]), labels[a]))
    if kind == "cat":
        n = len(components)
        for j in range(n):
            for k in range(j + 1, n):
                cj, ck = components[j], components[k]
                for i in range(ndim):
                    dq = abs(cj.q0[i] - ck.q0[i])
                    dk = abs(cj.k0[i] - ck.k0[i])
                    if dq > 0 and d[ndim + i] > pi*hbar/dq/2.:
                        warns.append(
                            "interference fringes of components %d and %d "
                            "(%s-period %.4g) are under-resolved by d%s = %.4g"
                            % (j + 1, k + 1, labels[ndim + i],
                               2.*pi*hbar/dq, labels[ndim + i], d[ndim + i]))
                    if dk > 0 and d[i] > pi*hbar/dk/2.:
                        warns.append(
                            "interference fringes of components %d and %d "
                            "(%s-period %.4g) are under-resolved by d%s = %.4g"
                            % (j + 1, k + 1, labels[i],
                               2.*pi*hbar/dk, labels[i], d[i]))
    return warns
