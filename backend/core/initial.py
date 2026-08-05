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

import numpy
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


# -- expression initial conditions -------------------------------------------
#
# Two further kinds, both auto-normalized: an arbitrary analytic W on the phase
# space, and an arbitrary analytic (complex) psi on configuration space, from
# which W comes by the Wigner transform. core/expr.py is the shared safe-compile
# boundary; nothing here parses anything itself.

# How much transient one block of the build may cost, and what a cell of that
# block is assumed to need.
#
# THE BUILD IS BLOCKED OVER SPATIAL AXIS 0, and that is a correctness property
# rather than a tuning knob: the transient of an ARBITRARY expression is a
# function of its own tree width, i.e. of user input. Measured at 1024^2 on the
# host, unblocked: 80 B/cell for a one-term psi, 96 for three terms, 112 for a
# 246-character nine-term one — and MAX_EXPR_LEN is 500. A fixed per-cell
# constant under an unbounded input is exactly the shape of the 256^4 preview
# OOM that routers/preview.py's fit checks exist to prevent, so there must not
# be one. Blocked, the peak is the float64 output plus this budget, whatever
# the user typed, which is what lets PREVIEW_BYTES_PER_CELL be a BOUND.
#
# The transform is over the momentum axes only and is independent per spatial
# point, so a block of spatial rows can run the whole pipeline on its own.
# BYTES_PER_CELL is deliberately generous against the 48 a plain psi needs: a
# block bounds the SHAPE, so a pathologically wide expression scales the peak by
# its own factor and the headroom is what keeps that bounded in practice.
IC_BLOCK_BYTES = 24*1024**2
IC_BLOCK_BYTES_PER_CELL = 128
# How far past the visible box psi is integrated for its normalization, in box
# widths per side, and how large the phi quadrature's kernel matrix may get.
# See _psi_lattice and _phi_along — both bounds exist because the Bopp
# half-width grows without limit as the momentum axis is refined.
NORM_PAD_MAX = 2
PHI_KERNEL_BYTES = 12*1024**2
# How small ∫W may be RELATIVE to ∫|W| before it stops being a normalizer.
#
# THE TEST HAS TO BE RELATIVE, because these kinds auto-normalize and therefore
# invite an arbitrary amplitude — "scale doesn't matter" is the whole selling
# point of the tabs. An absolute floor refused 1e-15*exp(-x^2-p^2), a perfectly
# good state, with a sentence claiming it had no positive total. What the check
# is actually for is CANCELLATION: an expression odd in any variable sums to
# roundoff against a mass of order 1, so the ratio is ~1e-16 there and ~0.5 for
# even the most oscillatory legitimate W (a cat's ∫|W| is ~2× its ∫W).
NORM_REL_EPS = 1e-9


class ICError(ValueError):
    """An initial condition that cannot be built — the CLIENT's problem, not the
    device's.

    Its own class because most of these can only be decided once W exists (an
    expression's integral is not knowable before it is summed), so unlike a
    parse error they are raised from inside the build. routers/preview.py must
    be able to tell them from a cupy OOM, whose answer is the CPU fallback:
    without that distinction a W integrating to zero would be retried on the CPU
    to fail identically, and then surface as a 500. It is the same reasoning that
    put components_of ahead of the device choice, applied to the part that
    cannot go there."""


@dataclass
class CompiledIC:
    """An initial condition reduced to everything decidable WITHOUT allocating
    a state — the expression twin of components_of, and split out for the same
    reason: routers/preview.py must answer "is this IC well formed?" before it
    picks a device, and routers/sessions.py before it starts four workers.

    Compiled ONCE and threaded through, exactly as session.compiled_potential
    is. Compiling inside the worker would have up to four threads parsing the
    same string simultaneously through sympy's global caches."""
    kind: str
    ndim: int
    components: tuple = ()
    src: str = ""
    expr: object = None          # sympy
    f: object = None             # numpy callable, array in -> array out
    f_gpu: object = None

    @property
    def is_expression(self):
        return self.kind in ("wexpr", "psi")

    def for_backend(self, backend):
        """The callable for this ArrayBackend — the CompiledPotential.for_backend
        contract, including the lazy cupy lambdify."""
        if not backend.is_gpu:
            return self.f
        if self.f_gpu is None:
            from . import expr as ex
            self.f_gpu = ex.lambdify(ex.symbols(_ic_var_names(self.kind,
                                                              self.ndim)),
                                     self.expr, "cupy")
        return self.f_gpu


@dataclass(frozen=True)
class ICNorm:
    """How an initial condition was normalized, and what that leaves worth
    reporting. `deficit` is meaningful only when `method` says the state had an
    analytic norm the grid could fall short of."""
    method: str          # "analytic" | "psi-extended" | "grid-sum"
    raw: float = 1.0     # the integral the state was divided by
    deficit: float = 0.0
    momentum_mass: float = 1.0   # psi only; see psi_wigner


def _ic_var_names(kind, ndim):
    """The variables each expression kind may use. A W lives on the whole phase
    space; a psi lives on configuration space only — psi(x,p) would be
    nonsense."""
    labels = ax.labels(ndim)
    return tuple(labels) if kind == "wexpr" else tuple(labels[:ndim])


def compile_ic(ic, hbar_eff, ndim, ranges=None):
    """Compile an ICSpec of any kind. Gaussian kinds go through components_of
    unchanged; the two expression kinds are screened, parsed, lambdified and
    probed for finiteness. Raises ValueError (ExprError is one) for anything a
    client must fix."""
    kind = ic.type
    if kind not in ("wexpr", "psi"):
        return CompiledIC(kind=kind, ndim=ndim,
                          components=tuple(components_of(ic, hbar_eff, ndim)))

    from . import expr as ex

    names = _ic_var_names(kind, ndim)
    label = "%s(%s)" % ("W" if kind == "wexpr" else "ψ", ",".join(names))
    src = (ic.expr or "").strip()
    if not src:
        raise ICError("%s: an expression is required" % label)
    # Lazily, so initial.py keeps the light import graph its callers rely on.
    # psi is COMPLEX by definition; W is real by definition, and a complex one
    # would make |exp| != 1 nonsense of every diagnostic downstream.
    ns = ex.namespace(names, label, complex_ok=(kind == "psi"))
    e = ex.parse(src, ns)
    syms = ex.symbols(names)
    f = ex.lambdify(syms, e, "numpy")

    if ranges is not None:
        # THE PROBE DTYPE MUST BE THE ONE THE BUILD USES, and the two kinds do
        # not agree. psi is evaluated at q -+ hbar*theta/2 inside the transform,
        # which is a complex array, so sqrt of a negative real is a finite
        # branch value there and probing it as real would refuse a legal psi —
        # the same argument compile_potential's quantum probe makes, and it
        # matters more here, since ONE nan is spread by the FFT over an entire
        # momentum axis and stops being localizable. A W is sampled on
        # grid.nat_mesh, which is float64, so probing IT complex is the mistake
        # in the other direction: sqrt(x)*exp(-p^2) passed this check and then
        # raised from inside wexpr_wigner, i.e. after POST /sessions had already
        # answered 200 and started four workers — precisely the failure this
        # compile exists to turn into a 422 at the door.
        dt = numpy.complex128 if kind == "psi" else None
        with numpy.errstate(all="ignore"):
            meshes = ex.probe_meshes(ranges, dtype=dt)
            v = numpy.asarray(f(*meshes))
            bad = ~numpy.isfinite(v)
            if bad.any():
                raise ICError(
                    "%s is not finite for %s (probed over %s)"
                    % (label, ex.bad_box(syms, meshes, bad),
                       ex.box_str(syms, ranges)))
    return CompiledIC(kind=kind, ndim=ndim, src=src, expr=e, f=f)


def probe_ranges(ic_type, grid_spec, hbar_eff):
    """The box compile_ic probes an expression for finiteness on — one place,
    because both routers must ask the same question of the same kind.

    A psi lives on configuration space and is evaluated at q -+ hbar*theta/2
    inside the transform, so its box is the EXTENDED spatial one — the same box
    the quantum potential is validated on, and for exactly the same reason. A W
    lives on the whole phase space and is only ever sampled on the grid itself.
    """
    if ic_type == "psi":
        return grid_spec.spatial_extended(hbar_eff)
    if ic_type == "wexpr":
        return tuple((a.lo, a.hi) for a in grid_spec.axes)
    return None


def _spatial_blocks(grid):
    """Slices of spatial axis 0 sized so one block's transient stays inside
    IC_BLOCK_BYTES. Yields at least one slice, always."""
    per_row = prod(grid.N[1:])
    rows = max(1, int(IC_BLOCK_BYTES //
                      max(1, per_row*IC_BLOCK_BYTES_PER_CELL)))
    for start in range(0, grid.N[0], rows):
        yield slice(start, min(start + rows, grid.N[0]))


def _block_mesh(grid, a, sl):
    """grid.nat_mesh(a), restricted to a block of spatial axis 0."""
    m = grid.nat_mesh(a)
    return m[sl] if a == 0 else m


def wexpr_wigner(grid, cic):
    """W straight from an analytic phase-space expression, normalized by its own
    integral over the grid — there is no analytic norm for an arbitrary
    expression, so the grid sum is the only one available.

    Returns (W_natural, raw_integral). Refuses a non-finite sample and an
    integral that cannot normalize.

    THE NORMALIZER IS BOUNDED FROM BOTH SIDES, and the upper bound is the one
    that was missing. Every sample can be finite while their SUM overflows —
    exp(-x^2-p^2)*exp(709) does it — and `raw = inf` then made `W /= raw`
    silently produce a W that is identically ZERO: a blank panel, ∫W = 0, and
    not one warning, because purity 0 and norm 0 trip nothing. So ∫|W| is taken
    first and its finiteness is what decides, since |raw| <= mass makes raw
    finite by construction once mass is. See NORM_REL_EPS for the lower test."""
    xp = grid.backend.xp
    f = cic.for_backend(grid.backend)
    W = xp.empty(grid.shape, dtype=xp.float64)
    for sl in _spatial_blocks(grid):
        args = [_block_mesh(grid, a, sl) for a in range(grid.n_axes)]
        v = xp.asarray(f(*args))
        W[sl] = xp.broadcast_to(v.real.astype(xp.float64), W[sl].shape)
    if not bool(xp.all(xp.isfinite(W))):
        raise ICError("W(%s) is not finite everywhere on this grid"
                         % ",".join(ax.labels(grid.ndim)))
    mass = float(xp.sum(xp.abs(W), dtype=xp.float64))*grid.dmu
    if not numpy.isfinite(mass):
        raise ICError(
            "this W overflows to infinity when summed over the grid, so it "
            "cannot be normalized. Its amplitude is far past what a float64 "
            "total can hold — scale the expression down, which costs nothing "
            "since it is normalized anyway.")
    if mass <= 0.0:
        raise ICError("this W is zero everywhere on the grid")
    raw = float(xp.sum(W, dtype=xp.float64))*grid.dmu
    if raw <= NORM_REL_EPS*mass:
        raise ICError(
            "this W integrates to %.3g against a total absolute mass of %.3g, "
            "so it cannot be normalized — an initial condition needs a positive "
            "total. An expression that is odd in any variable integrates to "
            "zero; a negative total would invert the state." % (raw, mass))
    W /= raw
    return W, raw


def _phi_along(xp, arr, axis, kvec, qvec, dq, h):
    """One axis of the phi quadrature: sum_q exp(-i k q / hbar) psi dq.

    CHUNKED OVER THE SOURCE AXIS, because the kernel is an (Nk x Nq) matrix and
    Nq is the extended lattice — measured asking for 13.7 GB in one piece at
    1D 4096^2 before NORM_PAD_MAX capped the extension, and still hundreds of MB
    after. The flop count is the same; only the live matrix is bounded.
    """
    nk, nq = len(kvec), len(qvec)
    chunk = max(1, int(PHI_KERNEL_BYTES//max(1, nk*16)))
    scale = dq/numpy.sqrt(2.*numpy.pi*h)
    out = None
    for s in range(0, nq, chunk):
        sl = slice(s, min(s + chunk, nq))
        E = xp.exp(-1j*kvec[:, None]*qvec[sl][None, :]/h)*scale
        part = xp.moveaxis(
            xp.tensordot(E, xp.take(arr, xp.arange(sl.start, sl.stop), axis=axis),
                         axes=([1], [axis])), 0, axis)
        out = part if out is None else out + part
    return out


def _psi_lattice(grid, cic, hbar_eff):
    """psi sampled on the EXTENDED spatial lattice, its normalization, and the
    fraction of its momentum content the grid's momentum box actually holds.

    Two things come out of here that the full state cannot supply.

    NORMALIZATION IS OVER THE EXTENDED SPATIAL BOX, not the visible one — the
    same lattice extended by whole cells until it covers grid.spatial_extended
    (the box psi is evaluated on inside the transform, since the Bopp arguments
    reach q -+ hbar*theta_amp/2). The visible points are then a strict SUBSET,
    so int W dmu (dmu = the phase-space measure, Grid.dmu) comes out as
    the in-box mass fraction and X-Wignerf-Norm-Deficit
    keeps exactly the meaning it has for the analytically-normalized Gaussian
    kinds. Normalizing over the visible box instead makes the deficit
    structurally zero and destroys the diagnostic (measured: 0.645236190989
    against an in-box mass fraction of 0.645236190989, versus a flat
    1.000000000000).

    MOMENTUM MASS is the one diagnostic that can see a wrong momentum box, and
    nothing else in this program can. The discrete Wigner transform is exactly
    N_k-periodic in the momentum index, so momentum content outside the box
    ALIASES BACK IN rather than being lost — the identity sum_k W d_k =
    |psi(q)|^2 holds to 3.3e-16 on ANY box. Measured on a packet at p0 = +2 with
    the momentum box [-10, -2], which excludes its mean momentum entirely: norm
    1.0000000000, purity 1.000000, max|W| 0.3172 against a correct 0.3171, and
    the p edge band 6.5e-06 — barely over its 1e-6 trigger and under the float32
    one. The packet is aliased to the wrong place at full height and every
    scalar diagnostic reads perfect. phi by direct quadrature on the box lattice
    reads 7.2e-07 against 0.9999999988 for a good box: a factor of 1e6, and the
    only honest answer.
    """
    xp = grid.backend.xp
    nd = grid.ndim
    h = float(hbar_eff)
    f = cic.for_backend(grid.backend)

    # The spatial lattice this normalization integrates over, per axis, on the
    # SAME spacing so the visible points stay a strict subset.
    #
    # CAPPED AT NORM_PAD_MAX BOX WIDTHS PER SIDE, and that cap is doing real
    # work: the Bopp half-width is hbar*pi/(2*dk), which GROWS as the momentum
    # axis is refined — at 1D 4096^2 over a [-8, 8] momentum box it is 402,
    # fifty times the spatial box, for a lattice of 210k points. Integrating out
    # there measures nothing (any state with mass at |x| = 400 is not being
    # simulated in a box of 8) and costs everything; it is what made the phi
    # quadrature below ask for a 13.7 GB kernel. What the extension is FOR is
    # catching the tail a state has just outside its box, and two box widths per
    # side is past any of that.
    ext = grid.spatial_extended(h)
    vs, ns = [], []
    for i in range(nd):
        d = grid.d[i]
        pad = int(numpy.ceil(max(0.0, grid.lo[i] - ext[i][0])/d))
        pad = min(pad, NORM_PAD_MAX*grid.N[i])
        v0 = grid.v[i]
        vs.append(float(v0[0]) + d*xp.arange(-pad, grid.N[i] + pad,
                                             dtype=xp.float64))
        ns.append(pad)

    shape = tuple(len(v) for v in vs)
    args = [vs[i].reshape(tuple(len(vs[i]) if j == i else 1
                                for j in range(nd))).astype(xp.complex128)
            for i in range(nd)]
    psi = xp.broadcast_to(xp.asarray(f(*args)), shape).astype(xp.complex128)
    if not bool(xp.all(xp.isfinite(psi))):
        raise ICError(
            "ψ(%s) is not finite on the extended spatial box %s the Wigner "
            "transform evaluates it on"
            % (",".join(ax.labels(grid.ndim)[:nd]),
               " × ".join("[%.4g, %.4g]" % r for r in ext)))
    dq = prod(grid.d[i] for i in range(nd))
    total = float(xp.sum(xp.abs(psi)**2, dtype=xp.float64))*dq
    # BOUNDED FROM ABOVE AS WELL AS BELOW. psi itself passed isfinite above, but
    # |psi|^2 need not: exp(-x^2/2)*exp(360) overflows only when it is SQUARED,
    # and total = inf then gave scale = 0, pre = 0, and a W of 0*inf = nan
    # everywhere. Nothing downstream could see that — every diagnostic compares
    # with `>` and NaN fails every comparison, so purity, the edge band and the
    # momentum mass all pass silently. The one warning that did fire blamed the
    # momentum box, which is the wrong remedy for an overflow.
    if not numpy.isfinite(total):
        raise ICError(
            "∫|ψ|² overflows to infinity on this grid, so ψ cannot be "
            "normalized. Its amplitude is far past what a float64 total can "
            "hold — scale the expression down, which costs nothing since it is "
            "normalized anyway.")
    if not (total > 1e-300):
        raise ICError("ψ is zero everywhere on this grid")
    # The SCALE is returned rather than applied here, because psi_wigner
    # transforms the lambdified callable itself, not this array. Normalizing
    # the copy and handing it back would silently drop the normalization from
    # the state that is actually built — and for an already-normalized psi it
    # would look perfect, which is how it would survive.
    scale = 1.0/numpy.sqrt(total)

    # in-box slice of the extended lattice
    inbox = tuple(slice(ns[i], ns[i] + grid.N[i]) for i in range(nd))
    inbox_frac = float(xp.sum(xp.abs(psi[inbox])**2,
                              dtype=xp.float64))*dq/total
    psi = psi*scale

    # phi on the grid's OWN momentum lattice, by quadrature — NOT by an FFT of
    # psi, whose dual lattice is 2*pi*hbar/(N*dq) wide and in general is not the
    # momentum axis the user chose. The kernel is separable, so this is one small
    # matrix product per dimension rather than a transform of the full state.
    phi = psi
    for i in range(nd):
        phi = _phi_along(xp, phi, i, grid.v[nd + i], vs[i], grid.d[i], h)
    dk = prod(grid.d[nd + i] for i in range(nd))
    momentum_mass = float(xp.sum(xp.abs(phi)**2, dtype=xp.float64))*dk
    return scale, inbox_frac, momentum_mass


def psi_wigner(grid, cic, hbar_eff):
    """W from an analytic wavefunction, by the discrete Wigner transform

        W(q,k) = (1/2pi)^n int d^n theta conj(psi(q + hbar*theta/2))
                                          psi(q - hbar*theta/2) e^{i k.theta}

    with theta_i the Fourier dual of k_i — the lattice Grid already builds, so
    the psi arguments span exactly grid.spatial_extended(hbar_eff), the same
    extended box the quantum potential is validated on.

    Discretely, with k_i = k0_i + m*d[k_i] and theta_j = -pi/d[k_i] + j*dtheta,

        e^{i k_i theta_j} = e^{i k0_i theta_j} * (-1)^m * e^{2 pi i m j / N}

    so each momentum axis is: multiply by the RAMP e^{i k0 theta}, inverse DFT,
    multiply by (-1)^m, scale by 1/d[k]. Take the real part (A is hermitian in
    theta, so the residue is the one unpaired lattice point plus roundoff —
    measured 3.5e-17).

    THE RAMP IS NOT OPTIONAL AND IS NOT A NON-SYMMETRIC-BOX SPECIAL CASE. k0 is
    the first LATTICE value and is never 0 for a box straddling the origin.
    Measured without it against the analytic coherent-state W: relative error
    1.0 — completely wrong — on a symmetric [-7, 7] box, while the norm stays
    exactly 1.000000. No norm-, energy- or purity-based assertion can see this;
    only a cell-by-cell comparison against the analytic form can.

    k0 is float(grid.v[a][0]) rather than grid.lo[a] because the identity needs
    the first lattice VALUE: on an anchored axis (any auto-expand regrid) v[0]
    materializes as anchor + d*(offset + 0) while lo comes off Axis.lo, and the
    two may differ in the last ulp.

    Returns (W_natural, ICNorm).
    """
    xp = grid.backend.xp
    nd = grid.ndim
    h = float(hbar_eff)
    f = cic.for_backend(grid.backend)

    scale, inbox_frac, momentum_mass = _psi_lattice(grid, cic, h)

    ths = [grid.nat_dual_mesh(nd + i) for i in range(nd)]
    ramps = [xp.exp(1j*float(grid.v[nd + i][0])*ths[i]) for i in range(nd)]
    alts = []
    for i in range(nd):
        a = nd + i
        s = [1]*grid.n_axes
        s[a] = grid.N[a]
        alts.append(((-1.)**xp.arange(grid.N[a])).reshape(tuple(s)))
    mo = tuple(range(nd, 2*nd))
    pre = scale*scale/prod(grid.d[nd + i] for i in range(nd))

    W = xp.empty(grid.shape, dtype=xp.float64)
    for sl in _spatial_blocks(grid):
        plus, minus = [], []
        for i in range(nd):
            q = _block_mesh(grid, i, sl).astype(xp.complex128)
            plus.append(q + 0.5*h*ths[i])
            minus.append(q - 0.5*h*ths[i])
        A = xp.asarray(f(*minus))*xp.conj(xp.asarray(f(*plus)))
        A = xp.broadcast_to(A, W[sl].shape).astype(xp.complex128)
        for r in ramps:
            A = A*r
        A = xp.fft.ifftn(A, axes=mo)
        for al in alts:
            A = A*al
        W[sl] = (pre*A).real

    # The same guard wexpr_wigner has, and for the same reason it is not
    # redundant with the checks above: _psi_lattice validates psi on the CAPPED
    # normalization lattice (NORM_PAD_MAX box widths), while the loop above
    # evaluates it at q -+ hbar*theta/2 over the FULL extended reach, which is
    # far wider once the momentum axis is fine — hbar*pi/(2*dk) is 402 against a
    # box of 8 at 1D 4096^2. Out there compile_ic's finite probe lattice is the
    # only cover, and it samples rather than proves. A NaN that got through
    # would be invisible: see the isfinite note in _psi_lattice.
    if not bool(xp.all(xp.isfinite(W))):
        raise ICError(
            "W is not finite everywhere: ψ is evaluated at %s -+ ℏθ/2 over the "
            "extended box %s, which reaches well past the visible domain when "
            "the momentum axes are fine. Widen the momentum cell (fewer points "
            "or a narrower momentum box) or remove ψ's singularity."
            % (",".join(ax.labels(grid.ndim)[:nd]),
               " × ".join("[%.4g, %.4g]" % r
                          for r in grid.spatial_extended(h))))

    return W, ICNorm(method="psi-extended", raw=1.0/(scale*scale),
                     deficit=abs(1.0 - inbox_frac),
                     momentum_mass=momentum_mass)


def wavefunction_cuts(grid_spec, cic, hbar_eff, backend, cut_axis=0, n=512):
    """ψ along one spatial axis and φ along its CONJUGATE momentum axis, as
    plain lists for the IC editor's two line charts.

    Two rules, both of which are silently wrong if broken.

    φ IS ON THE GRID'S OWN MOMENTUM LATTICE, by quadrature. An FFT of ψ lands on
    the dual of the SPATIAL lattice, 2πℏ/(N·dq) wide, which in general is not the
    momentum axis the user chose — and this chart is drawn directly under a W
    preview whose vertical axis is that axis.

    AT ndim=2 IT IS TRANSFORM-THEN-CUT, never cut-then-transform: φ(px, py0) is
    a slice of the full 2D transform of ψ, not the 1D transform of the slice
    ψ(x, y0). The two agree exactly at ndim=1 AND exactly for a separable ψ —
    i.e. for anything anyone is likely to type first — which is what would let
    the wrong one ship.
    """
    from .grid import Axis, Grid
    g = Grid(tuple(Axis(a.lo, a.hi, a.N) for a in grid_spec.axes), backend)
    xp = backend.xp
    nd = g.ndim
    h = float(hbar_eff)
    i = int(min(max(cut_axis, 0), nd - 1))
    j = ax.conjugate(nd, i)             # the dual of q_i, stated once in axes.py
    f = cic.for_backend(backend)

    args = [g.v[a].reshape(tuple(g.N[a] if b == a else 1
                                 for b in range(nd))).astype(xp.complex128)
            for a in range(nd)]
    psi = xp.broadcast_to(xp.asarray(f(*args)),
                          tuple(g.N[a] for a in range(nd))).astype(xp.complex128)
    if not bool(xp.all(xp.isfinite(psi))):
        raise ICError("ψ is not finite on this grid")
    dq = prod(g.d[a] for a in range(nd))
    total = float(xp.sum(xp.abs(psi)**2, dtype=xp.float64))*dq
    if not numpy.isfinite(total):
        raise ICError("∫|ψ|² overflows to infinity on this grid — scale the "
                      "expression down, which costs nothing since it is "
                      "normalized anyway.")
    if not (total > 1e-300):
        raise ICError("ψ is zero everywhere on this grid")
    # Normalize by the MODULUS only — a complex factor would make Re and Im jump
    # about under normalization while |ψ|² sat still, which is exactly the
    # information these two traces exist to show.
    psi = psi/numpy.sqrt(total)

    # THE SAME CHUNKED QUADRATURE _psi_lattice USES, and it has to be: the
    # kernel is an (N_k x N_q) matrix, so writing it out in one piece is
    # quadratic in the grid. Measured peak on this path before the switch, 1D:
    # 2.1 MiB at N=256, 32 at 1024, 128 at 2048, 512 at 4096 — and ~2 GiB at the
    # 8192 this host sets, on an endpoint that fires on every keystroke in the
    # psi box. _phi_along bounds the live matrix at PHI_KERNEL_BYTES for the
    # same flop count. Its qvec is the VISIBLE lattice here, not the extended
    # one: these are display traces on the axes the charts draw.
    phi = psi
    for a in range(nd):
        phi = _phi_along(xp, phi, a, g.v[nd + a], g.v[a], g.d[a], h)

    def _cut(arr, keep, coord_axis, vec):
        """The `keep` axis of a 1- or 2-D array, sliced at the sample nearest
        the origin on the other axis — nearestZero, the same rule the potential
        editor's 2D cuts use, and the title says where it was actually taken."""
        at = None
        if nd > 1:
            other = 1 - keep
            m = int(xp.argmin(xp.abs(vec[other])))
            at = float(backend.asnumpy(vec[other])[m])
            arr = arr[m, :] if other == 0 else arr[:, m]
        v = backend.asnumpy(vec[keep])
        a = backend.asnumpy(arr)
        step = max(1, len(v)//int(n))     # decimate; the chart cannot show more
        return {"axis": coord_axis, "at": at,
                "v": v[::step].tolist(),
                "re": a[::step].real.tolist(),
                "im": a[::step].imag.tolist()}

    qv = [g.v[a] for a in range(nd)]
    kv = [g.v[nd + a] for a in range(nd)]
    return (_cut(psi, i, i, qv), _cut(phi, i, j, kv))


def from_spec(grid_spec, ic, hbar_eff, backend, grid=None, compiled=None):
    """Build (Grid, W_natural, warnings, edge_axes, ICNorm) from
    protocol.GridSpec/ICSpec — `edge_axes` being the (axis_label, band_mass)
    pairs kept out of `warnings` so a caller can render them itself (see
    preview_warnings) (duck-typed: anything with the same attributes works).
    Shared by the IC preview endpoint and by each SolverWorker at session start;
    the worker passes its GridState-built `grid` so the lattice materialization
    matches the session's window bitwise.

    `compiled` is a CompiledIC from compile_ic. Pass it: it is what keeps the
    expression kinds from parsing once per worker thread, and it is how a bad
    expression becomes a 422 at the door rather than four dead workers."""
    from .grid import Axis, Grid
    g = grid if grid is not None else Grid(
        tuple(Axis(a.lo, a.hi, a.N) for a in grid_spec.axes), backend)
    cic = compiled if compiled is not None \
        else compile_ic(ic, hbar_eff, g.ndim)
    if cic.kind == "psi":
        W, norm = psi_wigner(g, cic, hbar_eff)
    elif cic.kind == "wexpr":
        W, raw = wexpr_wigner(g, cic)
        norm = ICNorm(method="grid-sum", raw=raw)
    elif cic.kind == "cat":
        W = cat_wigner(g, list(cic.components), hbar_eff)
        norm = ICNorm(method="analytic")
    else:
        W = mixture_wigner(g, list(cic.components))
        norm = ICNorm(method="analytic")
    edge = []
    warns = preview_warnings(g, cic.components, cic.kind, hbar_eff, W,
                             edge_axes=edge, norm=norm)
    return g, W, warns, edge, norm


def preview_warnings(grid, components, kind, hbar_eff=1.0, W=None,
                     edge_axes=None, norm=None):
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
    # The 4σ boxes are a property of Gaussian COMPONENTS; an expression IC has
    # none, and this loop is simply empty for it. Nothing is missing there — the
    # measure-based edge check below reads the sampled total W and covers every
    # kind, which is exactly why it was written.
    for n, c in enumerate(components):
        centre, sigma = c.center, c.sigma
        near = [labels[a] for a in range(grid.n_axes)
                if centre[a] - 4.*sigma[a] < lo[a]
                or centre[a] + 4.*sigma[a] > hi[a]]
        if near:
            warns.append("component %d is within 4σ of a domain edge in %s "
                         "(tails will wrap around)" % (n + 1, ", ".join(near)))
    # A wavefunction ALWAYS defines a valid pure state, so γ > 1 cannot mean
    # what it means for a hand-written W. There it can only mean the grid is
    # aliasing the state, and the standing remedy ("fine for the classical
    # variants") is the one reading that is certainly wrong. Same detector,
    # different sentence — the move boundaryTitle already makes for float32.
    from_psi = kind == "psi"
    # PURITY_EPS is tuned against the analytically exact Gaussian kinds. A
    # transform-built pure state overshoots it on a coarse lattice by ordinary
    # quadrature error, so accusing it of being no quantum state would be a
    # false positive on a perfectly good ψ.
    eps = 1e-4 if from_psi else 1e-6
    if W is not None:
        purity = (2.*pi*hbar)**ndim*float((W*W).sum())*grid.dmu
        if purity > 1. + eps:
            if from_psi:
                warns.append(
                    "Tr ρ² = %s = %.8g > 1 — a wavefunction is always a valid "
                    "pure state, so this is the GRID: the momentum axes cannot "
                    "resolve ψ and fold it back on itself. Widen or refine them"
                    % (ax.purity_expr(ndim), purity))
            else:
                warns.append(
                    "W is not a valid quantum state: Tr ρ² = %s = %.8g > 1 "
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
    # The ONLY detector for a momentum box in the wrong PLACE. See _psi_lattice:
    # the transform is exactly periodic in the momentum index, so content
    # outside the box aliases back in at full height and the norm, the purity
    # and the edge band above all read perfectly healthy.
    if norm is not None and norm.method == "psi-extended" \
            and norm.momentum_mass < 0.999:
        warns.append(
            "only %.4g of ψ's momentum content lies inside the %s box — the "
            "rest is not lost but ALIASED back into it, so W is a different "
            "state and no other diagnostic can tell. Widen the momentum axes"
            % (norm.momentum_mass, ", ".join(labels[ndim:])))
    return warns
