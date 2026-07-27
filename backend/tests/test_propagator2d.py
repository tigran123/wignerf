"""
Physics-invariant tests for 2D space (4D phase space, W(x,y,px,py)).

The correctness argument, strongest first:

1. test_matches_an_independent_schroedinger_run — evolve psi(x,y) by an
   ORDINARY split-operator Schroedinger solver, a completely different method,
   under a COUPLED potential, and require the Wigner run's plane reductions to
   reproduce |psi|^2 and |psi~|^2 cell by cell plus every observable. This is
   the only test here that discriminates the correlated multi-D Bopp shift from
   a pair of independent one-dimensional shifts: the two agree for EVERY
   quadratic U and first differ at third order in mixed derivatives (for
   U = x^2*y by exactly -2*a^2*b with a = hbar*thetax/2, b = hbar*thetay/2).
2. test_separable_run_equals_two_1d_runs — for U = Ux(x) + Uy(y) and a product
   IC, W_2D(t) must equal the outer product of two 1D solutions to roundoff,
   because a separable U makes dU a sum and the exponent factorise exactly. It
   validates the whole 2D pipeline against code the 1D suite already trusts.
3. test_quantum_equals_classical_with_a_cross_term — the 2D analogue of the 1D
   suite's strongest single check. NB it does NOT discriminate the shift
   structure (see 1); it pins that all second-order Bopp terms cancel,
   including the mixed 2*a*b*U_xy one.
4. test_angular_momentum — <Lz> conserved for a central U, demonstrably not for
   an anisotropic one. Physics that only exists in 2D.
5. test_observables_match_a_naive_computation — every reduction, marginal and
   moment against the naive full-array form, which is what catches 4-axis
   fftshift bookkeeping errors.
"""

from math import pi, sqrt

import numpy as np
import pytest

from core import axes as ax
from core import observables
from core.grid import Axis, Grid
from core.initial import (GaussianComponent, cat_wigner, minimal_sigma_p,
                         mixture_wigner)
from core.propagator import Propagator
from core.xp import ArrayBackend, C_AU

SIG = 1.0/sqrt(2.0)


@pytest.fixture(scope="module")
def backend():
    return ArrayBackend(device="cpu")


def grid2(backend, n=32, xlim=6.0, plim=7.0):
    return Grid((Axis(-xlim, xlim, n), Axis(-xlim, xlim, n),
                 Axis(-plim, plim, n), Axis(-plim, plim, n)), backend)


def grid1(backend, n=32, xlim=6.0, plim=7.0):
    return Grid.from_1d(-xlim, xlim, n, -plim, plim, n, backend)


def iso_harmonic():
    """U = (x^2 + y^2)/2 — separable AND central, so it serves both roles."""
    return dict(U=lambda x, y: x**2/2. + y**2/2.,
                gradU=(lambda x, y: x + 0.*y, lambda x, y: y + 0.*x))


def evolve(prop, W, dt, nsteps):
    expU, expT = prop.exponents(dt)
    for _ in range(nsteps):
        W = prop.solve_spectral(W, expU, expT)
    return W


# ---------------------------------------------------------------------------
# 1. the independent-method reference
# ---------------------------------------------------------------------------

def _schroedinger(xv, yv, psi, U, mass, hbar, dt, nsteps, c=None):
    """Split-operator TDSE on psi(x,y): exp(-iU dt/2) exp(-iT dt) exp(-iU dt/2)
    with T in the Fourier basis. Deliberately NOT the Wigner machinery — no Bopp
    shifts, no phase-space grid, nothing shared but numpy's FFT — so agreement is
    evidence, not tautology.

    c=None gives the non-relativistic T = (kx^2+ky^2)/2m; a c makes it the
    square-root (Salpeter) T = c*sqrt(p^2 + m^2c^2) with p = hbar*kappa, which a
    spectral method applies exactly because T is diagonal in the Fourier basis.
    The rest energy is a global phase there and cancels from |psi|^2 — the same
    cancellation the Wigner side gets from T entering only as a DIFFERENCE."""
    dx, dy = xv[1] - xv[0], yv[1] - yv[0]
    kx = 2.*pi*np.fft.fftfreq(len(xv), d=dx)[:, None]
    ky = 2.*pi*np.fft.fftfreq(len(yv), d=dy)[None, :]
    Umesh = U(xv[:, None], yv[None, :])
    halfU = np.exp(-0.5j*dt*Umesh/hbar)
    if c is None:
        expT = np.exp(-1j*dt*hbar*(kx**2 + ky**2)/(2.*mass))
    else:
        expT = np.exp(-1j*dt*c*np.sqrt(hbar**2*(kx**2 + ky**2)
                                       + mass**2*c**2)/hbar)
    for _ in range(nsteps):
        psi = halfU*psi
        psi = np.fft.ifft2(expT*np.fft.fft2(psi))
        psi = halfU*psi
    return psi


def _coherent_psi(xv, yv, q0, sigma, k0, hbar):
    """The same minimal packet cat_wigner assumes: a separable product of
    normalized Gaussians with momentum k0."""
    out = np.ones((len(xv), len(yv)), dtype=complex)
    for i, v in enumerate((xv, yv)):
        g = (2.*pi*sigma[i]**2)**(-0.25) \
            * np.exp(-(v - q0[i])**2/(4.*sigma[i]**2)
                     + 1j*k0[i]*(v - q0[i])/hbar)
        out = out*(g[:, None] if i == 0 else g[None, :])
    return out


HH_LAMBDA = 0.1


def _hh():
    """Henon-Heiles: couples x and y AND has a nonzero mixed third derivative
    U_xxy — exactly where the correlated Bopp shift and a pair of independent
    one-dimensional shifts diverge."""
    lam = HH_LAMBDA
    return (lambda x, y: (x**2 + y**2)/2. + lam*(x**2*y - y**3/3.),
            (lambda x, y: x + 2.*lam*x*y,
             lambda x, y: y + lam*(x**2 - y**2)))


def _wigner_vs_psi(backend, dt, nsteps, c=None):
    """Run both methods to the same t and return
    (relative rho error, obs, psi diagnostics). c=None is non-relativistic;
    passing a c runs BOTH sides relativistically (milestone M2)."""
    hbar, mass = 1.0, 1.0
    U, gradU = _hh()
    # p spans exactly the FFT frequency range of the Schroedinger x-grid, so
    # the momentum plane and |psi~|^2 land on the SAME lattice and can be
    # compared cell by cell rather than interpolated.
    g = grid2(backend, n=32, xlim=6.0, plim=pi*32/12.0)
    q0, sig, k0 = (1.0, 0.5), (SIG, SIG), (0.0, 0.6)
    prop = Propagator(g, quantum=True, mass=mass, hbar_eff=hbar,
                      U=U, gradU=gradU, relativistic=c is not None,
                      **({} if c is None else dict(c=c)))
    W = g.shift(cat_wigner(g, [GaussianComponent(q0, k0, sig, sig)], hbar))

    xv, yv = backend.asnumpy(g.v[0]), backend.asnumpy(g.v[1])
    psi = _coherent_psi(xv, yv, q0, sig, k0, hbar)

    W = evolve(prop, W, dt, nsteps)
    psi = _schroedinger(xv, yv, psi, U, mass, hbar, dt, nsteps, c=c)

    planes = observables.reduce_planes(W, g)
    obs = observables.compute(W, prop, planes)
    rho = backend.asnumpy(g.backend.ifftshift(planes[(0, 1)], axes=(0, 1)))
    rho_psi = np.abs(psi)**2
    err = float(np.max(np.abs(rho - rho_psi)))/float(np.max(rho_psi))
    return err, obs, (g, planes, psi, rho_psi, xv, yv, hbar, mass, U)


def test_matches_an_independent_schroedinger_run(backend):
    """THE 2D anchor: a completely different method, on a COUPLED potential.
    Two assertions, and the SECOND one is what makes it decisive.

    The residual is a discretization difference, not a structural
    disagreement, so it must FALL as O(dt^2): the Wigner step is
    expT·expU·expT and this TDSE step is expU·expT·expU, both 2nd order.
    Measured here — 1.45e-4 relative at dt = 0.02, 3.43e-5 at 0.01, ratio 4.22.

    Now the same numbers with the multi-D Bopp shift implemented WRONG, as a
    sum of two independent one-dimensional shifts (measured 2026-07-26 by
    monkeypatching Propagator.qd): 7.83e-3 at dt = 0.02 and 7.81e-3 at 0.01 —
    228x larger AND dt-INDEPENDENT (ratio 1.003), because a wrong shift is a
    different EVOLUTION OPERATOR (off by a term of order hbar^2*thetax^2*thetay,
    which is exactly -2*a^2*b/(i*hbar) — see the cubic test below), not a
    smaller time step away from the right one. Every quadratic-potential test in
    this file passes under that wrong implementation.

    Do NOT extend the dt sweep down to 0.005: there the ratio drops to 1.9
    because the residual has reached a dt-independent ~1.5e-5 floor set by the
    GRID, not the step — the Bopp shift samples U outside the box on the
    discrete theta lattice, which the TDSE's exp(-iU dt) never does, so the two
    schemes discretize the same continuous operator differently. The
    convergence check only means something in the regime where the splitting
    error dominates."""
    coarse, _, _ = _wigner_vs_psi(backend, 0.02, 50)
    fine, obs, extra = _wigner_vs_psi(backend, 0.01, 100)
    g, planes, psi, rho_psi, xv, yv, hbar, mass, U = extra

    assert fine < 1e-4, "densities disagree: %.3g" % fine
    assert coarse/fine > 3.0, (
        "the residual is not O(dt^2) — %.3g at dt=0.02 vs %.3g at 0.01. A "
        "dt-independent residual means the evolution operator is wrong, not the "
        "step size; suspect the multi-D Bopp shift." % (coarse, fine))

    # the momentum plane is |psi~|^2 on the matched lattice
    dx, dy = g.d[0], g.d[1]
    psit = np.fft.fftshift(np.fft.fft2(psi))*dx*dy/(2.*pi*hbar)
    nk = np.abs(psit)**2
    nk_w = backend.asnumpy(g.backend.ifftshift(planes[(2, 3)], axes=(0, 1)))
    assert np.max(np.abs(nk_w - nk)) < 5e-4*np.max(nk)   # measured 1.4e-4

    # and every observable, computed two completely different ways. Bounds are
    # ~3x the measured deltas at dt = 0.01 (norm 1.2e-15, <x> 5.5e-5, <y> 3.5e-5,
    # <px> -1.7e-4, <py> 3.5e-4, E 1.9e-5 relative, <Lz> 1.1e-4).
    def psi_mean(f):
        return float(np.sum(f*rho_psi))*dx*dy

    assert obs.norm == pytest.approx(float(rho_psi.sum())*dx*dy, abs=1e-9)
    assert obs.mean[0] == pytest.approx(psi_mean(xv[:, None]), abs=2e-4)
    assert obs.mean[1] == pytest.approx(psi_mean(yv[None, :]), abs=2e-4)

    kxv = 2.*pi*np.fft.fftshift(np.fft.fftfreq(len(xv), d=dx))
    kyv = 2.*pi*np.fft.fftshift(np.fft.fftfreq(len(yv), d=dy))
    npk = nk/float(nk.sum())
    assert obs.mean[2] == pytest.approx(
        float(np.sum(kxv[:, None]*npk)), abs=1e-3)
    assert obs.mean[3] == pytest.approx(
        float(np.sum(kyv[None, :]*npk)), abs=1e-3)

    # <H> from psi: <T> in the Fourier basis + <U> in configuration space
    E_psi = float(np.sum((kxv[:, None]**2 + kyv[None, :]**2)/(2.*mass)*nk)) \
        * (kxv[1] - kxv[0])*(kyv[1] - kyv[0]) \
        + psi_mean(U(xv[:, None], yv[None, :]))
    assert obs.E == pytest.approx(E_psi, rel=1e-4)   # measured 1.9e-5

    # <Lz> = -i hbar <psi| x d/dy - y d/dx |psi>, again straight from psi
    dpsidy = np.fft.ifft(1j*(2.*pi*np.fft.fftfreq(len(yv), d=dy))[None, :]
                         * np.fft.fft(psi, axis=1), axis=1)
    dpsidx = np.fft.ifft(1j*(2.*pi*np.fft.fftfreq(len(xv), d=dx))[:, None]
                         * np.fft.fft(psi, axis=0), axis=0)
    lz_psi = float(np.real(np.sum(np.conj(psi)*(-1j*hbar)*(
        xv[:, None]*dpsidy - yv[None, :]*dpsidx))))*dx*dy
    assert obs.lz == pytest.approx(lz_psi, abs=5e-4)   # measured 1.1e-4


# ---------------------------------------------------------------------------
# 2. separability against the validated 1D solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quantum", [True, False])
def test_separable_run_equals_two_1d_runs(backend, quantum):
    """U = x^2/2 + 0.3*y^4 is separable but NOT quadratic, so the quantum run
    carries real Moyal corrections in y — and must still factorise exactly.

    NB this anchor is NON-RELATIVISTIC and cannot be extended to qr/cr. It rests
    on the whole exponent factorising, which needs T separable as well as U:
    T = (px^2+py^2)/2m is a sum, but T = c*sqrt(px^2+py^2+m^2c^2) is not, so a
    relativistic 2D run is genuinely NOT the outer product of two 1D relativistic
    runs. Adding a relativistic parametrization here would fail against CORRECT
    code. The independent cross-check that replaces it for relativistic variants
    is test_relativistic_matches_an_independent_schroedinger_run."""
    U2 = lambda x, y: x**2/2. + 0.3*y**4
    g2 = grid2(backend)
    p2 = Propagator(g2, quantum=quantum, U=U2,
                    gradU=(lambda x, y: x + 0.*y, lambda x, y: 1.2*y**3 + 0.*x))
    W2 = g2.shift(mixture_wigner(g2, [GaussianComponent(
        (2.0, 0.5), (0.0, -1.0), (SIG, 0.6), (SIG, 0.8))]))

    gx, gy = grid1(backend), grid1(backend)
    px = Propagator(gx, quantum=quantum, U=lambda x: x**2/2.,
                    gradU=(lambda x: x,))
    py = Propagator(gy, quantum=quantum, U=lambda y: 0.3*y**4,
                    gradU=(lambda y: 1.2*y**3,))
    Wx = gx.shift(mixture_wigner(gx, [GaussianComponent(2.0, 0.0, SIG, SIG)]))
    Wy = gy.shift(mixture_wigner(gy, [GaussianComponent(0.5, -1.0, 0.6, 0.8)]))

    # the IC itself must already be the product (the builders factorise)
    prod = lambda a, b: np.asarray(a)[:, None, :, None]*np.asarray(b)[None, :, None, :]
    assert np.max(np.abs(np.asarray(W2) - prod(Wx, Wy))) < 1e-15

    dt, n = 0.01, 150
    W2 = evolve(p2, W2, dt, n)
    Wx = evolve(px, Wx, dt, n)
    Wy = evolve(py, Wy, dt, n)
    # Measured after 150 steps: 5.9e-11 of the peak (quantum) and 4.8e-10
    # (classical), and it IS pure roundoff — exp(a+b) against exp(a)*exp(b) over
    # the large phases the y^4 term produces, accumulated over the run. The
    # harmonic case, whose phases are far smaller, lands at 1.1e-13 instead.
    peak = float(np.max(np.abs(np.asarray(W2))))
    assert np.max(np.abs(np.asarray(W2) - prod(Wx, Wy))) < 1e-8*peak


# ---------------------------------------------------------------------------
# 3-4. the 2D-only physics
# ---------------------------------------------------------------------------

def test_quantum_equals_classical_with_a_cross_term(backend):
    """Quadratic H => the Moyal corrections vanish exactly, cross term and all.
    The x*y coupling makes the second-order Bopp expansion carry a mixed
    2*a*b*U_xy term that must cancel between the two shifted evaluations."""
    g = grid2(backend)
    U = lambda x, y: (x**2 + y**2)/2. + 0.3*x*y
    gradU = (lambda x, y: x + 0.3*y, lambda x, y: y + 0.3*x)
    q = Propagator(g, quantum=True, U=U, gradU=gradU)
    c = Propagator(g, quantum=False, U=U, gradU=gradU)
    assert float(np.max(np.abs(q.dU_im - c.dU_im))) < 1e-12
    assert float(np.max(np.abs(q.dT_im - c.dT_im))) < 1e-12

    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (1.5, -1.0), (0.0, 0.5), (SIG, SIG), (SIG, SIG))]))
    Wq = evolve(q, W0, 0.01, 120)
    Wc = evolve(c, W0, 0.01, 120)
    assert float(np.max(np.abs(Wq - Wc))) < 1e-11


def test_a_coupled_cubic_is_not_a_pair_of_1d_shifts(backend):
    """The structural check behind anchor 1, stated directly: for U = x^2*y the
    correlated shift and two independent 1D shifts differ by exactly
    -2*a^2*b/(i*hbar) with a = hbar*thetax/2, b = hbar*thetay/2. If dU ever
    reverts to a sum of per-axis differences, this fails while every quadratic
    test still passes."""
    g = grid2(backend, n=16)
    U = lambda x, y: x**2*y
    prop = Propagator(g, quantum=True, U=U,
                      gradU=(lambda x, y: 2.*x*y, lambda x, y: x**2 + 0.*y))
    h = prop.hbar_eff
    X, Y = g.Q
    a, b = h*g.Theta[0]/2., h*g.Theta[1]/2.
    joint = ((U(X - a, Y - b) - U(X + a, Y + b))/(1j*h)).imag
    apart = (((U(X - a, Y) - U(X + a, Y))
              + (U(X, Y - b) - U(X, Y + b)))/(1j*h)).imag
    np.testing.assert_allclose(prop.dU_im, np.broadcast_to(joint, g.shape),
                               atol=1e-12)
    diff = np.broadcast_to(joint - apart, g.shape)
    # and the two really are different: the discrepancy is the mixed term
    assert float(np.max(np.abs(diff))) > 1.0
    # rtol 1e-9, not tighter: `diff` is a difference of two O(10) meshes whose
    # cancellation leaves ~1e-12 relative, while the closed form is exact.
    np.testing.assert_allclose(diff, np.broadcast_to((-2.*a*a*b/(1j*h)).imag,
                                                     g.shape), rtol=1e-9)


@pytest.mark.parametrize("quantum,relativistic", [
    (True, False), (False, False), (True, True), (False, True)])
def test_angular_momentum(backend, quantum, relativistic):
    """<Lz> is conserved by a CENTRAL potential and not by an anisotropic one —
    a genuinely two-dimensional invariant, and one no 1D test can reach.

    It holds for the RELATIVISTIC variants too (milestone M2) and for the same
    reason: T depends on the momenta only through |k|, so it is invariant under a
    joint rotation of (x,y) and (px,py) exactly as p^2/2m is. That makes this the
    one existing 2D anchor that transfers to qr/cr unchanged — separability does
    not (see above), and quantum == classical does not either (T is not quadratic
    in k, so the kinetic Moyal corrections do not vanish)."""
    g = grid2(backend)
    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, 0.0), (0.0, 1.0), (SIG, SIG), (SIG, SIG))]))

    kw = dict(quantum=quantum, relativistic=relativistic)
    central = Propagator(g, **kw, **iso_harmonic())
    lz0 = observables.compute(W0, central).lz
    # x*py - y*px = 2*1 - 0; the 1.5e-7 deficit is Gaussian tail truncation at
    # the domain edge, the same bound test_mixture_normalization uses in 1D
    assert lz0 == pytest.approx(2.0, abs=2e-6)
    W = evolve(central, W0, 0.01, 300)
    # Measured 5.65e-6 over 300 steps — and IDENTICAL for the quantum and
    # classical variants, which is the tell that it is the square LATTICE
    # breaking rotational symmetry (and the periodic torus), not physics.
    assert observables.compute(W, central).lz == pytest.approx(lz0, abs=2e-5)

    aniso = Propagator(g, **kw,
                       U=lambda x, y: x**2/2. + 2.*y**2,
                       gradU=(lambda x, y: x + 0.*y, lambda x, y: 4.*y + 0.*x))
    Wa = evolve(aniso, W0, 0.01, 300)
    # measured drift 3.05, i.e. 150% of <Lz> — a 5e5 separation from the
    # central case's 5.65e-6
    assert abs(observables.compute(Wa, aniso).lz - lz0) > 0.5


def test_conservation_and_time_reversal(backend):
    g = grid2(backend)
    prop = Propagator(g, quantum=True, **iso_harmonic())
    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, 0.5), (0.0, -1.0), (SIG, SIG), (SIG, SIG))]))
    o0 = observables.compute(W0, prop)
    # a 2D coherent state is pure: (2 pi hbar)^2 int W^2 = 1
    assert o0.purity == pytest.approx(1.0, abs=1e-6)
    # E = sum over both dimensions of (q0^2 + k0^2)/2 + hbar*omega/2
    assert o0.E == pytest.approx((4. + 0.25 + 1.)/2. + 1.0, abs=1e-6)

    W = evolve(prop, W0, 0.01, 200)
    o = observables.compute(W, prop)
    assert abs(o.norm - o0.norm) < 1e-12
    assert abs(o.purity - o0.purity) < 1e-10
    assert abs(o.E - o0.E)/abs(o0.E) < 1e-4

    # Reversal is exact up to roundoff: each step's real() projection drops an
    # ~1e-16-relative imaginary residue. Measured 1.2e-7 over 400 steps here
    # against ~1e-9 over 600 in 1D — twice the transforms per step and four axes
    # of FFT roundoff over 1M cells. Far below any physical signal.
    back = evolve(prop, W, -0.01, 200)
    assert float(np.max(np.abs(back - W0))) < 1e-6


def test_isotropic_period(backend):
    """Both dimensions have omega = 1, so after t = 2*pi the whole state
    returns — a 2D revival, and a check that neither dimension's exponents got
    the other's mesh."""
    g = grid2(backend)
    prop = Propagator(g, quantum=True, **iso_harmonic())
    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, -1.5), (0.0, 0.0), (SIG, SIG), (SIG, SIG))]))
    n = 1200
    W = evolve(prop, W0, 2.*pi/n, n)
    rel = float(np.sum(np.abs(W - W0)))/float(np.sum(np.abs(W0)))
    assert rel < 0.02
    # at a quarter period the centre has rotated in each (q, k) plane
    W = evolve(prop, W0, (pi/2.)/400, 400)
    o = observables.compute(W, prop)
    assert o.mean[0] == pytest.approx(0.0, abs=2e-3)     # x: 2 -> 0
    assert o.mean[1] == pytest.approx(0.0, abs=2e-3)     # y: -1.5 -> 0
    assert o.mean[2] == pytest.approx(-2.0, abs=2e-3)    # px: 0 -> -x0
    assert o.mean[3] == pytest.approx(1.5, abs=2e-3)     # py: 0 -> -y0


# ---------------------------------------------------------------------------
# 5. reductions vs the naive computation (fftshift bookkeeping)
# ---------------------------------------------------------------------------

def test_observables_match_a_naive_computation(backend):
    """Every plane, marginal and moment against the naive natural-order form.
    This is what catches 4-axis fftshift errors, the likeliest porting bug
    after the shift pairing itself."""
    g = grid2(backend)
    prop = Propagator(g, quantum=True, **iso_harmonic())
    Wn = mixture_wigner(g, [GaussianComponent(
        (1.5, -0.5), (0.5, -1.0), (0.7, 0.6), (0.8, 0.9))])
    obs = observables.compute(g.shift(Wn), prop)
    planes = observables.reduce_planes(g.shift(Wn), g)

    W = np.asarray(Wn)                      # natural order
    d = g.d
    dV = g.dV
    v = [backend.asnumpy(x) for x in g.v]

    assert obs.norm == pytest.approx(W.sum()*dV, abs=1e-12)
    for a in range(4):
        m = W.sum(axis=tuple(b for b in range(4) if b != a)) \
            * (dV/d[a])
        np.testing.assert_allclose(obs.marg[a], m, atol=1e-14)
        assert obs.mean[a] == pytest.approx((v[a]*m).sum()*d[a], abs=1e-12)

    for plane in ax.planes(2):
        over = tuple(b for b in range(4) if b not in plane)
        want = W.sum(axis=over)*(d[over[0]]*d[over[1]])
        got = backend.asnumpy(g.backend.ifftshift(planes[plane], axes=(0, 1)))
        np.testing.assert_allclose(got, want, atol=1e-13)

    # E and <Lz> the naive way, weighting the whole 4D array
    X = v[0][:, None, None, None]
    Y = v[1][None, :, None, None]
    PX = v[2][None, None, :, None]
    PY = v[3][None, None, None, :]
    E = ((PX**2 + PY**2)/2. + (X**2 + Y**2)/2.)*W
    assert obs.E == pytest.approx(E.sum()*dV, abs=1e-10)
    assert obs.lz == pytest.approx(((X*PY - Y*PX)*W).sum()*dV, abs=1e-10)
    assert obs.purity == pytest.approx((2.*pi)**2*(W*W).sum()*dV, abs=1e-12)


def test_cat_state_is_a_valid_2d_quantum_state(backend):
    """A 2D cat built from the factorised cross-Wigner must be a pure state:
    (2 pi hbar)^2 int W^2 = 1, the ndim-th power of the 1D bound."""
    g = grid2(backend, n=48, xlim=8.0, plim=8.0)
    sq = (0.6, 0.6)
    sk = tuple(minimal_sigma_p(s, 1.0) for s in sq)   # derived, as cats require
    comps = [GaussianComponent((-2.0, 0.0), (0.0, 0.0), sq, sk),
             GaussianComponent((2.0, 0.0), (0.0, 0.0), sq, sk)]
    W = cat_wigner(g, comps, 1.0)
    assert float(W.sum())*g.dV == pytest.approx(1.0, abs=1e-6)
    assert (2.*pi)**2*float((W*W).sum())*g.dV == pytest.approx(1.0, abs=1e-4)
    assert float(W.min()) < -1e-3         # interference fringes go negative


# ---------------------------------------------------------------------------
# 6. relativistic variants in 2D (milestone M2)
# ---------------------------------------------------------------------------

def test_relativistic_matches_an_independent_schroedinger_run(backend):
    """The relativistic counterpart of anchor 1, and the anchor that REPLACES
    separability for qr/cr (T = c*sqrt(px^2+py^2+m^2c^2) does not factorise, so
    a 2D relativistic run is not the outer product of two 1D ones).

    Both sides run relativistically: the reference applies the square-root
    (Salpeter) T exactly in the Fourier basis, sharing nothing with the Wigner
    machinery but numpy's FFT.

    c = 10 with momenta of order 1, deliberately. Two things have to hold at once
    and they pull in opposite directions — the relativistic correction must be
    big enough to be the thing under test, and the splitting error must still
    dominate the dt-INDEPENDENT ~1.5e-5 grid floor that anchor 1's docstring
    documents. Measured relative density error and ratio:

        c = 10:  1.345e-4 at dt = 0.02, 2.936e-5 at 0.01, ratio 4.58  (O(dt^2))
        c = 5 :  1.135e-4,              3.282e-5,         ratio 3.46
        c = 3 :  8.522e-5,              3.218e-5,         ratio 2.65  <- floor

    At c = 3 the fine residual has reached that floor and the convergence check
    stops meaning anything, exactly as it does for the non-relativistic anchor
    below dt = 0.005. Do not lower c to make the physics "more relativistic"."""
    C = 10.0
    coarse, _, _ = _wigner_vs_psi(backend, 0.02, 50, c=C)
    fine, obs, extra = _wigner_vs_psi(backend, 0.01, 100, c=C)
    g, planes, psi, rho_psi, xv, yv, hbar, mass, U = extra

    assert fine < 1e-4, "densities disagree: %.3g" % fine
    assert coarse/fine > 3.0, (
        "the relativistic residual is not O(dt^2) — %.3g at dt=0.02 vs %.3g at "
        "0.01. A dt-independent residual means the 4D kinetic operator is wrong, "
        "not the step size." % (coarse, fine))

    # ...and the test can actually FAIL: a no-op `relativistic` flag, or a wrong
    # sqrt, is not a small perturbation here. The relativistic and
    # non-relativistic densities differ by 1.46e-2 of the peak at c = 10 (5.53e-2
    # at c = 5, 1.32e-1 at c = 3), i.e. ~500x the residual above, so agreement
    # with the relativistic reference is evidence and not a tautology.
    nonrel = _wigner_vs_psi(backend, 0.01, 100, c=None)[2][1][(0, 1)]
    rho_rel = planes[(0, 1)]
    sep = float(np.max(np.abs(np.asarray(rho_rel) - np.asarray(nonrel)))) \
        / float(np.max(np.abs(np.asarray(nonrel))))
    assert sep > 100.*fine, (
        "the relativistic run is indistinguishable from the non-relativistic "
        "one (%.3g), so this test could pass with T unchanged" % sep)

    # <H> against psi, with the rest energy handled explicitly on both sides:
    # observables subtract m*c^2 (it cancels inside the Wigner kinetic
    # DIFFERENCE), while <T> from psi carries it.
    dx, dy = g.d[0], g.d[1]
    psit = np.fft.fftshift(np.fft.fft2(psi))*dx*dy/(2.*pi*hbar)
    nk = np.abs(psit)**2
    kxv = 2.*pi*np.fft.fftshift(np.fft.fftfreq(len(xv), d=dx))
    kyv = 2.*pi*np.fft.fftshift(np.fft.fftfreq(len(yv), d=dy))
    T_psi = float(np.sum(
        (C*np.sqrt(kxv[:, None]**2 + kyv[None, :]**2 + mass**2*C**2)
         - mass*C**2)*nk))*(kxv[1] - kxv[0])*(kyv[1] - kyv[0])
    E_psi = T_psi + float(np.sum(U(xv[:, None], yv[None, :])*rho_psi))*dx*dy
    assert obs.E == pytest.approx(E_psi, rel=1e-3)


def test_the_kinetic_bopp_difference_survives_the_mc2_cancellation(backend):
    """The named M2 risk, settled by measurement rather than argument.

    Quantum relativistic dT is (T(K + hL/2) - T(K - hL/2))/(i*h)/2 — a difference
    of terms of magnitude m*c^2 = 1.878e4 at c = 137.036 yielding a result of
    order 1e2. The stable form of the same quantity has no cancellation at all:

        T+ - T- = c*(A - B)/(sqrt(A) + sqrt(B)),   A - B = 2*h*(K.L)

    Measured absolute error against it, 2026-07-27:

        ndim=1  N=32  max|dT_im| =  29.3   err 3.61e-12   rel 1.23e-13
        ndim=1  N=64  max|dT_im| =  58.5   err 3.60e-12   rel 6.16e-14
        ndim=2  N=32  max|dT_im| =  58.4   err 4.10e-12   rel 7.02e-14
        ndim=2  N=64  max|dT_im| = 116.6   err 4.21e-12   rel 3.61e-14

    The ABSOLUTE error is flat — 3.6e-12 in 1D against 4.2e-12 in 2D — because it
    is set by m^2c^2*eps ~ 1.9e-12, a couple of ulps, and that does not care how
    many momentum components enter the sum. So a second momentum argument does
    NOT worsen the cancellation, and the relative error actually improves because
    |dT| grows. This is why construction stays float64 (in float32 the same
    difference has 200% error) and why M1 must re-measure it, not inherit it."""
    for nd, gf in ((1, grid1), (2, grid2)):
        for n in (32, 64):
            g = gf(backend, n=n)
            kw = (dict(U=lambda x: x**2/2., gradU=(lambda x: x,)) if nd == 1
                  else iso_harmonic())
            p = Propagator(g, quantum=True, relativistic=True, mass=1.0,
                           c=C_AU, hbar_eff=1.0, **kw)
            h = p.hbar_eff
            A = sum((k + h*l/2.)**2 for k, l in zip(g.K, g.Lam)) + C_AU**2
            B = sum((k - h*l/2.)**2 for k, l in zip(g.K, g.Lam)) + C_AU**2
            kdotl = sum(k*l for k, l in zip(g.K, g.Lam))
            ref = np.asarray(-C_AU*kdotl/(np.sqrt(A) + np.sqrt(B)))
            got = np.asarray(p.dT_im)
            err = float(np.max(np.abs(got - ref)))
            # 1e-11 is ~2.5x the worst measured 4.21e-12; a bound on the ABSOLUTE
            # error, because that is the quantity m^2c^2*eps actually sets.
            assert err < 1e-11, (
                "ndim=%d N=%d: the kinetic Bopp difference lost %.3g to the mc^2 "
                "cancellation" % (nd, n, err))


def test_relativistic_matches_nonrel_at_large_c(backend):
    """The 2D mirror of the 1D check: c = 1e4 makes (p/mc)^2 ~ 1e-8 while keeping
    the cancellation error (~m*c^2*eps, see above) far below the phase scale."""
    g = grid2(backend)
    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, 0.0), (0.0, 0.0), (SIG, SIG), (SIG, SIG))]))
    nr = Propagator(g, quantum=True, relativistic=False, **iso_harmonic())
    re = Propagator(g, quantum=True, relativistic=True, c=1e4,
                    **iso_harmonic())
    Wn = evolve(nr, W0, 0.01, 100)
    Wr = evolve(re, W0, 0.01, 100)
    assert float(np.max(np.abs(Wn - Wr))) < 1e-6
    # and the rest energy must be subtracted from E: <U> = 2.5, <T> = 0.5
    assert observables.compute(Wr, re).E == pytest.approx(3.0, abs=1e-3)


def _shear_diag2(backend, dt, T, **kw):
    """Evolve the 2D coherent state at fixed dt to time T, sampling every 5
    steps. Returns (max(dx*dpx) - hbar/2, peak-to-peak E, max |purity drift|) —
    the three numbers that separate anharmonic shear from numerics, exactly as
    the 1D suite's _shear_diag does."""
    g = grid2(backend)
    prop = Propagator(g, **iso_harmonic(), **kw)
    expU, expT = prop.exponents(dt)
    W = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, 0.0), (0.0, 0.0), (SIG, SIG), (SIG, SIG))]))
    excess, E, gamma = [], [], []
    for i in range(int(T/dt)):
        W = prop.solve_spectral(W, expU, expT)
        if i % 5 == 0:
            obs = observables.compute(W, prop)
            excess.append(obs.std[0]*obs.std[2] - 0.5)
            E.append(obs.E)
            gamma.append(obs.purity)
    return max(excess), max(E) - min(E), max(abs(x - gamma[0]) for x in gamma)


def test_relativistic_uncertainty_shear(backend):
    """The diagnostic M2 exists to restore: relativistic variants grow dx*dpx
    away from hbar/2 and non-relativistic ones do not, because T's -p^4/(8m^3c^2)
    term makes the orbital frequency depend on energy and the ensemble shears.
    The packet sits at (2,0) with zero momentum, so the orbit is planar and the
    y dimension is a spectator — the 2D analogue of the 1D measurement.

    c = 10 rather than 137.036: the shear scales as 1/c^4, and at c = 137 it
    needs ~1200 steps to clear the noise, which at 57 ms/step over a 32^4 grid
    is not a test. Measured at dt = 0.05, T = 10 (2026-07-27):

        non-relativistic   shear 6.01e-6   E span 1.25e-3   dpurity 3.1e-11
        relativistic c=10  shear 6.215e-3  E span 1.25e-3   dpurity 1.2e-10
        the same at dt/2   shear 6.157e-3  E span 3.10e-4   dpurity 1.1e-10
        relativistic c=20  shear 3.816e-4  E span 1.24e-3   dpurity 3.6e-11

    Three independent tells that this is physics and not discretization, all
    asserted below: halving dt leaves the shear alone (0.95%) while the O(dt^2)
    splitting oscillation of E drops 4.03x; the purity stays flat, so the shear
    is symplectic; and c=10 against c=20 gives 16.3x, i.e. the 2^4 of the
    documented 1/c^4 law. Do not "fix" this; see the gotcha in CLAUDE.md."""
    rel = dict(quantum=False, relativistic=True, c=10.0)
    flat, _, _ = _shear_diag2(backend, 0.05, 10., quantum=False,
                              relativistic=False)
    shear, E_span, dgamma = _shear_diag2(backend, 0.05, 10., **rel)

    assert shear > 100.*flat            # measured 1034x
    assert dgamma < 1e-8                # symplectic: 1.2e-10

    # dt-invariance of the shear, against a 4x drop in the splitting oscillation
    half, E_span_half, _ = _shear_diag2(backend, 0.025, 10., **rel)
    assert abs(half - shear) < 0.1*shear                 # measured 0.95%
    assert E_span/E_span_half > 3.0, (
        "E's splitting oscillation is not O(dt^2) (%.3g vs %.3g), so the shear "
        "cannot be attributed to physics" % (E_span, E_span_half))

    # 1/c^4: doubling c must cut the shear ~16x
    slow, _, _ = _shear_diag2(backend, 0.05, 10., quantum=False,
                              relativistic=True, c=20.0)
    assert 8.0 < shear/slow < 32.0, "1/c^4 scaling broken: %.3g" % (shear/slow)


@pytest.mark.parametrize("nd", [1, 2])
def test_the_massless_gradient_reduces_to_the_1d_convention(backend, nd):
    """m = 0 gives T = c|k|, whose gradient is c times the UNIT vector k_i/|k| —
    0/0 at the origin, and the origin IS a lattice point (a symmetric box with
    even N puts an exact 0.0 on every axis). It is defined as 0 there.

    That is the existing 1D convention (c*sign(p), and sign(0) == 0) written
    generically, not a new 2D one, and this test pins that it stays BITWISE so:
    sqrt(k*k) == |k| exactly for every finite lattice value, so k_i/|k| is
    exactly +-1 off the origin. Only the CLASSICAL variant reaches the gradient;
    the quantum one differentiates T through qd()."""
    g = grid1(backend, n=32) if nd == 1 else grid2(backend, n=32)
    kw = (dict(U=lambda x: x**2/2., gradU=(lambda x: x,)) if nd == 1
          else iso_harmonic())
    p = Propagator(g, quantum=False, relativistic=True, mass=0.0, c=C_AU, **kw)
    _, grads = p._kinetic()

    kvs = [backend.asnumpy(g.v[nd + i]) for i in range(nd)]
    i0 = int(np.argmin(np.abs(kvs[0])))
    assert kvs[0][i0] == 0.0, "no exact zero on the momentum lattice"

    if nd == 1:
        gm = backend.asnumpy(np.asarray(grads[0](g.v[1])))
        assert np.all(gm == C_AU*np.sign(kvs[0]))       # bitwise
        assert gm[i0] == 0.0
    else:
        kx, ky = g.v[2][:, None], g.v[3][None, :]
        gx = backend.asnumpy(np.asarray(grads[0](kx, ky)))
        gy = backend.asnumpy(np.asarray(grads[1](kx, ky)))
        assert np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))
        j0 = int(np.argmin(np.abs(kvs[1])))
        assert kvs[1][j0] == 0.0
        # along ky = 0 the 2D gradient IS the 1D one, bitwise
        assert np.all(gx[:, j0] == C_AU*np.sign(kvs[0]))
        assert gx[i0, j0] == 0.0 and gy[i0, j0] == 0.0


@pytest.mark.parametrize("quantum", [True, False])
def test_a_massless_2d_run_is_finite_and_conserved(backend, quantum):
    """m = 0 in 2D is reachable for the first time now that relativistic
    variants are (protocol.py requires exclusively relativistic variants when
    mass == 0), so it gets its own end-to-end check rather than inheriting the
    massive case's. c = 1 keeps the unit-speed motion inside the box.

    MASSLESS COSTS PURITY, AND IT IS THE GRID NOT THE STEP. T = c|k| has a KINK
    at the origin, so its Bopp difference has slowly-decaying Fourier content in
    lambda that a finite lattice truncates. Measured over 100 steps at 32^4
    (2026-07-27):

        m=0 quantum    dt=0.01  -7.19e-6     dt=0.005  -6.99e-6
        m=0 classical  dt=0.01  -4.66e-5     dt=0.005  -3.43e-5
        m=1 c=1        dt=0.01  -3.27e-9     dt=0.005  -2.96e-9
        m=1 c=137.036  dt=0.01  -4.62e-12    dt=0.005  -3.27e-12

    Halving dt does NOT help (that is the assertion below), but refining the
    MOMENTUM grid does: the massless quantum drift falls from 7.19e-6 at N=32 to
    7.16e-7 at N=48, ~10x for a 1.5x refinement. So this is spectral truncation
    of a non-smooth T, not an unstable propagator — and the remedy for a user who
    needs a clean massless run is a finer momentum axis, not a smaller dt. Norm
    is conserved to machine precision throughout, which is what says the
    evolution is still exactly unitary; it is the RESOLUTION of the kink that is
    lossy, not the map."""
    g = grid2(backend)
    p = Propagator(g, quantum=quantum, relativistic=True, mass=0.0, c=1.0,
                   **iso_harmonic())
    assert np.all(np.isfinite(np.asarray(p.dT_im)))
    assert np.all(np.isfinite(np.asarray(p.dU_im)))
    assert p.rest_energy == 0.0          # m*c^2 with m = 0

    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, 0.0), (0.0, 0.0), (SIG, SIG), (SIG, SIG))]))
    o0 = observables.compute(W0, p)

    def run(dt, nsteps):
        o = observables.compute(evolve(p, W0, dt, nsteps), p)
        assert np.isfinite(o.E)
        # |expU| = |expT| = 1 regardless of how well the kink is resolved, so
        # norm is the invariant that must still hold to machine precision
        assert o.norm == pytest.approx(o0.norm, abs=1e-12)
        return abs(o.purity - o0.purity)

    coarse = run(0.01, 100)
    fine = run(0.005, 200)
    assert coarse < 1e-4, "massless purity loss %.3g is out of family" % coarse
    # dt-INDEPENDENT: a splitting error would fall ~4x here. If this ever starts
    # scaling with dt, the massless branch has acquired a time-integration bug
    # and the kink explanation above no longer applies.
    assert fine > 0.3*coarse, (
        "the massless purity drift fell with dt (%.3g -> %.3g), so it is no "
        "longer the kink-truncation effect this test documents" % (coarse, fine))
