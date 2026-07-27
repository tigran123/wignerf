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
from core.xp import ArrayBackend

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

def _schroedinger(xv, yv, psi, U, mass, hbar, dt, nsteps):
    """Split-operator TDSE on psi(x,y): exp(-iU dt/2) exp(-iT dt) exp(-iU dt/2)
    with T = (kx^2+ky^2)/2m in the Fourier basis. Deliberately NOT the Wigner
    machinery — no Bopp shifts, no phase-space grid, nothing shared but numpy's
    FFT — so agreement is evidence, not tautology."""
    dx, dy = xv[1] - xv[0], yv[1] - yv[0]
    kx = 2.*pi*np.fft.fftfreq(len(xv), d=dx)[:, None]
    ky = 2.*pi*np.fft.fftfreq(len(yv), d=dy)[None, :]
    Umesh = U(xv[:, None], yv[None, :])
    halfU = np.exp(-0.5j*dt*Umesh/hbar)
    expT = np.exp(-1j*dt*hbar*(kx**2 + ky**2)/(2.*mass))
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


def _wigner_vs_psi(backend, dt, nsteps):
    """Run both methods to the same t and return
    (relative rho error, obs, psi diagnostics)."""
    hbar, mass = 1.0, 1.0
    U, gradU = _hh()
    # p spans exactly the FFT frequency range of the Schroedinger x-grid, so
    # the momentum plane and |psi~|^2 land on the SAME lattice and can be
    # compared cell by cell rather than interpolated.
    g = grid2(backend, n=32, xlim=6.0, plim=pi*32/12.0)
    q0, sig, k0 = (1.0, 0.5), (SIG, SIG), (0.0, 0.6)
    prop = Propagator(g, quantum=True, mass=mass, hbar_eff=hbar,
                      U=U, gradU=gradU)
    W = g.shift(cat_wigner(g, [GaussianComponent(q0, k0, sig, sig)], hbar))

    xv, yv = backend.asnumpy(g.v[0]), backend.asnumpy(g.v[1])
    psi = _coherent_psi(xv, yv, q0, sig, k0, hbar)

    W = evolve(prop, W, dt, nsteps)
    psi = _schroedinger(xv, yv, psi, U, mass, hbar, dt, nsteps)

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
    carries real Moyal corrections in y — and must still factorise exactly."""
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


@pytest.mark.parametrize("quantum", [True, False])
def test_angular_momentum(backend, quantum):
    """<Lz> is conserved by a CENTRAL potential and not by an anisotropic one —
    a genuinely two-dimensional invariant, and one no 1D test can reach."""
    g = grid2(backend)
    W0 = g.shift(mixture_wigner(g, [GaussianComponent(
        (2.0, 0.0), (0.0, 1.0), (SIG, SIG), (SIG, SIG))]))

    central = Propagator(g, quantum=quantum, **iso_harmonic())
    lz0 = observables.compute(W0, central).lz
    # x*py - y*px = 2*1 - 0; the 1.5e-7 deficit is Gaussian tail truncation at
    # the domain edge, the same bound test_mixture_normalization uses in 1D
    assert lz0 == pytest.approx(2.0, abs=2e-6)
    W = evolve(central, W0, 0.01, 300)
    # Measured 5.65e-6 over 300 steps — and IDENTICAL for the quantum and
    # classical variants, which is the tell that it is the square LATTICE
    # breaking rotational symmetry (and the periodic torus), not physics.
    assert observables.compute(W, central).lz == pytest.approx(lz0, abs=2e-5)

    aniso = Propagator(g, quantum=quantum,
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
