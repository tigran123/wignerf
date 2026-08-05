"""
The two EXPRESSION initial conditions: an arbitrary analytic W on the phase
space, and an arbitrary analytic psi from which W comes by the Wigner transform.

The anchors here validate the discrete transform against code this suite already
trusts — initial.cat_wigner and initial.mixture_wigner, whose closed forms are
exact on the grid. That is the same tactic the 2D suite uses (a separable 2D run
against two 1D runs): compare new machinery with old machinery rather than with a
tolerance nobody can justify.
"""

from math import pi, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from core.expr import ExprError
from core.grid import Axis, Grid
from core.initial import (GaussianComponent, ICError, cat_wigner, compile_ic,
                          minimal_sigma_p, mixture_wigner, preview_warnings,
                          psi_wigner, wexpr_wigner)
from core.xp import ArrayBackend

BACKEND = ArrayBackend(device="cpu")


def _ic(kind, expr):
    """An ICSpec-shaped duck — from_spec and compile_ic are duck-typed over
    anything with .type/.components/.expr, which is what lets the physics tests
    stay free of pydantic."""
    return SimpleNamespace(type=kind, components=[], expr=expr)


def _compiled(kind, expr, ndim=1):
    return compile_ic(_ic(kind, expr), 1.0, ndim)


def _comp(x0, p0, sx, weight=1.0, phase=0.0):
    return GaussianComponent(x0, p0, sx, minimal_sigma_p(sx), weight, phase)


def _psi_src(x0, p0, s, hbar=1.0, var="x"):
    """The minimal packet initial.py's cat builder assumes, as a string:
    N exp(-(x-x0)^2/(4 s^2) + i p0 (x-x0)/hbar). `var` names the coordinate —
    a str.replace on the result would rewrite 'exp' into 'eyp'."""
    return ("(2*pi*%r^2)^(-1/4)*exp(-(%s-(%r))^2/(4*%r^2) + I*%r*(%s-(%r))/%r)"
            % (s, var, x0, s, p0, var, x0, hbar))


# The momentum box is NON-SYMMETRIC on purpose in most of what follows: it is
# the case the textbook shifted-FFT recipe gets wrong, and a symmetric box
# cannot tell the two apart. See test_the_momentum_ramp_is_not_optional.
def _grid(x=(-8.0, 8.0, 128), p=(-5.0, 9.0, 256)):
    return Grid((Axis(*x), Axis(*p)), BACKEND)


# -- the transform against the analytic closed forms --------------------------

@pytest.mark.parametrize("pbox", [(-9.0, 9.0, 256), (-5.0, 9.0, 256)],
                         ids=["symmetric-p", "shifted-p"])
def test_a_psi_expression_reproduces_the_analytic_coherent_state(pbox):
    """p0 != 0 is the point, not decoration: conjugating the WRONG argument
    reflects W in k, which every p0 = 0 check passes.

    BOTH boxes hold the packet ~8.4 sigma_p clear of the nearest momentum edge,
    and that is what the tolerance below is really about. The transform lives on
    a torus and the analytic reference does not, so the residual is the wrapped
    tail and tracks exp(-n^2/2) in sigma_p to the nearest edge exactly: measured
    1.52e-08 at 6 sigma (predicted 1.5e-08), 9.5e-15 at 8.4 sigma. Tightening
    the box instead of the tolerance is what keeps this a test of the TRANSFORM
    rather than of Gaussian tails."""
    g = _grid(p=pbox)
    x0, p0, s = 1.3, 2.0, 0.6
    W, norm = psi_wigner(g, _compiled("psi", _psi_src(x0, p0, s)), 1.0)
    ref = mixture_wigner(g, [_comp(x0, p0, s)])
    assert np.abs(W - ref).max()/np.abs(ref).max() < 1e-10
    assert norm.momentum_mass == pytest.approx(1.0, abs=1e-6)


def test_a_psi_expression_reproduces_the_analytic_cat_state():
    """Including the interference fringes — the amplitude test below is what
    stops a version that reproduced the envelope and lost the cross term."""
    g = Grid((Axis(-10.0, 10.0, 256), Axis(-9.0, 7.0, 256)), BACKEND)
    a, b = _comp(-2.0, 0.5, 0.7, 1.0, 0.0), _comp(2.0, -1.0, 0.5, 0.6, 0.9)
    src = "%s + sqrt(0.6)*exp(I*0.9)*%s" % (_psi_src(-2.0, 0.5, 0.7),
                                            _psi_src(2.0, -1.0, 0.5))
    W, _ = psi_wigner(g, _compiled("psi", src), 1.0)
    ref = cat_wigner(g, [a, b], 1.0)
    assert np.abs(W - ref).max()/np.abs(ref).max() < 1e-10
    assert W.min() < -0.05*W.max()      # the fringes are really there


def test_the_momentum_ramp_is_not_optional():
    """The exp(i*k0*theta) factor, dropped, is not a small error: it is a
    DIFFERENT state. Measured relative error 1.0 on a SYMMETRIC box — while the
    norm stays exactly 1 — so no norm-, energy- or purity-based assertion can
    see it and only this cell-by-cell comparison can. Do not 'simplify' the
    ramp away on the grounds that the box looks symmetric."""
    g = _grid(p=(-7.0, 7.0, 256))
    x0, p0, s = 1.3, 2.0, 0.6
    cic = _compiled("psi", _psi_src(x0, p0, s))
    ref = mixture_wigner(g, [_comp(x0, p0, s)])
    good, _ = psi_wigner(g, cic, 1.0)
    assert np.abs(good - ref).max()/np.abs(ref).max() < 1e-7

    # the same build with the ramp removed
    th = g.nat_dual_mesh(1)
    q = g.nat_mesh(0).astype(complex)
    f = cic.f
    A = f(q - 0.5*th)*np.conj(f(q + 0.5*th))
    A = np.fft.ifft(A, axis=1)*((-1.0)**np.arange(g.N[1])).reshape(1, -1)
    bad = (A/g.d[1]).real
    bad = bad/(bad.sum()*g.dmu)
    assert np.abs(bad - ref).max()/np.abs(ref).max() > 0.5
    assert bad.sum()*g.dmu == pytest.approx(1.0, abs=1e-9)   # norm is blind


def test_hbar_eff_scales_the_psi_transform():
    for h in (0.5, 1.0, 2.0):
        g = Grid((Axis(-9.0, 9.0, 256), Axis(-6.0, 10.0, 256)), BACKEND)
        s = 0.8
        W, _ = psi_wigner(g, _compiled("psi", _psi_src(1.0, 2.0, s, h)), h)
        ref = mixture_wigner(g, [GaussianComponent(1.0, 2.0, s,
                                                   minimal_sigma_p(s, h))])
        assert np.abs(W - ref).max()/np.abs(ref).max() < 1e-7
        assert 2.*pi*h*(W*W).sum()*g.dmu == pytest.approx(1.0, abs=1e-9)


def test_a_psi_expression_is_pure_and_fock_states_are_exact():
    """W_n(0,0) = (-1)^n/(pi*hbar) for a Fock state is an exact analytic
    identity, so it pins the transform's NORMALIZATION as well as its shape."""
    g = Grid((Axis(-10.0, 10.0, 512), Axis(-10.0, 10.0, 512)), BACKEND)
    for n in (0, 1, 3):
        W, _ = psi_wigner(g, _compiled("psi", "hermite(%d,x)*exp(-x^2/2)" % n),
                          1.0)
        assert 2.*pi*(W*W).sum()*g.dmu == pytest.approx(1.0, abs=1e-9)
        i = int(np.argmin(np.abs(g.v[0])))
        j = int(np.argmin(np.abs(g.v[1])))
        assert W[i, j] == pytest.approx((-1.)**n/pi, abs=1e-9)


# -- what the deficit can and cannot see --------------------------------------

def test_the_momentum_box_aliases_rather_than_truncating():
    """A packet whose mean momentum lies OUTSIDE the box is not lost — it is
    folded back in at full height, and the norm, the purity and the edge band
    all read perfectly healthy. Measured: norm 1.0000000000, purity 1.000000,
    p-edge band 6.5e-06 against its own 1e-6 trigger. The direct-quadrature
    momentum mass is the only thing that sees it (7.2e-07 against 1.0), which
    is why _psi_lattice computes it and why this test exists."""
    src = _psi_src(1.3, 2.0, 0.6)
    good, ngood = psi_wigner(_grid(p=(-7.0, 7.0, 256)), _compiled("psi", src), 1.0)
    g = _grid(p=(-10.0, -2.0, 256))
    bad, nbad = psi_wigner(g, _compiled("psi", src), 1.0)

    assert bad.sum()*g.dmu == pytest.approx(1.0, abs=1e-9)      # norm: blind
    assert 2.*pi*(bad*bad).sum()*g.dmu == pytest.approx(1.0, abs=1e-6)  # purity: blind
    assert np.abs(bad).max() == pytest.approx(np.abs(good).max(), rel=1e-2)

    assert ngood.momentum_mass > 0.999                          # ...and this sees it
    assert nbad.momentum_mass < 1e-3
    warns = preview_warnings(g, (), "psi", 1.0, bad, edge_axes=[], norm=nbad)
    assert any("ALIASED" in w for w in warns)


def test_the_psi_norm_deficit_is_the_spatial_mass_outside_the_box():
    """Normalizing over the EXTENDED spatial box is what keeps the deficit
    meaningful; normalizing over the visible one makes it a structural zero."""
    g = Grid((Axis(0.0, 6.0, 128), Axis(-8.0, 8.0, 256)), BACKEND)
    W, norm = psi_wigner(g, _compiled("psi", "exp(-(x-0.2)^2/(4*0.6^2))"), 1.0)
    assert norm.deficit > 0.3                     # most of this packet is at x<0
    assert W.sum()*g.dmu == pytest.approx(1.0 - norm.deficit, abs=1e-9)


# -- the W expression ---------------------------------------------------------

def test_a_w_expression_reproduces_the_analytic_gaussian():
    g = _grid()
    s, sk = 0.6, minimal_sigma_p(0.6)
    src = ("exp(-(x-1.3)^2/(2*%r^2) - (p-2.0)^2/(2*%r^2))" % (s, sk))
    W, raw = wexpr_wigner(g, _compiled("wexpr", src))
    ref = mixture_wigner(g, [_comp(1.3, 2.0, s)])
    assert np.abs(W - ref).max()/np.abs(ref).max() < 1e-9
    assert raw == pytest.approx(2.*pi*s*sk, rel=1e-6)   # unnormalised on input


def test_an_unnormalised_w_expression_is_normalised():
    g = _grid()
    a, _ = wexpr_wigner(g, _compiled("wexpr", "exp(-x^2-p^2)"))
    b, raw = wexpr_wigner(g, _compiled("wexpr", "17.5*exp(-x^2-p^2)"))
    np.testing.assert_allclose(a, b, atol=1e-15)
    assert a.sum()*g.dmu == pytest.approx(1.0, abs=1e-12)
    assert raw == pytest.approx(17.5*pi, rel=1e-6)


@pytest.mark.parametrize("src", ["x*p*exp(-x^2-p^2)", "-exp(-x^2-p^2)"],
                         ids=["odd-integrates-to-zero", "negative-total"])
def test_a_w_expression_with_a_non_positive_integral_is_refused(src):
    # ICError, not ValueError: the CLASS is the load-bearing property. It exists
    # only so routers/preview.py can tell a client error from a cupy OOM — one
    # gets a 422, the other falls back to the CPU — so a test that accepts any
    # ValueError would stay green while that distinction was removed.
    with pytest.raises(ICError, match="cannot be normalized"):
        wexpr_wigner(_grid(), _compiled("wexpr", src))


def test_a_w_expression_may_be_an_invalid_quantum_state_and_only_warns():
    """A hand-written W is almost never a state; that is the point of the tab,
    and the purity bound reports it rather than refusing it."""
    g = _grid()
    W, _ = wexpr_wigner(g, _compiled("wexpr", "exp(-4*x^2-4*p^2)"))
    warns = preview_warnings(g, (), "wexpr", 1.0, W, edge_axes=[])
    assert any("not a valid quantum state" in w for w in warns)


def test_a_psi_is_never_accused_of_not_being_a_quantum_state():
    """Same detector, different sentence: a wavefunction always IS a valid pure
    state, so an over-1 purity can only mean the grid is aliasing it."""
    g = Grid((Axis(-8.0, 8.0, 128), Axis(-7.0, 7.0, 64)), BACKEND)
    W, norm = psi_wigner(g, _compiled("psi", "exp(-x^2/(4*0.06^2))"), 1.0)
    warns = preview_warnings(g, (), "psi", 1.0, W, edge_axes=[], norm=norm)
    assert not any("not a valid quantum state" in w for w in warns)
    assert any("GRID" in w or "ALIASED" in w for w in warns)


# -- compilation: the safety boundary and the per-kind namespaces --------------

@pytest.mark.parametrize("kind", ["wexpr", "psi"])
@pytest.mark.parametrize("bad", [
    "__import__('os').system('id')", "x.__class__", "open('/etc/passwd')",
    "eval('1')", "lambda: 1", "'abc'", "x = 1", "", "9**9**9",
    "hermite(5000, x)", "hermite(x, x)",
])
def test_the_screen_refuses_the_same_things_for_every_kind(kind, bad):
    with pytest.raises(ValueError):
        _compiled(kind, bad)


def test_the_imaginary_unit_is_a_psi_privilege():
    _compiled("psi", "exp(-x^2)*I")                  # fine
    for src in ("exp(-x^2-p^2)*I", "sqrt(-1)*exp(-x^2-p^2)"):
        with pytest.raises(ExprError, match="must be a real expression"):
            _compiled("wexpr", src)


def test_each_kind_gets_only_its_own_variables():
    _compiled("wexpr", "exp(-x^2-p^2)")
    _compiled("wexpr", "exp(-x^2-y^2-px^2-py^2)", ndim=2)
    _compiled("psi", "exp(-x^2-y^2)", ndim=2)
    # psi lives on configuration space: psi(x,p) would be nonsense
    with pytest.raises(ExprError, match="may appear as a variable"):
        _compiled("psi", "exp(-x^2-p^2)")
    # ...and a 1D W does not know about px
    with pytest.raises(ExprError, match="may appear as a variable"):
        _compiled("wexpr", "exp(-x^2-px^2)")


def test_a_non_finite_expression_is_refused_with_its_box():
    with pytest.raises(ICError, match="not finite"):
        compile_ic(_ic("psi", "1/x"), 1.0, 1, ranges=((-6.0, 6.0),))
    with pytest.raises(ICError, match="not finite"):
        compile_ic(_ic("wexpr", "1/(x^2+p^2)"), 1.0, 1,
                   ranges=((-6.0, 6.0), (-6.0, 6.0)))


def test_a_w_is_probed_on_the_dtype_it_is_BUILT_on():
    """The probe dtype must match the build, and the two kinds disagree.

    psi is evaluated at q -+ hbar*theta/2, a COMPLEX array, so sqrt of a
    negative real is a finite branch value there and a real probe would refuse a
    legal psi. A W is sampled on grid.nat_mesh, which is float64, so probing IT
    complex accepts what the build then cannot produce: sqrt(x) passed
    compile_ic, POST /sessions answered 200, and every SolverWorker died in
    from_spec afterwards — the exact failure compiling at the door prevents.
    """
    box = ((-6.0, 6.0), (-6.0, 6.0))
    with pytest.raises(ICError, match="not finite"):
        compile_ic(_ic("wexpr", "sqrt(x)*exp(-p^2)"), 1.0, 1, ranges=box)
    # ...and the same expression is fine as a psi, on the same span
    compile_ic(_ic("psi", "sqrt(x)*exp(-x^2)"), 1.0, 1, ranges=box[:1])


@pytest.mark.parametrize("kind,src", [
    ("wexpr", "exp(-x^2-p^2)*exp(709)"),
    ("psi", "exp(-x^2/2)*exp(360)"),
], ids=["wexpr-sum-overflows", "psi-square-overflows"])
def test_an_overflowing_normalizer_is_refused_not_silently_applied(kind, src):
    """The normalizer is bounded from ABOVE as well as below.

    Every sample can be finite while the total is not, and the amplitude that
    does it is reachable in a dozen characters precisely because these kinds
    auto-normalize ("scale doesn't matter" is the selling point). Dividing by
    inf was silent in both directions: a wexpr came out identically ZERO (blank
    panel, integral 0, no warning at all) and a psi came out all-NaN, which no
    diagnostic can see either — every one of them compares with `>` and NaN
    fails every comparison. See test_a_nan_state_is_invisible_to_every_warning.
    """
    g = _grid()
    with pytest.raises(ICError, match="overflow"):
        if kind == "wexpr":
            wexpr_wigner(g, _compiled("wexpr", src))
        else:
            psi_wigner(g, _compiled("psi", src), 1.0)


def test_a_small_amplitude_w_is_normalised_rather_than_refused():
    """The lower bound is RELATIVE, because the quantity is about to be
    normalized. An absolute floor refused 1e-15*exp(-x^2-p^2) — a positive,
    well-formed, perfectly ordinary state — with a sentence claiming it had no
    positive total."""
    g = _grid()
    W, raw = wexpr_wigner(g, _compiled("wexpr", "1e-15*exp(-x^2-p^2)"))
    assert W.sum()*g.dmu == pytest.approx(1.0, abs=1e-12)
    assert raw == pytest.approx(1e-15*pi, rel=1e-6)


def test_a_nan_state_is_invisible_to_every_warning():
    """Why the builders must refuse rather than leave it to the diagnostics.

    This is the property that makes the two guards above load-bearing instead of
    belt-and-braces: preview_warnings cannot report a NaN state, and neither can
    the purity bound, the edge band or the momentum-mass detector, because all of
    them are `>` comparisons. Nothing downstream sees it either — worker._finite
    checks the rate meshes, never W."""
    g = _grid()
    W = np.full(g.shape, np.nan)
    from core.initial import ICNorm
    nan = float("nan")
    warns = preview_warnings(g, (), "psi", 1.0, W, edge_axes=[],
                             norm=ICNorm(method="psi-extended", raw=1.0,
                                         deficit=nan, momentum_mass=nan))
    assert warns == []


# -- 2D ------------------------------------------------------------------------
#
# Anisotropic axis counts and BOTH momentum boxes shifted, the discipline
# gen_fixture.py uses: a transposed theta<->k pairing is the classic multi-D
# error (core/axes.conjugate) and a cubic grid cannot see it.

def _grid2():
    """ANISOTROPIC is what matters here — it is what a transposed pairing
    cannot survive — and the counts are then chosen fine enough that the
    assertion is about the transform rather than about quadrature: measured
    9.7e-05 at 24×20×32×28 and 6.2e-15 at exactly twice that, i.e. converging
    hard, so the coarse grid was testing Gaussian sampling and nothing else."""
    return Grid((Axis(-7.0, 7.0, 48), Axis(-6.0, 6.0, 40),
                 Axis(-5.0, 7.0, 64), Axis(-6.0, 8.0, 56)), BACKEND)


def test_a_2d_psi_expression_reproduces_the_separable_analytic_state():
    """A separable psi is the outer product of two 1D packets, so cat_wigner's
    closed form is the reference — the same tactic
    test_separable_run_equals_two_1d_runs uses for the propagator."""
    g = _grid2()
    src = "(%s)*(%s)" % (_psi_src(1.0, 0.5, 0.9),
                         _psi_src(-0.5, 1.0, 0.8, var="y"))
    W, norm = psi_wigner(g, _compiled("psi", src, ndim=2), 1.0)
    ref = mixture_wigner(g, [GaussianComponent(
        (1.0, -0.5), (0.5, 1.0), (0.9, 0.8),
        (minimal_sigma_p(0.9), minimal_sigma_p(0.8)))])
    assert np.abs(W - ref).max()/np.abs(ref).max() < 1e-9
    assert norm.momentum_mass == pytest.approx(1.0, abs=1e-6)


def test_a_coupled_2d_psi_is_normalised_and_pure():
    """NON-separable, which is the case the 2D suite's separability anchor is
    structurally blind to: it validates the 4D pipeline by factorising, so a
    state that does not factorise is exactly what it cannot check."""
    g = Grid(tuple(Axis(-6.0, 6.0, 32) for _ in range(4)), BACKEND)
    W, norm = psi_wigner(
        g, _compiled("psi", "exp(-(x^2+y^2+0.8*x*y)/2 + I*0.7*x*y)", ndim=2),
        1.0)
    assert W.sum()*g.dmu == pytest.approx(1.0, abs=1e-9)
    assert (2.*pi)**2*(W*W).sum()*g.dmu == pytest.approx(1.0, abs=1e-6)
    assert norm.momentum_mass > 0.999
    # ...and the coupling is really in the state: the (x,y) density is not the
    # outer product of its own two marginals.
    n_q = W.sum(axis=(2, 3))*g.d[2]*g.d[3]
    outer = np.outer(n_q.sum(axis=1)*g.d[1], n_q.sum(axis=0)*g.d[0])
    assert np.abs(n_q - outer).max()/n_q.max() > 0.05


def test_a_2d_w_expression_is_normalised_on_the_whole_phase_space():
    g = _grid2()
    W, raw = wexpr_wigner(
        g, _compiled("wexpr", "3.5*exp(-x^2-y^2-px^2-py^2)", ndim=2))
    assert W.sum()*g.dmu == pytest.approx(1.0, abs=1e-9)
    assert raw == pytest.approx(3.5*pi**2, rel=1e-4)
