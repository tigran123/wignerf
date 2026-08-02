"""
The opt-in float32 ("preview") solver mode.

The rest of the suite pins float64: every other fixture builds an
ArrayBackend at its default precision, so the correctness anchors in
test_propagator.py — above all test_relativistic_matches_nonrel_at_large_c
and test_relativistic_uncertainty_shear, whose claims this mode is documented
as unable to support — keep asserting their 1e-12/1e-10 bounds against double
arithmetic and are untouched by the feature. This module is the other half:
it pins that float32 does what it promises and nothing more.

Two of these tests assert DTYPES rather than numbers, which looks fussy until
you notice both of the mode's failure modes are invisible in results:

  - hand a complex64 array to a complex128 pyFFTW plan and auto_align_input
    copies it up, returning a correct complex128 answer at complex128 speed;
  - `B *= expT` with B complex64 and expT complex128 is legal in numpy AND
    cupy, and yields a correct complex64 B through a full complex128
    temporary.

Both produce right answers, so no physics assertion can catch either. Only
the dtype can.
"""

from math import pi, sqrt

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core import boundary, describe, observables
from core.grid import Axis, Grid, embed_window
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.protocol import TOL_MIN_F32
from core.xp import ArrayBackend
from main import app

SIG = 1.0/sqrt(2.0)
HARMONIC = dict(U=lambda x: x**2/2., gradU=(lambda x: x,))
# the 2D counterpart: isotropic, so it is both separable and central
HARMONIC_2D = dict(U=lambda x, y: (x**2 + y**2)/2.,
                   gradU=(lambda x, y: x + 0.*y, lambda x, y: y + 0.*x))


def _harmonic(ndim):
    """Propagator kwargs at either dimensionality. Not interchangeable — a
    1-argument U with a 1-tuple gradU is REFUSED at ndim=2 by _check_grad, which
    is the point of keeping them apart rather than broadcasting one."""
    return HARMONIC if ndim == 1 else HARMONIC_2D


def _grid(precision, n=64, ndim=1):
    """The 1D default n=64 is unchanged; 2D callers pass their own, because n
    is PER AXIS and 64^4 is 16.8M cells against 64^2's 4096."""
    b = ArrayBackend(device="cpu", precision=precision)
    if ndim == 1:
        return Grid.from_1d(-6.0, 6.0, n, -7.0, 7.0, n, b)
    return Grid((Axis(-6.0, 6.0, n), Axis(-6.0, 6.0, n),
                 Axis(-7.0, 7.0, n), Axis(-7.0, 7.0, n)), b)


def _coherent(grid):
    """The same physical state at either dimensionality: a coherent packet at
    q = (2, 0…) with zero momentum, so a 2D run is the 1D one with a spectator
    second dimension and the two are directly comparable."""
    nd = grid.ndim
    c = (GaussianComponent(2.0, 0.0, SIG, SIG) if nd == 1
         else GaussianComponent((2.0, 0.0), (0.0, 0.0), (SIG, SIG), (SIG, SIG)))
    return grid.shift(mixture_wigner(grid, [c]))


def _evolve(prop, W, dt, nsteps):
    expU, expT = prop.exponents(dt)
    for _ in range(nsteps):
        W = prop.solve_spectral(W, expU, expT)
    return W


# -- what must NOT change in float32 -------------------------------------


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("quantum,relativistic",
                         [(True, False), (True, True), (False, False), (False, True)])
def test_exponent_construction_stays_double(quantum, relativistic, ndim):
    """The rate meshes and H are built in float64 in BOTH modes, and are
    bitwise identical between them.

    This is the load-bearing property of the whole design. Relativistic dT
    built in float32 has max abs error 455 against max |dT| = 228 — 200% —
    because mc^2 cancels inside a difference of ~1.9e4-magnitude terms, so a
    float32 session that let single precision reach _rate_mesh would not be a
    less accurate simulation, it would be a different one. The parametrization
    covers the relativistic variants precisely because they are the ones that
    cannot survive it.

    ndim is the OTHER parametrization, and it is what milestone M1 turned on:
    the multi-D Bopp shift moves every spatial argument of U together
    (propagator.qd), so "construction stays double" had to be re-verified at 4
    axes rather than inherited from 2. It holds bitwise, which is the honest
    reason float32 could be allowed in 2D at all — nothing in the mixed split is
    dimension-aware, and this is the assertion that says so."""
    props = {}
    for precision in ("float64", "float32"):
        g = _grid(precision, n=16 if ndim > 1 else 64, ndim=ndim)
        props[precision] = Propagator(g, quantum=quantum,
                                      relativistic=relativistic,
                                      **_harmonic(ndim))
    a, b = props["float64"], props["float32"]
    for name in ("dU_im", "dT_im", "U_mesh", "T_mesh"):
        assert getattr(b, name).dtype == np.float64, name
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_observables_stay_float64():
    """Reductions accumulate and leave in float64 whatever the solver does, so
    rho/phi keep the dtype history.py's byte accounting and the <f4 wire codec
    were written against."""
    g = _grid("float32")
    prop = Propagator(g, quantum=True, **HARMONIC)
    obs = observables.compute(_coherent(g), prop)
    assert obs.rho.dtype == np.float64 and obs.phi.dtype == np.float64
    assert isinstance(obs.norm, float) and isinstance(obs.purity, float)


def test_the_2d_reduction_path_does_not_leak_float32():
    """The same claim at 4 axes, where there is far more of it to get wrong: six
    plane reductions and four marginals rather than one plane and two.

    Run FIRST among the 2D cases when something looks wrong, because it does no
    time stepping at all — it reduces a freshly built IC, so a failure here is a
    dtype leak in the reduction path and nothing else. That separation matters:
    every other float32 measurement in this file mixes the reduction path with
    accumulated stepping error and could not tell you which one moved.

    The values must also match float64 to full double tolerance, which they can:
    initial.py builds in float64 at either precision and only worker.py casts the
    state, so at record 0 the two modes are reducing arrays that differ by
    nothing at all."""
    got = {}
    for precision in ("float64", "float32"):
        g = _grid(precision, n=16, ndim=2)
        prop = Propagator(g, quantum=True, **HARMONIC_2D)
        W = _coherent(g)
        planes = observables.reduce_planes(W, g)
        obs = observables.compute(W, prop, planes)
        for p in planes.values():
            assert p.dtype == np.float64, precision
        for m in obs.marg:
            assert m.dtype == np.float64, precision
        assert len(planes) == 6 and len(obs.marg) == 4
        got[precision] = obs
    a, b = got["float64"], got["float32"]
    assert b.norm == pytest.approx(a.norm, abs=1e-12)
    assert b.purity == pytest.approx(a.purity, abs=1e-12)
    assert b.E == pytest.approx(a.E, abs=1e-12)
    assert b.lz == pytest.approx(a.lz, abs=1e-12)


def test_regrid_keeps_the_working_dtype():
    """embed_window must not upcast a float32 state back to float64 — that
    would silently double the working set at exactly the moment (auto-expand)
    the domain just got bigger."""
    from core.grid import GridState
    old = GridState.from_1d(x0=-6.0, p0=-7.0, dx=12.0/64, dp=14.0/64,
                            ox=0, op=0, Nx=64, Np=64)
    new = old.moved(0, -32, 128)                # doubled along x, support centred
    for dtype in (np.float64, np.float32):
        Wn = np.ones((64, 64), dtype=dtype)
        out = embed_window(Wn, old, new, np)
        assert out.dtype == dtype
        # and the copy is still bitwise exact, which is the whole point of
        # the fixed-lattice regrid
        assert np.array_equal(out[32:96, :], Wn)


# -- what float32 must actually DO ---------------------------------------


def test_float32_really_is_single_precision():
    """The spectral path is complex64 end to end. Asserting the pyFFTW plan
    dtype is the point: a complex128 plan silently upcasts its input and
    returns a correct answer at double speed and double memory."""
    g = _grid("float32")
    assert g.backend.complex_dtype is np.complex64
    prop = Propagator(g, quantum=True, **HARMONIC)
    expU, expT = prop.exponents(0.01)
    assert expU.dtype == np.complex64 and expT.dtype == np.complex64
    W = prop.solve_spectral(_coherent(g), expU, expT)
    assert W.dtype == np.float32
    if g.backend.fft_provider == "pyfftw":
        probe = np.zeros((64, 64), dtype=np.complex64)
        assert prop._fft_sp(probe).dtype == np.complex64
        assert prop._ifft_mo(probe).dtype == np.complex64


def test_float32_really_is_single_precision_in_2d():
    """The same end-to-end dtype claim at 4 axes, and it is NOT redundant: the
    2D path takes fft_pair's MULTI-AXIS branch (pyfftw.builders.fftn planned on
    aligned buffers) where 1D takes the one-dimensional entry points, so the plan
    whose dtype could be wrong is a different object built by different code.

    The failure it guards against is silent in every result: a complex64 array
    handed to a complex128 plan is copied up by auto_align_input and comes back
    correct, at double speed and double memory — i.e. float32 would appear to
    work while buying nothing, which at 4D is the entire reason to choose it."""
    g = _grid("float32", n=16, ndim=2)
    prop = Propagator(g, quantum=True, **HARMONIC_2D)
    expU, expT = prop.exponents(0.01)
    assert expU.dtype == np.complex64 and expT.dtype == np.complex64
    assert expU.shape == (16, 16, 16, 16)
    W = prop.solve_spectral(_coherent(g), expU, expT)
    assert W.dtype == np.float32
    if g.backend.fft_provider == "pyfftw":
        probe = np.zeros((16,)*4, dtype=np.complex64)
        assert prop._fft_sp(probe).dtype == np.complex64
        assert prop._ifft_mo(probe).dtype == np.complex64


def test_the_massless_2d_gradient_is_unchanged_by_float32():
    """The float64-mesh half of the mixed split, checked where it is checkable
    BITWISE rather than to a tolerance.

    T = c|k| at m = 0 has a 0/0 gradient at the lattice origin and is defined as
    0 there (see propagator._kinetic). That value comes off the grid meshes,
    which stay float64 in both modes — so the whole gradient must be byte-for-byte
    identical between them, at 4 axes as at 2. A tolerance-based version of this
    could not distinguish "built in double" from "built in single and close"."""
    for ndim in (1, 2):
        out = {}
        for precision in ("float64", "float32"):
            g = _grid(precision, n=16, ndim=ndim)
            p = Propagator(g, quantum=False, relativistic=True, mass=0.0,
                           **_harmonic(ndim))
            _, grads = p._kinetic()
            ks = [g.v[ndim + i] for i in range(ndim)]
            mesh = [k.reshape((1,)*i + (-1,) + (1,)*(ndim - 1 - i))
                    for i, k in enumerate(ks)]
            out[precision] = [np.asarray(grads[i](*mesh)) for i in range(ndim)]
        for i, (a, b) in enumerate(zip(out["float64"], out["float32"])):
            assert a.dtype == np.float64 and b.dtype == np.float64
            assert np.array_equal(a, b), "ndim=%d axis %d" % (ndim, i)


def test_float64_is_the_default():
    b = ArrayBackend(device="cpu")
    assert b.precision == "float64" and b.complex_dtype is np.complex128


def test_host_precision_is_the_schema_default(monkeypatch):
    """WIGNERF_PRECISION must actually DRIVE the default, not merely be
    advertised. It was decorative once: `/api/device` reported float32 while
    every session that omitted the field was built with a hard-coded float64,
    so the setting silently did nothing.

    Both baselines are pinned with monkeypatch rather than assumed: run the
    suite ON a float32 host (WIGNERF_PRECISION=float32, which this project
    supports) and a test that reads the ambient default fails for a reason that
    has nothing to do with what it is checking."""
    import config
    from core.protocol import SessionCreate
    monkeypatch.setattr(config, "PRECISION", "float64")
    assert SessionCreate(**_cfg()).precision == "float64"
    monkeypatch.setattr(config, "PRECISION", "float32")
    assert SessionCreate(**_cfg()).precision == "float32"
    # an explicit choice still wins over the host's
    assert SessionCreate(**_cfg(precision="float64")).precision == "float64"


def test_the_resolution_rule_is_the_same_at_every_ndim(monkeypatch):
    """The resolution rule, in the one place it is written down.

    It USED to read "float64 at ndim=2, the host default at ndim=1", because
    float32 was refused there (M1) and a default is not a request: resolving one
    straight into a gate refused every 2D session on a float32 host over a value
    the client never sent. M1 landed on 2026-07-27 and took the special case with
    it, so there is now one rule — and a 2D session on a float32 host gets
    float32, which is the whole point, since 2D is where single precision's
    memory saving actually matters (96 B/cell against 176).

    The auto-expand gate is still here and still float64-only, and it is checked
    alongside deliberately: it is the remaining reason the two can disagree, and
    THAT one refuses rather than resolves, because it is asked for explicitly.
    Note it is a PRECISION gate at either ndim — M3 removed the 2D one, so 2D +
    float64 + auto_expand is now a perfectly ordinary session."""
    import config
    from core.protocol import SessionCreate, MSG_EXPAND_F32
    monkeypatch.setattr(config, "PRECISION", "float32")
    g2 = {"ndim": 2, "axes": [{"lo": -6.0, "hi": 6.0, "N": 16}]*2
                             + [{"lo": -7.0, "hi": 7.0, "N": 16}]*2}
    ic2 = {"type": "mixture", "components": [
        {"q0": [1.0, 0.0], "k0": [0.0, 0.0],
         "sigma_q": [0.7, 0.7], "sigma_k": [0.7, 0.7]}]}
    two_d = _cfg(grid=g2, ic=ic2, potential="(x^2+y^2)/2")
    assert SessionCreate(**two_d).precision == "float32"
    assert SessionCreate(**_cfg()).precision == "float32"
    # an explicit choice still wins over the host's, at either ndim
    assert SessionCreate(**dict(two_d, precision="float64")).precision \
        == "float64"
    # 2D + float64 + auto-expand is ordinary since M3...
    assert SessionCreate(**dict(two_d, precision="float64",
                                auto_expand=True)).auto_expand is True
    # ...while the FLOAT32 refusal stands, and stands at ndim=2 as well: it is
    # about single-precision noise passing the detector, which 2D never changed
    with pytest.raises(ValidationError) as e:
        SessionCreate(**dict(two_d, precision="float32", auto_expand=True))
    assert MSG_EXPAND_F32[:40] in str(e.value)


def test_bad_host_precision_falls_back_to_float64(monkeypatch):
    """A typo must not become a precision. Falling back to the SAFE value is
    never dangerous; passing `flaot32` through would have advertised it to
    every client and handed it to SessionCreate's default."""
    import importlib
    import config
    monkeypatch.setenv("WIGNERF_PRECISION", "flaot32")
    assert importlib.reload(config).PRECISION == "float64"
    monkeypatch.setenv("WIGNERF_PRECISION", "float32")
    assert importlib.reload(config).PRECISION == "float32"
    monkeypatch.delenv("WIGNERF_PRECISION")
    assert importlib.reload(config).PRECISION == "float64"


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="unknown precision"):
        ArrayBackend(device="cpu", precision="float16")


# -- how far the accuracy actually falls ---------------------------------


def test_coherent_state_revival_survives_float32():
    """The coarse physics still holds: after one period the coherent state
    returns to itself within the same 2% the float64 test demands. What
    float32 costs is the 1e-12 diagnostics, not the qualitative dynamics —
    which is exactly what makes it usable as a preview."""
    g = _grid("float32")
    prop = Propagator(g, quantum=True, **HARMONIC)
    W0 = _coherent(g)
    n = 2000
    W = _evolve(prop, W0, 2.*pi/n, n)
    rel_l1 = float(np.sum(np.abs(W - W0), dtype=np.float64)) \
        / float(np.sum(np.abs(W0), dtype=np.float64))
    assert rel_l1 < 0.02


def test_float32_drift_is_bounded_and_worse_than_float64():
    """Upper bounds, so a regression to something worse than measured fails —
    and a lower bound against float64, so this test also fails if float32 is
    silently not being used at all (the pyFFTW/exponent upcast traps)."""
    drift = {}
    for precision in ("float64", "float32"):
        g = _grid(precision)
        prop = Propagator(g, quantum=True, **HARMONIC)
        W0 = _coherent(g)
        o0 = observables.compute(W0, prop)
        o1 = observables.compute(_evolve(prop, W0, 0.01, 200), prop)
        drift[precision] = (abs(o1.norm - o0.norm), abs(o1.purity - o0.purity))
    # measured 2026-07-25 at 64^2/200 steps: float32 (7.5e-8, 3.4e-5),
    # float64 (3.3e-16, 4.0e-14)
    assert drift["float32"][0] < 1e-5, "norm drift"
    assert drift["float32"][1] < 1e-3, "purity drift"
    assert drift["float64"][0] < 1e-12 and drift["float64"][1] < 1e-10
    assert drift["float32"][1] > 100*drift["float64"][1]


def _centered(g):
    """A packet at the ORIGIN, so every axis's band mass is numerical by
    construction and any reading above the floor is the detector's own noise."""
    c = (GaussianComponent(0.0, 0.0, SIG, SIG) if g.ndim == 1
         else GaussianComponent((0.0, 0.0), (0.0, 0.0), (SIG, SIG), (SIG, SIG)))
    return g.shift(mixture_wigner(g, [c]))


def test_float32_noise_reaches_the_float64_edge_trigger():
    """The measurement behind the precision-keyed threshold and behind the
    auto_expand refusal — pinned, because it is the one place where float32
    does not merely lose accuracy but tells an active lie.

    A coherent state parked at the ORIGIN of a 256² domain is ~8 sigma from
    the edge band; its true band mass is ~1e-15, and float64 reports exactly
    that, flat, forever. float32's own spectral noise climbs past the float64
    trigger within a few hundred steps."""
    masses = {}
    for precision in ("float64", "float32"):
        g = _grid(precision, n=256)
        prop = Propagator(g, quantum=True, **HARMONIC)
        obs = observables.compute(_evolve(prop, _centered(g), 0.01, 600), prop)
        es = boundary.edge_report(obs.marg, g.d, precision)
        masses[precision] = max(es.x_mass, es.p_mass)
        # whatever the noise, the SESSION's own threshold must stay quiet —
        # a permanent boundary alarm on a contained state is a broken warning
        assert not es.triggered, (precision, es)
    # float64 stays at the ringing floor the boundary docstring claims
    assert masses["float64"] < 1e-12
    # ...and float32 would have tripped the float64 trigger. If this ever
    # stops being true, the raised threshold and the auto_expand refusal can
    # both be revisited — until then they are load-bearing.
    assert masses["float32"] > boundary.EDGE_THRESHOLD


def test_float32_in_2d_moves_real_mass_to_the_edge_band():
    """What M1 could NOT make go away, pinned so it cannot be forgotten.

    In 2D single precision genuinely migrates mass outward, and at a coarse grid
    it is enough to raise the boundary warning on a state that is nowhere near an
    edge. Measured 2026-07-27 on the shipping 2D default over 12000 steps: 1.0e-3
    of the integral in the band at 32^4 against 3.1e-5 for the IDENTICAL state in
    float64, latching the warning within ~7 s of wall clock, with purity down 2%.
    48^4 stays clear and 64^4 flickers, so it is the coarse grid that cannot
    carry single precision, not 2D as such.

    The mass is REAL — float64 measures the truth and float32 has 30x more of it
    — so the reading is not a detector artifact and no threshold was moved for
    it: a band mass that grows with step count outruns any fixed value, and one
    high enough to be safe would be past the point where real wrap does damage.
    What changed instead was the WORDS: SimulatorView.boundaryTitle names single
    precision as a cause in a float32 2D session, because the standing advice
    ("restart with a larger domain") is the one remedy that cannot help here.

    This asserts the ORDERING, not an absolute — the absolute grows with the run
    length and would make a brittle test — plus the fact that float64 at the same
    size stays clean, which is what identifies the precision as the cause."""
    mass = {}
    for precision in ("float64", "float32"):
        g = _grid(precision, n=32, ndim=2)
        prop = Propagator(g, quantum=True, **HARMONIC_2D)
        obs = observables.compute(_evolve(prop, _centered(g), 0.01, 300), prop)
        es = boundary.edge_report(obs.marg, g.d, precision, labels=g.labels)
        # EdgeState's x_mass/p_mass are 1D-only spellings and RAISE at 4 axes
        mass[precision] = max(es.mass)
        with pytest.raises(AttributeError):
            es.x_mass
    assert mass["float32"] > 10*mass["float64"], mass
    # ...and float64 at this grid is orders under its own trigger, so the
    # difference is the precision and not the coarseness of the lattice
    assert mass["float64"] < boundary.EDGE_THRESHOLD


def test_float32_drift_in_2d_is_bounded_and_worse_than_float64():
    """The 4D counterpart, and the honest cost of M1.

    Same shape as the 1D test above, same anti-tautology lower bound. 32^4 and
    50 steps rather than 16^4: at 16^4 the GRID's own error swamps the
    precision's — measured 1.076e-3 purity drift in float64 against float32's
    1.114e-3, a ratio of 1.04, so the test would pass with float32 switched off.
    At 32^4 the ratio is 5.4e6.

    Measured on CPU, 2026-07-27, 32^4 over 50 steps at dt = 0.01:
        float32   norm 8.4e-8   purity 1.76e-5
        float64   norm 4.4e-16  purity 3.3e-12
    So single precision costs ~2x the purity per step that it costs in 1D
    (8.4e-5 there over 200 steps at 64^2), which is what a 4-axis Strang step
    doing more arithmetic per element looks like."""
    drift = {}
    for precision in ("float64", "float32"):
        g = _grid(precision, n=32, ndim=2)
        prop = Propagator(g, quantum=True, **HARMONIC_2D)
        W0 = _coherent(g)
        o0 = observables.compute(W0, prop)
        o1 = observables.compute(_evolve(prop, W0, 0.01, 50), prop)
        drift[precision] = (abs(o1.norm - o0.norm), abs(o1.purity - o0.purity))
    assert drift["float32"][0] < 1e-6, "norm drift"
    assert drift["float32"][1] < 1e-3, "purity drift"
    assert drift["float64"][0] < 1e-12 and drift["float64"][1] < 1e-10
    assert drift["float32"][1] > 100*drift["float64"][1]


@pytest.mark.parametrize("ndim,n", [(1, 256), (2, 32), (2, 48)])
def test_the_adjust_step_residual_floor_stays_under_the_tol_minimum(ndim, n):
    """TOL_MIN_F32 must sit above the floor of the quantity it bounds, at every
    grid the solver offers — this is the measurement M1 had to take before
    float32 could be allowed at 4 axes, and the reason the constant did NOT move.

    adjust_step shrinks dt until one full step and two half steps agree to a
    relative tol. In single precision that comparison has a roundoff floor, and
    below it the controller burns all 15 shrink attempts every 20 steps and never
    converges — a run that grinds to a halt and reads as a solver bug. The floor
    was measured at 256^2 only, and MSG_TOL_F32 said it grew with grid size, so
    4D (16.8M cells at 64^4, 256x a 256^2 run) was the open question.

    It SATURATES rather than growing (RTX 3090, 2026-07-27, dt down to 0.0025):
        1D  256^2 9.2e-7   1024^2 1.2e-6   4096^2 1.2e-6
        2D   32^4 2.5e-6    48^4  2.0e-6    64^4  2.2e-6
    Worst case 2.5e-6 against TOL_MIN_F32 = 1e-5, i.e. 4x margin everywhere.
    This test takes the same reading at the sizes it can afford on CPU; the
    assertion is the MARGIN, because that is the property the constant needs."""
    g = _grid("float32", n=n, ndim=ndim)
    prop = Propagator(g, quantum=True, **_harmonic(ndim))
    W = _evolve(prop, _coherent(g), 0.01, 20)      # a settled state, not the IC
    # dt small enough that splitting error is far below the roundoff floor, so
    # what is left IS the floor (float64 is at ~5e-9 here and still falling)
    dt = 0.0025
    eU, eT = prop.exponents(dt)
    eUn, eTn = prop.exponents(0.5*dt)
    W1 = prop.solve_spectral(W, eU, eT)
    W2 = prop.solve_spectral(prop.solve_spectral(W, eUn, eTn), eUn, eTn)
    rel = float(np.sum(np.abs(W1 - W2), dtype=np.float64)
                / np.sum(np.abs(W1), dtype=np.float64))
    assert rel < TOL_MIN_F32/2.0, (
        "the float32 step-controller floor is %.3g at %s, too close to "
        "TOL_MIN_F32 = %g — the controller would stop converging" % (
            rel, "%d^%d" % (n, 2*ndim), TOL_MIN_F32))


def test_float32_refuses_auto_expand():
    """Detection still runs and still warns; only the automatic REGRID is
    refused, because it sizes the new domain from a support scan whose 1e-8
    threshold is two orders below the same noise."""
    from pydantic import ValidationError
    from core.protocol import SessionCreate
    with pytest.raises(ValidationError, match="auto-expand is not available"):
        SessionCreate(**_cfg(precision="float32", auto_expand=True))
    # both halves of the pair are individually fine
    SessionCreate(**_cfg(precision="float32"))
    SessionCreate(**_cfg(auto_expand=True))
    with TestClient(app) as client:
        r = client.post("/api/sessions",
                        json=_cfg(precision="float32", auto_expand=True))
        assert r.status_code == 422 and "auto-expand" in r.text


def test_live_auto_expand_toggle_is_refused_in_float32():
    """The schema rule is two clicks away otherwise — auto_expand is a live
    ParamChange field, so the live path needs the same guard as create."""
    import json
    with TestClient(app) as client:
        info = client.post("/api/sessions",
                           json=_cfg(precision="float32")).json()
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(json.dumps({"type": "set_params",
                                     "params": {"auto_expand": True}}))
            for _ in range(200):
                m = ws.receive()
                if not m.get("text"):
                    continue
                d = json.loads(m["text"])
                if d["type"] == "error" and d.get("code") == "bad_auto_expand":
                    break
                assert d["type"] != "params_applied", d
            else:
                raise AssertionError("the refusal never arrived")
        assert client.get("/api/sessions/%s" % sid).json()["auto_expand"] is False
        client.delete("/api/sessions/%s" % sid)


# -- the API surface ------------------------------------------------------


GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
IC = {"type": "mixture",
      "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.70711,
                      "sigma_p": 0.70711}]}


def _cfg(**over):
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": IC, "variants": ["qn"],
           "record_dt": 0.05, "delay": 0.0}
    cfg.update(over)
    return cfg


def test_precision_defaults_to_float64_and_rides_in_status():
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json=_cfg()).json()["session_id"]
        assert client.get("/api/sessions/%s" % sid).json()["precision"] == "float64"
        sid = client.post("/api/sessions",
                          json=_cfg(precision="float32")).json()["session_id"]
        assert client.get("/api/sessions/%s" % sid).json()["precision"] == "float32"


def test_tol_below_the_float32_floor_is_refused():
    """The combination is unreachable, not merely unwise: adjust_step compares
    a full step against two half steps, which agree only to ~1e-7 in single
    precision, so a smaller tol makes the controller shrink dt forever."""
    with TestClient(app) as client:
        r = client.post("/api/sessions",
                        json=_cfg(precision="float32", tol=1e-8))
        assert r.status_code == 422
        assert "float32 roundoff floor" in r.text
        # the same tol is perfectly fine in float64
        assert client.post("/api/sessions", json=_cfg(tol=1e-8)).status_code == 200
        # and float32 at or above the floor is fine
        assert client.post("/api/sessions",
                           json=_cfg(precision="float32",
                                     tol=TOL_MIN_F32)).status_code == 200


def test_unknown_device_is_a_clean_422():
    with TestClient(app) as client:
        r = client.post("/api/sessions", json=_cfg(device="cuda:99"))
        assert r.status_code == 422 and "device" in r.text
        r = client.post("/api/sessions", json=_cfg(device="quantum-annealer"))
        assert r.status_code == 422 and "device" in r.text
        # cpu is always available
        assert client.post("/api/sessions",
                           json=_cfg(device="cpu")).status_code == 200


def test_device_outside_the_host_pool_is_refused(monkeypatch):
    """WIGNERF_DEVICE has to be a POLICY, not a suggestion. It only checked that
    the spec parsed and the card existed, so a host pinned to one device — or to
    cpu, to keep its cards free for something else — could be overridden by any
    client that asked for another. The refusal names the pool, because "device
    not available" without saying what IS available is a dead end."""
    import config
    monkeypatch.setattr(config, "DEVICE", "cpu")
    with TestClient(app) as client:
        # in the pool
        assert client.post("/api/sessions",
                           json=_cfg(device="cpu")).status_code == 200
        # outside it — refused whether or not this host physically has the card
        r = client.post("/api/sessions", json=_cfg(device="cuda:0"))
        assert r.status_code == 422
        assert "WIGNERF_DEVICE=cpu" in r.text and "cpu" in r.text
        # and every member of a list is checked, not just the first
        r = client.post("/api/sessions", json=_cfg(device="cpu,cuda:0"))
        assert r.status_code == 422 and "cuda:0" in r.text


def test_history_cap_is_clamped_to_the_host_ceiling():
    """A session may narrow the host's RAM policy, never widen it."""
    import config
    with TestClient(app) as client:
        sid = client.post("/api/sessions", json=_cfg(history_mb=64)) \
            .json()["session_id"]
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["history_cap_bytes"] == 64*1024*1024
        assert st["history_mb_max"] == config.HISTORY_MB
        sid = client.post("/api/sessions",
                          json=_cfg(history_mb=config.HISTORY_MB*100)) \
            .json()["session_id"]
        st = client.get("/api/sessions/%s" % sid).json()
        assert st["history_cap_bytes"] == config.HISTORY_MB*1024*1024


def test_setup_document_and_export_block_carry_the_precision():
    """A preview run's video outlives the session that could have explained
    it, so the metadata must say so — and a float64 run's block must not grow
    a line of noise."""
    from core.protocol import SessionCreate
    f32 = SessionCreate(**_cfg(precision="float32"))
    f64 = SessionCreate(**_cfg())
    assert describe.setup_document(f32, [])["config"]["precision"] == "float32"
    lines32 = describe.param_lines(f32, [], 0, 10)
    lines64 = describe.param_lines(f64, [], 0, 10)
    assert any("PREVIEW" in ln and "float32" in ln for ln in lines32)
    assert not any("precision" in ln for ln in lines64)
    assert len(lines32) == len(lines64) + 1


def test_device_endpoint_reports_bare_specs():
    """The Setup panel's device select needs the round-trippable spec, not
    just the human name."""
    with TestClient(app) as client:
        d = client.get("/api/device").json()
        assert d["precision"] and d["pool"]
        for entry in d["devices"]:
            assert entry["spec"] in ("cpu",) or entry["spec"].startswith("cuda:")
