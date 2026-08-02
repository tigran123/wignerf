"""
The worker's SUBSTEP SCHEDULE — the code M7 changed, and the one part of the
solver the physics suite structurally cannot see.

tests/test_propagator*.py and test_precision.py drive Propagator directly
through a private fixed-dt `evolve()` helper, so they never enter
worker._advance: they were bitwise unchanged by M7 and would be just as
unchanged by a scheduler that never landed on tau_k at all. Before this file
`_exponents` and its slots had zero test references anywhere in the suite, which
made the -32 B/cell that M7 buys a property nothing protected.

What is pinned here: a record is walked at ONE committed substep size, that size
does not materially exceed the adaptive dt, the substeps really do cover the
record, an exact divisor is not split one step finer, one cached exponent pair
serves the production mesh, a record needing many substeps still TERMINATES, and
the schedule works in both time directions.
"""

import threading
from math import ceil, nextafter, sqrt

import numpy as np
import pytest

from core.grid import Grid
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.worker import SolverWorker
from core.xp import ArrayBackend

RECORD_DT = 0.05
SIG = 1.0/sqrt(2.0)
# deliberately NOT a divisor of RECORD_DT: 0.05/0.0043 = 11.63, so the old
# scheme took 11 steps of 0.0043 plus a 0.0027 straggler and needed a second
# exponent slot for it, while this one takes 12 of 0.0041667.
DT_ODD = 0.0043


@pytest.fixture(scope="module")
def backend():
    return ArrayBackend(device="cpu")


@pytest.fixture(scope="module")
def grid(backend):
    return Grid.from_1d(-6.0, 6.0, 64, -7.0, 7.0, 64, backend)


def _prop(grid, quartic=False):
    # a MILD quartic: 0.3*x^4 reaches |dU/dx| = 265 at the box edge, which
    # drives p past the momentum edge and wraps, and a boundary artefact is not
    # a convergence signal (measured ratio 1.42 there, against 4.51 here)
    if quartic:
        return Propagator(grid, quantum=True, U=lambda x: x**2/2. + 0.02*x**4,
                          gradU=(lambda x: x + 0.08*x**3,))
    return Propagator(grid, quantum=True, U=lambda x: x**2/2.,
                      gradU=(lambda x: x,))


def _w0(grid):
    return grid.shift(mixture_wigner(grid, [GaussianComponent(2.0, 0.0, SIG, SIG)]))


class _Cfg:
    record_dt = RECORD_DT


class _Session:
    """The two attributes SolverWorker.__init__ and _advance actually reach
    for. A real session would need a clock, a history and four threads to test
    arithmetic that depends on none of them."""
    id = "substep-test"
    cfg = _Cfg()


def _worker(dt=DT_ODD):
    w = SolverWorker(_Session(), "qn", 0, "cpu")
    w.dt = dt
    w.force_adjust = False   # no adjust_step: this file tests the SCHEDULE
    w.steps_total = 1        # ...and 1 % 20 != 0 keeps it that way
    return w


def _run(w, prop, W, t, t_tgt):
    """Advance one record, returning the substeps _advance actually took."""
    seen = []
    real = w._exponents

    def spy(p, dts):
        seen.append(dts)
        return real(p, dts)

    w._exponents = spy
    W, t = w._advance(prop, W, t, t_tgt)
    del w._exponents
    return W, t, seen


def test_every_substep_in_a_record_is_the_same_size(grid):
    """The property M7 is named for, and the whole reason one slot suffices."""
    w = _worker()
    _, t, seen = _run(w, _prop(grid), _w0(grid), 0.0, RECORD_DT)
    assert len(set(seen)) == 1, "record walked at %d distinct step sizes" % len(set(seen))
    assert len(seen) == ceil(RECORD_DT/DT_ODD) == 12
    assert t == RECORD_DT
    # the substeps must really COVER the record — _advance returns t_tgt
    # regardless, so a gap here would be invisible in `t` above
    assert sum(seen) == pytest.approx(RECORD_DT, abs=1e-15)


def test_the_substep_never_exceeds_the_adaptive_dt(grid):
    """n = ceil(|rem|/|dt|) rounds UP, so the step only ever shrinks against
    self.dt. That is what makes the uniform schedule no less accurate than the
    n-equal-plus-a-straggler one it replaced."""
    w = _worker()
    _, _, seen = _run(w, _prop(grid), _w0(grid), 0.0, RECORD_DT)
    assert all(abs(d) <= abs(DT_ODD) for d in seen)
    assert seen[0] == pytest.approx(RECORD_DT/12, rel=1e-15)


def test_an_exact_divisor_only_discounts_one_ulp_of_roundoff(grid):
    """A quotient rounded one ulp above an integer stays exact, but a genuinely
    smaller adaptive cap must add a substep rather than be exceeded."""
    prop, W0 = _prop(grid), _w0(grid)
    # exact
    _, _, seen = _run(_worker(RECORD_DT/8), prop, W0, 0.0, RECORD_DT)
    assert len(seen) == 8
    # ...and a cap one representable value smaller, where the quotient rounds
    # one ulp above eight.
    one_ulp_smaller = nextafter(RECORD_DT/8, 0.0)
    _, _, seen = _run(_worker(one_ulp_smaller), prop, W0, 0.0, RECORD_DT)
    assert len(seen) == 8
    # A cap that is materially smaller must not be exceeded just to avoid a
    # ninth step.
    smaller = RECORD_DT/8*(1 - 1e-13)
    _, _, seen = _run(_worker(smaller), prop, W0, 0.0, RECORD_DT)
    assert len(seen) == 9
    assert all(abs(d) <= abs(smaller) for d in seen)


def test_one_cached_exponent_pair_serves_a_whole_record(grid):
    """The test that protects the -32 B/cell: a production record caches ONE
    pair, so there is nothing for a second slot to hold.

    The adaptive controller may build temporary trial pairs at a record
    boundary, but those are dropped before the production mesh starts and do
    not occupy a second worker slot."""
    assert not hasattr(_worker(), "_exp_odd"), "the second slot is back"
    prop = _prop(grid)
    built = []
    real = prop.exponents
    prop.exponents = lambda dt: (built.append(dt), real(dt))[1]

    # one record, and 12 substeps from steps_total=1 never trips the
    # every-20-steps adjust (which would add two exponents() calls of its own)
    _run(_worker(), prop, _w0(grid), 0.0, RECORD_DT)
    assert len(built) == 1, "%d rebuilds in one record" % len(built)


def test_a_forced_adjustment_probes_without_committing_a_nonuniform_step(grid):
    """A forced boundary adjustment chooses dt, then the entire record runs on
    its one quotient. Returning NaNs as the trial state proves _advance drops
    it rather than treating it as the first production substep."""
    w = _worker()
    w.force_adjust = True
    prop = _prop(grid)
    selected = []
    real = prop.adjust_step

    def probe(dt, W):
        _, chosen, eU, eT = real(dt, W)
        selected.append(chosen)
        return np.full_like(W, np.nan), chosen, eU, eT

    prop.adjust_step = probe
    W, t, seen = _run(w, prop, _w0(grid), 0.0, RECORD_DT)
    assert len(selected) == 1
    assert len(set(seen)) == 1
    assert sum(seen) == pytest.approx(RECORD_DT, abs=1e-15)
    assert all(abs(d) <= abs(selected[0]) for d in seen)
    assert np.isfinite(W).all(), "the controller's trial state was committed"
    assert t == RECORD_DT
    assert not w.force_adjust and not w._adjust_pending


def test_a_periodic_adjustment_waits_for_the_next_record_boundary(grid):
    """Crossing step 20 mid-record must set a pending probe, not split the
    current record. The next record probes once, then remains uniform too."""
    w = _worker()
    w.steps_total = 16
    prop = _prop(grid)
    selected = []
    real = prop.adjust_step

    def probe(dt, W):
        _, chosen, eU, eT = real(dt, W)
        selected.append(chosen)
        return np.full_like(W, np.nan), chosen, eU, eT

    prop.adjust_step = probe
    W, t, first = _run(w, prop, _w0(grid), 0.0, RECORD_DT)
    assert selected == []
    assert len(set(first)) == 1
    assert sum(first) == pytest.approx(RECORD_DT, abs=1e-15)
    assert w._adjust_pending

    W, t, second = _run(w, prop, W, t, 2*RECORD_DT)
    assert len(selected) == 1
    assert len(set(second)) == 1
    assert sum(second) == pytest.approx(RECORD_DT, abs=1e-15)
    assert all(abs(d) <= abs(selected[0]) for d in second)
    assert np.isfinite(W).all(), "the controller's trial state was committed"
    assert t == 2*RECORD_DT
    assert not w._adjust_pending


def test_a_record_that_needs_many_substeps_still_terminates():
    """The substep SIZE is cached for the whole record, so `_advance` must
    iterate on the COUNT — a residual-driven loop has no clamped last step to
    converge on, the way the pre-M7 `min(|dt|, |rem|)` one did.

    Once n accumulations of rem/n land further than `eps = 1e-12·max(1, |t_tgt|)`
    from tau_k, such a loop takes ANOTHER full substep, marches PAST the target
    and never returns: dts keeps its sign and |t_tgt - t| only grows. n ~ 25000
    at t ~ 5 is enough, which two maximal `adjust_step` contractions reach
    (0.7^15 per call, so record_dt/8 -> record_dt/1684 -> record_dt/354610).
    Measured against the residual loop: 11.5 million substeps for a record that
    wants 50000, still running when a watchdog stopped it.

    No physics here on purpose — 50000 real spectral solves would be a minute of
    FFTs to test float arithmetic. t0 = 5.0 is a real record boundary of a
    default run (record 100 at record_dt = 0.05), and it is where eps is
    tightest relative to the accumulated roundoff.
    """
    class FakeProp:
        def exponents(self, dts):
            return (dts, dts)

        def solve_spectral(self, W, eU, eT):
            return W

    n_want = 50000
    w = _worker(RECORD_DT/n_want)
    t0 = 5.0
    watchdog = threading.Timer(20.0, w.stop_evt.set)
    watchdog.start()
    try:
        _, t = w._advance(FakeProp(), object(), t0, t0 + RECORD_DT)
    finally:
        watchdog.cancel()
    assert not w.stop_evt.is_set(), "_advance never returned on its own"
    assert w.steps_total - 1 == n_want
    assert t == t0 + RECORD_DT


def test_the_schedule_runs_backwards_too(grid):
    """dts carries rem's sign, so nothing here depends on dt_sign except
    through the direction flip _advance does above it."""
    w = _worker(-DT_ODD)
    _, t, seen = _run(w, _prop(grid), _w0(grid), RECORD_DT, 0.0)
    assert all(d < 0 for d in seen) and len(set(seen)) == 1
    assert t == 0.0
    assert sum(seen) == pytest.approx(-RECORD_DT, abs=1e-15)


def test_a_worker_record_converges_on_the_reference_as_dt_squared(grid):
    """Strang splitting is O(dt^2), and a scheduler that mis-sized or dropped a
    substep would break that while still landing on tau_k. Modelled on
    test_propagator2d's dt-ratio anchor: the RATIO is the assertion, because a
    wrong schedule is a different evolution rather than a smaller step away
    from the right one. Quartic U, so the splitting error is not degenerate.

    Measured: 4.51 (errors 1.83e-6 -> 4.05e-7) over four records."""
    prop, W0 = _prop(grid, quartic=True), _w0(grid)
    t_end = 4*RECORD_DT

    def run(dt):
        w, W, t = _worker(dt), W0, 0.0
        # This is the fixed-cap convergence anchor. Boundary probing is covered
        # above; letting its cadence choose different caps for the two runs
        # measures the controller schedule, not Strang's dt^2 error.
        w._probe_adjust = lambda *args: None
        for k in range(1, 5):
            W, t, _ = _run(w, prop, W, t, RECORD_DT*k)
        return np.asarray(W)

    ref = run(RECORD_DT/512)
    coarse = float(np.max(np.abs(run(DT_ODD) - ref)))
    fine = float(np.max(np.abs(run(DT_ODD/2) - ref)))
    assert t_end == pytest.approx(0.2)
    assert coarse/fine > 3.0, (
        "the worker's substep schedule is not O(dt^2) — %.3g at dt=%.4g vs "
        "%.3g at half that. A dt-independent residual means the record is "
        "being walked wrongly, not merely coarsely." % (coarse, DT_ODD, fine))


def test_a_direction_flip_returns_the_state(grid):
    """Forward four records then back four, through the flip at worker.py's
    top-of-loop sign check — which clears the slot and forces an adjust, so the
    reverse pass is NOT the forward one played backwards and this is O(dt^2),
    not roundoff. Untested in either scheme before M7.

    Measured 1.31e-6 against the 1e-4 asserted here; the fixed-dt propagator
    equivalent (test_propagator.test_time_reversal) gets ~1e-9 because it has
    no scheduler in the way."""
    prop, W0 = _prop(grid), _w0(grid)
    prop.tol = 1e-3        # loose: let the flip's forced adjust keep |dt|
    w, W, t = _worker(), W0, 0.0
    for k in range(1, 5):
        W, t = w._advance(prop, W, t, RECORD_DT*k)
    for k in range(3, -1, -1):
        W, t = w._advance(prop, W, t, RECORD_DT*k)
    assert t == 0.0
    assert float(np.max(np.abs(np.asarray(W) - np.asarray(W0)))) < 1e-4
