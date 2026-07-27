"""
Boundary watch: the spectral domain is a torus, so when W's support reaches
an edge it re-enters from the opposite side and the run silently evolves the
wrong (torus) problem.

WHAT THE NUMBER IS. `EdgeState.mass[a]` is the PROBABILITY in the outer band of
axis a: the natural-order marginal density of that axis (rho(x) = int W over
every other axis, which integrates to 1) summed over the outer `edge_band(N)`
cells at EACH end, times the cell size. So 3.8e-06 means 0.00038% of the state
lies within edge_band(N)*d of that boundary. "mass" here is measure-theoretic —
probability mass — and has nothing to do with the particle mass `m` in the
Physics panel; the user-facing strings say "probability" for exactly that
reason. Detection is measure-based and essentially free — the
outer-band mass of the marginals rho(x), phi(p) that observables.compute()
already produces per record — and runs unconditionally every record; the
session's `auto_expand` toggle governs only the RESPONSE (Phase 3: exact
fixed-lattice move/double regrids).

Shared by the runtime check (worker._emit), the IC-time check
(initial.preview_warnings) and the regrid planner (session), so all three
agree on what "too close to the edge" means.

Thresholds: expansion prevents wrap, it cannot repair it — mass that has
already crossed the seam re-enters on the wrong side. So the trigger must
fire while edge mass is still negligible. EDGE_THRESHOLD = 1e-6 matches the
project's health budget (an IC norm deficit >> 1e-6 is the documented wrap
tell) and sits far above float64 marginal ringing (~1e-12); for a Gaussian
it corresponds to the ~4.8 sigma point crossing the band's INNER edge, at
which point the mass actually at the seam is still ~1e-8.

THE NOISE FLOOR, and why the trigger is gated on a MEASURED one. A marginal
density is non-negative exactly (rho(x) = int |psi|^2 over the other axes), so
every negative value in one is pure numerical error and their total magnitude
MEASURES the floor the band mass is being read against. That is not a
theoretical nicety: a COARSE grid puts the floor above the trigger, because the
Nyquist edge of the lambda axis truncates the state's own Fourier tail at
exp(-(pi*sigma_q/dx)^2/2). Measured 2026-07-26, a coherent state parked at
|q| = 2 in an isotropic well, nowhere near an edge, 4D:

  N=32  dx=0.500  predicted 5.17e-05   measured band-mass noise 5.35e-05
  N=48  dx=0.333  predicted 2.27e-10   measured                 1.66e-10
  N=64  dx=0.250  predicted 7.15e-18   measured                 2.29e-13

At N=32 that is 50x the trigger, so the band mass is NOISE — negative in 96 of
204 readings — and its sign flips every record or two. Ungated, that produced
79 boundary state changes in a 201-record run (measured 243 WS events in 25 s in
the browser, each rewriting the header warning, which wraps the header and moves
the W panels by a line). So an axis trips only when its band mass clears
max(EDGE_THRESHOLD, EDGE_NOISE_MARGIN * noise), and session.report_edge
additionally requires EDGE_CONFIRM consecutive records either way. False
announcements over those 201 records against the record a state that really does
reach the edge is announced on (margin x confirm):

           K=1   K=2   K=3   K=4   K=6        real approach announced at
   none     79    73    69    58    50        record 0 / 1 / 2 / 3 / 5
   4x       4     2     2     0     0
   8x       2     0     0     0     0
   12x      0     0     0     0     0

8x + 4 is the choice: zero, with margin against the 4.2 band/noise ratio a 32^4
browser run still posted at 4x, and clear of the 11 a state genuinely over the
edge measures. The confirmation delay costs nothing because a real approach
grows monotonically over tens of records, whereas noise does not — and 12x is
not taken, because it would start to encroach on that 11.

Note what the gate does NOT change: wherever the detector already worked it is
inert. Any grid resolving its state to ~2 cells per sigma has a floor below
EDGE_THRESHOLD/EDGE_NOISE_MARGIN, so the 1e-6 threshold alone decides — which
covers every 1D session (N = 256..4096) and 2D from N=48 up (8 * 1.66e-10 is
1.3e-9, three orders under the trigger). It engages exactly where the
measurement was impossible, and 32^4 is a size this project recommends for
exploration. It composes with EDGE_THRESHOLD_BY_PRECISION rather than replacing
it: a float32 run's larger negative part raises its own floor further, on top of
the 1e-4 that mode already asks for.
"""

from dataclasses import dataclass
from math import isfinite

import numpy


def edge_band(n):
    """Outer band width in cells per side (8 at N=256 — ~3% of the axis)."""
    return max(4, n // 32)


EDGE_THRESHOLD = 1e-6   # band mass (both sides) that trips an axis (float64)
SUPPORT_EPS = 1e-8      # per-side tail mass excluded from the support box

# How far above the marginal's own measured noise a band mass must sit before it
# counts as edge mass, and how many consecutive records must agree (the latter
# enforced by session.report_edge). Both come from the sweep in the docstring
# below; 8 leaves margin against the 4.2 ratio a 32^4 browser run still posted
# at 4, and stays clear of the 11 a state genuinely over the edge measures.
EDGE_NOISE_MARGIN = 8
EDGE_CONFIRM = 4

# A float32 session's own spectral noise reaches the float64 trigger. Measured
# 2026-07-25, harmonic U, a coherent state parked at the ORIGIN — nowhere near
# an edge — with band mass that should be ~1e-15:
#
#   N=256   step 0: 1.8e-15   step 200: 6.1e-07   step 600: 1.6e-06  TRIGGERED
#   N=512   step 0: 1.8e-15   step 200: 7.6e-07   step 600: 1.8e-06  TRIGGERED
#   (float64, same runs: 1.5e-15 throughout, flat)
#
# So the float64 threshold would latch a permanent boundary warning on every
# float32 run within a few hundred steps. 1e-4 restores the margin and is still
# an EARLY warning — at 1e-4 band mass the mass actually at the seam is ~1e-6,
# well before wrap does damage. What it cannot do is drive auto-expand: that
# needs support_cells, whose SUPPORT_EPS = 1e-8 is two orders below the same
# noise (measured: the support scan reports the WHOLE axis by step 200 in
# float32, against an exact [43, 214) in float64), so the planner would size a
# window from garbage. Hence float32 refuses auto_expand outright at the
# schema — see protocol.SessionCreate._check.
EDGE_THRESHOLD_BY_PRECISION = {"float64": EDGE_THRESHOLD, "float32": 1e-4}


def edge_threshold(precision="float64"):
    return EDGE_THRESHOLD_BY_PRECISION.get(precision, EDGE_THRESHOLD)


@dataclass(frozen=True)
class EdgeState:
    """Edge-band mass of one variant's marginals at one record, one entry per
    phase-space axis, with the threshold it was measured against — the
    threshold is a property of the SESSION's precision, so it must travel with
    the reading rather than be read from a module constant at the point of
    comparison. `labels` travels with it for the same reason: which axes are
    named "x, p" and which "x, y, px, py" is a property of the grid.

    `noise` is the per-axis numerical floor measured from the same marginals
    (see the module docstring): the reading also carries what it can be trusted
    to, because on a coarse grid that is larger than `threshold` and it is the
    floor, not the threshold, that decides."""
    mass: tuple
    labels: tuple
    threshold: float = EDGE_THRESHOLD
    noise: tuple = ()

    def floor(self, a):
        """The value axis `a`'s band mass must clear to count as edge mass."""
        n = self.noise[a] if a < len(self.noise) else 0.0
        return max(self.threshold, EDGE_NOISE_MARGIN*n)

    @property
    def tripped(self):
        """Axis INDICES over their floor (what the session debounces on)."""
        return tuple(a for a, m in enumerate(self.mass) if m > self.floor(a))

    @property
    def axes(self):
        return [self.labels[a] for a in self.tripped]

    @property
    def triggered(self):
        return bool(self.tripped)

    def as_dict(self):
        """Wire/status shape: the per-axis masses keyed by axis name."""
        return {self.labels[a]: self.mass[a] for a in range(len(self.mass))}

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------

    def _only1d(self, name):
        if len(self.mass) != 2:
            raise AttributeError("EdgeState.%s is a 1D-only spelling; this "
                                 "reading has %d axes — use mass/as_dict()"
                                 % (name, len(self.mass)))

    @property
    def x_mass(self):
        self._only1d("x_mass"); return self.mass[0]

    @property
    def p_mass(self):
        self._only1d("p_mass"); return self.mass[1]


def _band_mass(marginal, d):
    b = edge_band(len(marginal))
    # An axis shorter than 8 bands (N < 32 at the 4-cell minimum band)
    # cannot separate "edge" from "center": the band pair would cover over
    # a quarter of the axis — at N <= 8 the slices overlap and a uniform
    # marginal reads as mass 2.0. Such degenerate toy axes never report
    # edge mass (they would otherwise warn always and, with auto_expand
    # on, double-storm to the cap).
    if len(marginal) < 8*b:
        return 0.0
    return float(marginal[:b].sum() + marginal[-b:].sum())*d


def _noise_mass(marginal, d):
    """Magnitude of the marginal's NEGATIVE part, times the cell size — the
    numerical floor of this reading, since a density cannot be negative.

    Elementwise only ((|m| - m)/2 is the negative part), so it costs one
    reduction and no boolean indexing: identical on numpy and on cupy, and no
    device sync beyond the one `float()` the band sum already pays. Worker
    marginals are host arrays by this point anyway (observables brings them
    over), so on the runtime path it is host arithmetic over sum(N) cells."""
    return float((abs(marginal) - marginal).sum())*0.5*d


def edge_report(margs, d, precision="float64", labels=None):
    """Edge-band mass from the natural-order marginal DENSITIES, one per
    phase-space axis (numpy or cupy arrays), against their cell sizes `d`, with
    the numerical floor each was measured against. A non-finite sum means the
    run diverged — report clear rather than trigger a regrid storm."""
    thr = edge_threshold(precision)
    if labels is None:
        from .axes import labels as axis_labels
        labels = axis_labels(len(margs)//2)
    mass = tuple(_band_mass(m, d[a]) for a, m in enumerate(margs))
    noise = tuple(_noise_mass(m, d[a]) for a, m in enumerate(margs))
    if not all(isfinite(m) for m in mass + noise):
        return EdgeState((0.0,)*len(mass), tuple(labels), thr)
    return EdgeState(mass, tuple(labels), thr, noise)


# ---------------------------------------------------------------------------
# Regrid planning (pure integer arithmetic on the frozen lattice)
# ---------------------------------------------------------------------------

def support_cells(marginal, d, eps=SUPPORT_EPS):
    """Smallest [lo, hi) cell window of a natural-order marginal density
    holding all but eps of the mass per side (tiny negative ringing is
    clipped). Returns (0, N) when the marginal is non-finite or empty."""
    m = numpy.clip(numpy.asarray(marginal, dtype=numpy.float64), 0.0, None)*d
    c = numpy.cumsum(m)
    total = float(c[-1])
    if not isfinite(total) or total <= 0.0:
        return 0, len(m)
    lo = int(numpy.searchsorted(c, eps))
    hi = int(numpy.searchsorted(c, total - eps)) + 1
    hi = min(hi, len(m))
    return min(lo, hi - 1), hi


@dataclass(frozen=True)
class AxisPlan:
    offset: int   # new window offset, in cells from the lattice anchor
    n: int        # new cell count (power of 2 preserved: n in {N, 2N, ...})
    kind: str     # "move" | "double" | "capped"


def plan_axis(offset, n, lo, hi, cap):
    """New window for one axis, given the current window (`offset`, `n` in
    anchor cells) and the support [lo, hi) RELATIVE to it. Doubles until the
    support fits in half the axis (regrids stay rare: >= n/4 free cells per
    side afterwards), then centers the support — a combined move+double
    falls out naturally, and a pure move still works at the cap (ballistic
    translation keeps getting relief forever). Returns kind "capped" when
    even the capped size cannot hold the support with 2 edge bands free per
    side — the caller warns and keeps computing."""
    w = hi - lo
    new_n = n
    while w > new_n//2 and new_n*2 <= cap:
        new_n *= 2
    if w > new_n - 4*edge_band(new_n):
        return AxisPlan(offset, n, "capped")
    new_offset = offset + (lo + hi - new_n)//2   # support centered
    return AxisPlan(new_offset, new_n, "double" if new_n != n else "move")
