"""
Boundary watch: the spectral domain is a torus, so when W's support reaches
an edge, mass wraps through the seam and the run silently evolves the wrong
(torus) problem. Detection is measure-based and essentially free — the
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
"""

from dataclasses import dataclass
from math import isfinite

import numpy


def edge_band(n):
    """Outer band width in cells per side (8 at N=256 — ~3% of the axis)."""
    return max(4, n // 32)


EDGE_THRESHOLD = 1e-6   # band mass (both sides) that trips an axis (float64)
SUPPORT_EPS = 1e-8      # per-side tail mass excluded from the support box

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
    """Edge-band mass of one variant's marginals at one record, with the
    threshold it was measured against — the threshold is a property of the
    SESSION's precision, so it must travel with the reading rather than be
    read from a module constant at the point of comparison."""
    x_mass: float
    p_mass: float
    threshold: float = EDGE_THRESHOLD

    @property
    def axes(self):
        return [a for a, m in (("x", self.x_mass), ("p", self.p_mass))
                if m > self.threshold]

    @property
    def triggered(self):
        return self.x_mass > self.threshold or self.p_mass > self.threshold


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


def edge_report(rho, phi, dx, dp, precision="float64"):
    """Edge-band mass from natural-order marginal DENSITIES rho(x) = int W dp
    and phi(p) = int W dx (numpy or cupy arrays). A non-finite sum means the
    run diverged — report clear rather than trigger a regrid storm."""
    thr = edge_threshold(precision)
    mx = _band_mass(rho, dx)
    mp = _band_mass(phi, dp)
    if not (isfinite(mx) and isfinite(mp)):
        return EdgeState(0.0, 0.0, thr)
    return EdgeState(mx, mp, thr)


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
