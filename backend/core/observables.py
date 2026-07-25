"""
Observables of a propagator state: norm, marginals, energy and phase-space
moments. All reductions run on the device; only the results cross to the
host. Input W is in the propagator's fftshifted order.

Every reduction accumulates in float64 and every result LEAVES in float64,
whatever the solver's working precision. In double mode that is what already
happened; in a float32 session it is what keeps E/purity/moments worth reading
(these are sums over up to 16.7M elements) and what keeps rho/phi identical in
dtype to before — history.py's byte accounting and the wire's <f4 codec then
behave the same in both modes.
"""

from dataclasses import dataclass

import numpy
from numpy import pi


@dataclass
class Observables:
    norm: float
    E: float          # <H> minus rest energy (subtracted for relativistic variants)
    x_mean: float
    x_std: float
    p_mean: float
    p_std: float
    purity: float     # gamma = 2*pi*hbar_eff * int W^2 dx dp  (= Tr rho^2)
    rho: numpy.ndarray   # rho(x) = integral W dp, natural x order, host array
    phi: numpy.ndarray   # phi(p) = integral W dx, natural p order


def _base(W, grid, hbar_eff):
    b = grid.backend
    xp = b.xp
    f8 = xp.float64
    dxdp = grid.dx*grid.dp
    Xm = float(xp.sum(grid.X*W, dtype=f8))*dxdp
    X2 = float(xp.sum(grid.X**2*W, dtype=f8))*dxdp
    Pm = float(xp.sum(grid.P*W, dtype=f8))*dxdp
    P2 = float(xp.sum(grid.P**2*W, dtype=f8))*dxdp
    return dict(
        norm=float(xp.sum(W, dtype=f8))*dxdp,
        x_mean=Xm, x_std=float(numpy.sqrt(abs(X2 - Xm*Xm))),
        p_mean=Pm, p_std=float(numpy.sqrt(abs(P2 - Pm*Pm))),
        purity=2.*pi*float(hbar_eff)*float(xp.sum(W*W, dtype=f8))*dxdp,
        rho=b.asnumpy(b.ifftshift(xp.sum(W, axis=1, dtype=f8)))*grid.dp,
        phi=b.asnumpy(b.ifftshift(xp.sum(W, axis=0, dtype=f8)))*grid.dx,
    )


def compute(W, prop):
    """Full observables for a propagator state (uses its H and rest energy)."""
    g = prop.grid
    d = _base(W, g, prop.hbar_eff)
    xp = g.backend.xp
    E = float(xp.sum(prop.H*W, dtype=xp.float64))*g.dx*g.dp \
        - prop.rest_energy*d["norm"]
    return Observables(E=E, **d)


def compute_basic(W, grid, hbar_eff=1.0):
    """Moments/marginals only (E = 0) — used by the initial-Wigner preview,
    which has no potential attached."""
    return Observables(E=0.0, **_base(W, grid, hbar_eff))
