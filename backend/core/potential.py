"""
Safe compilation of user-entered analytic potentials U(x) / U(x, y).

Pipeline: token screen (the security boundary) -> sympy parse in a
whitelisted namespace -> per-variant-family validity -> lambdify for
numpy (and cupy when a GPU backend is active). The first two steps and the
probe machinery live in core/expr.py, which the initial-condition expression
kinds share — one screen, three kinds of expression. What stays here is the
VALIDITY model, which is specific to a potential:

- quantum-valid  = U real and finite on the extended real BOX
  [q1_i - hbar*theta_amp_i/2, q2_i + hbar*theta_amp_i/2] over every spatial
  axis (numeric probe; Abs(x) is quantum-valid, no analyticity required).
  Complex values there (branch cuts) make the evolution non-unitary: reported
  as a warning, since an absorbing potential may be intended.
- classical-valid = EVERY partial dU/dq_i well-defined on the visible box with
  no DiracDelta (so a Heaviside step potential is quantum-only).

The Bopp arguments of the quantum differential are REAL (q_i -+ hbar*theta_i/2,
complex dtype only), which is what makes the extended box the right probe box.
"""

from dataclasses import dataclass, field

import numpy
import sympy as sp

from . import expr as ex
from .expr import ExprError

# Re-exported for callers that predate core/expr.py. describe._func_names()
# reads these to build the mathtext rewrite's function-name alternation, and it
# must keep seeing one list whether a name arrived here or in expr.
_SMOOTH = ex.SMOOTH
_NONSMOOTH = ex.NONSMOOTH
MAX_EXPR_LEN = ex.MAX_EXPR_LEN

# The spatial symbols a potential may use, per ndim.
SYMBOLS = {1: ex.symbols(("x",)), 2: ex.symbols(("x", "y"))}

# "y" is in the namespace at EVERY ndim on purpose — see core/expr.namespace.
_NS = {nd: ex.namespace(tuple(s.name for s in SYMBOLS[nd]),
                        "U(%s)" % ",".join(s.name for s in SYMBOLS[nd]))
       for nd in SYMBOLS}


class PotentialError(ExprError):
    """Kept as its own class because three routers and the session catch it by
    name; subclassing ExprError means `except ExprError` sees it too, which is
    what lets a shared compile helper raise one kind of error."""


@dataclass
class CompiledPotential:
    expr_str: str
    expr: object
    ndim: int
    grad_exprs: tuple = ()       # empty when classical-invalid
    quantum_valid: bool = True
    classical_valid: bool = True
    reasons: list = field(default_factory=list)    # hard per-family failures
    warnings: list = field(default_factory=list)
    latex: str = ""
    grad_latex: tuple = ()
    U: object = None             # numpy callable(*q), array in -> array out
    grad: tuple = None           # one callable per spatial axis
    U_gpu: object = None         # cupy callables, built on demand
    grad_gpu: tuple = None

    def for_backend(self, backend):
        """(U, grad) callables for the given ArrayBackend."""
        if not backend.is_gpu:
            return self.U, self.grad
        if self.U_gpu is None:
            syms = SYMBOLS[self.ndim]
            self.U_gpu = ex.lambdify(syms, self.expr, "cupy")
            self.grad_gpu = tuple(ex.lambdify(syms, e, "cupy")
                                  for e in self.grad_exprs) or None
        return self.U_gpu, self.grad_gpu

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------
    # They RAISE past ndim=1 like every other compatibility spelling in this
    # codebase. Returning grad[0] there instead reads as "the gradient" and is
    # only its x component: routers/preview.py shipped that as `dudx_latex` on
    # 2D responses, describing dU/dx as though U had one variable.

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("CompiledPotential.%s is a 1D-only spelling; "
                                 "this potential is %dD — use grad/grad_exprs/"
                                 "grad_latex" % (name, self.ndim))

    @property
    def dUdx(self):
        self._only1d("dUdx")
        return self.grad[0] if self.grad else None

    @property
    def dUdx_expr(self):
        self._only1d("dUdx_expr")
        return self.grad_exprs[0] if self.grad_exprs else None

    @property
    def dUdx_latex(self):
        self._only1d("dUdx_latex")
        return self.grad_latex[0] if self.grad_latex else ""


def compile_potential(expr_str, ndim=1, ranges=None, extended=None):
    """Compile a U(x) / U(x, y) expression string.

    ndim      -- number of spatial dimensions (1 or 2).
    ranges    -- ((lo, hi), ...) per spatial axis: the visible grid box, used
                 for the classical gradient probe.
    extended  -- ((lo, hi), ...) per spatial axis: the box the quantum
                 propagator evaluates U on (lo_i = q1_i - hbar*theta_amp_i/2,
                 ...); when None, the quantum probe is skipped (validity
                 assumed).

    Raises PotentialError for expressions that are rejected outright
    (bad syntax, forbidden names, wrong free symbols). Per-family validity
    failures are reported in the returned object, not raised.
    """
    if ndim not in SYMBOLS:
        raise PotentialError("unsupported ndim %r" % (ndim,))
    syms = SYMBOLS[ndim]
    for name, box in (("ranges", ranges), ("extended", extended)):
        if box is not None and len(box) != ndim:
            raise PotentialError("%s has %d entries, need one per spatial "
                                 "axis (%d)" % (name, len(box), ndim))

    # One wrap, so every refusal the shared screen raises reaches callers as the
    # class they catch by name.
    try:
        expr = ex.parse(expr_str, _NS[ndim])
        U = ex.lambdify(syms, expr, "numpy")
    except ExprError as e:
        raise PotentialError(str(e)) from None

    cp = CompiledPotential(expr_str=expr_str.strip(), expr=expr, ndim=ndim,
                           latex=sp.latex(expr))
    cp.U = U

    # -- classical family: the gradient ------------------------------------
    grads = []
    for s in syms:
        try:
            d = sp.diff(expr, s).doit()
        except Exception as e:
            cp.classical_valid = False
            cp.reasons.append("classical: cannot differentiate by %s (%s)"
                              % (s.name, e))
            grads = None
            break
        if d.has(sp.DiracDelta):
            cp.classical_valid = False
            cp.reasons.append("classical: dU/d%s contains a Dirac delta "
                              "(hard wall) - not representable as a force"
                              % s.name)
            grads = None
            break
        if d.has(sp.Heaviside):
            cp.warnings.append("classical: dU/d%s has a finite jump "
                               "(Heaviside) - force is discontinuous" % s.name)
        grads.append(d)
    if grads is not None:
        cp.grad_exprs = tuple(grads)
        cp.grad_latex = tuple(sp.latex(d) for d in grads)
        cp.grad = tuple(ex.lambdify(syms, d, "numpy") for d in grads)

    # -- symbolic pole detection (numeric probes can straddle a pole without
    # ever sampling it) ---------------------------------------------------
    if extended is not None:
        pts = ex.poles_in(expr, syms, extended)
        if pts:
            cp.quantum_valid = False
            cp.reasons.append("quantum: U is singular at %s inside the "
                              "extended range %s the propagator evaluates it on"
                              % ("; ".join(pts), ex.box_str(syms, extended)))
    if cp.classical_valid and cp.grad_exprs and ranges is not None:
        for i, d in enumerate(cp.grad_exprs):
            pts = ex.poles_in(d, syms, ranges)
            if pts:
                cp.classical_valid = False
                cp.reasons.append("classical: dU/d%s is singular at %s"
                                  % (syms[i].name, "; ".join(pts)))

    # -- numeric probes -----------------------------------------------------
    with numpy.errstate(all="ignore"):
        if cp.classical_valid and cp.grad and ranges is not None:
            meshes = ex.probe_meshes(ranges)
            for i, gi in enumerate(cp.grad):
                dv = numpy.asarray(gi(*meshes))
                bad = ~numpy.isfinite(dv)
                if bad.any():
                    cp.classical_valid = False
                    cp.reasons.append("classical: dU/d%s is not finite for %s"
                                      % (syms[i].name,
                                         ex.bad_box(syms, meshes, bad)))
        if extended is not None:
            meshes = ex.probe_meshes(extended, dtype=numpy.complex128)
            uv = numpy.asarray(cp.U(*meshes))
            bad = ~numpy.isfinite(uv)
            if bad.any():
                cp.quantum_valid = False
                cp.reasons.append(
                    "quantum: U is not finite on the extended range the "
                    "propagator evaluates it on (%s)"
                    % ex.bad_box(syms, meshes, bad))
            else:
                im = numpy.abs(uv.imag) > 1e-12*(1. + numpy.abs(uv.real))
                if numpy.any(im):
                    cp.warnings.append(
                        "quantum: U takes complex values for %s (extended "
                        "range) - evolution will be non-unitary (absorbing) "
                        "there" % ex.bad_box(syms, meshes, im))
    return cp


def sample_potential(cp, x1, x2, n=400):
    """Sample U along the first spatial axis over [x1, x2] for the preview
    plot, with any other spatial variable held at 0. Non-finite values are
    mapped to None (JSON null) so the client can show gaps."""
    xs = numpy.linspace(float(x1), float(x2), int(n))
    args = [xs] + [numpy.zeros_like(xs)]*(cp.ndim - 1)
    with numpy.errstate(all="ignore"):
        us = numpy.broadcast_to(
            numpy.asarray(cp.U(*args), dtype=numpy.float64), xs.shape)
    ok = numpy.isfinite(us)
    return xs.tolist(), [float(u) if o else None for u, o in zip(us, ok)]


def sample_potential_grid(cp, ranges, n=128):
    """Sample U(x, y) on an n x n lattice over `ranges` for the 2D preview
    heatmap (from which the client also reads its axis cuts). Returns
    (xs, ys, rows) with non-finite values as None."""
    if cp.ndim != 2:
        raise PotentialError("sample_potential_grid needs a 2D potential")
    xs = numpy.linspace(*map(float, ranges[0]), int(n))
    ys = numpy.linspace(*map(float, ranges[1]), int(n))
    with numpy.errstate(all="ignore"):
        us = numpy.broadcast_to(
            numpy.asarray(cp.U(xs[:, None], ys[None, :]), dtype=numpy.float64),
            (len(xs), len(ys)))
    return (xs.tolist(), ys.tolist(),
            [[float(v) if numpy.isfinite(v) else None for v in row]
             for row in us])
