"""
Safe compilation of user-entered analytic potentials U(x) / U(x, y).

Pipeline: token screen (the security boundary) -> sympy parse in a
whitelisted namespace -> per-variant-family validity -> lambdify for
numpy (and cupy when a GPU backend is active).

Validity model (see plan): the Bopp arguments of the quantum differential
are REAL (q_i -+ hbar*theta_i/2, complex dtype only), so:
- quantum-valid  = U real and finite on the extended real BOX
  [q1_i - hbar*theta_amp_i/2, q2_i + hbar*theta_amp_i/2] over every spatial
  axis (numeric probe; Abs(x) is quantum-valid, no analyticity required).
  Complex values there (branch cuts) make the evolution non-unitary: reported
  as a warning, since an absorbing potential may be intended.
- classical-valid = EVERY partial dU/dq_i well-defined on the visible box with
  no DiracDelta (so a Heaviside step potential is quantum-only).

The numeric probe lattice deliberately uses an ODD point count per axis and
forces an exact 0.0 onto any axis whose range straddles the origin. The poles
that matter in 2D sit on the axes and at the origin (1/sqrt(x^2+y^2), 1/x,
log(x)), and an even lattice steps straight over them — sympy's singularity
machinery is one-dimensional, so past ndim=1 the lattice is most of the guard.
"""

import io
import math
import tokenize
from dataclasses import dataclass, field

import numpy
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        convert_xor)

MAX_EXPR_LEN = 500
PROBE_POINTS = 4096    # ndim=1: one axis, so a dense sweep is free
PROBE_POINTS_ND = 129  # per axis past 1D (129^2 = 16641 points); ODD on purpose
MAX_POW_DIGITS = 300   # numeric powers must stay below ~1e300 (float range)

_X = sp.Symbol("x", real=True)
_Y = sp.Symbol("y", real=True)

# The spatial symbols a potential may use, per ndim.
SYMBOLS = {1: (_X,), 2: (_X, _Y)}

_SMOOTH = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log,
    "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
    "pi": sp.pi, "E": sp.E,
}
_NONSMOOTH = {
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceiling": sp.ceiling,
    "Max": sp.Max, "Min": sp.Min, "Mod": sp.Mod,
    "Piecewise": sp.Piecewise, "Heaviside": sp.Heaviside,
}
# "y" is in the namespace at EVERY ndim on purpose: a 1D session that types
# x*y should get the free-symbol message naming what is allowed, not a
# tokenizer refusal that reads like a typo.
_LOCALS = {**_SMOOTH, **_NONSMOOTH, "x": _X, "y": _Y,
           "True": sp.true, "False": sp.false}   # Piecewise conditions
_ALLOWED_NAMES = set(_LOCALS)

_TRANSFORMS = standard_transformations + (convert_xor,)


class PotentialError(ValueError):
    pass


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
            self.U_gpu = _lambdify(syms, self.expr, "cupy")
            self.grad_gpu = tuple(_lambdify(syms, e, "cupy")
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


def _screen_tokens(src):
    """Reject anything but whitelisted names, numbers and operators BEFORE
    sympy's parse_expr (which evals) ever sees the string."""
    if len(src) > MAX_EXPR_LEN:
        raise PotentialError("expression too long (max %d characters)" % MAX_EXPR_LEN)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError as e:
        raise PotentialError("cannot tokenize expression: %s" % e) from None
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string not in _ALLOWED_NAMES:
            raise PotentialError("name '%s' is not allowed" % tok.string)
        if tok.type == tokenize.OP and tok.string in (".", "=", ":=", ";", "@"):
            raise PotentialError("operator '%s' is not allowed" % tok.string)
        if tok.type == tokenize.STRING:
            raise PotentialError("string literals are not allowed")


def _screen_powers(expr):
    """Reject numeric powers of astronomical magnitude BEFORE evaluation:
    parse_expr(evaluate=True) of e.g. 9**9**9 would try to materialize a
    ~4e8-digit integer, pinning a CPU and gigabytes of RAM. Called on the
    unevaluated parse; the magnitude estimate uses float logs only."""
    for node in sp.preorder_traversal(expr):
        if not (isinstance(node, sp.Pow) and node.base.is_number
                and node.exp.is_number):
            continue
        try:
            b = abs(complex(node.base.evalf(8)))
            p = abs(complex(node.exp.evalf(8)))
        except (OverflowError, TypeError, ValueError):
            raise PotentialError("numeric power is too large") from None
        if b > 0.0 and b != 1.0 and p*abs(math.log10(b)) > MAX_POW_DIGITS:
            raise PotentialError(
                "numeric power is too large (|result| would exceed 1e%d)"
                % MAX_POW_DIGITS)


def _xp_of(vs):
    """The array module the inputs came from (cupy on a GPU backend)."""
    for v in vs:
        if getattr(type(v), "__module__", "").startswith("cupy"):
            import cupy
            return cupy
    return numpy


def _lambdify(syms, expr, modules):
    """lambdify, plus one guarantee callers rely on: array in, array out.

    A CONSTANT expression is the exception sympy makes — it compiles to a
    function returning a python scalar, which then has no shape for
    sample_potential to zip over. Broadcasting a scalar with broadcast_to is a
    VIEW, so honouring the contract costs nothing even when the inputs span a
    4D state; the branch is decided once, at compile time, so the hot path
    (any expression with a free symbol) is the bare lambdified function."""
    f = sp.lambdify(syms, expr, modules=modules)
    if expr.free_symbols:
        return f

    def const(*vs, _f=f):
        xp = _xp_of(vs)
        shape = numpy.broadcast_shapes(*(getattr(v, "shape", ()) for v in vs))
        return xp.broadcast_to(xp.asarray(_f(*vs)), shape)
    return const


def _axis_probe(lo, hi, n):
    """`n` samples of [lo, hi], with an exact 0.0 substituted for the nearest
    sample when the range straddles the origin — see the module docstring."""
    xs = numpy.linspace(float(lo), float(hi), int(n))
    if lo < 0.0 < hi:
        xs[int(numpy.argmin(numpy.abs(xs)))] = 0.0
    return xs


def _probe_meshes(ranges, dtype=None):
    """Broadcast probe meshes, one per spatial axis: a dense sweep at ndim=1,
    an odd square lattice past it."""
    nd = len(ranges)
    n = PROBE_POINTS if nd == 1 else PROBE_POINTS_ND
    out = []
    for i, (lo, hi) in enumerate(ranges):
        xs = _axis_probe(lo, hi, n)
        if dtype is not None:
            xs = xs.astype(dtype)
        shape = [1]*nd
        shape[i] = n
        out.append(xs.reshape(tuple(shape)))
    return tuple(out)


def _bad_box(meshes, bad, ndim):
    """Where a probe failed, as a readable per-axis bounding box."""
    shape = numpy.broadcast_shapes(*(m.shape for m in meshes))
    bad = numpy.broadcast_to(bad, shape)
    parts = []
    for i, m in enumerate(meshes):
        v = numpy.broadcast_to(numpy.real(m), shape)[bad]
        parts.append("%s ∈ [%.4g, %.4g]"
                     % (SYMBOLS[ndim][i].name, v.min(), v.max()))
    return ", ".join(parts)


def _box_str(ranges, ndim):
    return " × ".join("%s ∈ [%.4g, %.4g]" % (SYMBOLS[ndim][i].name, lo, hi)
                      for i, (lo, hi) in enumerate(ranges))


def _poles_in(expr, ndim, ranges):
    """Symbolic singularities inside the box, best effort. sympy's
    `singularities` is one-dimensional, so past ndim=1 each variable is scanned
    with the others pinned at a few sample values — it cannot decide every
    whitelisted expression (Piecewise, for one), which is why the numeric
    lattice above forces the axes and the origin."""
    syms = SYMBOLS[ndim]
    found = []
    for i, s in enumerate(syms):
        lo, hi = ranges[i]
        others = [j for j in range(ndim) if j != i]
        for combo in _pin_combos(ranges, others):
            e = expr.subs(dict(zip((syms[j] for j in others), combo))) \
                if others else expr
            try:
                sing = sp.singularities(e, s, sp.Interval(lo, hi))
            except Exception:
                continue
            if not isinstance(sing, sp.sets.sets.FiniteSet):
                continue
            for v in sing:
                if not v.is_real:
                    continue
                where = "%s = %.4g" % (s.name, float(v))
                if others:
                    where += " (at %s)" % ", ".join(
                        "%s = %.4g" % (syms[j].name, c)
                        for j, c in zip(others, combo))
                if where not in found:
                    found.append(where)
    return found


def _pin_combos(ranges, others):
    """Sample values to pin the other variables at: both ends, the midpoint,
    and 0 when the range straddles it."""
    if not others:
        return [()]
    per = []
    for j in others:
        lo, hi = ranges[j]
        vals = [lo, (lo + hi)/2., hi]
        if lo < 0.0 < hi:
            vals.append(0.0)
        per.append(vals)
    combos = [()]
    for vals in per:
        combos = [c + (v,) for c in combos for v in vals]
    return combos


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

    src = expr_str.strip()
    if not src:
        raise PotentialError("empty expression")
    _screen_tokens(src)
    # two-phase parse: an unevaluated pass feeds the power screen (nothing
    # is materialized), then the real evaluated parse
    try:
        unev = parse_expr(src, local_dict=_LOCALS, transformations=_TRANSFORMS,
                          evaluate=False)
    except Exception as e:
        raise PotentialError("parse error: %s" % e) from None
    if isinstance(unev, sp.Basic):
        _screen_powers(unev)
    try:
        expr = parse_expr(src, local_dict=_LOCALS, transformations=_TRANSFORMS,
                          evaluate=True)
    except Exception as e:
        raise PotentialError("parse error: %s" % e) from None
    if not isinstance(expr, sp.Expr):
        raise PotentialError("not a valid expression in %s"
                             % ", ".join(s.name for s in syms))
    allowed = ", ".join("'%s'" % s.name for s in syms)
    if not expr.free_symbols <= set(syms):
        extra = ", ".join(sorted(str(s) for s in expr.free_symbols - set(syms)))
        raise PotentialError("only %s may appear as a variable (got: %s)"
                             % (allowed, extra))
    if expr.has(sp.I):
        raise PotentialError("U(%s) must be a real expression"
                             % ", ".join(s.name for s in syms))

    cp = CompiledPotential(expr_str=src, expr=expr, ndim=ndim,
                           latex=sp.latex(expr))
    cp.U = _lambdify(syms, expr, "numpy")

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
        cp.grad = tuple(_lambdify(syms, d, "numpy") for d in grads)

    # -- symbolic pole detection (numeric probes can straddle a pole without
    # ever sampling it) ---------------------------------------------------
    if extended is not None:
        pts = _poles_in(expr, ndim, extended)
        if pts:
            cp.quantum_valid = False
            cp.reasons.append("quantum: U is singular at %s inside the "
                              "extended range %s the propagator evaluates it on"
                              % ("; ".join(pts), _box_str(extended, ndim)))
    if cp.classical_valid and cp.grad_exprs and ranges is not None:
        for i, d in enumerate(cp.grad_exprs):
            pts = _poles_in(d, ndim, ranges)
            if pts:
                cp.classical_valid = False
                cp.reasons.append("classical: dU/d%s is singular at %s"
                                  % (syms[i].name, "; ".join(pts)))

    # -- numeric probes -----------------------------------------------------
    with numpy.errstate(all="ignore"):
        if cp.classical_valid and cp.grad and ranges is not None:
            meshes = _probe_meshes(ranges)
            for i, gi in enumerate(cp.grad):
                dv = numpy.asarray(gi(*meshes))
                bad = ~numpy.isfinite(dv)
                if bad.any():
                    cp.classical_valid = False
                    cp.reasons.append("classical: dU/d%s is not finite for %s"
                                      % (syms[i].name,
                                         _bad_box(meshes, bad, ndim)))
        if extended is not None:
            meshes = _probe_meshes(extended, dtype=numpy.complex128)
            uv = numpy.asarray(cp.U(*meshes))
            bad = ~numpy.isfinite(uv)
            if bad.any():
                cp.quantum_valid = False
                cp.reasons.append(
                    "quantum: U is not finite on the extended range the "
                    "propagator evaluates it on (%s)"
                    % _bad_box(meshes, bad, ndim))
            else:
                im = numpy.abs(uv.imag) > 1e-12*(1. + numpy.abs(uv.real))
                if numpy.any(im):
                    cp.warnings.append(
                        "quantum: U takes complex values for %s (extended "
                        "range) - evolution will be non-unitary (absorbing) "
                        "there" % _bad_box(meshes, im, ndim))
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
