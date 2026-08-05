"""
Safe compilation of the analytic expressions this program evaluates on a user's
behalf — THE security boundary, and deliberately the only one.

Three kinds of expression reach it, differing only in which names they may use
and in what their caller then considers valid:

    U(x) / U(x, y)              a real potential        core/potential.py
    W(x, p) / W(x,y,px,py)      an initial Wigner fn    core/initial.py
    psi(x) / psi(x, y)          an initial wavefunction core/initial.py

Pipeline, in this order and for a reason: a tokenize screen that rejects
anything but whitelisted names, numbers and operators BEFORE sympy's parse_expr
(which evals) ever sees the string; an unevaluated parse feeding a numeric-power
screen (9**9**9 would otherwise materialize a ~4e8-digit integer, pinning a CPU
and gigabytes of RAM); then the real evaluated parse and a free-symbol check.

Split out of potential.py on 2026-08-04, when the initial condition gained its
own expression tabs. It holds no VALIDITY policy of its own — "U must be real
and finite on the extended Bopp box" and "psi may be complex and lives on
configuration space only" are different questions — but safety is one question
with one answer, so the screen lives here once and every caller goes through it.

A `Namespace` bundles the three things that vary: the free symbols an expression
may use, the sympy objects the parser may name, and whether the imaginary unit
is among them. Everything below takes one, rather than indexing a module-level
per-ndim table — which is what let the probe helpers move here at all, since
their old form knew only about spatial axes.
"""

import io
import math
import tokenize
from dataclasses import dataclass

import numpy
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        convert_xor)

MAX_EXPR_LEN = 500
MAX_POW_DIGITS = 300   # numeric powers must stay below ~1e300 (float range)
# Orthogonal-polynomial order cap. hermite(n, x) is a Function in the
# UNEVALUATED parse, so screen_powers cannot see it, and the evaluated parse
# materializes the polynomial: measured 0.028 s / 101 terms at n = 200, 0.293 s
# / 501 terms at n = 1000, superlinear, with n arriving straight from a form
# that recompiles on every keystroke. 200 is far past any state anyone plots.
MAX_POLY_ORDER = 200

# Probe lattice size per axis, by how many axes there are. ODD on purpose, and
# shrinking fast: the phase-space namespaces have 2*ndim axes, so the old flat
# 129-per-axis would be 129^4 = 2.8e8 points for a 2D W(x,y,px,py) — minutes of
# probing on every keystroke. 33^4 = 1.19e6 costs milliseconds and still forces
# the axes and the origin, which is what the lattice is actually for.
PROBE_POINTS = {1: 4096, 2: 129, 3: 49, 4: 33}


class ExprError(ValueError):
    """A user expression that is rejected outright — bad syntax, a forbidden
    name, the wrong free symbols. Per-use-case VALIDITY failures are not this:
    callers report those on their own result objects."""


# -- the whitelisted namespace ------------------------------------------------

SMOOTH = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log,
    "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
    "pi": sp.pi, "E": sp.E,
}
NONSMOOTH = {
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceiling": sp.ceiling,
    "Max": sp.Max, "Min": sp.Min, "Mod": sp.Mod,
    "Piecewise": sp.Piecewise, "Heaviside": sp.Heaviside,
}
# hermite(n, x) with a NUMERIC n auto-expands to a polynomial under
# evaluate=True, so lambdify needs nothing special for it; the harmonic
# oscillator's eigenstates are the first thing anyone reaches for after a
# coherent state, and spelling H_3 out by hand is how a typo becomes a physics
# result. A SYMBOLIC order is refused by screen_poly_orders — it survives the
# free-symbol check (its arguments are allowed names) and lambdify accepts it
# happily, producing a callable that only fails when it is called.
SPECIAL = {"hermite": sp.hermite}
IMAGINARY = {"I": sp.I, "i": sp.I}

_TRANSFORMS = standard_transformations + (convert_xor,)

# Every axis name this program can ever offer, in every namespace, at every
# ndim: `y` is in a 1D potential's namespace and `px` in a 1D wavefunction's on
# PURPOSE. A 1D session that types x*y or px should get the free-symbol message
# naming what IS allowed here, not a tokenizer refusal that reads like a typo
# about a name the program obviously knows.
_ALL_AXIS_NAMES = ("x", "y", "p", "px", "py")
_SYM = {n: sp.Symbol(n, real=True) for n in _ALL_AXIS_NAMES}


def symbols(names):
    """The shared real Symbol for each axis name — shared so that two
    namespaces built from the same name compare and substitute as one symbol."""
    return tuple(_SYM[n] for n in names)


@dataclass(frozen=True)
class Namespace:
    """What one kind of expression may contain.

    `free` are the symbols it may use as variables; `locals` is what parse_expr
    may name (which is a SUPERSET of `free` — see _ALL_AXIS_NAMES); `label` is
    how the expression is named in a refusal ("U(x)", "psi(x,y)").
    """
    free: tuple
    locals: dict
    label: str
    allow_complex: bool = False

    @property
    def allowed(self):
        return frozenset(self.locals)

    @property
    def names(self):
        return ", ".join("'%s'" % s.name for s in self.free)


def namespace(free_names, label, *, complex_ok=False):
    """Build a namespace over `free_names`.

    THE PARSER'S VOCABULARY IS THE SAME FOR EVERY KIND; only `free` and
    `complex_ok` differ. That is the rule potential.py already stated for `y`
    ("a 1D session that types x*y should get the free-symbol message naming
    what is allowed, not a tokenizer refusal that reads like a typo"), applied
    consistently: every axis name, and `I` too. So `I*x` in a potential is
    refused as "U(x) must be a real expression" rather than "name 'I' is not
    allowed" — the user knows perfectly well what I means, and the real check
    was never the tokenizer anyway (sqrt(-1) and log(-1) evaluate to an explicit
    I without ever naming it, and only the has(I) test below sees those).

    The tokenizer then refuses exactly what it should: names this program does
    not know at all.
    """
    loc = {**SMOOTH, **NONSMOOTH, **SPECIAL, **IMAGINARY,
           **{n: _SYM[n] for n in _ALL_AXIS_NAMES},
           "True": sp.true, "False": sp.false}   # Piecewise conditions
    return Namespace(free=symbols(free_names), locals=loc, label=label,
                     allow_complex=complex_ok)


# -- the screens --------------------------------------------------------------

def screen_tokens(src, ns):
    """Reject anything but whitelisted names, numbers and operators BEFORE
    sympy's parse_expr (which evals) ever sees the string."""
    if len(src) > MAX_EXPR_LEN:
        raise ExprError("expression too long (max %d characters)" % MAX_EXPR_LEN)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError as e:
        raise ExprError("cannot tokenize expression: %s" % e) from None
    allowed = ns.allowed
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string not in allowed:
            raise ExprError("name '%s' is not allowed" % tok.string)
        if tok.type == tokenize.OP and tok.string in (".", "=", ":=", ";", "@"):
            raise ExprError("operator '%s' is not allowed" % tok.string)
        if tok.type == tokenize.STRING:
            raise ExprError("string literals are not allowed")


def screen_powers(expr):
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
            raise ExprError("numeric power is too large") from None
        if b > 0.0 and b != 1.0 and p*abs(math.log10(b)) > MAX_POW_DIGITS:
            raise ExprError(
                "numeric power is too large (|result| would exceed 1e%d)"
                % MAX_POW_DIGITS)


def screen_poly_orders(expr):
    """Refuse an orthogonal polynomial of astronomical order, on the
    UNEVALUATED parse and for the same reason screen_powers exists: the
    evaluated parse is what materializes it, so by the time it could be
    measured it has already been paid for. See MAX_POLY_ORDER."""
    for node in sp.preorder_traversal(expr):
        if not (isinstance(node, sp.Function) and node.args
                and type(node).__name__ in ("hermite", "chebyshevt",
                                            "chebyshevu", "legendre",
                                            "laguerre")):
            continue
        name, n = type(node).__name__, node.args[0]
        if not n.is_number:
            # A SYMBOLIC order — hermite(x, x) — passes the free-symbol check
            # (its arguments are allowed names) and lambdify accepts it without
            # complaint, returning a callable that raises only when it is
            # called, from inside the build. Refuse it here, where there is
            # still a sentence to attach.
            raise ExprError("the order of %s must be a number, not an "
                            "expression" % name)
        try:
            order = abs(float(n))
        except (OverflowError, TypeError, ValueError):
            raise ExprError("the order of %s is not a usable number"
                            % name) from None
        # NB outside the try: ExprError IS a ValueError, so raising it in there
        # would be caught by that except and reported as the wrong problem.
        if order > MAX_POLY_ORDER:
            raise ExprError("%s order %s is too large (max %d)"
                            % (name, n, MAX_POLY_ORDER))


def parse(src, ns):
    """Screen, parse and check free symbols. Returns the sympy expression.

    Raises ExprError for everything a caller cannot act on; per-use-case
    validity (finiteness on a box, a differentiable gradient, realness) is the
    caller's job and is deliberately not decided here.
    """
    src = src.strip()
    if not src:
        raise ExprError("empty expression")
    screen_tokens(src, ns)
    # Two-phase parse: an unevaluated pass feeds the screens (nothing is
    # materialized), then the real evaluated parse.
    #
    # BOTH the keyword and the context manager are needed, and the keyword alone
    # is a trap. parse_expr(evaluate=False) suppresses only ARITHMETIC
    # evaluation — a FUNCTION application still runs, so hermite(400, x) arrives
    # at the screen already expanded into the polynomial the screen exists to
    # refuse: measured 0.116 s to parse against 0.002 s inside sp.evaluate(False),
    # and hermite(50000, x) stays a bare node under it rather than materializing
    # coefficients past CPython's 4300-digit int-to-str limit.
    try:
        with sp.evaluate(False):
            unev = parse_expr(src, local_dict=ns.locals,
                              transformations=_TRANSFORMS, evaluate=False)
    except Exception as e:
        raise ExprError("parse error: %s" % e) from None
    if isinstance(unev, sp.Basic):
        screen_powers(unev)
        screen_poly_orders(unev)
    try:
        expr = parse_expr(src, local_dict=ns.locals, transformations=_TRANSFORMS,
                          evaluate=True)
    except Exception as e:
        raise ExprError("parse error: %s" % e) from None
    if not isinstance(expr, sp.Expr):
        raise ExprError("not a valid expression in %s"
                        % ", ".join(s.name for s in ns.free))
    free = set(ns.free)
    if not expr.free_symbols <= free:
        extra = ", ".join(sorted(str(s) for s in expr.free_symbols - free))
        raise ExprError("only %s may appear as a variable in %s (got: %s)"
                        % (ns.names, ns.label, extra))
    if not ns.allow_complex and expr.has(sp.I):
        raise ExprError("%s must be a real expression" % ns.label)
    return expr


# -- evaluation ---------------------------------------------------------------

def _xp_of(vs):
    """The array module the inputs came from (cupy on a GPU backend)."""
    for v in vs:
        if getattr(type(v), "__module__", "").startswith("cupy"):
            import cupy
            return cupy
    return numpy


def lambdify(syms, expr, modules):
    """lambdify, plus one guarantee callers rely on: array in, array out.

    A CONSTANT expression is the exception sympy makes — it compiles to a
    function returning a python scalar, which then has no shape for a sampler to
    zip over. Broadcasting a scalar with broadcast_to is a VIEW, so honouring the
    contract costs nothing even when the inputs span a 4D state; the branch is
    decided once, at compile time, so the hot path (any expression with a free
    symbol) is the bare lambdified function.

    A lambdify FAILURE is an ExprError, not a traceback from somewhere in
    sympy's printer: an expression can pass every screen above and still have no
    numpy spelling — hermite(x, x) is the reachable case, since a symbolic order
    is made of allowed names and free symbols the check accepts.
    """
    try:
        f = sp.lambdify(syms, expr, modules=modules)
    except Exception as e:
        raise ExprError("cannot evaluate this expression numerically: %s"
                        % e) from None
    if expr.free_symbols:
        return f

    def const(*vs, _f=f):
        xp = _xp_of(vs)
        shape = numpy.broadcast_shapes(*(getattr(v, "shape", ()) for v in vs))
        return xp.broadcast_to(xp.asarray(_f(*vs)), shape)
    return const


# -- probe lattices -----------------------------------------------------------

def axis_probe(lo, hi, n):
    """`n` samples of [lo, hi], with an exact 0.0 substituted for the nearest
    sample when the range straddles the origin.

    The poles that matter past one dimension sit on the axes and at the origin
    (1/sqrt(x^2+y^2), 1/x, log(x)) and an even lattice steps straight over them —
    sympy's singularity machinery is one-dimensional, so past ndim=1 this
    lattice is most of the guard."""
    xs = numpy.linspace(float(lo), float(hi), int(n))
    if lo < 0.0 < hi:
        xs[int(numpy.argmin(numpy.abs(xs)))] = 0.0
    return xs


def probe_meshes(ranges, dtype=None):
    """Broadcast probe meshes, one per axis: a dense sweep over one axis, an odd
    lattice over several (see PROBE_POINTS for why it thins so fast)."""
    na = len(ranges)
    n = PROBE_POINTS.get(na, PROBE_POINTS[max(PROBE_POINTS)])
    out = []
    for i, (lo, hi) in enumerate(ranges):
        xs = axis_probe(lo, hi, n)
        if dtype is not None:
            xs = xs.astype(dtype)
        shape = [1]*na
        shape[i] = n
        out.append(xs.reshape(tuple(shape)))
    return tuple(out)


def bad_box(syms, meshes, bad):
    """Where a probe failed, as a readable per-axis bounding box."""
    shape = numpy.broadcast_shapes(*(m.shape for m in meshes))
    bad = numpy.broadcast_to(bad, shape)
    parts = []
    for i, m in enumerate(meshes):
        v = numpy.broadcast_to(numpy.real(m), shape)[bad]
        parts.append("%s ∈ [%.4g, %.4g]" % (syms[i].name, v.min(), v.max()))
    return ", ".join(parts)


def box_str(syms, ranges):
    return " × ".join("%s ∈ [%.4g, %.4g]" % (syms[i].name, lo, hi)
                      for i, (lo, hi) in enumerate(ranges))


def poles_in(expr, syms, ranges):
    """Symbolic singularities inside the box, best effort. sympy's
    `singularities` is one-dimensional, so past one variable each is scanned
    with the others pinned at a few sample values — it cannot decide every
    whitelisted expression (Piecewise, for one), which is why the numeric
    lattice above forces the axes and the origin."""
    found = []
    for i, s in enumerate(syms):
        lo, hi = ranges[i]
        others = [j for j in range(len(syms)) if j != i]
        for combo in pin_combos(ranges, others):
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


def pin_combos(ranges, others):
    """Sample values to pin the other variables at: both ends, the midpoint,
    and 0 when the range straddles it.

    THINNED past two variables, because the combination count is the product.
    With one other variable that is 4 values and 4 sympy `singularities` calls;
    with three it would be 4^3 = 64 per symbol, 256 in total, each of them a
    symbolic solve — seconds of CPU on every keystroke in a form that fires on
    every keystroke. Past two the ends are dropped and only the midpoint and the
    origin are kept, which is where the poles that matter actually sit (see the
    axis_probe docstring); the numeric lattice remains the real guard there and
    always was.
    """
    if not others:
        return [()]
    lean = len(others) > 1
    per = []
    for j in others:
        lo, hi = ranges[j]
        vals = [(lo + hi)/2.] if lean else [lo, (lo + hi)/2., hi]
        if lo < 0.0 < hi:
            vals.append(0.0)
        per.append(vals)
    combos = [()]
    for vals in per:
        combos = [c + (v,) for c in combos for v in vals]
    return combos
