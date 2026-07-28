"""
Human-readable descriptions of a session: the analytic initial condition and
the parameter block that make an exported video self-contained (everything
needed to reproduce the run is ON the frame).

Plain Unicode text ONLY in THIS module — never matplotlib mathtext — for two
reasons, and the first is the one that cannot be measured away:

1. U(x) is user input ("x^2/2", "Abs(x)", ...) and would either fail to parse
   or render as nonsense in math mode. (A "$" cannot actually reach here — the
   potential tokenizer refuses it — but the block also wraps by CHARACTER
   count, and "$p_x$" is five characters that render as two glyphs, so the
   column widths would stop meaning anything.)
2. mathtext is not free per artist: measured 2026-07-28 at 4.05 ms against
   plain text's 2.21 ms, i.e. 1.83x. That matters wherever text is re-rendered
   often.

NB the second reason used to be stated here as "mathtext is SLOW ... it was
measured to slow the export's plots down enormously", and as a blanket claim
that is WRONG — it was the reason the exported frame spelled its momentum axes
"px"/"py" instead of with the real subscripts the SPA shows. Re-measured for
M4: the figure BLITS, so every title and axis label is a STATIC artist baked
into the background once, and putting real subscripts on all of them costs
+147 ms on ONE figure build and 1.00-1.02x per frame — nothing. render_mpl
therefore uses mathtext for the axis names (axes.sub_math) and this module
still does not, for reason 1. usetex is the genuinely expensive one, 12x, and
also needs a LaTeX install and a third spelling of every string.

The same facts also go into the mp4 metadata tag as JSON (`config_json`),
which is what a machine should read back.

No matplotlib import here on purpose: this module is pure string work and
is unit-tested as such.
"""

import json
import time


def _func_names():
    """The function names the potential tokenizer admits, from its OWN
    namespace rather than a copy that can drift. In math mode an unwrapped
    "exp" renders as the product e·x·p in italics, so each has to be marked
    upright. Longest first, so the alternation cannot prefer "sin" inside
    "asin". `pi`/`E` are constants, not functions, and are handled separately.

    Imported lazily to keep this module's import graph free of sympy — it is
    pure string work and is unit-tested as such (see the module docstring),
    and the same lazy-import rule already applies to `axes`."""
    from .potential import _NONSMOOTH, _SMOOTH
    return sorted((set(_SMOOTH) | set(_NONSMOOTH)) - {"pi", "E"},
                  key=len, reverse=True)


def potential_math(src):
    """The user's U expression, typeset for matplotlib mathtext — as a LEXICAL
    rewrite of the string they typed, NOT a round trip through sympy.

    Measured 2026-07-28 with sympy.latex on the parsed expression: it renders
    every real potential without a mathtext parse error, and it is still the
    wrong tool here for two reasons.
      - It CANONICALISES. `x^2/2 + 0.3*x^4` comes back as `0.3x^4 + x^2/2`, and
        `10*(1-exp(-0.5*(x-1)))^2` as `10(1 - 1.64872127070013 e^{-0.5x})^2` —
        sympy evaluated exp(0.5) at parse time. That is mathematically the same
        function and visually a different one, and this block's job is "how to
        reproduce this run": the source string is what you paste back into the
        U(x) box, and the video has to read like that box.
      - It emits \\frac, which is TALL. The metadata block advances by a fixed
        META_LINESPACING, so a stacked fraction overprints its neighbours.
    The rewrite below has neither problem: it preserves the source token for
    token and introduces no construct taller than a superscript.

    Returns None if the result would not be an improvement, in which case the
    caller keeps the plain string. Whatever comes back is still validated by
    the caller against matplotlib's own parser — this is display formatting of
    user input, so it must not be the only line of defence.
    """
    import re
    if not src or "$" in src or "\\" in src:
        return None            # nothing we are prepared to typeset
    out = src
    # multi-character exponents need braces: x^12 is x^1 followed by a 2
    out = re.sub(r"\^\s*(\w+)", lambda m: "^{%s}" % m.group(1), out)
    # upright function names, before mathtext's italic-variable default
    # applies to their letters
    out = re.sub(r"\b(%s)\b" % "|".join(_func_names()),
                 lambda m: "\\mathrm{%s}" % m.group(1), out)
    out = re.sub(r"\bpi\b", r"\\pi", out)
    out = re.sub(r"\bE\b", r"\\mathrm{e}", out)
    # explicit multiplication reads as a thin space in maths, not an asterisk
    out = out.replace("*", r"\,")
    return out if out != src else None


def _num(v):
    """Compact number formatting: 2 stays "2", 0.70711 keeps its digits."""
    f = float(v)
    if f == int(f) and abs(f) < 1e16:
        return "%d" % int(f)
    return "%.6g" % f


def _shift(var, v):
    """"x − 2.5" / "x + 2.5" / "x" — never the "x−−2.5" a bare _num gives."""
    f = float(v)
    if f == 0.0:
        return var
    return "%s %s %s" % (var, "−" if f > 0 else "+", _num(abs(f)))


def _sigma_k_of(comp, ic_type, hbar_eff):
    """sigma_k per dimension as the solver uses it (derived for cat states —
    see initial.minimal_sigma_p; components carry their own for mixtures)."""
    if ic_type == "cat":
        return tuple(hbar_eff/(2.*s) for s in comp.sigma_q)
    return tuple(comp.sigma_k)


def _ax_labels(ndim):
    from . import axes as ax
    return ax.labels(ndim)


def _mathify(lines, ndim, math):
    """Subscript the axis names in already-assembled lines. Applied LAST, so
    the line contents and their character positions are decided by the plain
    code path and only the spelling of "px"/"py" changes — a no-op at ndim=1."""
    if not math:
        return lines
    from . import axes as ax
    return [ax.sub_math_text(l, ndim) for l in lines]


def _vec(parts, sep=", "):
    """"2" for one value, "(2, 1.67)" for several — tuple notation only where
    there IS a tuple, so 1D text reads exactly as it always did."""
    return parts[0] if len(parts) == 1 else "(%s)" % sep.join(parts)


def ic_expression(ic, hbar_eff, ndim=None, math=False):
    """The initial condition as an analytic expression, with the concrete
    numbers substituted. Returns a list of text lines.

    - mixture: W itself (initial.mixture_wigner) — a weighted sum of
      normalized Gaussian blobs, one factor per phase-space axis.
    - cat: the WAVEFUNCTION psi (initial.cat_wigner builds W from the
      pairwise cross-Wigner closed form, which is far longer to print and
      carries no extra information): W = Wigner[psi] is the complete and
      compact specification. In 2D each packet is the separable product over
      dimensions that cat_wigner assumes, so the same one-line form per
      dimension is exact.
    """
    from . import axes as ax
    comps = list(ic.components)
    hbar = float(hbar_eff)
    nd = ndim if ndim is not None else comps[0].ndim
    labels = ax.labels(nd)
    args = ",".join(labels)
    # psi lives on configuration space only — "psi(x,p,0)" would be nonsense
    qargs = ",".join(labels[:nd])

    if ic.type == "cat":
        terms = []
        for c in comps:
            amp = "√%s·" % _num(c.weight) if c.weight != 1.0 else ""
            phase = "e^(i%s)·" % _num(c.phase) if c.phase else ""
            factors = []
            for i in range(nd):
                q, s = labels[i], _num(c.sigma_q[i])
                u = "(%s)" % _shift(q, c.q0[i])
                factors.append("(2π·%s²)^(−1/4)·exp(−%s²/(4·%s²) + i·%s·%s/ℏ)"
                               % (s, u, s, _num(c.k0[i]), u))
            terms.append("%s%s%s" % (amp, phase, "·".join(factors)))
        lines = ["IC (cat state, ℏ = %s):  W(%s,0) = Wigner[ψ],  ψ(%s,0) = "
                 "S^(−1/2)·[" % (_num(hbar), args, qargs)]
        lines.append("    " + "  +  ".join(terms) + " ]")
        # "σpx" has no word boundary between the sigma and the axis name, so
        # the blanket pass at the end cannot reach it — subscript it here
        klabel = _vec(["σ%s" % (ax.sub_math(l) if math else l)
                       for l in labels[nd:]], sep=",")
        sig = ", ".join(
            "σ%d = %s (%s = %s)"
            % (j + 1, _vec([_num(s) for s in c.sigma_q]), klabel,
               _vec([_num(s) for s in _sigma_k_of(c, "cat", hbar)]))
            for j, c in enumerate(comps))
        lines.append("    with %s;  S = ⟨ψ|ψ⟩ (analytic normalization)" % sig)
        return _mathify(lines, nd, math)

    wtot = sum(c.weight for c in comps)
    terms = []
    for c in comps:
        sk = _sigma_k_of(c, "mixture", hbar)
        amp = c.weight/wtot
        for i in range(nd):
            amp /= 2.*3.141592653589793*c.sigma_q[i]*sk[i]
        parts = []
        for i in range(nd):
            parts.append("(%s)²/(2·%s²)" % (_shift(labels[i], c.q0[i]),
                                            _num(c.sigma_q[i])))
            parts.append("(%s)²/(2·%s²)" % (_shift(labels[nd + i], c.k0[i]),
                                            _num(sk[i])))
        terms.append("%s·exp(−%s)" % (_num(amp), " − ".join(parts)))
    return _mathify(["IC (Gaussian mixture):  W(%s,0) = " % args,
                     "    " + "  +  ".join(terms)], nd, math)


# wire field -> the label the Setup panel puts on it
FIELD_LABEL = {"U": "U(x)", "mass": "m", "c": "c", "hbar_eff": "ℏ",
               "tol": "tol", "dt_sign": "t dir", "auto_expand": "auto-expand"}
# wire field -> SessionCreate attribute (dt_sign has none: it is not config)
FIELD_ATTR = {"U": "potential", "mass": "mass", "c": "c",
              "hbar_eff": "hbar_eff", "tol": "tol",
              "auto_expand": "auto_expand"}


def _value(field, v):
    if field == "auto_expand":
        return "on" if v else "off"
    if field == "dt_sign":
        return "backward" if v < 0 else "forward"
    if field == "U":
        return str(v)
    return _num(v)


def state_at(cfg, param_log=(), k0=None):
    """The physics as of record `k0`, rewound from the session's CURRENT
    values through the log's `before` entries. Without this the block on a
    frame computed at ℏ = 1 would carry the ℏ = 100 the run happened to end
    with. Returns {attr: value} overrides for the cfg fields."""
    out = {}
    if k0 is None:
        return out
    for e in reversed(list(param_log)):
        if e["at_record"] <= k0:
            break
        for field, v in (e.get("before") or {}).items():
            if field in FIELD_ATTR:
                out[FIELD_ATTR[field]] = v
    return out


def _U_line(nd, text, math):
    """"U(x,y) = <expr>", typeset when asked and when the expression is one we
    can typeset (see potential_math). The ARGUMENT list is subscripted either
    way at ndim=2 — it is our own string, not the user's."""
    from . import axes as ax
    args = ",".join(ax.sub_math(l) if math else l for l in _ax_labels(nd)[:nd])
    if math:
        tex = potential_math(text)
        if tex is not None:
            return "U(%s) = $%s$" % (args, tex)
    return "U(%s) = %s" % (args, text)


def param_lines(cfg, param_log=(), k0=None, k1=None, math=False):
    """The physics/setup block, describing the state at record `k0` (not the
    session's latest). `param_log` entries inside [k0, k1] are listed: a live
    U/mass/ℏ change mid-range would otherwise make the block a lie about the
    frames that follow it.

    `math` typesets U(x,y) and the axis names for matplotlib mathtext — the
    exported frame asks for it, the mp4 comment tag and the setup document
    never do (they are machine-readable, and the raw expression is what a
    reader pastes back into the U(x) box)."""
    st = state_at(cfg, param_log, k0)
    at = lambda f: st.get(f, getattr(cfg, f))
    nd = cfg.grid.ndim
    # label the fields exactly as the UI does: ℏ (not hbar_eff — the Physics
    # panel calls it ℏ). The mode's wire value ("batch"/"interactive") is also
    # its display label, so no remapping is needed.
    lines = [
        _U_line(nd, at("potential"), math),
        "m = %s   c = %s   ℏ = %s   tol = %s"
        % (_num(at("mass")), _num(at("c")), _num(at("hbar_eff")),
           _num(at("tol"))),
        "t₁ = %s   record_dt = %s   mode = %s%s   auto-expand: %s"
        % (_num(cfg.t1), _num(cfg.record_dt), cfg.mode,
           ("  t₂ = %s" % _num(cfg.t2)) if cfg.t2 is not None else "",
           "on" if at("auto_expand") else "off"),
    ]
    # A line of its own, and only when it applies. Tacked onto the line above
    # it would be the least conspicuous thing there, and this is the one fact
    # about a preview run that a viewer must not miss — the video outlives the
    # session that could have told them.
    if getattr(cfg, "precision", "float64") == "float32":
        lines.append("precision: float32 — PREVIEW run, reduced accuracy "
                     "(purity/energy drift and uncertainty noise ~1e-4)")
    changes = [e for e in param_log
               if (k0 is None or e["at_record"] >= k0)
               and (k1 is None or e["at_record"] <= k1)]
    for e in changes:
        before = e.get("before") or {}
        parts = []
        for field, v in e["applied"].items():
            label = ("U(%s)" % ",".join(_ax_labels(nd)[:nd]) if field == "U"
                     else FIELD_LABEL.get(field, field))
            # a live U change quotes the expression verbatim on both sides of
            # the arrow, and typesetting only one of a "before → after" pair
            # would read as two different KINDS of thing
            if field in before:
                parts.append("%s %s → %s" % (label, _value(field, before[field]),
                                             _value(field, v)))
            else:
                parts.append("%s = %s" % (label, _value(field, v)))
        lines.append("live change at record %d: %s"
                     % (e["at_record"], ", ".join(parts)))
    return lines


SETUP_FORMAT = "wignerf-setup"
SETUP_VERSION = 1


def setup_document(cfg, param_log=()):
    """The exchangeable "initial conditions" of a run: exactly the config the
    session was CREATED with, whatever happened to it since.

    Live changes mutate session.cfg and auto-expand moves the grid, so the
    log's `before` values are rewound at k0 = -1 (i.e. every entry) to get
    back what POST /api/sessions was given. Live changes are deliberately NOT
    part of this — a mid-run ℏ change is not a starting state; the exported
    video's metadata block is where they are recorded."""
    conf = json.loads(cfg.model_dump_json())
    conf.update(state_at(cfg, param_log, -1))
    return {"format": SETUP_FORMAT, "version": SETUP_VERSION,
            "generator": "wignerf",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": conf}


def config_json(cfg, param_log=(), at_record=None, **extra):
    """Machine-readable twin of the visible block (mp4 `comment` tag).

    `at_record` rewinds the physics to that record (see state_at), so the
    JSON says what the visible block says — the log's before/after entries
    carry the rest."""
    conf = json.loads(cfg.model_dump_json())
    conf.update(state_at(cfg, param_log, at_record))
    d = {"generator": "wignerf", "config": conf, "param_log": list(param_log)}
    d.update(extra)
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
