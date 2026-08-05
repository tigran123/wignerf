# Expression initial conditions — the measurements

`wexpr` (an arbitrary analytic W on phase space) and `psi` (an arbitrary complex
wavefunction, Wigner-transformed), added 2026-08-04. Split out of `CLAUDE.md` on
2026-08-05 (see `notes/precision.md` for why). What stayed there is the rule;
this is the measurement behind each one. Read it before touching
`initial.psi_wigner`, `_psi_lattice`, `wexpr_wigner` or `core/expr.py`.


## The expression-IC gotcha, as it stood before the split

- **EXPRESSION INITIAL CONDITIONS (`wexpr`, `psi`, 2026-08-04): the transform is
  four lines and every trap is in what the diagnostics CANNOT see.**
  `initial.psi_wigner` builds W(q,k) = (1/2π)^n ∫dⁿθ ψ*(q + ħθ/2) ψ(q − ħθ/2)
  e^{ik·θ} on the θ lattice `Grid` already owns (so ψ's arguments span exactly
  `spatial_extended`, the box the quantum U probe uses). Per momentum axis:
  multiply by the ramp e^{i k₀ θ}, inverse DFT, multiply by (−1)^m, scale by
  1/d[k]. Verified against `mixture_wigner`/`cat_wigner` at ~1e-14, Fock states
  give W(0,0) = −1/π exactly, and a coupled 2D ψ comes out pure to 1e-6.
  **THE RAMP IS NOT OPTIONAL AND NOT A NON-SYMMETRIC-BOX SPECIAL CASE** — k₀ is
  `float(grid.v[a][0])`, the first LATTICE value (not `lo`, which can differ by
  an ulp on a regridded axis), and it is never 0 for a box straddling the
  origin. Dropped, the relative error is **1.0** on a symmetric [−7, 7] box
  while the norm stays exactly 1.000000, so only a cell-by-cell comparison
  against the analytic form can see it (`test_the_momentum_ramp_is_not_optional`).
  **THE MOMENTUM BOX CANNOT PRODUCE A NORM DEFICIT, AND NOTHING ELSE SEES IT
  EITHER.** The transform is exactly N_k-periodic in the momentum index, so
  content outside the box ALIASES back in rather than being lost — the identity
  Σ_k W·d_k = |ψ(q)|² holds to 3.3e-16 on any box. Measured on a packet at
  p₀ = **+2** with the momentum box **[−10, −2]**, which excludes its mean
  momentum entirely: norm 1.0000000000, purity 1.000000, max|W| 0.3172 against a
  correct 0.3171, and the p edge band 6.5e-06 — barely over its own 1e-6 trigger
  and under the float32 one. Every scalar diagnostic reads perfect on a
  completely wrong state. So `_psi_lattice` also computes the **momentum mass**
  from a direct-quadrature φ on the box lattice, which reads 7.2e-07 against
  0.9999999988 for a good box. It is the only detector there is; do not remove
  it because "the edge band already covers that".
  **ψ IS NORMALISED OVER THE EXTENDED SPATIAL BOX**, so ∫W dμ is the in-box mass
  fraction and the norm deficit keeps the meaning it has for the Gaussian kinds
  (measured 0.645236190989 against an in-box fraction of 0.645236190989).
  Normalising over the visible box makes it a structural zero. The extension is
  **capped at `NORM_PAD_MAX` box widths per side** because the Bopp half-width
  is ħπ/(2dk) and GROWS as the momentum axis is refined — at 1D 4096² over a
  [−8, 8] momentum box it is 402, fifty times the spatial box, and the φ
  quadrature's kernel asked for **13.7 GB** before the cap.
  **BOTH BUILDS ARE BLOCKED AGAINST FIXED BYTE BUDGETS**, and that is what lets
  `PREVIEW_BYTES_PER_CELL` stay one number per ndim: an arbitrary expression's
  unblocked transient grows with its own tree width (80 → 96 → 112 B/cell for
  1, 3 and 9 terms at 1024², with MAX_EXPR_LEN = 500), so there is no honest
  constant for it. Blocked, the peak is the float64 output plus the budget and
  is largest at the SMALLEST grid — measured on the 3090: ψ 36.4 B/cell at
  1D 1024² falling to 10.3 at 4096², 20.1 at 2D 32⁴ falling to 9.0 at 64⁴, all
  under the cat build's 88/56.
  **A ψ IS NEVER ACCUSED OF NOT BEING A QUANTUM STATE.** A wavefunction always
  defines a valid pure state, so γ > 1 there can only mean the grid is aliasing
  it — same detector, different sentence, the move `boundaryTitle` already makes
  for float32. Its tolerance is looser too (1e-4 against 1e-6): the Gaussian
  figure was tuned against analytically exact W and a transform-built pure state
  overshoots it on a coarse lattice.
  **`initial.ICError` is its own class** so `routers/preview.py` can tell a
  client error from a cupy OOM. Most of these are only knowable once W exists (an
  expression's integral is not knowable before it is summed), so they are raised
  from inside the build — where the CPU fallback would otherwise retry them, fail
  identically and surface as a 500.
  The IC is compiled ONCE in the router and threaded through
  (`session.compiled_ic`), like `compiled_potential`: `worker._run` runs per
  variant, so compiling there is four threads parsing one string through sympy's
  global caches — and a bad expression must be a 422 at the door rather than four
  dead workers.



## The potentials bullet, as it stood before the split

- **Potentials** (`core/potential.py`, on `core/expr.py`): tokenize-screen
  (THE security boundary, now shared with the two expression IC kinds — one
  screen, three kinds of user expression; `potential.py` keeps the per-family
  VALIDITY model, which is a potential's alone). The parser's vocabulary is the
  same for every kind and only the free symbols and complex-ness differ, which
  is the `y`-at-every-ndim rule applied consistently: `I*x` in a U is refused as
  "U(x) must be a real expression", not "name 'I' is not allowed", because the
  user knows what I means and the real check was never the tokenizer anyway
  (`sqrt(-1)` evaluates to an explicit I without naming it). `hermite` is
  whitelisted and its ORDER is capped — it is a Function in the unevaluated
  parse, so the power screen cannot see it, and the evaluated parse materialises
  the polynomial (0.293 s / 501 terms at n = 1000). Suppressing that needs BOTH
  `parse_expr(evaluate=False)` and the `sp.evaluate(False)` CONTEXT: the keyword
  alone stops arithmetic evaluation but not function application, so the screen
  received the expanded polynomial it exists to refuse. Then: tokenize-screen
  → sympy parse → per-family validity. The Bopp arguments are REAL
  (q_i ∓ ħθ_i/2, complex dtype only): quantum needs U real+finite on the
  extended BOX (per spatial axis, [q1 − πħ/(2dk), q2 + πħ/(2dk)] with the
  CONJUGATE axis's spacing; Abs is quantum-valid); classical needs EVERY
  partial ∂U/∂q_i DiracDelta-free (Heaviside steps are quantum-only). At
  ndim=2 the symbols are (x, y) and `grad_exprs` is the gradient tuple; `y` is
  in the tokenizer's namespace at every ndim on purpose, so a 1D session that
  types `x*y` gets the free-symbol message naming what IS allowed rather than a
  tokenizer refusal that reads like a typo. **The numeric probe lattice uses an
  ODD count per axis and forces an exact 0.0 onto any axis straddling the
  origin**: the poles that matter in 2D sit on the axes and at the origin
  (`1/sqrt(x^2+y^2)`, `1/x`, `log(x)`) and an even lattice steps straight over
  them — and sympy's `singularities` is one-dimensional, so past ndim=1 the
  lattice is most of the guard (the symbolic scan pins the other variables at a
  few sample values, best effort).
  **The preview endpoint's PLOT window and its VALIDITY boxes are two different
  things and must not be conflated.** `POST /api/preview/potential` takes
  `x1/x2` (and `y1/y2`) as what to SAMPLE — the editor zooms them, and zooming
  out past the domain is how the interesting part of U is found — while both
  validity boxes come from `req.grid`: `spatial_ranges()` for the classical
  gradient probe and `spatial_extended()` for the quantum one. That is the same
  pair `routers/sessions.compile_for` uses at create time, which is the point:
  tie the classical probe to the zoom instead and the panel stops predicting the
  API. Measured symptom — `1/x` on a grid of [-6, 6], zoomed to [1, 6]: the
  badge reads `classical ✓`, the Solve gate opens, and `POST /sessions` 422s on
  the potential the editor had just approved. Pinned by
  `test_the_validity_probe_follows_the_GRID_not_the_zoom`.
  **At ndim=2 the editor draws the two axis cuts on TWO charts, not two traces**
  (`PotentialEditor.vue`): uPlot's `AlignedData` has ONE shared abscissa and
  these cuts do not share one — U(x, 0) is indexed by x over the zoom window,
  U(0, y) by y over the grid's own y extent. Overlaid, the y cut was drawn at the
  x sample positions, i.e. rescaled by (x2−x1)/(y2−y1) — invisible on the
  isotropic default box, which is exactly why it looked right. Each chart's title
  also names the coordinate its cut was actually TAKEN at (`nearestZero` picks
  the sample closest to the origin, and a window zoomed away from it has none),
  so `U(x, 0.4)` rather than a false `U(x, 0)`. 1D is untouched: one chart, one
  trace.
