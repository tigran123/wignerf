# 2D milestones M1–M4: what it cost to retire each gate

Split out of `CLAUDE.md` on 2026-08-02, which had grown past the 150k-char
limit at which Claude Code stops loading it. Everything here was in CLAUDE.md
verbatim; nothing has been rewritten. CLAUDE.md keeps the operative facts —
the ones that change what you would write today — and points here for the
measurements behind them.

**Read this before touching**: precision handling or `BYTES_PER_CELL_2D` (M1),
the relativistic kinetic term or a new 2D physics anchor (M2), the auto-expand
regrid guard `core/fit.py` or `worker._apply_regrid` (M3), or the export
figure in `core/render_mpl.py` (M4).

## Three lessons from retiring the four, which no single one holds


- **The verification is the work; the gate removal is a few lines.** The physics
  core was already generic every time. What took the time was measuring what the
  gate stood in for — and M1's measurement contradicted the prediction twice, in
  opposite directions (54% of the memory rather than 58%, but 1.5–2.6× the speed
  rather than ~3.4×). A gate retired by argument ships the wrong number.
- **Look for the accounting the gate was hiding**, and expect the search to
  produce false positives. Each milestone's real work was a constant nobody had
  listed: M1's `BYTES_PER_CELL_2D` (feeds `_fit_error`; left at 208 it would go
  on refusing exactly the grids float32 affords), M4's `RangeStats.scale` (one
  colour scale per variant — right at ndim=1, five of six panels blank at 2,
  silently), M3's `session.max_cells` (stored and reported since the first 2D
  cut, consulted by the planner never). But M3's most promising candidate —
  `SUPPORT_EPS = 1e-8` reading noise on a coarse 2D grid, by the same argument
  `MSG_EXPAND_F32` makes for float32 — was refuted outright when measured: the
  planner scans only TRIPPED axes, so the scenario is unreachable. A plausible
  chain from two documented measurements still got it wrong. Predict, then
  measure the prediction, at the point the code actually runs.
- **A gate can carry a stale REASON.** M4's row said axis subscripts "must NOT
  use mathtext", citing a `describe.py` measurement; re-measured it was backwards

## M1 — float32 in 2D (landed 2026-07-27)

- **FLOAT32 IN 2D (M1, landed 2026-07-27): choose it for MEMORY, not for speed,
  and know that it raises the boundary warning on a contained state at 32⁴.**
  Like M2, the physics core needed no change — the mixed scheme (float64 meshes
  and `qd()` evaluations, complex64 only for the spectral array and the exponent
  phases) has no dimension-aware line in it, `fft_pair` picks its dtype and its
  multi-axis entry point independently, and every `observables` reduction already
  forced `dtype=float64`. So M1 was measurement plus one piece of accounting.
  Four things worth keeping:
  **The mixed rules hold BITWISE at 4 axes.** The rate meshes are byte-identical
  between the two modes for all four variants at ndim=2 exactly as at ndim=1,
  which is the property that had to be checked because the multi-D Bopp shift
  moves every spatial argument of U together. Pinned by
  `test_exponent_construction_stays_double`, now parametrized over ndim.
  **MEMORY: 112 B/cell against float64's 208** — 54%, flat across 32⁴–80⁴, and
  identical for the relativistic variants. That is better than the ~58% predicted
  from 1D, and it survives `exponents()`' cast (which builds the phase in
  complex128 and rounds down, so its *transient* peak is a mesh HIGHER than
  float64's per call while the pool high-water still lands at 112 — measured, not
  reasoned). 3.25 → 1.75 GiB/worker at 64⁴, and 80⁴ becomes reachable at all.
  **SPEED: 1.48× at 64⁴, not the ~3.4× the milestone predicted** — 2.63× at 32⁴,
  2.09× at 48⁴, 1.48× at 64⁴, 1.64× at 80⁴. The 1D control on the same card in
  the same session reproduced 3.76 / 3.40 / 3.29× at 1024²/2048²/4096², so this
  is not the machine: at the SAME 16.8M cell count 1D gets 3.29× and 2D gets
  1.48×. The 2D step transforms two axes of a 4D array at a time (`fft_pair`'s
  `fftn` branch) where 1D transforms one axis of a 2D array, and cuFFT's
  single-precision advantage for that strided layout is far smaller. Do not quote
  the 1D figure for 2D.
  **It puts REAL mass in the edge band, and at 32⁴ that latches the boundary
  warning within ~7 s.** Measured on the shipping 2D default, a fully contained
  state: at 32⁴ the band mass reaches 1.0e-3 of the integral after 12000 steps
  (ratio 10 against the float32 threshold, a 505-record streak, 6 announced state
  changes) against **3.1e-5 for the IDENTICAL state in float64**, with purity down
  2%. 48⁴ stays clear (ratio 0.12) and 64⁴ flickers (1.72, streak 5) — so it is
  the COARSE grid that cannot carry single precision, not 2D as such. **No
  threshold was moved for this**, deliberately: the mass is genuinely there
  (float64 measures the truth and float32 has 30× more of it), it GROWS with step
  count so any fixed threshold is outrun by a long enough run, and one high enough
  to be safe would sit past the point where real wrap does damage. What changed is
  the WORDS — `SimulatorView.boundaryTitle` names single precision as a cause in a
  float32 2D session, because the standing remedy ("restart with a larger domain")
  is the one thing that cannot help here. Pinned by
  `test_float32_in_2d_moves_real_mass_to_the_edge_band`.
  **`TOL_MIN_F32` did NOT move, and that is measured too.** The `adjust_step`
  residual floor SATURATES rather than growing with cells — 1D 9.2e-7 at 256² then
  1.2e-6 flat from 1024² to 4096², 2D 2.0-2.5e-6 flat from 32⁴ to 64⁴ — so the old
  "larger at larger grids" reading was only the first step of that curve. Worst
  case 2.5e-6 against 1e-5 is 4× margin everywhere. 2D sits ~2× above 1D because a
  4-axis Strang step does more arithmetic per element.

## M2 — relativistic 2D, `qr`/`cr` (landed 2026-07-27)

- **RELATIVISTIC 2D (M2, landed 2026-07-27): what it cost to retire the gate,
  and the one anchor that does NOT transfer.** The physics was already generic —
  `_kinetic()` built T = c√(Σkᵢ² + m²c²) and its gradient for any ndim, and the
  streaming/observables/frame paths never cared — so M2 was a gate removal plus
  the verification the gate named. Four things worth keeping:
  **SEPARABILITY IS UNAVAILABLE FOR `qr`/`cr`, and a natural test would fail
  against correct code.** `test_separable_run_equals_two_1d_runs` rests on the
  whole exponent factorising, which needs T separable as well as U:
  (px²+py²)/2m is a sum, c√(px²+py²+m²c²) is not. A 2D relativistic run is
  genuinely NOT the outer product of two 1D relativistic runs. Since separability
  is what validates the 4D pipeline against trusted 1D code, the replacement is
  `test_relativistic_matches_an_independent_schroedinger_run` — the same TDSE
  reference as the Bopp anchor, with the square-root (Salpeter) T applied exactly
  in the Fourier basis. Measured at c = 10: 1.345e-4 relative at dt = 0.02 and
  2.936e-5 at 0.01, **ratio 4.58** (O(dt²)); at c = 5 the ratio is 3.46 and at
  c = 3 it is 2.65, because the residual has reached the same dt-independent
  ~1.5e-5 GRID floor the non-relativistic anchor hits below dt = 0.005. Do not
  lower c to make it "more relativistic". The test can also FAIL: relativistic
  and non-relativistic densities differ by 1.46e-2 of the peak at c = 10, ~500×
  the residual, so a no-op `relativistic` flag cannot pass it.
  **The mc² cancellation does NOT worsen in 4D.** Quantum relativistic dT is a
  difference of m²c² ≈ 1.878e4-magnitude terms; against the stable form
  `(A−B)/(√A+√B)` the measured ABSOLUTE error is 3.61e-12 (1D N=32), 3.60e-12
  (1D N=64), 4.10e-12 (2D N=32), 4.21e-12 (2D N=64) — flat, because it is
  m²c²·eps ≈ 1.9e-12, a couple of ulps, and that does not care how many momentum
  components enter the sum. Relative error actually IMPROVES in 2D (1.2e-13 →
  3.6e-14) since |dT| grows. M1 did NOT have to re-measure this after all, and
  the reason is the stronger result: float32 never reaches `_rate_mesh` at all —
  construction stays double at every ndim, verified bitwise — so there is no
  single-precision version of this number to measure.
  **⟨Lz⟩ transfers, quantum ≡ classical does not.** T depends on the momenta only
  through |k|, so rotational symmetry holds and `test_angular_momentum` extends
  to qr/cr unchanged — it is the one existing 2D anchor that does. But qr ≠ cr
  for relativistic even under a quadratic U: the Moyal corrections vanish because
  U is quadratic, and T is not quadratic in k, so the kinetic Bopp difference and
  the gradient genuinely differ. Never assert quantum ≡ classical for qr/cr.
  **The shear diagnostic works, at c = 10 not c = 137.** Shear goes as 1/c⁴ and
  at c = 137.036 needs ~1200 steps to clear the noise, which at 57 ms/step over
  32⁴ is not a test. Measured at dt = 0.05, T = 10: non-relativistic 6.01e-6,
  relativistic **6.215e-3 (1034×)**, the same at dt/2 **6.157e-3 (unchanged,
  0.95%) while E's splitting oscillation drops 1.25e-3 → 3.10e-4 (4.03×)**, and
  c = 20 gives 3.816e-4 — **16.3× ≈ 2⁴, the 1/c⁴ law confirmed independently**.
  Purity flat at ~1e-10 throughout, so the shear is symplectic, exactly as in 1D.
  **Relativistic is FREE in memory and in time**, so `BYTES_PER_CELL_2D` did not
  move: measured on the 3090 at float64, `qn` and `qr` arenas are identical to
  the byte (176.0 B/cell at 32⁴, 48⁴ and 64⁴ — that harness omits the frame
  build the 208 figure includes) and throughput is within noise (34.8 vs 35.6
  steps/s at 64⁴, against the 35.1 in the config table). A √ over meshes that
  already exist costs nothing; the FFTs are still the whole cost. Massless is the
  same again — 176.0 B/cell.

## M3 — auto-expand in 2D (landed 2026-08-01)

- **AUTO-EXPAND IN 2D (M3, landed 2026-08-01): the orchestration was free, the
  guard was the work, and TWO of the three predictions were wrong.**
  `_schedule_regrid` already looped `range(gs.n_axes)` from the first 2D cut, and
  `embed_window`/`GridState`/`plan_axis`/the v4 wire header/`regridFlash`/
  `adoptLive` were all tuple-shaped. Seven things worth keeping:
  **THE GUARD IS THE CREATE-TIME CHECK ASKED AGAIN MID-RUN, with the arithmetic
  shared so the two cannot drift** (`core/fit.py`; `routers/sessions._fit_error`
  keeps its three messages and the session writes its own, because at create time
  the advice is "drop a variant, change device" and mid-run none of that is
  available). Per device, for `n` of our workers on it:
  `n·per_new·REGRID_PEAK ≤ (F + n·per_old)·FIT_MARGIN`. **The `+ n·per_old` term
  is load-bearing, not a refinement**: at 64⁴ float64 on the real pair, a
  single-worker doubling on the 2080 Ti needs 6.50 GiB against 7.15 free — which
  `F` alone refuses and `F + per_old` correctly allows, while the 2-worker case
  (13.0 GiB) stays correctly refused either way. It is also why release-before-
  allocate had to happen: the term is only honest if the old memory comes back.
  **IT IS ASKED OF A DOUBLING AND OF NOTHING ELSE, and that is the inequality
  talking, not an optimization.** A whole-cell window SHIFT keeps `per_new ==
  per_old`, where the same expression reduces to `F ≥ (REGRID_PEAK/FIT_MARGIN −
  1)·n·per_old` = **0.222·n·per_old** — 1.44 GiB free demanded at 64⁴ float64
  with 2 workers on a card, to slide a window that allocates nothing (and the
  `+ n·per_old` term is wrong there too: `Propagator.set_grid` takes its `else`
  branch on an unchanged shape and releases nothing). So `fit.regrid_shortfall`
  returns `[]` for any non-growing window BEFORE it reads the driver, and
  `_schedule_regrid` takes a reading only when the composed plan grows. Pinned by
  `test_a_pure_move_is_never_refused_for_memory` and, end to end,
  `test_a_window_slides_even_when_the_device_is_full`. The reading is taken
  ONCE per attempt and passed into the greedy walk-back below: a card re-read
  between candidates could accept a plan that neither reading on its own allows.
  **AND A MOVE COSTS THE CARD NOTHING, measured 2026-08-01 with
  `scripts/bench.py --ndim 2 --regrid move`** — the earlier reasoned figure here
  (~48 B/cell) was wrong twice over, understating the transient (it omitted
  `qd()`'s ~80 B/cell Bopp construction inside `rebuild()`, the largest item) and
  overstating what the card sees (nothing). On a 3090 at 32⁴/48⁴/64⁴:
  **float64 peak/steady 1.000, driver +0 MiB; float32 1.143, +16/82/256 MiB** —
  and float32 is unreachable, auto-expand being float64-only. In float64 every
  block a move's peak needs is a size class the 208 B/cell arena already holds
  from `adjust_step`. **Hoisting the mesh drop out of `set_grid`'s `reshaped`
  branch was tried and measured a NO-OP in both arms** (it swaps the mesh-pair
  overlap for `rebuild()`'s own float64 intermediate, both 16 B/cell), so it was
  not shipped — and `_release_pool()` on a move would be actively worse, clearing
  a cuFFT plan still VALID at an unchanged shape whose work area is real VRAM. A
  guard here would also be incoherent: the same rebuild runs at ~168 B/cell on any
  live parameter change (`set_physics`, both exponent slots still resident),
  unguarded, at either dimensionality, and a 1D doubling is unguarded outright.
  `Propagator.set_grid` had NO direct test until this was settled — only
  end-to-end coverage, where neither branch's ordering is observable;
  `test_a_move_rebuilds_without_releasing_its_plans_or_the_pool` now pins both
  branches and the bitwise-equality contract they rest on.
  **RELEASE BEFORE ALLOCATE, measured with `scripts/bench.py --ndim 2 --regrid`.**
  `worker._apply_regrid` used to hold the old state, the old exponent slots
  (64 B/cell, the largest single item) and the old propagator meshes while
  `rebuild()` allocated the new ones — `_exp_clear()` ran *after* `set_grid`.
  Reordered (slots first; the state dropped through a one-element box so `_run`'s
  own local does not pin it; `Propagator.set_grid` dropping its meshes, its FFT
  plans and this thread's cuFFT plan cache and freeing the pool before
  rebuilding): **peak/new 1.269 → 1.038 (float64) and 1.071 (float32), recovered
  0.46 → 0.92 (float64) and 0.86 (float32)**, flat across 32⁴/48⁴/64⁴. Quote both
  precisions or the three places carrying this number drift apart, which they
  briefly did. `REGRID_PEAK` = 1.10 is the rounded ceiling. ALLOCATION ORDER ONLY
  — every array rebuilt is a pure function of (grid, physics), so ndim=1 is
  bitwise unaffected and the 1D suite passes untouched, which is the claim.
  **WHAT THE GUARD CANNOT SEE IS ACCEPTED, and `fit.py`'s docstring names all of
  it** rather than leaving it to be rediscovered as a bug. `device_free_bytes`
  asks the driver, so anything ALLOCATED is counted; what is not is (a) another
  process, or our own IC preview's private pool, taking the card between the
  reading and the allocation it licensed, (b) **another SESSION in this process
  whose growing plan is committed but has not landed** — `k_star` is frontier + 2
  records, tens of ms at 32⁴ and up to ~1 s at 64⁴, inside which two sessions on
  one card can each be told the same bytes are free — and (c) the bigger unguarded
  transients elsewhere (`set_physics`; a 1D doubling at all). `FIT_MARGIN` does
  not cover (a) or (b) and must not be read as if it did: a tenth of free memory
  against a sibling's whole doubling is an order of magnitude short. What makes it
  acceptable is the FAILURE MODE — losing the race is a cupy OOM inside
  `worker._apply_regrid`, fatal by design, so `run()` posts the traceback and
  pauses that one session loudly, which every candidate mechanism would trade for
  a quiet refusal. A registry walk over `SESSIONS` was designed and rejected on
  exactly that: it could only NARROW (b), and a claim derived from a pending plan
  outlives its plan on a PAUSED session — `apply_params` commits one while paused,
  pinned by `test_toggle_while_paused_schedules_immediately` — which would refuse
  every OTHER session's expansion silently and for good, the shape of bug
  `_no_room_posted` already had once. So both the refusal AND the accept path log
  the reading and the ask, which is what makes an OOM at `k_star` diagnosable
  after the fact; the loudness itself is pinned by
  `test_a_failed_regrid_stops_the_run_instead_of_going_quiet`, because the day
  that becomes silent the limitation stops being acceptable.
  **PREDICTION THAT FAILED #1: `SUPPORT_EPS` was NOT a hidden constant.** The plan
  argued that since `MSG_EXPAND_F32` refuses float32 for making the 1e-8 support
  scan "report the WHOLE axis", and 2D N=32 has a measured float64 noise floor of
  5.35e-05 (5000× that), the planner would size 2D windows from noise. It does
  not, for the reasons now in `boundary.py` beside `SUPPORT_EPS`: the planner
  scans only TRIPPED axes, so the contained marginal carrying that noise is never
  scanned; at the trigger point a `(0, N)` support is CORRECT because ~1e-7 of the
  mass has genuinely re-entered at the far edge on the torus (measured with noise
  identically **0.00e+00 in 1D**, where it is shipping behaviour); and at 32⁴ the
  noise and the real wrapped mass are the same order, so no gate can separate
  them. Across 1D N=64/256 and 2D N=32/48/64, mid-box, drifting and edge-placed, a
  gate changed **no plan kind** — only one 32⁴ window's centring, by 4 cells, in a
  direction that cannot be shown to be more correct. **Measure at the TRIGGER
  point**: a contained state reproduces a scenario the planner never reaches,
  which is exactly the error the original synthetic check made.
  **PREDICTION THAT FAILED #2: greedy degradation helps less than it looks.**
  Giving up a doubling and re-planning that axis as a pure move IS
  `plan_axis(offset, n, lo, hi, cap=n)` — neat, and already pinned by
  `test_support_cells_and_plan_axis`. But because the support legitimately spans
  the axis at the trigger point (above), the fallback is usually `"capped"`
  rather than a useful move (`plan_axis(0,32,0,31,cap=32) → capped`). It is kept
  because it is right for the case it was chosen for — several axes trip and only
  some doublings fit — and **the warning has to say WHICH of the two happened**:
  a denied doubling can still leave a shift to commit, so `no_room` carries
  `denied` (the axes given up) and `applied` (what went out anyway, `{}` when
  nothing) and `SimulatorView.boundaryText` picks one of two sentences from it.
  One sentence for both read as a self-contradiction on screen — "the domain
  cannot be expanded" beside the ⤢ flash saying it moved to [-4, 12].
  **`no_room` IS ITS OWN BOUNDARY ACTION, not a flavour of `capped`.** `capped`
  means the domain reached the per-axis `WIGNERF_MAX_GRID` ceiling; `no_room`
  means the guard made us give a doubling up. Opposite advice (a smaller grid vs a
  bigger card or fewer variants), so opposite messages and tooltips, rendered
  separately. **An ACTION is what happened to the PLAN, and WHICH ceiling ran out
  is a FIELD** — `limit` ∈ {`device`, `cells`} — because `_afford_regrid` has two
  guards whose remedies have nothing in common: for the `WIGNERF_MAX_CELLS_2D`
  rail, freeing the card does nothing and neither does float32 (a cell count is
  precision-independent), so `boundaryTitle` says exactly that instead of the
  device advice. Returned as a bare sentence under one action, a rail denial
  arrived telling the user to free a card with 100+ GiB spare; a third action
  would have duplicated the whole `denied`/`applied` payload contract and the
  latch to vary two clauses of prose. Pinned by
  `test_a_cell_rail_denial_does_not_blame_the_card`. **It is a MESSAGE latch, like
  `_capped_posted`, and NOT a scheduling gate like `_invalid_posted`** — a
  distinction worth the sentence, because it gated `report_edge`'s `want` for one
  revision on the argument that it kept a driver query off the per-record path.
  That query is a `cudaMemGetInfo` and (above) is not even taken unless the plan
  grows, where `_invalid_posted` stands in front of a ~ms sympy probe under the
  edge lock; and the gate's cost was total, because the tripped axes that reset it
  never clear on their own — a state at the edge is what tripped them — so one
  refusal switched auto-expand off for the rest of the run, including the pure
  shifts that were never the problem. Pinned by
  `test_a_denial_does_not_disable_auto_expand_for_the_rest_of_the_run`, which
  frees the card after the refusal and requires the next attempt to expand. A
  plan that commits with nothing denied clears the latch, so a LATER denial is
  announced rather than swallowed. The `WIGNERF_MAX_CELLS_2D` rail, **stored on
  the session and consulted by nothing until M3**, is now checked on the new
  window too: it is the only guard on a host whose free memory cannot be read.
  **The mp4 metadata block absorbed it.** A regrid adds the "axes follow each
  record (auto-expand); widest: …" line, which at four axes wraps to two — the
  obvious candidate to blow the 11-line budget. Measured both ways: with a short
  IC the left column goes 9 → 11 and lands exactly on the 11 that fit at the full
  8 pt; with a long 2D cat IC the right column is 12 either way and the union line
  changes nothing. Neither approaches the 5 pt elision floor.

## M4 — mp4 export of 2D runs (landed 2026-07-28)

- **2D mp4 EXPORT (M4, landed 2026-07-28): the frame is a SELECTION, real
  subscripts turned out to be free, and the blit decides where static art may
  live.** Four things worth keeping:
  **MATHTEXT IS FREE HERE, and the note saying otherwise was backwards.**
  `describe.py` used to state that mathtext "was measured to slow the export's
  plots down enormously" (hence exported frames spelling the momentum axes
  `px`/`py` while the SPA showed real subscripts). Re-measured on the shipping
  figure: **plain 2.21 ms per text artist, mathtext 4.05 ms (1.83×), usetex
  26.6 ms (12×)** — but the figure BLITS, so every title and axis label is a
  STATIC artist baked into the background once. In situ that is **build
  ×0.91–1.34 and steady-state ×0.98–1.05 (noise) in every configuration from
  1D/4-panel to 2D/24-panel at FHD and 4K**. So `axes.sub_math` typesets the
  two-letter axis names as `$p_x$` and everything else — ∬, ⨌, γ, ℏ, ρ, φ, ⟨⟩ —
  stays the same Unicode the screen uses, so the two cannot drift and 1D
  (single-letter names) is untouched at the byte level. Every title function
  takes a `math=` flag, the counterpart of its frontend mirror's `html=`.
  **usetex is NOT used**, and not only for the 12×: it needs a LaTeX install
  (the VPS has none), it cannot render our Unicode without a THIRD spelling of
  every string, and `text.usetex` is global, so the metadata block's
  user-supplied U(x) would go through LaTeX too.
  **AND IT HAS TO BE THE WHOLE FRAME, not the plot titles only** — half the job
  reads worse than none, because the same axis then appears twice on one screen
  in two spellings. So the header's per-record geometry readout
  (`geom_line(math=True)`), the metadata block (`_extents`, the panel list,
  `diagnostic_label`) and `describe.ic_expression` all take the same flag.
  Three things that fell out of it:
  **The metadata block is WRAPPED PLAIN and typeset afterwards** (`_emit`).
  `$p_x$` is five characters that draw as two glyphs, so letting the typeset
  form into `textwrap` makes every column width a guess; which characters land
  on which line has to be decided by the plain text. A logical line that fit
  unbroken is swapped wholesale for its typeset twin; one that had to be broken
  keeps its fragments with only the per-token axis-name substitution (valid on
  any fragment, unlike U's single `$…$` span).
  **And it must not break a line MID-FACT.** `px ∈ [-7, 7]` is one fact and was
  being split between the axis name and the ∈; so were
  `(6 planes × 1 variant = 6)` and the units figure. Each such group is joined
  with `_NB` (`\x00`) and `_emit` restores real spaces after wrapping — it has
  to be a character `textwrap` cannot see as whitespace at all, and a Unicode
  NBSP is `\s` so it does not work. A test asserting the groups survive must
  check them **per line**: joining a column with " " is exactly what
  reconstitutes a group that was split, so a whole-column search cannot see the
  bug. Pinned by `test_the_metadata_block_never_breaks_a_line_mid_fact`.
  **The header readout and the block deliberately DISAGREE.** The header's
  geometry follows the PAINTED record (`geom_line` off the record handed to
  `update`), as the SPA follows the painted frame; the block's is labelled "at
  record k0" and stays there, because it is part of "what this video is" and a
  per-frame rewrite would make it a moving target — the union is quoted on its
  own line instead. They can only diverge across an auto-expand regrid; pinned at
  **1D** by `test_the_header_follows_the_painted_record_and_the_block_does_not`
  and, since M3 made a 2D regrid reachable, at **2D** by
  `test_the_header_and_block_diverge_in_2d_too` — which is also where the
  geometry line's equal-extent GROUPING earns its keep: before the switch all
  four axes share one group, and doubling x alone splits it into
  `x ∈ [-12, 12]  y,pₓ,p_y ∈ [-6, 6]`.
  **U(x,y) is typeset by a LEXICAL rewrite of the user's own string**
  (`describe.potential_math`), NOT by round-tripping through `sympy.latex`. That
  was measured too: sympy renders every real potential without a mathtext parse
  error, and is still wrong here twice over. It CANONICALISES — `x^2/2 + 0.3*x^4`
  comes back as `0.3x^4 + x^2/2`, and `10*(1-exp(-0.5*(x-1)))^2` as
  `10(1 − 1.64872127070013·e^{−0.5x})^2`, sympy having evaluated `exp(0.5)` at
  parse time — the same function and a different picture, where this block's job
  is "how to reproduce this run": the source string is what you paste back into
  the U(x) box. And it emits `\frac`, which is TALL, where the block advances by
  a fixed `META_LINESPACING`. The rewrite (brace multi-char exponents,
  `\mathrm{}` the tokenizer's own function names, `*` → thin space) preserves the
  source token for token and introduces nothing taller than a superscript. The
  one 1D-visible change in all of this is `U(x) = $x^{2}/2$`.
  **A typeset line built from USER input cannot be the only line of defence**:
  a mathtext parse error raises at DRAW time and would take the export down, so
  every candidate goes through `render_mpl.mathtext_ok` (matplotlib's own
  parser, per `$…$` span, cached) and falls back to plain per line.
  `describe.config_json` and the setup document stay plain either way — they are
  machine-readable, and `param_lines`/`ic_expression` only typeset when asked.
  **THE BLIT DECIDES WHERE STATIC ART MAY LIVE.** Past `CBAR_MAX_CELLS` = 8
  panels a colorbar is a ~9 px strip with 6.5 pt ticks, so each panel's fixed
  scale is stated instead — and the first version put it in the corner of the
  heatmap, where it rendered as NOTHING. `update()` restores the static
  background and then `draw_artist`s the images on top, so anything static drawn
  INSIDE the axes box is painted over every frame; that is exactly why the panel
  grid lines are in `_dynamic` and ordered after the images. The caption went
  into the panel's `loc="right"` TITLE instead (outside the axes box, hence still
  static and free); making it dynamic would have cost 24 text renders a frame for
  something that never changes. Pinned by
  `test_a_dense_grid_states_its_scales_in_the_titles_not_over_the_heatmap`,
  which also asserts no panel has an in-axes text at all.
  **COST: 2D is CHEAPER per frame than 1D, not dearer.** Measured on a real
  32⁴ 4-variant run at FHD with the parallel pool: **9.8 fps for 4 panels,
  8.7 for the 6-panel phase portrait, 6.4 for the full 24-panel matrix, 10.1
  for panels-only** — against 1D's ~9–10 fps at 1080p. A 2D plane is at most
  128×128 (`WIGNERF_MAX_GRID_2D`) where a 1D W is up to 4096², so the `imshow`
  upsampling that dominates a 1D frame barely registers, and 24 panels cost only
  ~35% over 4 because the total image AREA is what matters. The pool's per-task
  pickle payload shrinks too — ~50 KiB per variant per record at 64⁴ against
  32 MiB at 1D/4096² — so `POOL_MIN_FRAMES` and the w+2 window needed no change.
  **The metadata block did NOT overflow, measured rather than assumed.** 2D
  makes every line longer (four axes in the grid line and in the IC) and adds
  two of its own ("panels: …", "plots omitted: …"), so the 11-line budget was the
  obvious thing to blow. Worst realistic case — 4-variant 2D cat, 4 live changes,
  float32, all six planes — measures **10 lines against the 11 that fit at 8 pt**:
  no shrink, no elision, one line of margin. `_meta_fontsize` / `_meta_fit` still
  degrade gracefully past that and are pinned at 2D.
