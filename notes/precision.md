# float64 / float32 — the measurements

Split out of `CLAUDE.md` on 2026-08-05, when the file passed the 150k-char limit
at which Claude Code stops loading it (the same split `notes/2d-milestones.md`
came from on 2026-08-02). What stayed there is the operative rule; what is here
is how each number in it was arrived at. Read this before changing
`WIGNERF_PRECISION`, `TOL_MIN_F32`, `applyPrecisionInvariants`, or anything in
`Propagator.exponents` / `_rate_mesh`.


## The float64/float32 gotcha, as it stood before the split

- **The solver is float64 BY DEFAULT and float32 only when explicitly asked
  — and the difference was measured, not assumed.** `SessionCreate.precision`
  (`float64` | `float32`, host default `WIGNERF_PRECISION`) is restart-only and
  picks the SPECTRAL working dtype. float32 must never be the default and never
  the setting a physics claim is made from; the UI badges it permanently and
  every exported mp4 says so on its own metadata line.
  - **What it costs.** complex64 stepping destroys the diagnostics this project
    navigates by: over 2000 steps at 256², Δpurity −2.4e-4 and ΔE +9.4e-4, both
    SECULAR — exactly the boundary-wrap signature in the gotcha below, from a
    perfectly contained state — with ΔX·ΔP noise of 1.3e-3, 150× the ~7e-6
    relativistic shear that `test_relativistic_uncertainty_shear` pins. (float64
    for comparison: +6.7e-13, bounded +4.2e-5, +5.1e-8.)
  - **What it buys, measured 2026-07-25 with `scripts/bench.py --precision both`
    on the real propagator, RTX 3090: 3.84× at 1024², 3.39× at 2048², 3.29× at
    4096².** NOT the "~5×" this file used to quote, and there is no "4.8×" —
    reproduce it from the repo rather than citing a session log. The 2080 Ti
    lands in the same 3.3-3.7× band. **Those are 1D figures and they do NOT
    carry to 2D — see the float32-in-2D gotcha, where the same card gives
    1.5-2.6×.** **On CPU it buys nothing**: pyFFTW through
    the `builders` API `fft_pair` actually uses measures 6.01 ms (c128) vs
    5.80 ms (c64) at 1024². This is a CUDA feature.
  - **It is a MIXED scheme, and that is not a compromise — it is required and
    it is free.** Only the spectral working array and the exponent PHASES are
    single; the grid meshes, both `qd()` evaluations, `dU_im`/`dT_im` and `H`
    stay float64. Required, because relativistic `dT` built in float32 has max
    abs error 455 against max |dT| = 228 (200%: mc² cancels inside a difference
    of ~1.9e4-magnitude terms) — and because keeping construction double is what
    lets `_rate_mesh`'s 1e-13 gate and the frozen-lattice regrid arithmetic stay
    exact with no dtype-scaled tolerances anywhere. Free, because the FFTs are
    the cost: mixed measures 3.72/3.62/3.22× against 3.80/3.69/3.27× for float32
    everywhere. `test_precision.py` asserts the rate meshes are BITWISE
    identical between the two modes, relativistic variants included.
  - **Two failure modes are invisible in results, so they are pinned by DTYPE
    assertions.** A complex64 array handed to a complex128 pyFFTW plan is
    silently upcast by `auto_align_input` (correct answer, complex128 speed);
    and `B *= expT` with B complex64 and expT complex128 is legal in both numpy
    and cupy (correct answer, via a full complex128 temporary). No physics
    assertion can catch either. Hence `fft_pair` takes an explicit dtype and
    `exponents()` casts.
  - **Memory drops to ~54%, not 50%** — `dU_im`/`dT_im` stay float64 and are
    irreducible. Measured per-worker arena on the 3090 with `bench.py
    --footprint` (float64 → float32, post-M7): 768 → 420 MiB at 2048²,
    3.00 → 1.63 GiB at 4096², 12.0 → 6.5 GiB at 8192², i.e. **192 vs 104
    B/cell** — 224 vs 120 before M7 dropped the second exponent slot. Note the
    frame history is NOT affected: it is already uint16 via `core/quantize.py`,
    so `WIGNERF_HISTORY_MB` buys the same record count either way.
  - **float32 REFUSES auto-expand, and `tol` below 1e-5** (`protocol.py`
    `MSG_EXPAND_F32` / `MSG_TOL_F32`, enforced at create AND on the live
    ParamChange path, because both fields are reachable live). Auto-expand,
    because single-precision noise passes its own detector: measured at 256²
    with a coherent state parked at the ORIGIN (true band mass ~1e-15), edge
    mass climbs 1.8e-15 → 6.1e-7 (step 200) → 1.6e-6 (step 600, TRIGGERED),
    while the 1e-8 support scan reads the WHOLE axis by step 200 against an
    exact [43, 214) in float64 — so the planner would size a new domain from
    noise. Detection still WARNS, on a raised threshold
    (`boundary.EDGE_THRESHOLD_BY_PRECISION`, 1e-4 for float32; at that band
    mass the mass actually at the seam is still ~1e-6, so it remains an early
    warning). `tol`, because `adjust_step`'s full-step-vs-two-half-steps
    residual has a measured float32 floor of ~7.4e-7 (flat in dt, and larger at
    larger grids) against 1.6e-15 in float64 — below that the controller shrinks
    dt through all 15 tries every 20 steps and never converges.
    **Both refusals are also enforced in the FORM, not just answered with a
    422**: `lib/config.applyPrecisionInvariants` clears `auto_expand` and raises
    `tol` to `TOL_MIN_F32` (the frontend mirror of `protocol.TOL_MIN_F32` — move
    both together), and the Setup panel disables the checkbox and lowers the tol
    input's `min`. The config-level invariant is the load-bearing half: the panel
    can be unmounted, and `probeHost`/`mergeConfig` reach the same combinations
    from outside it. **How the gates are EXPLAINED is a settled three-part
    pattern, and a standing paragraph is not part of it** — two permanent notes
    beside controls you are not allowed to change were crowding the actual
    controls out of a narrow column. Instead: a compact permanent marker in the
    label that costs no line ("auto-expand (f64)", "tol ≥1e-5"; there was a
    "(1D)" marker beside auto-expand until M3 retired the 2D gate on
    2026-08-01); the full reason in the control's `title`;
    **`:disabled` on the control itself** wherever the value is not merely
    discouraged but overridden — the precision select was left enabled at ndim=2
    and `applyNdimInvariants` put its value back in `payload()`, so picking
    float32 lit an amber "restart to apply" against a value no restart could
    send, and on a fresh session made `syncFreshSessionToForm` build TWO of them
    (its loop re-reads the form after `restart()`, which had moved it back); and
    the reason ONCE in amber
    (`f32Applied`, recorded at the moment of the switch so it names only what
    actually changed — a form already at tol = 0.01 had nothing raised) while
    `runDiffers('precision')` holds, so "Restart session" clears it and the
    header badge carries the one permanent fact from then on. The amber note is
    not garnish: it is the only path on a touch device, which has no hover.
    But it renders ONLY when `f32Applied` is non-empty, and it no longer opens
    with "single precision mode" — that phrase said nothing the precision
    select, its `title` and the header's float32 badge do not already say, and
    it appeared even when the switch had changed nothing else, which is the
    common case (a form already at tol ≥ 1e-5 with auto-expand off). Measured
    on the real panel: a float64 → float32 switch after a 399-record run used
    to raise THREE amber paragraphs and now raises one, the COMPUTE line naming
    what is running.
    Clearing `auto_expand` in the form is NOT enough on its own, because it
    applies LIVE — `SimulatorView` watches `cfg.precision` and sends
    `auto_expand: false` to a running session, since the status→form watcher
    cannot (`status.auto_expand` does not CHANGE, so it never fires) and the
    checkbox is by then disabled, which left a session quietly expanding behind
    an unchecked, unreachable box. **`cfg.grid.ndim` was in that key too until
    M3 (2026-08-01)**, and it mattered more there than precision does: 2D
    refused auto-expand, `applyNdimInvariants` cleared the form, and `dims` is
    restart-only — so the old 1D session could go on regridding for as long as
    the switch waited for its restart, far longer than the float32 case ever
    lasts. That gate is gone, so precision is the whole condition again.
    **And the invariants must be applied SYNCHRONOUSLY at the point of change,
    never from a watcher.** `SetupPanel`'s precision select calls
    `onPrecisionChange` directly, because a watcher is too late: a child's setup
    runs during the parent's render, so `SimulatorView`'s own `cfg.precision`
    watcher holds a lower id and its pre-flush job runs FIRST — and on a fresh
    session it restarts (`syncFreshSessionToForm`) inside that same flush,
    serializing a config the panel had not fixed up yet. Symptom: picking
    float32 with auto-expand on 422'd immediately in BOTH run modes (both start
    fresh, so both take the auto-restart path). `payload()` therefore calls
    `applyPrecisionInvariants` too — the form must be self-consistent before it
    is serialized, and that is the last place it can be guaranteed regardless of
    which watcher ran first.
  - **Do NOT "optimize" `exponents()` by casting the ANGLE instead of the
    result.** `exp(1j*θ).astype(complex64)` is safe for any finite θ because the
    modulus is 1; `exp(1j*θ.astype(float32))` is NaN for θ ~ 1e91, which a steep
    U on the extended Bopp range reaches at large grids — and `worker._finite`
    checks the float64 rate meshes, so nothing would see it.
- The exponent generators dU, dT are EXACTLY purely imaginary (max|Re| = 0
  in all four variants), so they are stored as the real rate meshes
  `dU_im`/`dT_im` and `exponents()` rebuilds the phase — half the bytes,
  bitwise-identical results. `Propagator._rate_mesh` REFUSES a generator whose
  real part exceeds 1e-13 relative to its imaginary part, rather than
  truncating it: a real part means |expU| ≠ 1, an evolution that quietly
  gains or loses norm.
