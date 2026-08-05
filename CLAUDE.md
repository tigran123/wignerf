# wignerf — Interactive Wigner Function Simulator

Live client-server simulator of W(x,p,t) in 1D phase space, evolved by the
spectral split-operator method of Cabrera, Bondar, Jacobs, Rabitz (2015)
(arXiv:1212.3406; PDF at `docs/Efficient-Method-2015.pdf`). The propagator is
a direct port of the math in a validated batch implementation of that method.

**This repo is self-contained** — split out of `quantum-infodynamics` on
2026-07-22 (`git filter-repo --subdirectory-filter wignerf`). Never read or
write anything under `quantum-infodynamics`; everything needed is here.
`docs/` is git-ignored reference material, not part of the program: the paper
above, and `solve4D.py`, an old batch 4D solver kept only as a historical
reference for the eventual multi-D work (read it for ideas if useful; it is
not a spec, and the 2D version should be written fresh). Install/build steps
live in this repo's `README.md`.

Units are Hartree atomic units everywhere (ħ = mₑ = e = 1). `c` is a session
parameter (default 137.035999; `c=1` reproduces the old natural-unit toy
runs). `hbar_eff` (default 1) scales the quantum differential — a
classical-limit dial, not a unit change. SI (fs/Å/eV) appears in display
labels only (`frontend/src/lib/units.ts`).

## Layout and stack

- `backend/` — FastAPI (uv-managed Python 3.12 venv, port **8010**;
  urantia-library owns 8000). `main.py` is pure wiring; routes in
  `routers/`; all physics/infra in `core/` (no FastAPI imports there).
  Two `core/` modules are the dimensionality seams: `axes.py` (what the axes
  ARE — labels, array order, conjugation, the plane set, every display title)
  and `frame.py` (state → the quantized 2D planes + marginals + scalars of one
  record). Read `axes.py` first when touching anything multi-D.
- `frontend/` — Vue 3 + TypeScript + Vite 8 + Tailwind 4, composables (no
  Pinia), hash routing. Built SPA is served statically by FastAPI when
  `frontend/dist` exists; in dev, Vite proxies `/api` (incl. WebSocket).
- `start.sh` — prod launcher: runs uvicorn only (guards that `backend/.venv`
  and `frontend/dist` exist, else errors). Install/build is manual and
  pre-service — see `README.md`.
- `notes/` — tracked long-form material split OUT of this file, which stops
  being loaded around 150k characters. It is not auto-loaded, so **what remains
  here is what changes the code you would write today; HOW it was measured goes
  to `notes/`, behind a pointer naming when to go and read it.** Anything moved
  there stays a pointer here — never a silent deletion.
  **THE BUDGET IS 130,000 CHARACTERS** (`wc -c CLAUDE.md`, in the Commands
  block). It is stated as a number because "keep it short" alone has now failed
  twice: the first split (2026-08-02) bought 19,321 chars and 83% of that was
  spent within two commits, putting the file back over the line by 2026-08-05.
  When a feature's retrospective would push it over, that retrospective is what
  moves — not the rule it justifies.
  - `2d-milestones.md` — M1–M4 and M7: the auto-expand memory guard, the
    float32 and relativistic 2D verification, the export figure's typesetting,
    and the one-exponent-slot before/after.
  - `precision.md` — the float64/float32 bench figures, the mixed-scheme
    derivation, the two dtype-invisible failure modes, the auto-expand-noise and
    tol-floor sweeps.
  - `export.md` — the mp4 encoder probe, pool sizing, blit timings and metadata
    line-budget arithmetic.
  - `ui-notices.md` — the transient-notice, boundary-warning, header-strip,
    parameter-policy and theming archaeology, with the browser reproductions
    and the edge noise-floor sweep.
  - `session-lifecycle.md` — every RSS/VRAM figure behind "a closed session must
    actually free", the browser receive ceiling, and the streaming transport.
  - `memory-and-devices.md` — the device fit check, the per-cell footprint
    tables and the IC preview's VRAM release.
  - `expression-ics.md` — the `wexpr`/`psi` transform verification and the
    potential validity model.
  - `downsampling.md` — the pyramid/crop benchmarks, the browser A/B that
    justifies protocol v5, and the two traps in reproducing it.

Dependency workflow (same as urantia-library): edit `requirements*.in`,
then `uv pip compile requirements[-x].in -o requirements[-x].txt` and
`uv pip sync --compile-bytecode requirements.txt requirements-dev.txt
[requirements-gpu.txt]`. The `--compile-bytecode` is not optional in a
deployment: the systemd unit runs with a read-only home, so Python can never
write `__pycache__/*.pyc` at runtime — install-time precompilation (plus
`.venv/bin/python -m compileall main.py config.py core routers` for our own
source) is what keeps startup fast. See `README.md`.

## Architecture (see the plan in git history / memory for rationale)

- **Dimensionality: ONE generic core over `ndim ∈ {1, 2}`** (added 2026-07-26).
  1D solves W(x,p,t) on `(Nx,Np)`; 2D solves W(x,y,px,py,t) on
  `(Nx,Ny,Npx,Npy)`. There is no second solver — 1D is `ndim=1` of the same
  code, which is deliberate: this project navigates by diagnostics, and two
  propagators that could quietly disagree would poison exactly that.
  `core/axes.py` is the single source of axis truth (labels, the array-order
  split, the plane set, every display title), mirrored by
  `frontend/src/lib/axes.ts` and by `render_mpl.py` — change a title string in
  one and change it in all three. **ARRAY ORDER is (q…, k…)**: all spatial axes
  then all momentum axes, so `ndim=1` is `(x,p)` exactly as before.
  **CONJUGATION is index-matched** — θ_i is the dual of k_i and lives on array
  axis `ndim+i`, λ_i is the dual of q_i and lives on axis `i` — stated once in
  `axes.conjugate` and derived everywhere from it, because a wrong pairing is
  the classic multi-D error and is silent.
  `Grid` takes a tuple of `Axis(lo, hi, n, d, anchor)` and exposes `N/lo/hi/d`
  tuples, `v` (natural-order vectors), `C`/`D` (shifted coordinate and dual
  meshes) and `dmu`, the phase-space measure ∏d[a] — dμ rather than dV
  because it is a measure on 2·ndim-dimensional phase space, not a spatial
  volume, and the two would read alike at ndim=1 where they differ least.
  Every 1D-only spelling — `grid.dx`, `gs.ox`,
  `geom.Nx`, `vf.rho`, `obs.x_std`, `es.x_mass`, `cp.dUdx` — survives as a
  compatibility property that **RAISES at ndim > 1** rather than returning axis
  0/1, so a call site nobody generalized fails loudly instead of computing a
  wrong number. That is what made the migration tractable; do not soften it.
  **NOTHING IS DEFERRED IN 2D ANY MORE.** M2 (relativistic `qr`/`cr`) and M1
  (float32) landed 2026-07-27, M4 (mp4 export) on 2026-07-28 and **M3
  (auto-expand) on 2026-08-01** — see the relativistic-2D, float32-in-2D,
  2D-mp4 and 2D-auto-expand gotchas below for what was measured to retire each.
  `applyNdimInvariants` (`lib/config.ts`), which mirrored those gates in the
  FORM, is **deleted** rather than kept as an empty hook: an invariant helper
  that enforces nothing is a place for a future gate to be added silently,
  which is the opposite of the marker/tooltip/amber pattern. What survives is
  `applyPrecisionInvariants`, because float32 still refuses auto-expand and
  `tol < 1e-5` — at EITHER dimensionality. The remaining M5/M6 rows are
  enhancements, not gates: nothing refuses them, they simply do not exist yet.
- **2D streams PLANE REDUCTIONS, never the state.** W(x,y,px,py) can be neither
  drawn nor sent, so each worker reduces it on the device to the six pairwise
  2D projections (`axes.PLANES`) plus one 1D marginal per axis, and the 4D array
  never leaves the GPU. **Measured at 64⁴: 50.0 KiB per variant per record
  against 33 MiB for the raw state** — so the browser-receive ceiling and
  `WIGNERF_HISTORY_MB` both stop being 2D constraints, and scrubbing is instant.
  At `ndim=1` the single plane's complement is EMPTY, so that plane *is* W: 1D
  is the general case, not a special one, and `quantize` sees the identical
  array it always did. Each plane keeps its OWN range and therefore its own
  colorbar — the six reductions of one state differ by orders of magnitude (a
  spatial density against a signed reduced Wigner function), so one shared range
  would render most panels blank. `core/frame.py` builds a record, shared by
  `worker._emit` and `routers/preview.py` so a session frame and an IC preview
  frame come out of one code path. Observables follow: everything but the purity
  comes from reductions (⟨H⟩ = ∫U·n_q + ∫T·n_k, moments from the 1D marginals,
  ⟨Lz⟩ from the two mixed planes), leaving ∫W² as the only full-array pass —
  which is also why `Propagator` no longer keeps a full-shape `H` mesh, in 1D
  either.
  **`PanelGrid`'s two readings are `variants` and `phase`** — one plane across
  every variant, or every plane of one variant. The second is called "**phase
  portrait**" in the UI and never just "portrait", because that is already the
  name of a LAYOUT orientation two controls away in the same header; the stored
  `wignerf.panelMode` value migrates from the old `'portrait'` rather than
  resetting the choice. **"link zoom" is hidden in the phase portrait**
  (`canLink`): coupling one fractional window across panels only means something
  when they share an axis pair, and there they show six different ones — (x,y)
  against (px,py) against (x,py) — different quantities in different units, so
  one window would map to six unrelated physical regions. The stored preference
  is left alone, so it comes back on the way to `variants`.
- **Streaming**: solver workers append records to an in-RAM byte-capped
  `FrameHistory`; the WS streamer (`routers/stream.py`) sends the newest
  lockstep-complete record (live, coalescing — slow clients skip frames) or
  exact sequential records (replay/scrub). Computation ALWAYS runs at full speed
  in both modes — neither the dial nor a slow client ever throttles the workers;
  `delay` paces only the display. The dial's "0" (default) means one record per
  display refresh: the client measures its refresh interval
  (lib/perf.ts) and sends that as the delay, and every dial position is clamped
  to at least it, so delivery never outpaces painting. **At 4096²/8192² that is
  NOT enough and there is deliberately no client-side pacing loop** — an
  adaptive pacer keyed on paint time was built and REMOVED on 2026-07-23,
  because paint time is not the binding constraint there; see the
  browser-receive-ceiling gotcha. Don't rebuild it: the constraint is the
  browser's per-message receive cost, so the fix is smaller messages, and that
  fix is now BUILT: **the client sends a `view` (physical window + pixel size
  per panel) and the server ships only that** — a decimated crop out of a
  per-record mip pyramid (`core/pyramid.py`, `core/planeview.py`, protocol v5).
  Measured at 4096²: 3.1 rec/s at 99 MiB/s before (i.e. AT the ceiling), 8.0 at
  4.2 after; at 8192² the before case is not slow but ZERO — no 128 MiB record
  arrives at all. Four rules: the area mean happens ON THE DEVICE once per
  record (250 ms/plane on the host); the send-time crop is a CONTIGUOUS slice of
  a pre-reduced level (0.07 ms against 1.1 for a strided gather), which is the
  only reason a pyramid is built; every level is quantized against the FULL
  plane's range, so a level change cannot repaint the colorbar under someone who
  only scrolled; and the window is PHYSICAL, which is what survives auto-expand
  — `_views_for` resolves per RECORD, so a scrub across a regrid needs no
  bookkeeping. A plane absent from the request is NOT SENT (`na = 0`,
  header only), so the phase portrait stops paying for planes it is not
  showing — which also means **the client must pick a plane by PAIR, not by
  list position**. A reduced panel says so (`↓8×` with the reason in its
  title): area-averaging is honest but invisible, and fringes below one pixel
  vanish with nothing on screen to suggest it. The viewport is re-sent on
  reconnect, because `send` no-ops on a closed socket and the server drops a
  departing client's. Numbers in `notes/downsampling.md`.
  Replay never skips a record; it slips on WS
  backpressure. The UI dial is "0" plus a log range 20 ms–1.5 s.
  **THAT BACKPRESSURE IS OURS AND MUST STAY OURS** (`protocol.AckCmd`, 2026-08-05).
  `ws.send_bytes` does NOT wait: uvicorn's `websockets-sansio` impl — what
  `--ws auto` now picks — sets its `writable` event once and never clears it and
  has no `pause_writing`/`resume_writing` at all. So a replay buffered the whole
  history, the keepalive PING went out behind it, could not be answered inside
  `--ws-ping-timeout`, and **the server killed its own socket** — every "streamer
  send failed / disconnected" pair in that day's journal, at an exact multiple of
  the ping interval after accept. The client therefore acks each
  PAINTED record and the sender stops past `INFLIGHT_MAX_BYTES`, which must admit
  ≥2 records or it degenerates to stop-and-wait. Pacing arms on the FIRST ack, so
  a non-acking client (ws_smoke, any raw consumer) is unpaced.
  `start.sh` pinned the ping VALUES against an impl swap and that was not what
  mattered; an app-level credit cannot be swapped out. NB `_guard_send`'s old
  `code=1006` was a HARDCODED CONSTANT, not a wire code, and it cost a whole
  investigation. Measurements in `notes/session-lifecycle.md`.
  Client frame fan-out is rAF-timed (decode per message, paint one per animation
  frame; small FIFO, drop-to-newest as a burst valve), so texture uploads, uPlot
  updates and Vue reactivity run per PAINTED frame by construction. That drop-to-newest is why the timeline readout shows painted/s
  AND received/s: when they diverge the client is SKIPPING records, which reads
  on screen as fast playback and is really loss — one number alone cannot tell
  the two apart, and the live path makes that worse by design (the `delay` gate
  applies only to replay while live coalesces to the newest record, so computing
  legitimately animates faster than paced playback). It also shows the SERVER's
  bytes/s (`status.sent_bytes_per_s`), the only number separating "the server
  sent less" from "this client received less". A playback-only run must
  never coalesce to the frontier while sequential records are unsent, and its
  auto-pause is delivery-aware — it fires only after the frontier record was
  **PAINTED** (the client's newest ack), else the newest sent for a client that
  does not ack. "Sent" was the honest best available before the credit below,
  but with no transport backpressure it meant "buffered", so the gate that
  exists to stop the cursor outrunning unseen records read a number that had.
  **`loop` repeats that pass instead of pausing** (`LoopCmd`, a checkbox beside
  Solve/Play/Pause, echoed in `status`). It exists because the auto-pause is
  correct but easy to walk into: playback stops at the frontier, the button
  becomes "Solve", and the Space that replayed a second ago now COMPUTES. A
  DISPLAY policy like `delay`, it rewinds to `loop_from` — the cursor captured
  when the pass STARTED — so "again" means the region you asked to watch, not
  all of history. Two things are load-bearing: it reuses the
  auto-pause's delivery gate, so a slow client is never rewound past frames it
  has not been sent; and `browsed` stays True across the wrap, or the next tick
  re-attaches to the frontier and rolls into computation. **Rewinding `cursor`
  alone STALLS the loop silently** — the sender walks forward from `last_sent`,
  which is still at the frontier, so nothing sends and the display freezes.
  Hence `loop_epoch`, bumped on each wrap, which the sender watches to rearm
  `last_sent`. NB a test that counts arrivals at the FRONTIER cannot see that
  failure — the live frame already in flight when the seek was sent is itself
  the frontier, so a dead loop reads as two passes; count arrivals at the START.
  Pinned by `test_loop_replays_the_same_region_instead_of_stopping`.
  **AN ARMED `loop` ALWAYS LOOPS, including when `loop_from` IS the frontier.**
  The wrap gate was `loop_from < latest_complete`, which a FINISHED BATCH run
  fails by construction — there play at the frontier is playback, not Solve, so
  `set_running` captures `loop_from = int(cursor) = frontier` — and it fell
  through to the pause, the checkbox silently doing nothing. A reconnect reaches
  that state unattended: detaching pauses and `recover()` re-issues `play`. It
  now falls back to the oldest RETAINED record. Pinned by
  `test_loop_wraps_even_when_the_pass_started_at_the_frontier`.
  The transport must stay responsive under full frame backpressure: control JSON
  is flushed BEFORE frame sends each tick, play/pause are echoed at once, replay
  batches are wall-clock-budgeted (~0.2 s) and preempted by pause/seek, and the
  client flips the transport button optimistically. The delay dial is
  settable only while PAUSED and its thumb is local UI state, re-synced from
  status when idle. Binary layout in `core/protocol.py`, mirrored by
  `frontend/src/lib/protocol.ts` and cross-checked via `scripts/gen_fixture.py`
  + the frontend vitest. Measurements in `notes/session-lifecycle.md`.
- **Record grid**: τ_k = t1 + k·record_dt. Each variant (1–4 worker
  threads: quantum/classical × rel/non-rel) integrates with its own
  adaptive dt (`adjust_step`, every 20 steps) but lands exactly on each τ_k
  by clamping the final substep. Same k ⇒ same physical t across variants.
- **State convention**: W is float64 and fftshifted INSIDE the propagator, and
  **natural (unshifted) order everywhere outside it** — `frame.build` unshifts
  on the device, which is free there. That is a precondition for cropping, not
  a tidy-up: a crop of a shifted array straddles the seam, so the old
  convention and display downsampling could not coexist without a third
  convention or two client paths. The shader's half-period offset and its
  toroidal `%` wrap went with the shift they undid (it clamps now; `viewWindow`
  never shows past the seam anyway), and so did `render_mpl`'s unshift.
  (`render/WignerRenderer.ts`: R16UI texture, manual bilinear, diverging LUT
  centered at W=0.)
  **Every W plot autoscales to its OWN range and therefore carries its OWN
  colorbar**, overlaid in its corner (`Colorbar.vue`, taking an explicit
  min/max). `WignerPanel` uploads `f.variants[variantIndex]` and the renderer
  sets `q = [v.wmin, v.wmax]` from it, so the four variants' scales drift apart
  as they evolve — measured on a cat state in x²/2 + 0.3x⁴ at t = 15: QN wmin
  −1.87e-1 vs CN −2.70e-1 (44% apart), wmax +3.18e-1 vs +3.51e-1. A single
  shared bar read `variants[0]` and so mislabelled the other three panels; it
  agreed with all of them only at record 0, which IS the IC, which is exactly
  why it looked right. The IC preview has one too — it is a W plot with its own
  range. Overlaid rather than stacked because it then costs no layout height:
  the bar used to head the diagnostics column, the tallest of the three in
  portrait (808 px against setup 758, IC 557), so a row spent there was a row
  the panels started later by.
- **Boundary watch / auto-expand** (`core/boundary.py`): detection is
  ALWAYS on — every record, each worker sums the outer edge band of the ρ/φ
  marginals it already computed (host-side, O(Nx+Np), no extra device sync) and
  `session.report_edge` posts a `boundary` WS event on state change (band =
  max(4, N/32) cells/side, trigger 1e-6 in float64 and 1e-4 in float32 via
  `EDGE_THRESHOLD_BY_PRECISION`; **gated on a MEASURED noise floor and confirmed
  over `EDGE_CONFIRM`=4 records** — see the edge-noise-floor gotcha, which is
  what stops a coarse 2D grid strobing the warning). Expansion prevents wrap, it
  cannot repair it, so it must fire while edge mass is negligible.
  The `auto_expand` toggle (SessionCreate field AND live-appliable via
  ParamChange) governs only the RESPONSE: an exact fixed-lattice regrid. **It is
  refused outright in float32** — single precision cannot supply the
  measurements it needs.
  dx/dp and the lattice anchor are FROZEN at session creation (`GridState`,
  integer window arithmetic; extents materialize as anchor + integer·dx, so
  overlap lattice points are bitwise-identical across regrids); move =
  whole-cell window shift, expand = double an axis (powers of 2, support
  centered, combined move+double; NO shrink, NO interpolation ever — norm/E/
  purity survive to machine precision minus the ≤threshold dropped tails). The
  session commits a `RegridPlan(epoch, k_star, state)` with k_star past every
  in-flight record; each worker applies it before computing its first record ≥
  k_star (`embed_window` + `Propagator.set_grid`), so the switch is
  lockstep-uniform and records <k_star stay old-geometry. U is revalidated on
  the union extended Bopp range BEFORE commit (refusal ⇒ `invalid_potential`
  warning, keep computing).
  **Plan commits and physics commits are mutually exclusive** (both hold
  `_edge_lock` for their whole body, and `apply_params` orders physics BEFORE
  any immediate schedule): U/hbar_eff move the Bopp range, a plan validated
  under stale physics would hit the deliberately-fatal non-finite check at
  k_star (a per-worker rollback there would desync lockstep geometry), so a
  pending plan's union window is revalidated under incoming physics and the
  change is REJECTED if it does not hold — this also closes the race of a plan
  committing during the streamer's ~ms validation compile. Expansion caps at
  `WIGNERF_MAX_GRID` (`capped` warning, keep computing; pure moves still work at
  the cap).
  Geometry is a PER-RECORD fact: protocol headers carry the axis counts and
  extents, history keeps geom per record, the streamer packs from the record
  (never the session), and the frontend follows the PAINTED frame (panels,
  overlays and marginal axes re-derive per frame; zoom windows remap to the same
  physical region), so scrubbing across a regrid boundary just works. Each
  doubling ≈ 4× step cost and 4× bytes/record.
- **Export panel** (header button "⤓ export") carries two things: the mp4
  below, and the run's SETUP — `GET /sessions/{id}/setup` serves
  `describe.setup_document`, the config the session was CREATED with
  (`state_at(cfg, log, -1)` rewinds every live change; live changes are not
  part of a starting state — the video's metadata block records them). Import fills the setup form and marks the session
  restart-dirty, never restarts by itself (`lib/config.importConfig`, in-place
  merge on the reactive cfg), and accepts that .json OR an exported .mp4: `lib/mp4meta.ts` scans the file's head for the same document in the
  `comment` tag (faststart keeps it there, byte ~3.5k), so a kept video is
  self-restoring. Its confirmation says `press "Restart session" to run
  it` and NOT "or Solve": an import moves grid/IC/variants, which are
  SessionCreate-only, so Solve would compute the old ones (the auto-restart in
  `syncFreshSessionToForm` fires only when the document ALSO moves
  mode/t₂/Δt rec/precision on a fresh idle session). And it clears on a
  `sessionId` change, because that IS the restart it asks for: it used to clear
  only at the top of the NEXT import, so "press Restart session to run
  it" stood over a session already running the imported setup. It survives a panel
  close/reopen on purpose — an import you have not acted on is still
  actionable.
  A render is destroyed by anything that moves the session on — Restart deletes
  the session (file unlinked mid-write) and computing new records evicts the
  ones behind the renderer. Both used to happen SILENTLY, so
  `SimulatorView.mayDiscardExport` confirms first (Restart, and a transport
  command whose action is `solve` — playback adds no records and is never
  gated) and cancels the job outright on "yes" rather than leaving it to die
  mid-file. The automatic restarts (first mount, backend recovery) never prompt.
- **mp4 export** (`core/videoexport.py` + `core/render_mpl.py` +
  `routers/export.py`): renders an ALREADY-COMPUTED record range on the
  BACKEND — matplotlib/Agg frames piped as raw RGBA into ffmpeg (system ffmpeg,
  absence ⇒ 503). PAUSED-only (409 while running): a running session evicts old
  records, and the feature is for filming a range you already played
  back. **Read `notes/export.md` before changing any of it**:
  every figure below was measured, and the note says how.
  **WHAT GOES IN THE FRAME IS A CHOICE** (`ExportSpec.planes` / `.diagnostics`),
  because a 2D record carries far more than a frame can hold: six planes × four
  variants is 24 panels against a column of NINE diagnostics.
  So **panels are the CARTESIAN PRODUCT of the selected planes and variants**,
  which makes `PanelGrid`'s two readings the two EDGES of one control rather
  than modes the renderer knows about — "compare variants" is one plane × every
  variant, "phase portrait" every plane × one variant, and the Export panel
  offers both as one-click presets setting the checkboxes.
  `render_mpl.panel_grid` REFLOWS by count when one dimension is 1, else lays
  out the matrix, rows = planes. Diagnostics are plot ids shared VERBATIM with
  `frontend/src/lib/plotPrefs.ts` (`marg0..marg3`, `E`, `uncertainty0/1`,
  `purity`, `lz`) — one vocabulary for the hidden-series preferences, the export
  wire and the metadata block — and an EMPTY list is legal. **The 2D default
  drops the four marginals, and that is a physical argument, not a space-saving
  one**: at ndim=2 the (x,y) and (px,py) PANELS already ARE the spatial and
  momentum densities. Past `DIAG_ROWS_MAX` = 7 the column splits in two and the
  panels pay in width (`diag_layout`); the panel states the resulting count and
  pixel size and warns about thumbnails.
  Both refusals live in `routers/export.py`, not the schema (what is available
  depends on the session's ndim, which the body does not carry), and both name
  what IS available.
  **NEITHER THE ENCODE NOR THE VIDEO SIZE IS THE COST — THE PLANE IS**, twice
  over, and both fixes are "only what a panel can draw". (1) matplotlib is priced
  by the ARRAY handed to it: 4096² cost 2247 ms/frame/worker against 256²'s 80
  while 1080p and 4K differed by 3%, so `render_mpl.plane_step` draws the
  coarsest mip level still covering the panel (`_panel_px`), never below it —
  4429 → 92 ms at 4096²×4 variants, 92.6% of pixels bit-identical on a fringed
  cat state. (2) the pool is fed by PICKLE and at 8192² a record is 170.8 MiB,
  70 ms to serialize alone: that was 3.2 fps *whatever the video size*, FHD and
  4K coming out identical being the tell. `ExportJob._trim` drops unshown
  planes' payloads and keeps only levels at or under the FIGURE's width — a
  bound no panel can exceed, so it cannot under-resolve and no layout arithmetic
  is mirrored. FHD 3.23 → 35.65 rec/s, 4K 3.53 → 8.89, and they now DIFFER, i.e.
  the render is the wall again. NB a SHORT job measures the pool warmup; read
  the rolling rate. So export renders
  frames across a **spawn** `ProcessPoolExecutor` (`WIGNERF_EXPORT_WORKERS`,
  auto = min(cpu, 8)) while this thread feeds ORDERED frames to one ffmpeg:
  a sliding window of ≤w+2 futures consumed FIFO, so workers run ahead with
  memory bounded. Spawn, NOT fork: the backend initializes CUDA and forking
  after that inherits a broken context. A small job (`< max(2·w,
  POOL_MIN_FRAMES=16)`) renders serially, skipping the warmup. Encoder via
  `choose_encoder`/`WIGNERF_EXPORT_ENCODER` — **auto is libx264 EVEN WHERE THE
  GPU WORKS**; see that row for why, and for the probe bug behind it.
  Two passes: a scan collects the E/ΔX·ΔP/γ series, the per-variant FIXED colour
  scale (no brightness flicker), the fixed marginal amplitudes and the widest
  window any record used, and proves every record is still retained before ffmpeg
  starts; then one figure update per frame. Only VALUE scales are
  export-wide — the SPATIAL axes follow each record's own geometry
  (`_apply_geom`, which re-captures the blit background too, ticks being static
  art), as the SPA follows the painted frame. The figure is built ONCE and
  BLITTED (static background + ~15 animated artists), the difference between
  minutes and half an hour for 1000 frames.
  Sizes offered: FHD / QHD / 4K UHD. The figure is always 19.2×10.8 in and the
  RESOLUTION RIDES ON THE DPI (`FrameFigure.REF_WIDTH`) — font sizes are in
  points, so a fixed dpi renders every label at half its relative size at 4K.
  The downloaded name is descriptive (`Content-Disposition`) while the on-disk
  path keeps session+job ids, so two exports of one range cannot collide mid
  download.
  **The video must READ like the screen**: every plot title comes from
  `core/axes.py`, the source `lib/axes.ts` mirrors (γ keeps the UI's "purity
  γ(t) = 2πℏ∬W²dxdp", never an equivalent like Tr ρ²), field labels match the
  Setup panel, and the series y-window + tick decimals reproduce that
  component's `scales.y.range` rule (`render_mpl.series_ylim`) — matplotlib's
  autoscale renders a 2e-5 purity drift as a dramatic dive with a "×10⁻⁵+1"
  offset where the UI shows a flat line at 1.000000, from identical data.
  The "grid lines on plots" toggle rides along in `ExportSpec.show_grid` and
  governs EVERY plot in the frame — charts get uPlot's grid stroke, the W panels
  `GridOverlay.vue`'s theme-INDEPENDENT lines drawn AFTER the image (matplotlib
  puts the axes grid under it, hence the heatmaps first had none). Mirror any change to those rules on both sides.
  The metadata block carries U(x), parameters, the IC as an analytic expression
  (`core/describe.py`) and any live parameter change inside the range
  (`session.param_log`), so one frame documents the whole run; the same facts
  go into the mp4 `comment` tag as JSON. Anchored at `FrameFigure.META_TOP`
  with `va="top"`, growing DOWNWARD, and **11 lines fit at 8 pt**; a realistic
  4-variant run sits at the edge, so `_meta_fontsize` shrinks to fit (one size
  for BOTH columns) and past a 5 pt floor `_meta_fit` elides with a pointer to
  the comment tag — honest, since `describe.config_json` really carries it.
  `describe.IC_SRC_MAX` caps an IC expression against the same budget, in
  PHYSICAL lines. Static art in the blit background, so free per frame.
  Progress: `export` events on the session WS plus a REST poll, carrying
  `render_fps` — **the RENDER rate, NOT `fps`, which is the finished mp4's** —
  ROLLING while running, because a cumulative average spends the render
  climbing out of the pool warmup and reads as a slowdown that is not
  happening. The file lives in `WIGNERF_EXPORT_DIR` until downloaded: TTL
  30 min, session close, or shutdown.
  The header button stays ENABLED while computing (a disabled button
  explained only by a tooltip is how this feature first read as broken): the
  panel states the gate, and "Pause & render" pauses, waits for the server's
  confirmation and re-seeds an untouched range before posting. Rendering continues
  while the popover is CLOSED, so the button IS the notification — "⤓ export
  42%" while running, emerald "⤓ export ready" (red "failed") when finished. The
  panel re-reads the extent from `GET /sessions/{id}` when it opens: the
  streamed status lags a frame burst by seconds after a pause, and seeding the
  range from it silently exported half the history.
  **The mp4 export follows the UI theme** (`ExportSpec.theme`, defaulted from
  the app each time the Export panel opens, overridable per job, never persisted
  or it would stop tracking). Schema default `light`, like the SPA's. It threads
  the same path `show_grid` does, to `render_mpl.FrameFigure(theme=)`, which
  resolves `PALETTE[theme]` once.
  `render_mpl`'s `PALETTE`/`VARIANT_COLORS` MIRROR the `--wf-*` values (it
  cannot read our stylesheet): change one side, change the other.
- **Theming (light/dark)**: a header button (`☀ light` / `☾ dark`) flips the
  whole UI; **light is the DEFAULT** and the choice persists in
  `localStorage.wignerf.theme`, a sibling of `wignerf.layout`/`wignerf.grid` and
  so untouched by "Reset setup to defaults". Details and the migration numbers
  are in `notes/ui-notices.md`.
  **CSS is the single source of truth**: `frontend/src/style.css` defines every
  colour once per theme as `--wf-*` on `:root` (light) and `.dark`, and a
  Tailwind 4 `@theme inline` block turns them into semantic utilities
  (`bg-panel`, `text-fg-3`, `border-line`, `text-warn`, …). **`inline` is
  load-bearing** — a plain `@theme` copies the VALUE in at build time and a
  runtime override then changes nothing. There are ~15 roles, not 200 literals.
  State lives in `frontend/src/lib/theme.ts` — a module-singleton `ref`, NOT a
  prop like `showGrid`, because the uPlot option builders need it too.
  `chartPalette()` reads the `--wf-chart-*` properties back off the document, so
  JS duplicates no value; it must run AFTER the root class is applied, hence a
  function called per chart build (one `getComputedStyle`, never per frame) with
  literal fallbacks for the DOM-less unit tests. That ordering is why
  `setTheme`/`toggleTheme` apply the root class SYNCHRONOUSLY as well as from
  the watch: `chartPalette` CACHES per theme name, so one read taken before the
  class landed would pin the wrong palette for the life of the page — not a
  one-frame glitch. Nothing should rest on a Vue scheduler detail, and `apply`
  is idempotent.
  **The charts destroy+rebuild** on a theme change — uPlot takes axis/grid/
  series colours at construction only — by widening the watch the grid-lines
  toggle already had; all of them re-apply their existing data to the new chart
  rather than re-fetching it. The theme is deliberately NOT in `SimulatorView`'s
  `plotsKey`: a flip must never remount/blank the W panels.
  `index.html` applies the stored class in a **blocking inline script**, because
  `main.ts` is a deferred module and a dark user would otherwise get a white
  flash on every load; it also declares `color-scheme`, which fixes native
  `<select>`/`<input type=range>` widgets rendering in the OS's LIGHT style.
  **What does NOT follow the theme, on purpose**: the bwr heatmap LUT
  (blue-white-red with W = 0 at white is the physics convention) and so
  everything drawn ON the heatmap rather than the page — `GridOverlay.vue`'s
  grey lines, `Colorbar.vue`'s chrome, the panels' overlay labels, `ICEditor`'s
  IC-marker rings. Saturated filled action buttons stay put too; they read
  correctly on white. What DOES change and is easy to miss: **variant curve
  colours** (`lib/variants.ts`, `*-400` on dark → `*-600` on light, because
  `#fbbf24` amber on white is unreadable) via `variantColor()`, and the
  Timeline readouts' halo (`--wf-label-shadow` inverts).
  **A data trace gets its OWN role, never a borrowed one.** `--wf-wave-re/im/abs`
  are three roles for three series precisely so adjusting the time cursor cannot
  silently repaint |ψ|² in a chart it has nothing to do with.
- **Parameter policy**: U, c, mass, hbar_eff, tol, dt_sign, auto_expand
  apply live at the frontier; **ndim**/grid/IC/variant-set and the whole COMPUTE
  group (precision, device, history_mb — the Setup panel's third section)
  require a session restart, because each is fixed at worker construction (FFT
  plan dtype, `ArrayBackend` device, `FrameHistory` cap). Auto-expand moves the
  LIVE grid; the Setup panel shows it and offers "adopt" to copy it into the
  form. **The panel's own layout arithmetic and the UI archaeology behind these
  rules are in `notes/ui-notices.md`** — including why COMPUTE is the one
  section with labels above its controls, and why the column is 320px.
  ndim is the most restart-only of them all — it decides the array rank — and
  switching it in the form rebuilds the grid and the IC (`config.setNdim`,
  mirroring each component's second dimension on its first) and replaces U
  **only if it was still the default**, since `x^2/2` cannot silently become a
  two-variable expression and a hand-written potential must never be discarded.
  The same rule governs the two IC EXPRESSION drafts, with one extra condition:
  "still at the other ndim's default" is a good proxy for "untouched" only on an
  EXPLICIT switch, so `conformICExprToNdim` takes the set of drafts the caller
  was just handed and never repoints those (`DEFAULT_IC_EXPR[1].psi` is itself a
  legal 2D expression, so on a page load or an import the proxy discarded text
  the user had typed, or the document had carried).
  **The BOX follows the same "only if untouched" rule, for a sharper reason**: a
  still-default box is replaced by the TARGET ndim's default, because carrying
  [-6,6] into 2D reproduces exactly what `DEFAULT_AXES[2]` was widened to [-8,8]
  to avoid — the edge band is max(4, N/32) CELLS, so at N=64 the 4-cell floor
  leaves only 4.60σ from the default packet, and a FRESH 2D default tripped its
  own boundary warning on the first Restart. A box the user
  CHOSE still carries over untouched: silently widening someone's domain is
  worse than a warning. **N lands inside the TARGET's own select list**: their
  own choice when it is offerable there, capped at that ndim's default, and the
  target's DEFAULT when it is not — never the list's floor, which would quarter
  the resolution of anyone who started at the 1D default and merely looked at
  2D. Pinned in `config.test.ts`.
  **Every restart-only field goes amber when it disagrees with the session**,
  COMPUTE included — a form reading `cuda:0` over a session on `cuda:1` is the
  same trap as one reading float64 over a float32 run. `precision` gets it via
  `LiveRun`, but `device` and `history_mb` cannot: the form holds a REQUEST
  (`''` = the host's pool, `0` = its ceiling) while `status` reports what was
  GRANTED. So `SetupPanel` resolves the request the way the server would — `''`
  against `/api/device`'s pool, a bare `cuda` to `cuda:0` to match
  `resolve_devices`, `history_mb` clamped to `history_mb_max` — and only then
  compares. One amber line names what is running, and only while something
  differs. **Each section's summary line covers its OWN fields**: `precision`
  lives in `LiveRun` because that is where `status` carries it, but it is a
  COMPUTE control, and is deliberately absent from `runStale` and the RUN line.
  The steady-state facts are NOT repeated in the panel — the devices and the
  history cap ride the timeline's own readout, which is drawn anyway.
  `apply_params` echoes a fresh `status` right after its `params_applied`,
  exactly as play/pause do — the periodic one is only every `STATUS_PERIOD`
  (1 s), so a field the form marks amber could stay amber for a second after the
  "✓ applied" flash had confirmed the change. Pinned by
  `test_params_applied_is_followed_by_a_fresh_status`.
  `apply_params` compares against what is LIVE and drops fields that did not
  change — no worker command, no `param_log` entry, no `params_applied`, nothing
  at all if the message is a no-op ("Apply live" always carries the U(x) draft,
  so the log used to fill with U changes that never happened and an export's
  "how to reproduce this" block lied about its own frames). Entries carry `before` as well as `applied`, so the
  block renders "ℏ 1 → 2" and `describe.state_at` rewinds the header physics to
  the FIRST exported record. Live changes are visible in the UI: the header
  flashes "✓ applied …", and any live-appliable field whose form value differs
  from `status` renders amber.
  **U(x) is a LIVE-appliable parameter and has exactly ONE button.** It used to
  have two, both confusing for the same reason: the draft was local to the
  editor, so the FORM did not mean what it showed until you pressed one — while
  every other setting in the panel is bound straight to `cfg`. The draft now
  auto-commits to `cfg.potential` from `compile()`'s success path, gated on the
  server's verdict (a half-typed `x^2/` must never reach `cfg`, which is
  persisted and is what a restart computes from) and on the response still
  describing the CURRENT text. So U(x) is not restart-dirtying at all.
  **Solve carries the form's U(x).** `SimulatorView.sendCommand` pushes
  `set_params {U: cfg.potential}` before a `play` whose action is `solve`,
  whenever the form's validated U differs from `status.potential`. Without it
  the form was authoritative for nothing: 369 records computed under `x^2/2`
  behind a form reading `x^2/2 + 5*x^4`, the only signal an amber input in a
  panel that can be hidden. Playback is excluded — it computes nothing. So "Apply live" is gated on `status.computing`, NOT on `live` and NOT
  on `running`: while nothing computes there is no live run to reach and Solve
  does the job. `computing`, because `running` is true during pure PLAYBACK too,
  where the button's own tooltip would be false. What is left is one emerald
  "Apply live", disabled unless the session is COMPUTING AND the draft is valid
  AND it differs from `status.potential`, with the reason in its `title` and **no
  standing paragraph** (see [no-mystery-disabled-controls]).
  The setup form gates the transport: while the potential draft is invalid for
  the active variant families or the IC preview errors, Solve (button AND Space)
  is disabled and "Apply live" greyed — nothing may compute behind a visibly
  broken form.
  **Every saturated action button carries `.wf-solid`** (`style.css`): it
  supplies `color: #fff` — those buttons never set a text colour, they INHERITED
  the shell's light text, which went invisible ("black on blue") once the shell
  could be light — and a disabled state dropping to the neutral raised surface,
  because `disabled:opacity-40` over a saturated fill is pale colour under
  equally pale text, unreadable in either theme.
- **Run modes: `interactive` vs `batch`** (SessionCreate `mode`; `batch`
  requires `t2`). Both start paused. INTERACTIVE computes until paused and
  streams a coalesced live preview (the newest complete record) so you can
  watch/zoom in real time. BATCH (renamed from `runahead` on 2026-07-24)
  computes flat-out until t2 and streams NO frames while computing — the
  heatmaps + marginals dim and the streamer sends only a throttled
  (`PROGRESS_PERIOD` = 0.25 s) JSON `progress` message (record, t, percent,
  per-variant steps/s, AND the frontier record's per-variant observables —
  E/x_std/p_std/purity; ~400 bytes). This is for heavy runs where transferring
  hundreds of MiB/record of live preview measurably slowed compute and hit the
  browser-receive ceiling; the progress report is ~1000× cheaper on the event
  loop and the workers are untouched. Batch's `status` carries `computing`
  (`running and not stop_at_frontier`) which drives the frontend dimming; the
  observable SERIES (E/ΔX·ΔP/γ) stay LIVE during batch compute because they
  poll `GET /sessions/{id}/series` (cheap, frame-independent), and so do the
  control bar's numeric readouts, from the progress message above — that is
  why the observables ride it. They cost nothing (the worker computed them at
  emit; `history.get` returns references, no array copies) and without them the
  bar read "—" for a whole batch run while the plots two panels away were live:
  the same data on screen, just missing from the one place showing its CURRENT
  value. **AND THEY MUST SURVIVE THE PAUSE THAT ENDS THE RUN**, which is when
  "what t did I stop at?" is actually asked: WHICH source a readout describes is
  decided by **arrival order** (`useSession.readoutSource` → `lib/readout.
  pickReadout`), never by record index. Gating it on `computing` blanked t, E,
  ΔX·ΔP, γ and the batch-% badge together the moment a mid-run Pause landed —
  interactive mode never showed it because there `lastFrame` survives a pause.
  "Highest record wins" is the wrong repair: scrubbing to record 50 in a
  FINISHED batch run would print the retained frontier report's t over a panel
  painted at 50. `_sender` also emits a FINAL report when a batch compute stops
  (on the compute→stop flip, and again per record that lands after it), because
  the periodic one is up to `PROGRESS_PERIOD` old and in-flight records land
  after the pause — while a batch PLAYBACK's pause must emit NONE, or the
  readouts jump off the browsed frame onto the frontier. Batch's live
  branch never sends a frame (computing OR paused-at-frontier) — you review a
  finished batch run via explicit playback (seek + sequential replay, which DO
  stream frames). Its t2 auto-stop is NOT delivery-gated (unlike playback):
  batch has no `delivered` frames, so the frontier reaching t2 is itself the
  completion signal.
- **Sessions always start paused** (both modes): computation begins only on
  the explicit Solve/Play command. The transport button label predicts its
  effect: Solve = will compute, Play = pure history playback, Pause while
  running. Playback-only runs (play pressed behind the frontier, or after a
  finished batch run) auto-pause AT the frontier — they never roll into
  computation; only an explicit Solve does (`SessionClock.stop_at_frontier`).
  A batch run that REACHES t2 ends the run too (`advance_cursor`): its workers
  already idle there, and leaving `running` set froze the transport on "Pause"
  forever and locked out every paused-only action (pinned by
  `test_batch_starts_paused_and_stops_at_t2`).
  Setup persists in browser localStorage; "↺ defaults" (IC editor) restores the
  IC in the form and marks the session restart-dirty, but **"Reset setup to
  defaults" (Setup panel) RESTARTS** — nearly all it restores is SessionCreate-only
  (grid, IC, variants), so marking the form dirty and stopping there left the one
  button whose job is "put everything back, ready to compute" needing a second
  click to mean anything: reported from a session at 8192×4096 under a form
  reading 1024², where Solve would still have computed the old grid. It emits
  `dirty` FIRST all the same, because the restart can be declined (mp4 render in
  flight) or refused (422), and then the form really does disagree.
  **EVERY form-driven restart goes through ONE serialized loop**
  (`SimulatorView.restartLoop`, once `syncFreshSessionToForm`): a reset moves the
  grid AND the run fields, so the fresh-session watcher fires in the same flush
  as the panel's own request, and two overlapping `create()`s each destroy the
  other's session, orphaning one that holds its workers' VRAM until the idle
  sweeper reaps it. Serializing alone is NOT enough — `requestRestart` samples
  `restartsStarted` SYNCHRONOUSLY at the click and drops its request if a create
  started after it, or the reset computes the same session twice in a row
  (measured: 3 creates for one click, the third redundant).
- **Potentials** (`core/potential.py`, on `core/expr.py`): tokenize-screen
  (THE security boundary, shared with the two expression IC kinds — one screen,
  three kinds of user expression; `potential.py` keeps the per-family VALIDITY
  model, which is a potential's alone). The parser's vocabulary is the same for
  every kind and only the free symbols and complex-ness differ, which is the
  `y`-at-every-ndim rule applied consistently: `I*x` in a U is refused as "U(x)
  must be a real expression", not "name 'I' is not allowed", because the user
  knows what I means and the real check was never the tokenizer anyway
  (`sqrt(-1)` evaluates to an explicit I without naming it). `hermite` is
  whitelisted and its ORDER is capped — it is a Function in the unevaluated
  parse, so the power screen cannot see it, and the evaluated parse materialises
  the polynomial. Suppressing that needs BOTH `parse_expr(evaluate=False)` and
  the `sp.evaluate(False)` CONTEXT: the keyword alone stops arithmetic
  evaluation but not function application, so the screen received the expanded
  polynomial it exists to refuse. Then: tokenize-screen → sympy parse →
  per-family validity.
  The Bopp arguments are REAL (q_i ∓ ħθ_i/2, complex dtype only): quantum needs
  U real+finite on the extended BOX (per spatial axis, with the CONJUGATE axis's
  spacing; Abs is quantum-valid); classical needs EVERY partial ∂U/∂q_i
  DiracDelta-free (Heaviside steps are quantum-only). At ndim=2 the symbols are
  (x, y) and `grad_exprs` is the gradient tuple. **The numeric probe lattice uses
  an ODD count per axis and forces an exact 0.0 onto any axis straddling the
  origin**: the poles that matter in 2D sit on the axes and at the origin
  (`1/sqrt(x^2+y^2)`, `1/x`, `log(x)`) and an even lattice steps straight over
  them — and sympy's `singularities` is one-dimensional, so past ndim=1 the
  lattice is most of the guard.
  **The preview endpoint's PLOT window and its VALIDITY boxes are two different
  things and must not be conflated.** `POST /api/preview/potential` takes
  `x1/x2` (and `y1/y2`) as what to SAMPLE — the editor zooms them, and zooming
  out past the domain is how the interesting part of U is found — while both
  validity boxes come from `req.grid`: `spatial_ranges()` for the classical
  gradient probe and `spatial_extended()` for the quantum one, the same pair
  `routers/sessions.compile_for` uses at create time. Tie the classical probe to
  the zoom instead and the panel stops predicting the API: `1/x` on [-6, 6]
  zoomed to [1, 6] reads `classical ✓`, opens the Solve gate, and 422s on the
  potential the editor had just approved. Pinned by
  `test_the_validity_probe_follows_the_GRID_not_the_zoom`.
  **At ndim=2 the editor draws the two axis cuts on TWO charts, not two traces**
  (`PotentialEditor.vue`): uPlot's `AlignedData` has ONE shared abscissa and
  these cuts do not share one — U(x, 0) is indexed by x over the zoom window,
  U(0, y) by y over the grid's own y extent. Overlaid, the y cut was drawn at
  the x sample positions, i.e. rescaled by (x2−x1)/(y2−y1) — invisible on the
  isotropic default box, which is exactly why it looked right. Each chart's
  title also names the coordinate its cut was actually TAKEN at (`nearestZero`),
  so `U(x, 0.4)` rather than a false `U(x, 0)`. 1D is untouched.
- **ICs** (`core/initial.py`): Gaussian mixtures (independent σ per axis) and
  cat states (analytic pairwise cross-Wigner; σ_k derived = ħ/(2σ_q) per
  dimension). **BOTH FACTORISE OVER DIMENSIONS, which is why 2D needed no new
  closed form**: the packets are separable products, the Wigner transform of a
  tensor product is the product of the transforms, so a 2D pair's cross-Wigner
  is the OUTER PRODUCT of the two one-dimensional cores (each carrying its own
  1/(2πħ), giving (2πħ)^−ndim automatically) and ⟨ψ_k|ψ_j⟩ is the product of the
  per-dimension overlaps. The 1D closed forms are reused verbatim, per
  dimension — and because those cores are 2D arrays, only their product is ever
  full size, which is why the 2D IC preview measures LESS per cell than the 1D
  one (56 B/cell for a cat, 32 for a mixture, against 88 in 1D).
  **TWO FURTHER KINDS ARE EXPRESSIONS** (2026-08-04): `wexpr`, an arbitrary
  analytic W on the phase space, and `psi`, an arbitrary complex ψ on
  configuration space from which W comes by the Wigner transform. Both
  auto-normalise, both work at either ndim, and `core/expr.py` — the token
  screen lifted out of `potential.py` — is now the ONE security boundary all
  three kinds of user expression go through. See the expression-IC gotcha below
  before touching `psi_wigner`.

- **Purity** γ = (2πℏ_eff)^ndim ∫W² over the whole phase space (= Tr ρ²; so
  2πℏ∬W²dxdp in 1D and (2πℏ)²⨌W²dxdydpxdpy in 2D — the power per dimension is
  easy to drop and makes a pure state read as γ ≈ 6.28 instead of 1) is
  computed per record and
  streamed/plotted. Both the Moyal flow (unitarity) and the classical
  Liouville flow (incompressibility) conserve it for closed systems, so
  until the Lindblad term exists it is a solver-fidelity diagnostic (a
  contained state holds it to ~1e-12); quantum validity of an IC is a
  property of the TOTAL W (γ ≤ 1 necessary), never of its components.

## GPU
`WIGNERF_DEVICE=auto|cpu|cuda:N|comma list` (config.py) names a device POOL.
`core/xp.resolve_devices` expands it fastest-first (`auto` = all CUDA devices
ranked by SM count; an explicit list is trusted as written) and
`core/session.assign_devices` spreads variant workers over it: costliest
variants (relativistic, then quantum) and the larger share go to the fastest
card; each worker owns its own `ArrayBackend`, so no propagator code is
device-aware. `core/xp.py` pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match
nvidia-smi (RTX 3090 = cuda:1, the display-driving 2080 Ti = cuda:0 here).

GPU deps: `cupy-cuda13x[ctk]` — the `[ctk]` extra is REQUIRED (cupy JIT-compiles
kernels at runtime via NVRTC, never nvcc, and needs the PyPI CUDA headers/libs;
NO system CUDA Toolkit anywhere, only the driver). CUDA 13 dropped
Maxwell/Pascal/Volta, so the dev workstation's GTX 1060 needs
`cupy-cuda12x[ctk]` instead.

Throughput, RTX 3090: ~2400 steps/s at 512², ~550 at 1024², ~134 at 2048²;
2080 Ti ~390 at 1024²; CPU (pyfftw) ~75 at 512². Multi-GPU is worth it:
4-worker lockstep at 1024² measured 135 steps/s all-on-3090 against 191 split
2+2 (+41%, and 2+2 beats 3+1's 181 — the even chunk is right).

**WHETHER A SESSION STARTS IS DECIDED BY ASKING THE DRIVER, not by a cell
count** (`routers/sessions._fit_error`) — **and since M3 the same question is
asked again whenever auto-expand wants to DOUBLE the grid.** The arithmetic
behind both lives in `core/fit.py` so the two cannot drift; the messages do not,
because the create-time advice ("drop a variant, change device") is unavailable
mid-run. The cell rails are only rails: the operative check runs `assign_devices` to
learn which devices this session's workers land on, counts them per device, and
compares
`n·cells·bytes_per_cell(ndim, precision) + CONTEXT_BYTES` (300 MiB of CUDA context + cuFFT
plan cache, per process per device) against `xp.device_free_bytes(dev)` ×
`FIT_MARGIN` (0.9). Free memory comes from the driver (`mem_info`), or `MemAvailable` for `cpu`, so
whatever else is on the card is already counted. Two load-bearing properties:
**the SMALLER card binds**, which no per-session cell count can express; and
**unknown free memory does NOT refuse**, because there the rail is the only
guard and guessing would be worse.
**IT RUNS AT EVERY ndim SINCE 2026-08-05.** It used to skip ndim=1 "because
`WIGNERF_MAX_GRID` already bounds a worker" — true of that var's DEFAULT (4096²,
~3.0 GiB) and not of the var, which is tunable to 16384 and which `wignerf.env`
sets to 8192, i.e. ~12 GiB/worker against an 11 GiB 2080 Ti. An unguarded 1D
session duly started and then killed a worker with a cupy OOM mid-run, the exact
outcome this guard replaces with a sentence at the door. Measured 192/104 B/cell
at ndim=1 (`bench.py --ndim 1 --footprint`, flat across sizes and unchanged by
`--relativistic`), so `config.bytes_per_cell` is keyed by ndim as well as
precision and `/api/device` reports both. **A gate can carry a stale REASON:
re-measure the claim it rests on, not only the risk it names** — the M1–M4 lesson
firing again. Pinned by `test_the_device_fit_check_is_the_operative_guard`.

**Its refusal describes the POOL, and the ROOMIEST device decides which of two
stories it tells.** If the per-worker footprint exceeds EVERY device's budget it
says *no device in the pool can hold even one*, names the roomiest with its
free/installed figures, and says dropping a variant or changing device will not
help — because neither will, and only the grid is left. Otherwise a worker does
fit somewhere, making it a DISTRIBUTION problem: it names the over-subscribed
device and points at the one with room, with a count. It used to name whichever
assigned device sorted first and close with "pick a device with more room",
which on the real pair implied a roomier card that could not hold one either. Pinned by
`test_the_fit_refusal_describes_the_POOL_not_the_first_device`.

**The IC preview is BOUNDED the same two ways** — by `protocol.grid_limit_error`
(the shared rail: a grid a session would refuse is one the preview must not
allocate either) and, before the CPU fallback, by `preview._cpu_fit_error`,
which asks `xp.device_free_bytes("cpu")` what `_fit_error` asks of a card. Both
are needed: the preview builds the FULL state at the requested grid and fires on
every form change, long before anyone presses Restart, while `_fit_error` runs
only at creation. The GPU path was never the hazard (`_pick_device` declines);
the CPU fallback is what needed the check. **`/preview/wavefunction` is behind the same rail**,
and its φ quadrature is chunked (`initial._phi_along`, `PHI_KERNEL_BYTES`)
because an (N_k × N_q) kernel written in one piece is quadratic in the grid.

**The IC preview runs on a GPU too, and hands the VRAM straight back**
(`routers/preview.py`). CPU-only was the right instinct and the wrong trade: the
preview is built at the SESSION's grid, so at 8192² it is the same 67M-cell
array the solver evolves — **25.9 s on the CPU against 0.50 s on the 3090**,
paid on every page reload AND every IC edit. (That asymmetry is the tell if it
regresses: big panel instant, small IC panel slow.) What matters is the
transient PEAK, so `_pick_device` takes the CUDA device with the most FREE VRAM
and only if the build fits with 1.4× headroom, GPU previews are serialized
(`_gpu_lock`) so two peaks cannot stack, and ANY failure falls back to the CPU.
The release works only because `_build_frame` keeps every device array in its
own frame, so they die on return before `free_all_blocks()`.

**Two things about that release are load-bearing.** **The preview allocates
from its OWN pool** (`_pool`, via `cupy.cuda.using_allocator`, which is
thread-local — previews run in starlette's threadpool). `free_all_blocks()`
acts on whichever pool it is handed and there is no per-backend allocator, so
releasing the process DEFAULT pool also returned the running workers' cached
blocks to the driver, on every IC keystroke — the exact opposite of what the
free-VRAM check is for. Isolation is free.
**And the failure path needs a SECOND release, after the `except` handler has
exited.** While an exception propagates, its traceback still references
`_build_frame`'s frame and every device array in it, so the `finally`'s
`free_all_blocks()` frees nothing — and a release *inside* the handler is no
better, the exception being live for the whole handler. Hence a client error
(`initial.ICError`) records `failed` and re-raises AFTER the release. Untreated,
a preview that OOMs at 8192² parks GiB on the card until the next SUCCESSFUL
one. Related: that handler logs `traceback.format_exc()` and deliberately NOT
`exc_info=True` — a LogRecord built with `exc_info` stores the traceback, and
any handler that retains records (pytest's log capture) then pins the frame
past the release.

Measurements — the per-cell footprint tables, the 88 B/cell preview figure and
the free/reserved readings behind every claim here — are in
`notes/memory-and-devices.md`.
## Configuration (environment variables, read by backend/config.py)

| Variable | Default | Meaning |
|---|---|---|
| `WIGNERF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda:N` \| comma list (`cuda:1,cuda:0`). Names the device pool; sessions spread variant workers across it. `auto` = all CUDA devices fastest-first if cupy imports, else CPU; a list's order IS the speed ranking. Indices are PCI order (match nvidia-smi). The **host default and an enforced POLICY**: `SessionCreate.device` (restart-only) may NARROW it per session but never widen it — a spec outside `xp.devices_allowed(WIGNERF_DEVICE)` (the pool **plus cpu**) is a 422 naming the pool, as is a malformed or absent one. It used to check only that the spec parsed and the card existed, so a host pinned to `cuda:1` (or to `cpu`, to keep its cards free) could be overridden by any client. `GET /api/device` returns BOTH `devices` (the pool) and `choices` — and `choices` IS `devices_allowed`, the same list the validator uses, so the Setup select can never offer a device the API refuses. That endpoint is where every HOST fact the form needs before it can create anything lives: the device lists, `WIGNERF_PRECISION`, and the per-ndim grid ceilings (`max_grid`, `max_cells`, `bytes_per_cell_2d`). Those sit OUTSIDE `_probe_backend`'s `lru_cache`, so they follow a monkeypatched `config` and ride the probe's error path. CPU is always a legal target but never appears in an `auto` pool on a CUDA host, hence the append. `resolve_devices` returns CANONICAL specs (a bare `cuda` → `cuda:0`), without which that membership test would reject a device the host does offer. |
| `WIGNERF_PORT` | `8010` | Backend port (8000 belongs to urantia-library). Used by start.sh; `uvicorn --port` otherwise. |
| `WIGNERF_PRECISION` | `float64` | Default spectral working precision (`float64` \| `float32`); the Setup form's **Compute** section overrides it per session (restart-only). float32 is a PREVIEW mode — see the float64/float32 gotcha and `notes/precision.md`. Do not make this `float32` on a host where anyone might read a result off it (setting it logs a WARNING; an unrecognized value falls back to float64 with one too). It reaches sessions through `SessionCreate.precision`, which is `Optional` and **resolved in `_check`, not by a `default_factory`** — a hard-coded literal there once made this var decorative, and a factory once refused sessions over a value the client never sent: a gate must refuse what was ASKED FOR. An omitted precision resolves to the host default at **every** ndim. **The SPA defers rather than guesses** (`lib/config.precisionForPayload`): until the user operates the control — or an IMPORT supplies one — the payload OMITS the field, so the host decides and the answer comes back in `status`. That is what makes the `/device` probe non-load-bearing. Two deliberate exceptions: an **imported** setup document marks the precision CHOSEN (`markPrecisionChosen`), because reproducing the run is what the document is for and without it the form showed a float32 that never happened behind a "restart to apply" no restart could clear; and a form with **auto-expand on** sends `float64` explicitly, because auto-expand is float64-only, so asking for it IS asking for float64. |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session (scrub/replay window). 32 GiB ≈ 4000 four-variant records at 1024², ≈ 64000 at 256². On the VPS set `16384`. The CEILING as well as the default: `SessionCreate.history_mb` may ask for less, never more, and status reports both `history_cap_bytes` and `history_mb_max`. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto (ncores/(2·n_variants), capped at 4). Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download. Under systemd (`PrivateTmp=yes`) the default is a private tmpfs — i.e. RAM, wiped on restart; point it at a disk path for long exports. Files are removed after download, on session close, at shutdown, or 30 min after finishing. |
| `WIGNERF_EXPORT_ENCODER` | `auto` | mp4 video encoder: `auto` \| `cpu` \| `nvenc`. **`auto` = `libx264 -preset veryfast`, even on a working GPU** — measured, the same 60-frame 1080p export takes 4.3 s either way and h264_nvenc writes 1.8× the bytes, because the bottleneck is frame RENDERING and the encode is a rounding error against it. `nvenc` selects the GPU encoder explicitly, as does `ExportSpec.encoder` per JOB. Its runtime probe used to encode a 64×64 clip, under NVENC's 145×49 minimum, and so answered "unavailable" everywhere — see `notes/export.md`. The right GPU path is the h264_nvenc ENCODER, NOT ffmpeg `-hwaccel` (a decode flag, irrelevant to rawvideo input). |
| `WIGNERF_EXPORT_WORKERS` | `0` | Export frame-render processes; `0` = auto (`min(cpu_count, 8)`; scaling flattens past the physical cores). Rendering dominates export time, so it is spread over a **spawn** `ProcessPoolExecutor` while one ffmpeg encodes the ordered stream. One export at a time (`_RENDER_LOCK`); a job below `max(2·workers, 16)` frames renders serially to skip pool warmup. |
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling — enforced at session creation AND for auto-expand doublings; tunable BOTH ways (schema rail: 16384). The UI's Nx/Np selects follow it — from **`GET /api/device`, per ndim**, NOT from `status`; see the `WIGNERF_MAX_GRID_2D` row for why. Lower it on VRAM-constrained hosts (`lib/config.axisFloor` clamps the 256 floor to the cap, and `setNdim` asks the same function). Measured peak per variant worker with the WHOLE-RECORD harness (`bench.py --footprint`): **192 B/cell in float64 and 104 in float32** — 0.19 / 0.75 / 3.00 / 12.00 GiB at 1024² / 2048² / 4096² / 8192², plus ~300 MiB of CUDA context per process per device. HIGHER than step-loop figures, and not a regression: a step loop misses `adjust_step`'s transient and the frame build. Workers spread over the pool, so what binds is the per-card share — 4 variants at 8192² is ~24 GiB/card at 2+2, which does **not** fit even the 3090, so cap by variant count and not just by grid. At the cap the session warns and keeps computing (moves still allowed). |
| `WIGNERF_MAX_GRID_2D` | `128` | Per-axis ceiling for **ndim=2** sessions. A sanity rail only — a 4D array grows as N⁴, so a per-axis cap is no guard at all. What binds is the per-device fit check, `routers/sessions._fit_error` (see the GPU section). **The UI's per-axis N selects follow this from `GET /api/device`, which reports every ndim's ceiling, NOT from `status`** — `status.max_grid`/`max_cells`/`bytes_per_cell` are resolved once for the ndim of the session that is RUNNING, while the form must describe the ndim it is SHOWING, and `dims` is restart-only so the two disagree until the restart. Reading them off `status` broke the panel in BOTH directions (a 2D form offering N up to 4096 with no footprint line at all, and a 1D form's select collapsing to one option). `lib/config.axisSizeOptions` is the extracted, unit-tested list — extracted because both bugs were reachable only through the DOM. **The list is FIXED per ndim: powers of two from `AXIS_N_FLOOR[ndim]` to this ceiling.** **Its 2D floor is 32, not 16**, because `boundary._band_mass` reports nothing below 32 cells per axis, so a 16⁴ session has no boundary watch and says so nowhere; 16⁴ stays reachable through the API. **`AXIS_N_FLOOR` is shared with `setNdim`**: a dims switch lands N *inside the target's list* — their own choice when offerable, else that ndim's DEFAULT, clamped by the target's cap. Capping from ABOVE alone left a HOLE in the select (1D→2D→1D ran a 1D session at 64²); falling back to the FLOOR is its quieter twin. Pinned in `config.test.ts`. |
| `WIGNERF_MAX_CELLS_2D` | `2**27` (134M) | **Total-cell** RAIL for ndim=2 — a cheap deterministic stop for absurd values, and the only guard on a host where free memory cannot be read. **Checked on an auto-expand DOUBLING as well as at create time since M3** — it was stored on the session and consulted by the planner nowhere, which was the accounting that milestone's gate had been hiding. It is deliberately NOT the operative limit: at the default it permits 22.0 GiB per worker, and a fixed cell count is wrong in both directions (refusing 128×128×64×64 on a 24 GiB card while permitting 5.5 GiB × 2 workers on an 11 GiB one). The real check asks the driver. Measured with `bench.py --ndim 2 --footprint`, which runs a whole worker record rather than a step loop: **176 B/cell in float64 and 96 in float32**, flat across sizes and identical for the relativistic variants — 0.17 / 0.87 / 2.75 / 6.71 GiB at 32⁴ / 48⁴ / 64⁴ / 80⁴ in float64, so **80⁴ is reachable only in float32**. **The STATE is only 5% of that** (W is real: 8 B/cell); the rest is the step's machinery at full shape. `config.BYTES_PER_CELL_2D` carries the breakdown and is **keyed by precision** because `_fit_error` reads it: a flat float64 figure would refuse precisely the grids float32 makes affordable. Throughput on the 3090: 610 steps/s at 32⁴, 130 at 48⁴, 35.1 at 64⁴, 13.8 at 80⁴ — **32⁴ is for exploration and 64⁴ is a serious run**. Both the rail's refusal and the fit check quote the estimate, and `/api/device`'s `bytes_per_cell` (per ndim, per precision) feeds the Setup panel's footprint line so a grid that cannot start says so BEFORE the restart — at either ndim, though the 1D line stays quiet under 0.5 GiB where the number decides nothing. |

## Commands

```sh
# CLAUDE.md budget — the file stops being loaded around 150k characters, and
# has crossed that line twice. Overflow goes to notes/, never deleted.
wc -c CLAUDE.md          # keep under 130000

# backend tests (GPU tests auto-skip without cupy/CUDA)
cd backend && .venv/bin/pytest

# live-server streaming smoke test (no browser); --ndim 2 exercises the 2D
# record path against a real uvicorn rather than a TestClient
.venv/bin/uvicorn main:app --port 8010 --ws-per-message-deflate false &
.venv/bin/python scripts/ws_smoke.py
.venv/bin/python scripts/ws_smoke.py --ndim 2

# throughput benchmark (add --precision both to reproduce the float32 speedup,
# -N to sweep other grids). --ndim 2 also reports the measured per-worker VRAM,
# which is what decides whether a 2D session starts at all
.venv/bin/python scripts/bench.py [cpu] [cuda:1]
.venv/bin/python scripts/bench.py --precision both -N 1024,2048,4096 cuda:1
.venv/bin/python scripts/bench.py --ndim 2 -N 32,48,64,80 cuda:1

# what an auto-expand DOUBLING costs: the transient peak as a factor on the new
# footprint, and how much of the old one the driver gets back. These are the two
# numbers core/fit.py budgets a 2D regrid with (REGRID_PEAK)
.venv/bin/python scripts/bench.py --ndim 2 --regrid --precision both -N 32,64 cuda:1
# ...and `--regrid move` is the other plan, the one the guard never refuses:
# there per_old == per_new, so the row asks whether the card sees the switch at
# all (float64: it does not — peak/steady 1.000, driver +0 MiB)
.venv/bin/python scripts/bench.py --ndim 2 --regrid move --precision both -N 32,64 cuda:1

# frontend: decoder golden test + typecheck + build
cd frontend && npm run test && npm run build

# mp4 export needs the system ffmpeg (libx264); its tests skip without it
ffmpeg -version

# dev loop: uvicorn (above) + `npm run dev`, open http://localhost:5173
# prod-style: ../start.sh, open http://localhost:8010
```

After changing the binary protocol: bump `VERSION` in BOTH protocol files,
regenerate the fixtures (`scripts/gen_fixture.py` writes TWO — `frame.bin`
(ndim=1) and `frame2d.bin` (ndim=2), both with deliberately anisotropic axis
counts so a transposed index cannot pass), and update the vitest, which runs
`describe.each` over both. A 1D-only golden would let a 2D decode bug ship.

UI debugging without touching the real display: drive the BUILT SPA with
headless Chrome via `puppeteer-core` (frontend devDep; system Chrome at
/usr/bin/google-chrome, flags `--no-sandbox --disable-gpu`). Seed
`localStorage['wignerf.cfg']` BEFORE the first navigation to choose the grid —
the wrong key looks like it worked and silently measures the default. The
series plots expose `window.__wfSeries.<which>()` and element screenshots of
`.wf-plot` reveal what uPlot painted — how the "flat purity line camouflaged on
a gridline" bug was found.
`window.__wfPerf.snapshot()/reset()` (lib/perf.ts) exposes frame-pipeline
counters: received/painted rates, MiB/s, queue drops, per-stage avg ms
(decode/upload/draw/plots/fanout), the GL renderer string (SwiftShader
here = software rendering, the classic cause of few-fps playback at large
grids) and the measured refresh interval. **Two things worth asserting from a
script rather than eyeballing**: the plot TITLES (they are the fastest check
that the frontend and the backend agree about ndim — `ρ(x) = ∫W dy dpx dpy`
against `ρ(x) = ∫W dp`), and the Setup column's real width — screenshot the
`<aside>` and assert nothing's rect exceeds it AND that
`scrollWidth === clientWidth`, which is what caught nothing at 320px after the
Grid section became a four-axis table.

**Do NOT trust a headless SCREENSHOT of a WebGL canvas, and never conclude
anything from one alone.** Under SwiftShader a canvas that redraws every frame
screenshots correctly while one that painted once comes out blank — measured
side by side, and the blank one's drawing buffer read back 8% coloured pixels,
i.e. the content was there all along. A whole "the IC preview is broken"
reproduction was that artifact. The reliable oracle is `gl.readPixels` into a
counter, and the reliable context probe holds the object **captured at
getContext time**: re-querying `canvas.getContext('webgl2')` later returns null
on a live context — which also means readback is UNAVAILABLE from outside the
renderer, so verify a panel through what it reports (see
`notes/downsampling.md`) rather than through its pixels.

**And the real bug that hunt uncovered: a remounted WebGL canvas leaks its
context until GC, and browsers cap live contexts (16 in Chrome) by silently
killing the OLDEST.** `WignerRenderer.dispose()` therefore calls
`WEBGL_lose_context.loseContext()`; deleting the textures and program is not a
substitute, the same way `gc.collect()` rather than refcounting is what frees a
closed session. Restart bumps `plotsKey` and remounts PanelGrid, so every
Restart built a fresh context per panel — SIX at ndim=2 against 1D's one — and
after two or three Chrome reclaimed the page's oldest, the IC editor's. Nothing
reported it: `webglcontextlost` fired on a component that never listened and
every later GL call succeeded as a no-op, so the IC preview was blank for the
rest of the page's life and a reload "fixed" it. Measured across six Restarts:
detached-but-alive contexts 2, 4, 6, 8, 10, 12 without the release, 0 with.

## Roadmap (v2, agreed 2026-07-19)

1. **Destructive forking**: resume computation from ANY record (end or
   intermediate; the abandoned branch is discarded), both modes. Requires
   periodic float64 checkpoints alongside the uint16 display history — a
   quantized frame must NEVER seed a propagator. "Continue past t2" is the
   fork-at-the-end special case.
2. **Save/load the whole simulation** to disk (config + history +
   checkpoints; own format, no legacy compatibility).
3. ~~Multi-GPU~~ — DONE 2026-07-19: variant workers spread across the
   `WIGNERF_DEVICE` pool (see GPU section); measured +41% (4 variants)
   and +39% (2 variants) at 1024² on the 3090 + 2080 Ti pair.
4. ~~mp4 export~~ — DONE 2026-07-21: backend-rendered export of any
   computed range (see the mp4 export bullet above). Later: Lindblad
   dissipation (the propagator's exponent construction is deliberately
   modular for it).
5. ~~**2D space / 4D phase space**~~ — FIRST CUT DONE 2026-07-26: `ndim ∈
   {1,2}` through one generic core, 2D streaming the six pairwise 2D
   projections instead of the 4D array (see the two Architecture bullets at
   the top). **COMPLETE as of 2026-08-01**: its four deferred follow-ups
   (M1–M4) have all landed and no 2D-only gate remains — their retrospectives
   are in `notes/2d-milestones.md`, their operative findings in the gotchas
   below. M5–M7 were enhancements rather than gates, and M7 landed 2026-08-02
   (one exponent slot, −32 B/cell at both dimensionalities), leaving M5 and M6.

## 2D follow-up milestones (deferred from the first 2D cut, 2026-07-26)
**M1-M4 and M7 have all landed, and nothing in the codebase refuses anything on
the grounds of `ndim == 2` any more.** What each cost is in its gotcha below;
the full retrospectives are in `notes/2d-milestones.md`. What is left (M5, M6)
are enhancements that simply do not exist yet at either dimensionality — no
refusal to relax, no half-feature to mistake for a working one.

Three lessons from retiring those five, kept because the next milestone will
need them: **the verification is the work and the gate removal is a few lines**
(the physics core was already generic every time, and M1's measurement
contradicted its prediction in both directions at once); **look for the
accounting the gate was hiding**, since each milestone's real work turned out to
be a constant nobody had listed (`BYTES_PER_CELL_2D`, `RangeStats.scale`,
`session.max_cells`, and M7's `fit.REGRID_PEAK`, which got RELATIVELY worse as
the worker got smaller) — but expect false positives there too, and settle them
by measuring at the point the code actually runs, not by reasoning from two
documented measurements; and **a gate can carry a stale REASON**, so re-measure
the claim it rests on and not only the risk it names (the 1D fit check, below,
is that lesson firing a second time).

| # | Milestone | What it needs |
|---|---|---|
| M5 | **Cuts / slices** | The wire reserves a per-plane `mode` byte and only `mode=0` (projection) is defined. Projections are EXACT for separable states but average away fringe contrast for entangled ones, precisely the interesting 2D regime; a cut at fixed (y, py) keeps the interference. Purely additive — a new `mode` value, no version bump, no new reduction cost (a cut is cheaper than a projection). |
| M6 | **FFT fusion** | The trailing inverse transform of step *n* and the leading forward transform of step *n+1* are inverses; staying in λ-space across step boundaries and merging the two half-`expT`s removes 4 of the 12 one-dimensional sweeps per 2D step, i.e. **+50%**. Not free: the per-step `real()` projection becomes per-record, changing numerics (and `test_time_reversal`'s ~1e-9 residue budget), so it needs its own verification pass at both ndim. |
## Conventions / gotchas

- Do not reference the old project website domain anywhere in wignerf —
  it expired (old code/comments elsewhere in the repo may keep theirs).
- Nx, Np must be even (shader unshift + fftshift symmetry); powers of 2
  for FFT speed. Grid warns, API schema enforces evenness.
- **A failed API call goes through `lib/apierror.apiErrorText`, never
  `data.detail` directly.** FastAPI's `detail` is a STRING for the
  `HTTPException`s we raise, but an ARRAY of pydantic error objects for any
  body-validation failure — so `detail ?? String(e)` rendered a schema refusal
  as the whole raw blob: `type`/`loc`/`ctx` plus a verbatim copy of the entire
  request `input` (grid, IC, every component), with the one readable sentence
  buried in the middle of it. The refusal messages in `core/protocol.py` are
  written to be read by a person; the helper strips pydantic's "Value error, "
  prefix, names the field for per-field errors, and de-duplicates. Used by both
  `SimulatorView.restart` and `ExportPanel`.
- **Live numeric readouts get FIXED decimals in a FIXED-width field** — the
  control bar's t/E (`.wf-fixnum`, tabular-nums) and the exported frames'
  header (`%*.3f` + a monospace family, widths from the export's own t
  range; a.u. and fs both at 3 decimals, same as the screen).
  `toPrecision`/`%g` print a different number of decimals as a
  value grows (0.02419 → 0.2419 → 2.419 fs), so the text changes length and
  everything after it slides sideways on every frame.
- Physics invariants in `tests/test_propagator.py` are the correctness
  anchor — harmonic quantum ≡ classical (Moyal terms vanish for quadratic
  H) is the strongest single check; run them after touching propagator,
  grid or fftshift bookkeeping. **`tests/test_propagator2d.py` is the 2D
  anchor** and its strongest check is different: see the next entry.
- **The multi-D Bopp shift is ONE simultaneous shift of every spatial
  argument, and quantum ≡ classical CANNOT detect getting it wrong.**
  `qd(f, xs, dxs)` evaluates `U(x − ħθx/2, y − ħθy/2)` — both arguments moved
  together, same sign, index-matched — which is what the 2D Moyal product
  gives. The plausible wrong version, a SUM of two independent 1D differences,
  agrees with it for **every quadratic U**, cross terms included, so
  `test_quantum_equals_classical_with_a_cross_term` passes under it; the two
  first differ at third order in mixed derivatives (for U = x²y by exactly
  −2a²b with a = ħθx/2, b = ħθy/2). What discriminates them is
  `test_matches_an_independent_schroedinger_run`: evolve ψ(x,y) with an
  ordinary split-operator TDSE — a different method sharing nothing but
  numpy's FFT — under a COUPLED potential (Hénon–Heiles), and compare the
  streamed (x,y) plane against |ψ|² cell by cell. Relative error at t = 1:
  correct shift **1.45e-4 at dt = 0.02 and 3.43e-5 at 0.01 (ratio 4.22,
  O(dt²))**; independent shift **7.83e-3 and 7.81e-3 (ratio 1.003, FLAT)**.
  228× worse and dt-independent, because a wrong shift is a different evolution
  OPERATOR, not a smaller time step away from the right one. So the assertion that matters is the dt RATIO, not a tolerance — do not
  "simplify" it to a single run. And do not extend the sweep to dt = 0.005:
  the ratio falls to 1.9 there because the residual has reached a
  dt-independent ~1.5e-5 floor set by the GRID (the Bopp shift samples U
  outside the box on the discrete θ lattice, which `exp(-iU dt)` never does),
  so the convergence check only means something where splitting error
  dominates.
- **The other 2D anchors, and what each is for.** Separability: for
  U = Ux(x) + Uy(y) and a product IC, W_2D(t) must equal the outer product of
  two 1D runs — a separable U makes dU a SUM, so the exponent factorises
  exactly, validating the whole 4D pipeline against code the 1D suite already
  trusts (5.9e-11 of the peak after 150 steps, pure `exp(a+b)` vs
  `exp(a)exp(b)` roundoff). ⟨Lz⟩: conserved by a central U to 5.65e-6 over 300
  steps and off by 3.05 (150%) for an anisotropic one — a 5e5 separation, and
  the drift is IDENTICAL for the quantum and classical variants, the tell that
  it is the square lattice breaking rotational symmetry rather than physics. And a reductions-vs-naive test
  over all six planes and four marginals, which is what catches 4-axis
  fftshift bookkeeping — the likeliest porting error after the shift pairing.
- **RELATIVISTIC 2D (`qr`/`cr`, M2)**: the physics core was already generic —
  `_kinetic()` builds T = c√(Σkᵢ² + m²c²) and its gradient at any ndim, and the
  streaming/observables/frame paths never cared. Four rules survive it;
  measurements in `notes/2d-milestones.md`.
  **SEPARABILITY IS UNAVAILABLE HERE, and the natural test would fail against
  correct code.** `test_separable_run_equals_two_1d_runs` needs T separable as
  well as U, and c√(px²+py²+m²c²) is not a sum — a 2D relativistic run is
  genuinely NOT the outer product of two 1D relativistic runs. Since
  separability is what validates the 4D pipeline against trusted 1D code, the
  replacement is `test_relativistic_matches_an_independent_schroedinger_run`,
  and what it asserts is the **dt RATIO**. Do not lower c to make it "more
  relativistic" — the residual meets a dt-independent ~1.5e-5 GRID floor.
  **Never assert quantum ≡ classical for qr/cr**: the Moyal corrections vanish
  for a quadratic U, but T is not quadratic in k, so the kinetic Bopp difference
  and the gradient genuinely differ. ⟨Lz⟩ DOES transfer — T depends on the
  momenta only through |k| — and `test_angular_momentum` extends unchanged; it
  is the one existing 2D anchor that does.
  **The mc² cancellation does NOT worsen in 4D** (it is m²c²·eps, a couple of
  ulps, and that does not care how many momentum components enter the sum), and
  exponent construction stays double at every ndim, verified bitwise.
  **Relativistic is FREE in memory and in time**, so `BYTES_PER_CELL_2D` did not
  move — a √ over meshes that already exist costs nothing and the FFTs are still
  the whole cost. The uncertainty-shear diagnostic works at **c = 10 and not
  c = 137** (shear goes as 1/c⁴, so 137.036 needs ~1200 steps, which at 32⁴ is
  not a test).
- **2D mp4 EXPORT (M4): the frame is a SELECTION, real subscripts are FREE, and
  the blit decides where static art may live.** **Read `notes/export.md` before
  editing `core/render_mpl.py`**; it carries the measurements and the
  archaeology. The rules:
  **MATHTEXT IS USED** (the figure BLITS, so titles and labels are static
  artists baked in once and free in situ) but **usetex is NOT** — 12×, needs a
  LaTeX install the VPS lacks, cannot render our Unicode, and is global, so the
  user's own U(x) would go through it. `axes.sub_math` typesets the two-letter
  axis names as `$p_x$`; ∬, ⨌, γ, ℏ, ρ, φ, ⟨⟩ stay the Unicode the screen uses,
  so the two cannot drift and 1D is untouched at the byte level. It has to be
  the WHOLE frame — half the job puts the same axis on one screen in two
  spellings.
  **The metadata block is WRAPPED PLAIN and typeset afterwards** (`_emit`),
  because `$p_x$` is five characters drawing as two glyphs; it must not break
  MID-FACT (groups joined with `_NB`, a character `textwrap` cannot see as
  whitespace at all — a Unicode NBSP is `\s` and fails) and `break_on_hyphens`
  is off, a minus sign not being a hyphen. Test it PER LINE.
  **U(x,y) is typeset by a LEXICAL rewrite of the user's own string**
  (`describe.potential_math`), NOT `sympy.latex`, which CANONICALISES — the
  block would stop being "the text you paste back" — and emits tall `\frac`.
  **A typeset line built from USER input cannot be the only defence** (a
  mathtext parse error raises at DRAW time), so every candidate goes through
  `render_mpl.mathtext_ok` and falls back to plain per line. **AN IC EXPRESSION
  IS NEVER TYPESET AT ALL**, and says so by returning `None` rather than a copy
  of the plain line: `_emit` skips substitution only on its single-fragment
  branch, so once a line WRAPPED a user's literal `px` became `$p_x$`.
  **The header readout and the block deliberately DISAGREE**: the header's
  geometry follows the PAINTED record as the SPA does; the block's is labelled
  "at record k0" and stays.
  **THE BLIT DECIDES WHERE STATIC ART MAY LIVE.** `update()` restores the static
  background then `draw_artist`s the images on top, so anything static drawn
  INSIDE the axes box is painted over every frame — hence the panel grid lines
  are in `_dynamic`, ordered after the images, and the per-panel scale caption
  (past `CBAR_MAX_CELLS` = 8 panels) lives in the panel's `loc="right"` TITLE,
  outside the axes box. Its first version sat in the heatmap's corner and
  rendered as NOTHING.
- **AUTO-EXPAND IN 2D (M3): the orchestration was free and the GUARD is the
  whole feature.** `core/fit.py` is the create-time check asked again mid-run,
  SHARED with `routers/sessions._fit_error` so the two cannot drift (each writes
  its own message: at create time the advice is "drop a variant, change device"
  and mid-run none of that is available). **Read `notes/2d-milestones.md` before
  editing `fit.py`, `worker._apply_regrid` or `Propagator.set_grid`** — it has
  the measurements. Five things that must not be undone:
  the per-device inequality is `n·per_new·REGRID_PEAK ≤ (F + n·per_old)·
  FIT_MARGIN`, and **the `+ n·per_old` term is load-bearing, not a refinement**
  — it is what correctly allows a single-worker 64⁴ doubling on a card that is
  ALREADY holding the worker wanting to grow, which `F` alone refuses (the
  figures moved with M7 and the shape did not; recompute them, do not carry
  them);
  **it is asked of a DOUBLING and nothing else** — for a pure window shift
  `per_new == per_old` and the same expression would demand free memory to slide
  a window that allocates nothing, so `fit.regrid_shortfall` returns `[]` for
  any non-growing window BEFORE it reads the driver; the reading is taken ONCE
  per attempt, since a re-read between candidates could accept a plan neither
  reading allows;
  **release before allocate** in `worker._apply_regrid` (exponent slots first,
  then the state through a one-element box so `_run`'s local does not pin it,
  then `set_grid` dropping meshes, FFT plans and this thread's cuFFT cache) —
  what makes `REGRID_PEAK` = 1.10 honest, and ALLOCATION ORDER ONLY, so ndim=1
  stays bitwise unaffected;
  **what the guard cannot see is ACCEPTED** — another process, a sibling session
  whose committed plan has not landed, the bigger unguarded transients
  elsewhere; `FIT_MARGIN` does not cover them and must not be read as if it did.
  What makes that acceptable is the failure mode: a lost race is a cupy OOM
  inside `_apply_regrid`, fatal by design, so the session pauses LOUDLY (pinned
  by `test_a_failed_regrid_stops_the_run_instead_of_going_quiet` — the day that
  goes silent the limitation stops being acceptable);
  and **`no_room` is its own boundary ACTION with a `limit` FIELD**, not a
  flavour of `capped`. `capped` = the per-axis `WIGNERF_MAX_GRID` ceiling,
  `no_room` = the guard made us give a doubling up, and their remedies are
  opposite. It carries `denied` and `applied`, because a denied doubling can
  still leave a shift to commit. `limit` ∈ {`device`, `cells`} because against
  the `WIGNERF_MAX_CELLS_2D` rail, freeing the card does nothing and neither
  does float32 (a cell count is precision-independent). It is a MESSAGE latch,
  like `_capped_posted`, and NOT a scheduling gate like `_invalid_posted`: as a
  gate, one refusal switched auto-expand off for the rest of the run, including
  the pure shifts that were never the problem. Do not re-propose gating
  `SUPPORT_EPS` on the 2D noise floor: measured and refuted.
- **FLOAT32 IN 2D (M1): choose it for MEMORY, not for speed, and know that it
  raises the boundary warning on a contained state at 32⁴.** The mixed scheme
  has no dimension-aware line in it and holds BITWISE at 4 axes, which had to be
  checked because the multi-D Bopp shift moves every spatial argument of U
  together; pinned by `test_exponent_construction_stays_double`, parametrized
  over ndim. **MEMORY: 96 B/cell against float64's 176** — 55%, flat across
  32⁴–80⁴, identical for the relativistic variants, and 80⁴ becomes reachable at
  all. **SPEED: 1.48× at 64⁴, not the ~3.4× predicted from 1D** — the 2D step
  transforms two axes of a 4D array where 1D transforms one axis of a 2D array,
  and cuFFT's single-precision advantage for that strided layout is far smaller.
  Do not quote the 1D figure for 2D. **`TOL_MIN_F32` did NOT move**: the
  `adjust_step` residual floor SATURATES rather than growing with cells.
  **At 32⁴ it puts REAL mass in the edge band and latches the boundary warning
  within ~7 s** — 1.0e-3 of the integral against 3.1e-5 for the IDENTICAL state
  in float64; 48⁴ stays clear and 64⁴ flickers, so it is the COARSE grid that
  cannot carry single precision, not 2D as such. **No threshold was moved**,
  deliberately: the mass is genuinely there, it GROWS with step count so any
  fixed threshold is outrun by a long enough run, and one high enough to be safe
  would sit past the point where real wrap does damage. What changed is the
  WORDS — `SimulatorView.boundaryTitle` names single precision as a cause in a
  float32 2D session, because the standing remedy ("restart with a larger
  domain") is the one thing that cannot help here. Pinned by
  `test_float32_in_2d_moves_real_mass_to_the_edge_band`; numbers in
  `notes/2d-milestones.md`.
- **EXPRESSION INITIAL CONDITIONS (`wexpr`, `psi`, 2026-08-04): the transform is
  four lines and every trap is in what the diagnostics CANNOT see.** Full
  measurements in `notes/expression-ics.md`; `core/expr.py` is the ONE security
  boundary all three kinds of user expression go through.
  `initial.psi_wigner` builds W(q,k) = (1/2π)^n ∫dⁿθ ψ*(q + ħθ/2) ψ(q − ħθ/2)
  e^{ik·θ} on the θ lattice `Grid` already owns. Per momentum axis: multiply by
  the ramp e^{i k₀ θ}, inverse DFT, multiply by (−1)^m, scale by 1/d[k].
  **THE RAMP IS NOT OPTIONAL AND NOT A NON-SYMMETRIC-BOX SPECIAL CASE** — k₀ is
  `float(grid.v[a][0])`, the first LATTICE value (not `lo`, which can differ by
  an ulp on a regridded axis), and it is never 0 for a box straddling the
  origin. Dropped, the relative error is **1.0** on a symmetric box while the
  norm stays exactly 1.000000, so only a cell-by-cell comparison against the
  analytic form can see it (`test_the_momentum_ramp_is_not_optional`).
  **THE MOMENTUM BOX CANNOT PRODUCE A NORM DEFICIT, AND NOTHING ELSE SEES IT
  EITHER.** The transform is exactly N_k-periodic in the momentum index, so
  content outside the box ALIASES back in rather than being lost. Measured on a
  packet at p₀ = +2 with a momentum box excluding its mean momentum entirely:
  norm 1.0000000000, purity 1.000000, edge band 6.5e-06 — every scalar
  diagnostic perfect on a completely wrong state, while `_psi_lattice`'s
  direct-quadrature **momentum mass** reads 4.8e-16. It is the only detector
  there is; do not remove it because "the edge band covers that".
  **ψ IS NORMALISED OVER THE EXTENDED SPATIAL BOX**, so ∫W dμ is the in-box mass
  fraction and the norm deficit keeps the meaning it has for the Gaussian kinds;
  normalising over the visible box makes it a structural zero. The extension is
  capped at `NORM_PAD_MAX` box widths per side because the Bopp half-width is
  ħπ/(2dk) and GROWS as the momentum axis is refined — at 1D 4096² it is 402
  against a box of 8, and the φ quadrature's kernel asked for 13.7 GB.
  **THE NORMALIZER IS BOUNDED FROM BOTH SIDES, AND THE UPPER BOUND IS THE ONE
  THAT IS EASY TO FORGET.** Every sample can be finite while the total is not —
  `exp(-x^2/2)*exp(360)` overflows only when SQUARED — and dividing by inf was
  silent both ways: a `wexpr` came out identically ZERO (blank panel, ∫W = 0, no
  warning) and a `psi` all-NaN, which **no diagnostic sees either, because every
  one is a `>` comparison and NaN fails all of them**. Reachable in eighteen
  characters, precisely because these kinds auto-normalise. The lower test is
  RELATIVE for the same reason — an absolute floor refused
  `1e-15*exp(-x^2-p^2)`, a perfectly good state. `psi_wigner` also checks `isfinite(W)` as `wexpr_wigner`
  does: `_psi_lattice` validates ψ only on the CAPPED lattice while the build
  evaluates it over the full extended reach.
  **THE PROBE DTYPE MUST BE THE ONE THE BUILD USES, and the two kinds differ.**
  ψ is evaluated at complex-dtype arguments inside the transform, so `sqrt` of a
  negative real is a finite branch value there; a W is sampled on
  `grid.nat_mesh`, which is float64. Probing a W complex accepted
  `sqrt(x)*exp(-p^2)`, and `POST /sessions` then answered 200 and killed four
  worker threads — the exact failure compiling at the door prevents.
  **A ψ IS NEVER ACCUSED OF NOT BEING A QUANTUM STATE.** A wavefunction always
  defines a valid pure state, so γ > 1 there can only mean the grid is aliasing
  it — same detector, different sentence, the move `boundaryTitle` already makes
  for float32. Its tolerance is looser too (1e-4 against 1e-6).
  **BOTH BUILDS ARE BLOCKED AGAINST FIXED BYTE BUDGETS**, which is what lets
  `PREVIEW_BYTES_PER_CELL` stay one number per ndim: an arbitrary expression's
  unblocked transient grows with its own tree width, so there is no honest
  constant for it.
  **`initial.ICError` is its own class** so `routers/preview.py` can tell a
  client error from a cupy OOM. Most of these are only knowable once W exists,
  so they are raised from inside the build — where the CPU fallback would
  otherwise retry them, fail identically and surface as a 500. That means the
  GPU handler must defer its pool release to `failed` like the OOM path does
  (a release inside a live `except` frees nothing) and must NOT fall through to
  the CPU. Tests assert the CLASS and COUNT the builds, or neither property is
  pinned.
  The IC is compiled ONCE in the router and threaded through
  (`session.compiled_ic`), like `compiled_potential`: `worker._run` runs per
  variant, so compiling there is four threads parsing one string through sympy's
  global caches — and a bad expression must be a 422 at the door. **The GRID
  rail is checked first**, in both routers, so one body cannot get two different
  errors depending on which endpoint it reached (the property
  `grid_limit_error`'s docstring is about) and an over-the-rail grid does not
  pay for a sympy parse and a 33⁴ probe first.
  **CARRYING BOTH SHAPES IS LEGAL, AND THE FOREIGN ONE IS NORMALISED AWAY
  RATHER THAN REFUSED.** The editor holds a default for every tab at once — that
  is what makes switching tabs non-destructive — so a form on the `wexpr` tab
  genuinely has Gaussian components behind it, as does every stored config and
  setup document written before `expr` existed. `ICSpec._check` requires the
  kind's own field and DROPS the other, keeping the dead one out of
  `describe.setup_document` and off `SessionCreate`'s per-component ndim loop.
  `lib/config.icPayload` sends only the active kind's shape too — belt, and the
  schema is the braces. **Refusing the foreign field was shipped and instantly
  broke a cold start**: with local data cleared, selecting the W(x,p) tab
  answered *"an IC of type 'wexpr' … carries no Gaussian components (got 1)"* —
  a schema telling a client its own defaults were invalid. A rule that only
  holds once every client has been rewritten is not a schema rule.
- **MASSLESS (m = 0) relativistic runs lose purity to the |k| KINK, and it is
  the GRID not the step.** m = 0 became reachable in 2D only with M2 (the schema
  requires exclusively relativistic variants there, since non-relativistic
  T = p²/2m diverges). The gradient c·kᵢ/|k| is 0/0 at the origin — which IS a
  lattice point, a symmetric box with even N puts an exact 0.0 on every axis — so
  it is defined as 0 there. That is not a new 2D convention: at ndim=1 it returns
  `c·sign(p)` BITWISE, because `sqrt(k*k) == |k|` exactly for every finite
  lattice value and `sign(0)` was already 0. Pinned by
  `test_the_massless_gradient_reduces_to_the_1d_convention`.
  What massless costs is purity, because T = c|k| is not smooth at the origin and
  its Bopp difference has slowly-decaying Fourier content in λ that a finite
  lattice truncates. Over 100 steps at 32⁴: **m=0 quantum −7.19e-6 at dt = 0.01
  and −6.99e-6 at dt = 0.005 (dt-INDEPENDENT)**, against m=1 c=1 at −3.27e-9.
  Halving dt does not help; refining the MOMENTUM grid does — **7.19e-6 at N=32
  falls to 7.16e-7 at N=48**, ~10× for a 1.5× refinement. So the remedy for a
  clean massless run is a finer momentum axis, never a smaller dt. Norm stays at
  machine precision throughout, which says the map is still exactly unitary and
  it is the RESOLUTION of the kink that is lossy. Only the
  CLASSICAL variant reaches the gradient at all; the quantum one differentiates
  T through `qd()`.
- **Always run uvicorn with `--ws-per-message-deflate false`** (start.sh
  does). uvicorn's default permessage-deflate zlib-compresses every
  multi-MiB frame bundle on the asyncio event loop and caps the stream at
  ~10-25 records/s — measured 12x slower than uncompressed on localhost
  (browsers silently negotiate the extension, so the slowdown looks like
  a rendering problem; `__wfPerf` showing tiny stage times with a low
  received_per_s is the tell).
- pyFFTW plans are per-`ArrayBackend`-instance and must not be shared
  across threads; each worker owns its backend.
- Relativistic variants: mc² cancels inside the propagator; observables
  subtract it from displayed E.
- **The solver is float64 BY DEFAULT and float32 only when explicitly asked
  — and the difference was measured, not assumed.** `SessionCreate.precision`
  (`float64` | `float32`, host default `WIGNERF_PRECISION`) is restart-only and
  picks the SPECTRAL working dtype. float32 must never be the default and never
  the setting a physics claim is made from; the UI badges it permanently and
  every exported mp4 says so on its own metadata line.
  **Read `notes/precision.md` before touching any of this** — it carries the
  bench figures, the mixed-scheme derivation and the two invisible failure
  modes behind every rule below.
  - **What it costs**: complex64 stepping destroys the diagnostics this project
    navigates by — over 2000 steps at 256², Δpurity −2.4e-4 and ΔE +9.4e-4, both
    SECULAR from a perfectly contained state, i.e. the boundary-wrap signature.
  - **What it buys**: 1D **3.3-3.8× on CUDA**, 2D only **1.5-2.6×** (see the
    float32-in-2D gotcha — do not quote the 1D figure for 2D), and **nothing at
    all on CPU**. Memory ~54%, not 50%.
  - **It is a MIXED scheme and that is REQUIRED, not a compromise.** Only the
    spectral working array and the exponent PHASES are single; the grid meshes,
    both `qd()` evaluations, `dU_im`/`dT_im` and `H` stay float64. Relativistic
    `dT` built in float32 has 200% error (mc² cancels inside a difference of
    ~1.9e4-magnitude terms), and keeping construction double is what lets
    `_rate_mesh`'s 1e-13 gate and the frozen-lattice regrid arithmetic stay
    exact with no dtype-scaled tolerances anywhere. It is also FREE — the FFTs
    are the cost. `test_precision.py` asserts the rate meshes are BITWISE
    identical between the two modes.
  - **Two failure modes are invisible in RESULTS, so they are pinned by DTYPE
    assertions**: a complex64 array handed to a complex128 pyFFTW plan is
    silently upcast, and `B *= expT` across dtypes silently allocates a
    complex128 temporary. Both give the right answer at the wrong speed, so no
    physics assertion can catch either. Hence `fft_pair` takes an explicit dtype
    and `exponents()` casts.
  - **float32 REFUSES auto-expand, and `tol` below 1e-5** (`protocol.py`
    `MSG_EXPAND_F32` / `MSG_TOL_F32`, enforced at create AND on the live
    ParamChange path, because both fields are reachable live). Auto-expand
    because single-precision noise passes its own detector and the planner would
    size a new domain from it; `tol` because `adjust_step`'s residual has a
    float32 floor of ~7.4e-7 against float64's 1.6e-15, below which the
    controller never converges. Detection still WARNS, on a raised threshold
    (`boundary.EDGE_THRESHOLD_BY_PRECISION`, 1e-4).
    **Both refusals are also enforced in the FORM**:
    `lib/config.applyPrecisionInvariants` clears `auto_expand` and raises `tol`
    to `TOL_MIN_F32` (the frontend mirror of `protocol.TOL_MIN_F32` — move both
    together), and the Setup panel disables the checkbox and lowers the tol
    input's `min`. The config-level invariant is the load-bearing half: the
    panel can be unmounted, and `probeHost`/`mergeConfig` reach the same
    combinations from outside it.
    **How the gates are EXPLAINED is a settled three-part pattern**: a compact
    permanent marker in the label that costs no line ("auto-expand (f64)",
    "tol ≥1e-5"); the full reason in the control's `title`; `:disabled` on the
    control itself wherever the value is not merely discouraged but overridden;
    and the reason ONCE in amber (`f32Applied`) while `runDiffers('precision')`
    holds. **A standing paragraph is not part of it** — see
    [no-mystery-disabled-controls].
    Clearing `auto_expand` in the form is NOT enough on its own, because it
    applies LIVE: `SimulatorView` watches `cfg.precision` and sends
    `auto_expand: false` to a running session, since the status→form watcher
    cannot (`status.auto_expand` does not CHANGE, so it never fires) and the
    checkbox is by then disabled.
    **And the invariants must be applied SYNCHRONOUSLY at the point of change,
    never from a watcher.** `SetupPanel`'s precision select calls
    `onPrecisionChange` directly: a child's setup runs during the parent's
    render, so `SimulatorView`'s own `cfg.precision` watcher holds a lower id and
    runs FIRST — and on a fresh session it restarts inside that same flush,
    serializing a config the panel had not fixed up yet. `payload()` therefore
    calls `applyPrecisionInvariants` too, as the last place self-consistency can
    be guaranteed regardless of which watcher ran first.
  - **Do NOT "optimize" `exponents()` by casting the ANGLE instead of the
    result.** `exp(1j*θ).astype(complex64)` is safe for any finite θ because the
    modulus is 1; `exp(1j*θ.astype(float32))` is NaN for θ ~ 1e91, which a steep
    U on the extended Bopp range reaches at large grids — and `worker._finite`
    checks the float64 rate meshes, so nothing would see it.
- **ONE exponent slot, because every committed substep inside a record is the
  same size (M7) — and the CACHED PLAN is what makes that true.** `_substep`
  divides what is left of a record into `n = ceil(|rem|/|dt|)` EQUAL steps of
  `rem/n`, so the straggler that needed a second slot is gone: −32 B/cell at
  both dimensionalities. Four rules, measurements in `notes/2d-milestones.md`:
  **Adaptive retuning is a NON-COMMITTING boundary probe** — `adjust_step`
  selects the cap before a record and its trial state and pair are DROPPED, so
  the cached pair always belongs to `_substep`'s quotient. Storing the pair, as
  the old code did for free, resurrects the second slot once per adjust.
  **`_substep` caches its plan against `t_tgt`, and without it the milestone
  BACKFIRES**: `rem` shrinks as the record is walked, so recomputing `rem/n`
  returns sizes differing in the last ulps — distinct float keys, so the one slot
  misses on nearly every step (5 sizes and 22 rebuilds over three records,
  against one cached pair). A stale plan is as wrong as a stale exponent, so
  `_exp_clear` drops both together; every site invalidating one invalidates the
  other.
  **Because the size is cached, `_advance` iterates on the substep COUNT and
  must never go back to summing toward τ_k.** The pre-M7 loop exited on
  `|τ_k − t| > eps` because its last step was clamped and therefore
  self-correcting; a cached size has no such step, so once n accumulations land
  further than eps from the target the loop takes ANOTHER full substep and
  marches past τ_k forever — 11.5 million substeps for a record wanting 50000,
  an unkillable spin with no error and no emitted record. Pinned by
  `test_a_record_that_needs_many_substeps_still_terminates`.
  **M7 buys MEMORY, not speed** — the production rebuild count is at most one
  per record either way; do not let the halved slot count imply a halved rebuild
  rate.
  **What the physics suite cannot see**: `test_propagator*.py` and
  `test_precision.py` drive `Propagator` through a private fixed-dt `evolve()`
  helper, so they never enter `_advance` and would be just as unchanged by a
  scheduler that never landed on τ_k. `tests/test_substep.py` exists for that
  gap, and it caught the missing plan cache.
- **`close()` must tear the streamer down, or the whole FrameHistory leaks.**
  The `ws_endpoint` coroutine holds `s` (hence its entire history) as a local;
  `close()` pops the session from `SESSIONS` and stops its workers but the
  streamer must be ended too, or it keeps the session fully resident —
  invisible to the TTL sweeper (already gone from `SESSIONS`) and surviving
  its own workers' death (tens of GB stranded at 8192² on 2026-07-22). TWO
  prongs, because the sender can be stuck in two different ways: `_sender`
  loops on `not recv_task.done() and not s.closed` and `close()` wakes an
  IDLE sender via `notify_frame`; AND `close()` cancels `s.stream_task` (the
  sender runs as a task) to interrupt a sender BLOCKED inside a large
  backpressured `send_bytes` — the loop-top `s.closed` check can never fire
  for a blocked send, which is exactly what strands a 1024²/8192² history
  where a 128 MiB frame stalls on a slow client. Pinned by
  `test_close_while_attached_unwinds_streamer` (which also asserts
  `stream_task` is set then cleared). Also: every ws send goes through
  `_guard_send`, which turns a send-after-disconnect `RuntimeError` (uvicorn,
  when a send races the client's close) into a normal `WebSocketDisconnect`
  and LOGS it — otherwise it surfaced as a "streamer failed" traceback and the
  frontend's auto-recover churned reconnects. `ws_endpoint` logs the
  disconnect code and `_sender` logs where it stopped (`last_sent`), so a
  mid-replay drop is diagnosable from the journal.
- **A departing browser must free its session PROMPTLY, not on the idle TTL.**
  A tab close / reload / navigate sends NO `DELETE` (Vue's `onBeforeUnmount`
  does not run on a real unload): the backend learns only via the WS close,
  whose `finally` merely *pauses + detaches* — it never calls `close()`. So the
  session lingers with alive-but-idle workers holding the full `FrameHistory`
  (RSS) and each worker's CuPy pool + cuFFT cache + exponent meshes (VRAM) until
  the sweeper reaps it, and a reload creates a NEW session at once that competes
  with its just-orphaned twin. THREE-part fix:
  (1) the frontend fires a `keepalive` `DELETE` on `pagehide`
  (`useSession.beaconDestroy`; skips `event.persisted` bfcache) — safe against
  `recover()`, which is a live-tab `sock.onclose`, never a pagehide. The
  `DELETE` path drops its own `s` local (`del s`) and calls `_collect_closed()`
  right after `close()`, off the event loop.
  **The `del s` and the arm-until-freed logic are not optional** — a closed
  session is a cycle, so `gc.collect()` frees NOTHING while any live reference
  roots it, and at DELETE time there are two: the handler's own local and, if a
  client was attached, the `ws_endpoint` streamer coroutine still unwinding.
  So `_collect_closed` keeps `_closed_since_sweep` ARMED while any `weakref` in
  `_closed_refs` is still alive. Clearing it unconditionally meant a collect
  running a beat before the streamer released left the flag down and the multi-GB
  history sat resident until a chance gen-2 gc ("RSS stuck at 13.2 GB, nothing in
  the logs"). Pinned by
  `test_collect_stays_armed_while_a_closed_session_is_rooted`. VRAM comes back
  at worker-join inside `close()`.
  (2) `WS_IDLE_TTL` is **90 s**, swept every 5 s. It was 20 s, justified as
  "well above `recover()`'s ~1.5 s reattach" — which assumed the client learns of
  a close as soon as the server does. **It does not**: the close frame is written
  at the TAIL of the transport buffer, so a client with a backlog sees it tens of
  seconds later, and a reattach that lost that race got a 404, a new session id
  and `0 / [0, 0]` — 100 computed records gone. The credit cap removes the
  backlog, but this grace must not DEPEND on that. Three supports: `recover()`
  probes immediately and backs off after; a reattach racing its predecessor's
  ~3 s teardown WAITS rather than returning 4409, which the client read as a
  plain close and answered with another reconnect; and the 404 path SAYS a
  session was lost, with how many records went with it.
  (3) `start.sh` PINS `--ws-ping-interval/timeout 20` (matching uvicorn's
  current defaults) so a HALF-OPEN drop — kill -9, laptop sleep, network
  partition, no TCP FIN — is detected by the keepalive and closed, running the
  `finally` instead of `receive_text()` blocking forever; bounded at ~60 s. Explicit so the keepalive cannot silently regress — though it is that
  same keepalive which killed the socket above when a PING queued behind a
  backlog, so these pins were necessary and never sufficient. Pinned by
  `test_detached_session_swept_after_grace_attached_is_shielded` (behaviour, not
  the number) and `test_a_reattach_waits_for_the_previous_streamer_to_let_go`.
- **The BROWSER'S WebSocket receive path is the large-grid wall, and it
  degrades with MESSAGE SIZE — not the server, not painting, not pacing.** At
  4096² it tops out at ~112 MiB/s with `queue_drops: 0` and the client IDLE,
  waiting on delivery, while the same server feeds a raw Python client at 402
  MiB/s; at 2048² the same browser sustains ≥480 MiB/s, so the cost is
  per-message. **This is what made display downsampling the only real fix, and
  why no pacing policy can help** — a pacer targets paint time, 33× off the
  real constraint. Also: `pack_frame` costs 28 ms/record at 4096² ON THE EVENT
  LOOP. NB the "~3 GB RSS hump, transient, not a leak" recorded here in
  2026-07-23 was the UNBOUNDED TRANSPORT BUFFER — bounded now by the credit
  cap, and the same backlog that was killing the socket. Numbers and that
  correction in `notes/session-lifecycle.md`.
- **Two more things kept a closed session's RAM resident, both found only by
  measuring RSS across a Restart.** (1) a `for` target outlives its loop, so
  `ttl_sweeper` iterating `SESSIONS` inline held the LAST session it examined
  across its sweep sleep — and FOREVER once SESSIONS emptied, since an empty loop
  never rebinds the name. The loop lives in `_sweep_idle`, whose frame dies on
  return (pinned structurally by
  `test_ttl_sweeper_never_binds_a_session_in_its_own_frame`). (2) glibc's mmap
  threshold is DYNAMIC (128 KiB, ratcheting up to each freed mmap'd block, capped
  at 32 MiB), so records just under that cap come from the arena and `free()`
  never lowers RSS; hence `_collect_closed`'s `malloc_trim(0)`. **Record size
  decides which of these you see**, so test memory at more than one grid: 4096²
  looked clean while 2048² sat at ~9.8 GB. Figures in
  `notes/session-lifecycle.md`.
- **A closed session's history is CYCLIC garbage — freeing it needs the
  collector, not refcounting.** `SimSession.workers` holds each `SolverWorker`
  and `worker.session` holds the session back, so after `close()` the pair (and
  the whole `FrameHistory` hanging off it) is unreachable but not refcount-free.
  On an idle server a gen-2 collection may not run for many minutes, so tens of
  GB stay resident long after Restart and look EXACTLY like a leak.
  `session._collect_closed()` makes it deterministic: `close()` sets
  `_closed_since_sweep` and the TTL sweeper does one `gc.collect()` per sweep
  that had a close (off the event loop; collection cost scales with tracked
  CONTAINERS, not with the bytes they point at). Pinned by
  `test_closed_history_needs_the_cyclic_collector`, which asserts BOTH halves —
  the history survives `close()` + `del`, and dies on `_collect_closed()` — so
  if the back-reference is ever removed that test fails loudly rather than
  silently keeping a now-pointless collect.
- **Do not chase "leaked" objects with `gc.get_referrers` alone — it cannot
  see frame locals.** In CPython 3.12 it does NOT report an object held by a
  plain local (fast locals are invisible unless `f_locals` was materialized), so
  "no coroutine frame holds it" is a conclusion that instrument can never
  support; a snapshot must also exclude its OWN containers. Use a `weakref` +
  explicit `gc.collect()` to decide whether something leaked, and
  `sys._current_frames()` to find who is still running. The hunt is in
  `notes/session-lifecycle.md`; every real lifecycle path leaks nothing once
  collected.
- **The boundary warning is ONE SHORT LINE, and the cells it names are
  drawable.** It can clear again within a couple of records as a state drifts
  back out of the band, so a sentence that also explained periodicity and
  offered a remedy was regularly gone before it had been read. What survives is
  only what cannot be got elsewhere: `⚠ W(x,y,px,py,t) has reached the px edge —
  1.3e-4 of its integral is in the outer 4 cells.` ("integral", not
  "probability": the quantity is ∫W over the band, and W is signed.) The
  reasoning and the remedy moved into the span's `title` (`boundaryTitle`),
  which a hover holds still. The cell COUNT comes from the server (`band` in the
  boundary payload, per axis) rather than being re-derived, because it must be
  the width the mass was actually measured with.
  **`lib/cells.ts` mirrors `boundary.edge_band` separately, for DRAWING only** —
  a second "cells" toggle (`wignerf.cells`, its own key, so "Reset setup to
  defaults" leaves it alone) paints the computed lattice faintly on the W panels
  and the IC preview with the edge-band cells brighter, so the number in that
  warning points at something visible. The mirror exists because the overlay
  follows the PAINTED frame, which during a scrub across an auto-expand boundary
  is not the live window the server's `band` describes. NB it draws the COMPUTED
  lattice from the record's `N`, not the samples actually sent, so it stays
  right under display downsampling — the panel says its own reduction. Ticks and cells are
  INDEPENDENT layers, and the lattice is dropped when more than ~200 of its
  lines would land in the visible window — a COUNT not a pixel test, so zooming
  in brings it back. **The three toggles share ONE row** (`flex`, natural
  widths, labels "auto-expand", "grid", "cells"), which is what pays for the
  third control in a 320px column. Cell lines are deliberately NOT in
  `ExportSpec`: the mp4 is a finished artefact, and this is a diagnostic you
  reach for.
  **The edge finding is stated ONCE, IN ONE PLACE, AND THE TIME ARGUMENT SAYS
  WHICH STATE IT IS ABOUT.** The detector runs on the IC preview and on the live
  record, and at record 0 those are the SAME state, so the fact was genuinely
  duplicated. Reporting it twice and suppressing one by DIFFERENCING the axis
  sets was worse than the duplication: the survivor flickered between two places
  as the live axes changed and read as two problems. Now the header says it once
  and `wOf` writes **`W(x,p,0)`** while nothing has been computed past the
  Cauchy data, `W(x,p,t)` after — one sentence whose argument distinguishes the
  initial condition from the evolved state. `preview_warnings`'s `edge_axes` and
  the `X-Wignerf-Edge` header stay (structured, so the client never
  pattern-matches a sentence); `ICEditor` EMITS the finding upward. The one case the session cannot cover — an IC
  you have edited but not restarted into — is `SimulatorView.icEdgeText`, gated
  on `restartNeeded` so it can never compete with the session's own reading.
  **The × dismisses BY TEXT.** Clearing `session.boundary` cannot dismiss this:
  the sentence may come from `icEdgeText` or from the standing `status.boundary`,
  neither of which that touches, and even for the transient event `sticky` keeps
  displaying the last value for its dwell. The button was inert on screen while
  looking perfectly wired in the source. Keyed on the text, so dismissing one
  warning never hides the next, different one.
  **DISPLAY-ONLY STATE THAT ADDRESSES AN AXIS MUST FOLLOW ndim DOWN.**
  `ICEditor.cutAxis` is not in `cfg` (it changes nothing computed) and its
  select is inside `v-if="ndim > 1"`, so a stale 1 at ndim=1 was both
  unreachable and unrecoverable — and not inert, because the ψ/φ titles fall
  back to it whenever the response is null, which is exactly when a 2D ψ stops
  compiling at ndim=1. The fallback then read `ψ(p)` and `φ(x)`.
  **A REQUEST SEQUENCE MUST BE BUMPED ONLY WHEN A REQUEST IS ACTUALLY SENT, AND
  EVERY INDEPENDENT REQUEST NEEDS ITS OWN KEY AS WELL AS ITS OWN COUNTER.**
  `ICEditor.refresh` incremented `seq` before the body de-dup could return, so
  every successful commit re-fired the deep watch, declined to repeat the
  request, and invalidated the responses still in flight from the one it had
  just declined to repeat. The ψ/φ traces were the casualty. The
  wavefunction call has its own counter AND its own `lastWaveBody`, because it
  depends on `cutAxis` and the W preview does not: sharing one key made the 2D
  cut selector INERT — the W body was byte-identical, so `refresh` returned at
  the de-dup and `refreshWave` below it was never reached. **And a de-dup key must
  be cleared when the request FAILS**, or a transient error is permanent: the
  identical body de-dups out and nothing can retry it.
  **And a failed ψ sample SAYS SO.** It gates nothing (the W response owns the
  Solve gate) but it must not blank two charts in silence. Their titles come from
  the CUT rather than the response for the same reason — a title that degrades
  whenever the data does turns a failed request into apparent rot.
  **A failed W preview must also CLEAR what the previous one left**: the
  warnings, the deficit, the norm readout and the emitted edge finding all
  describe a state that no longer exists, and leaving them put a stale ⚠ above
  the compile error and an edge claim in the HEADER about an IC that does not
  compile.
- **EVERY TRANSIENT NOTICE IS AN OVERLAY AND HAS A MINIMUM DWELL. This is a
  RULE, not a fact about the header** — it was written down for the header alone
  and then broken the next time a panel grew a conditional line, which is
  exactly what a rule stated as a special case invites.
  A notice derived from live state comes and goes at the rate that state
  changes, not at the rate a person reads, so:
  **(1) it must not occupy layout** — anchor it `absolute left-0 right-0
  top-full z-*` to a `relative` parent, overlaying the top of whatever follows.
  **(2) it must stay long enough to be read** — `lib/sticky`, MIN_DWELL_MS =
  5 s, holding the last non-empty value after the condition clears. A
  REPLACEMENT is never delayed (a newer message beats an older one and restarts
  the clock); only the clearing is. Display only, so the transport, the Solve
  gate and every test still see the instantaneous state.
  The steady READOUTS are the exception and stay in flow — the norm deficit, the
  normalisation scale — because they change only when the form does, which is
  the user's own doing and already a relayout. The IC editor's ⚠ strip is
  steady for the same reason and deliberately has NO dwell.
  **Three traps, all of which shipped once:**
  **An `overflow-hidden` ancestor deletes the strip from the screen and leaves
  it in the DOM.** The IC preview needs that class to keep the heatmap inside
  its rounded border, and it also clips any absolutely-positioned child hanging
  BELOW the box — where `top-full` puts one. Hence two nested boxes: an outer
  `relative` anchor and an inner clipping one. Do not merge them.
  **"Nothing moved" and "the text is in the DOM" together do NOT prove the
  message is visible.** Assert it is PAINTED: `document.elementFromPoint` inside
  the notice's own rect must land on the notice. Same discipline as the
  WebGL-canvas rule, in the other direction — there a screenshot lies, here the
  DOM does.
  **When several lines share one strip, the flickering one gets its OWN timer.**
  `sticky(() => a || b)` over two sources is not a dwell: losing `a` still
  leaves a non-empty value, which `sticky` correctly reads as a REPLACEMENT and
  applies at once — so the two sentences alternate at the rate the flickering
  one changes, and the × looks inert because dismissing one substitutes the
  other in the same flush. One `sticky` per source, combined afterwards. Pinned
  by `sticky.test.ts`. Reproductions in `notes/ui-notices.md`.
- **The header's TRANSIENT notices are an absolute overlay, not flow content —
  the W panels must never move because a message arrived.** `restartNeeded`,
  `boundaryText`, `paramFlash`, `regridFlash` and the lost-session notice live in
  an `absolute left-0 right-0 top-full z-30` strip anchored to the `relative`
  header, out of the vertical flow in BOTH layouts; as inline children each
  arrival wrapped the header and moved the panels 32 px. The trade is
  deliberate: the strip OVERLAYS the top ~24 px of the columns, briefly hiding a
  panel's label chip and the first plot's title — chrome, never data — far
  cheaper than relaying out the thing being watched, and dismissible. Full width
  so no message need be truncated.
  **The float32 badge stays inline on purpose** (a permanent property of the
  session, set before there is anything to watch), and so does `createError` —
  but INSIDE the header; see the failed-restart bullet. Do not "tidy" these back
  in. See `notes/ui-notices.md`.
- **A FAILED restart leaves the app session-less, and three things used to hide
  that.** `useSession.create` calls `destroy()` BEFORE it posts, so a 422
  deletes the old session and leaves `info`/`status` null with the form intact.
  Three consequences, all fixed and all easy to reintroduce: the server's
  refusal was rendered but PAINTED OVER by the transient strip; the header read
  **"connecting…"** forever, a false progress report on a socket that will never
  open (now suppressed while `createError` is set); and the transport button
  stayed pink "Solve" and ENABLED, doing nothing on click
  (`ControlBar.noSession` now disables it and says so in the title).
  **`createError` must be inline INSIDE the header** (`basis-full`, its own
  flex-wrap line), not after it — outside, it landed at exactly the y the
  `top-full` strip is anchored to and the amber "setup changed — restart to
  apply" was painted straight over it. That pairing is not exotic: **every
  restart that 422s sets both.** See `notes/ui-notices.md`.
  **And the Setup panel's 2D footprint line says when a grid cannot fit at
  all.** `GET /api/device` reports each pool device's `total_bytes`; the panel
  compares its per-device estimate against the SMALLEST of them — the workers
  spread, so the smaller card binds, the same property `_fit_error` rests on —
  and turns red. TOTAL, not free, deliberately: total is static, so the panel
  needs no polling and can never contradict the server, whose live-free refusal
  is a strict superset.
- **A marginal's NEGATIVE part measures the noise floor its edge-band mass is
  read against, and on a coarse grid that floor is ABOVE the trigger.** ρ(x) is
  a probability density, so any negative value in it is pure numerical error —
  which makes `(|ρ|−ρ)/2` summed over the axis a free, self-calibrating error
  bar (elementwise only, so it costs one reduction on numpy and cupy alike). It
  has to be used, because the floor is the Nyquist truncation of the state's own
  Fourier tail, `exp(−(πσ_q/dx)²/2)`: at 32⁴ — a size this project *recommends*
  for exploration — that is 50× the 1e-6 trigger, so the band mass IS noise,
  sign-flipping every record or two, and ungated it produced 79 boundary state
  changes in 201 records. That is what "the heatmaps jump while computing" was.
  Fix: `EdgeState` carries `noise` and an axis trips only above
  `max(threshold, EDGE_NOISE_MARGIN·noise)`, plus `session._confirm_edge`
  requires `EDGE_CONFIRM` consecutive records in BOTH directions.
  Three properties are load-bearing: the gate is **inert wherever the detector
  already worked** (1D N=256..4096 and 2D from N=48 have floors three orders
  under the trigger, so 1D behaviour is untouched); a **ratio-only** gate is not
  enough, because a state genuinely over the edge rings in proportion (band/
  noise 11 there against 4.2 for the worst noise blip, which is why the margin
  is 8 and not 12); and the **first reading per slot is exempt**, or an IC that
  starts at the edge — a paused session with exactly one record — would never
  warn. Sweep in `boundary.py`; measurements in `notes/ui-notices.md`.
- **Secular E drift + slow purity decay = boundary wrap, not a solver bug.**
  The spectral domain is a torus: when a state's orbit + ~5σ tails reach the x
  or p edge, mass wraps through the seam and the run faithfully evolves the
  WRONG (torus) problem. Tells: IC norm deficit >> 1e-6, the 4σ edge warning,
  secular (not oscillatory-bounded) drifts. Fix: enlarge the domain — or enable
  auto-expand, which detects the approach (edge-band mass of the total sampled
  W, also checked at IC-preview time — the per-component 4σ boxes alone miss
  interference terms) and regrids before mass wraps. Verified: same cat state,
  [-6,6]x[-7,7] gives E drift 2e-3; [-12,12]² gives 4e-6 with purity conserved
  to 5e-12 — the discrete map is exactly unitary for contained states (healthy E
  behavior is a BOUNDED O(dt²) oscillation from Strang splitting, never a
  drift).
- **Growing ΔX·ΔP in the RELATIVISTIC variants only = anharmonic shear, not
  a bug.** T = c√(p²+m²c²) carries a −p⁴/(8m³c²) term, so ω depends on E
  (δω = −3E/(8c²)) and the ensemble shears at k = t·r²·3/(8c²). The shear is
  symplectic: purity and det C are conserved and the LOWER envelope of ΔX·ΔP
  stays exactly at ħ/2 while the upper grows ∝ t² (modulated at 2ω). Tells that
  it is physics: halving dt leaves it identical while the E(t) splitting
  oscillation drops 4×, it scales as 1/c⁴, purity stays flat. Non-relativistic
  harmonic H is exactly quadratic ⇒ no shear.
  Measured: coherent state at (2,0) in x²/2 with c = 137.036 → 2e-5 at
  t = 100 (analytic σ²k²/2 = 1.6e-5). Pinned by
  `test_relativistic_uncertainty_shear` — and since M2 there is a 2D one of the
  same name in `test_propagator2d.py`, which measures the same four tells at
  c = 10 because 1/c⁴ makes c = 137 unaffordable over a 32⁴ grid; see the
  relativistic-2D gotcha above for its numbers.
