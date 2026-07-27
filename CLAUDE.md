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
  meshes) and `dV`. Every 1D-only spelling — `grid.dx`, `gs.ox`,
  `geom.Nx`, `vf.rho`, `obs.x_std`, `es.x_mass`, `cp.dUdx` — survives as a
  compatibility property that **RAISES at ndim > 1** rather than returning axis
  0/1, so a call site nobody generalized fails loudly instead of computing a
  wrong number. That is what made the migration tractable; do not soften it.
  **What is DEFERRED in 2D and why**: float32, auto-expand and mp4 export are
  milestones M1, M3 and M4 (see the milestone table below), each refused with a
  message naming its milestone at create time and, for auto-expand, on the live
  `ParamChange` path too. `applyNdimInvariants` (`lib/config.ts`) mirrors the
  first two in the FORM for the same reason `applyPrecisionInvariants` exists:
  the panel can be unmounted and an import can reach the combination from
  outside it. **M2 (relativistic `qr`/`cr`) landed 2026-07-27** and its gate is
  gone from both sides — see the relativistic-2D gotcha below for what was
  measured to retire it.
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
  exact sequential records (replay/scrub). Computation ALWAYS runs at full
  speed in both modes — neither the dial nor a slow client ever throttles
  the workers; `delay` (seconds injected between played-back frames)
  paces only the display. The dial's "0" position (default) means one
  record per display refresh — the fastest speed at which every frame is
  still painted: the client measures its refresh interval (lib/perf.ts)
  and sends that as the delay, and every dial position is clamped to at
  least it, so delivery never outpaces painting. **At 4096²/8192² that is
  NOT enough and there is deliberately no client-side pacing loop** — an
  adaptive pacer keyed on paint time was built and REMOVED on 2026-07-23
  because paint time is not the binding constraint there (8.7 ms/frame
  against a 285 ms delivery interval); see the browser-receive-ceiling
  gotcha. Don't rebuild it: the constraint is the browser's per-message
  receive cost, so the fix is smaller messages (display downsampling), not
  a smarter delay. Replay never skips a
  record; it slips on WS backpressure when the client can't keep up. The
  UI dial is "0" plus a log range 20 ms–1.5 s. Client frame fan-out is
  rAF-timed (useSession: decode per message, paint one frame per
  animation frame; small FIFO with drop-to-newest as a burst safety
  valve), so texture uploads, uPlot updates and Vue reactivity run per
  PAINTED frame by construction. That drop-to-newest is why the timeline
  readout shows painted/s AND received/s (`Timeline.vue`, `perfRates`):
  when they diverge the client is SKIPPING records, which reads on screen
  as fast playback and is really loss — one number alone cannot tell the
  two apart, and the live/compute path makes that worse by design (the
  `delay` gate applies only to replay — see `advance_cursor` — while live
  coalesces to the newest record, so computing legitimately animates
  faster than paced playback). A playback-only run must never coalesce to the
  frontier while sequential records are unsent (that would teleport
  playback to the end), and its auto-pause is delivery-aware — it fires
  only after the frontier record was SENT.
  **`loop` repeats that pass instead of pausing** (`LoopCmd`, a `loop` checkbox
  in the transport row beside Solve/Play/Pause, echoed in `status`). It exists
  because the auto-pause above is correct but easy to walk into: playback stops
  at the frontier, the button there becomes "Solve", and the Space that was
  replaying a second ago now COMPUTES. It is a DISPLAY policy like `delay` —
  never changes what is computed — and it rewinds to `loop_from`, the cursor
  captured when the pass STARTED, so "again" means the region you asked to
  watch rather than all of history. Two things are load-bearing. It reuses the
  auto-pause's delivery gate, so a slow client is never rewound past frames it
  has not been sent; and `browsed` stays True across the wrap, or the next tick
  re-attaches to the frontier and rolls into computation — precisely the
  confusion the feature removes. **Rewinding `cursor` alone STALLS the loop
  silently**: the sender walks forward from `last_sent`, which is still at the
  frontier, so nothing sends and the display freezes on the last record. Hence
  `loop_epoch`, bumped on each wrap, which the sender watches to rearm
  `last_sent` — the job `pending_seek` does for a seek. Measured with that
  rearm removed: `[8, 3, 4, 5, 6, 7, 8]` and then nothing, against 60 laps with
  it. NB a test that counts arrivals at the FRONTIER cannot see that failure —
  the live frame already in flight when the seek was sent is itself the
  frontier, so a dead loop reads as two passes; count arrivals at the START.
  Pinned by `test_loop_replays_the_same_region_instead_of_stopping`.
  The transport must stay
  responsive under full frame backpressure: control JSON (status echoes)
  is flushed BEFORE frame sends each tick, play/pause are echoed
  immediately, replay batches are wall-clock-budgeted (~0.2 s) and
  preempted by pause/seek, and the client flips the transport button
  optimistically on play/pause. The delay dial is settable only while
  PAUSED (pause → change → resume) and its thumb is local UI state,
  re-synced from status when idle. Binary layout in
  `core/protocol.py`, mirrored by `frontend/src/lib/protocol.ts` and
  cross-checked via `scripts/gen_fixture.py` + the frontend vitest.
- **Record grid**: τ_k = t1 + k·record_dt. Each variant (1–4 worker
  threads: quantum/classical × rel/non-rel) integrates with its own
  adaptive dt (`adjust_step`, every 20 steps) but lands exactly on each τ_k
  by clamping the final substep. Same k ⇒ same physical t across variants.
- **State convention**: W is float64, fftshifted along both axes on the
  backend; frames stream in shifted order and the *shader* unshifts via a
  half-period texture offset (`render/WignerRenderer.ts`, R16UI texture,
  manual bilinear with periodic wrap, diverging LUT centered at W=0).
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
  ALWAYS on — every record, each worker sums the outer edge band of the
  ρ/φ marginals it already computed (host-side, O(Nx+Np), no extra device
  sync) and `session.report_edge` posts a `boundary` WS event on state
  change (band = max(4, N/32) cells/side, trigger 1e-6 in float64 and 1e-4
  in float32 via `EDGE_THRESHOLD_BY_PRECISION`; **gated on a MEASURED noise
  floor and confirmed over `EDGE_CONFIRM`=4 records** — see the
  edge-noise-floor gotcha, which is what stops a coarse 2D grid strobing the
  warning — expansion
  prevents wrap, it cannot repair it, so it must fire while edge mass is
  negligible). The `auto_expand` toggle (SessionCreate field AND
  live-appliable via ParamChange) governs only the RESPONSE: an exact
  fixed-lattice regrid. **It is refused outright in float32** — single
  precision cannot supply the measurements it needs; see the float64/float32
  gotcha for the numbers. dx/dp and the lattice anchor are FROZEN at session
  creation (`GridState`, integer window arithmetic; extents materialize as
  anchor + integer·dx, and `Grid` takes explicit dx/dp + anchors so overlap
  lattice points are bitwise-identical across regrids); move = whole-cell
  window shift, expand = double an axis (powers of 2, support centered,
  combined move+double; NO shrink, NO interpolation ever — norm/E/purity
  survive to machine precision minus the ≤threshold dropped tails). The
  session commits a `RegridPlan(epoch, k_star, state)` with k_star past
  every in-flight record; each worker applies it before computing its
  first record ≥ k_star (`embed_window` + `Propagator.set_grid`), so the
  switch is lockstep-uniform and records <k_star stay old-geometry. U is
  revalidated on the union extended Bopp range BEFORE commit (refusal ⇒
  `invalid_potential` warning, keep computing). **Plan commits and physics
  commits are mutually exclusive** (both hold `_edge_lock` for their whole
  body, and `apply_params` orders physics BEFORE any immediate schedule):
  U/hbar_eff move the Bopp range, a plan validated under stale physics
  would hit the deliberately-fatal non-finite check at k_star (a per-worker
  rollback there would desync lockstep geometry), so a pending plan's union
  window is revalidated under incoming physics and the change is REJECTED
  if it does not hold — this also closes the race of a plan committing
  during the streamer's ~ms validation compile. Expansion caps at
  `WIGNERF_MAX_GRID` (`capped` warning, keep computing; pure moves still
  work at the cap). Geometry is a PER-RECORD fact: protocol v3 headers
  carry Nx/Np/x1/x2/p1/p2, history stores geom per record, the streamer
  packs from the record (never the session), and the frontend follows the
  PAINTED frame (panels/overlays/marginal axes re-derive per frame;
  zoom windows remap to the same physical region) — so scrubbing across a
  regrid boundary just works. Each doubling ≈ 4× step cost and 4×
  bytes/record (the history cap then holds ¼ the records).
- **Export panel** (header button "⤓ export") carries two things: the mp4
  below, and the run's SETUP — `GET /sessions/{id}/setup` serves
  `describe.setup_document`, the config the session was CREATED with
  (`state_at(cfg, log, -1)` rewinds every live change; live changes are
  deliberately not part of a starting state — the video's metadata block is
  where they are recorded). Import fills the setup form and marks the
  session restart-dirty, never restarts by itself (`lib/config.importConfig`,
  in-place merge on the reactive cfg), and accepts that .json OR an exported
  .mp4: `lib/mp4meta.ts` scans the file's head for the same document in the
  `comment` tag (faststart keeps it there — byte ~3.5k), so a kept video is
  self-restoring. Its confirmation line says `press "Restart session" to run
  it` and NOT "or Solve" — an import moves grid/IC/variants, which are
  SessionCreate-only, so Solve would compute the old ones (the auto-restart in
  `syncFreshSessionToForm` only fires when the document ALSO moves
  mode/t₂/Δt rec/precision on a fresh idle session). And it clears on a
  `sessionId` change, because that IS the restart it asks for: it used to
  clear only at the top of the NEXT import, so "press Restart session to run
  it" stood over a session already running the imported setup. It survives a
  panel close/reopen on purpose — an import you have not acted on yet is still
  actionable.
  A render is destroyed by anything that moves the session on — Restart
  deletes the session (`close` → `videoexport.close_session`, file unlinked
  mid-write) and computing new records evicts the ones behind the renderer
  (`record N is no longer retained`). Both used to happen SILENTLY, so
  `SimulatorView.mayDiscardExport` confirms first (Restart, and a transport
  command whose action is `solve` — playback adds no records and is never
  gated) and cancels the job outright on "yes", instead of leaving it to die
  mid-file. The automatic restarts (first mount, backend recovery) never
  prompt.
- **mp4 export** (`core/videoexport.py` + `core/render_mpl.py` +
  `routers/export.py`): renders an ALREADY-COMPUTED record range on the
  BACKEND — matplotlib/Agg frames piped as raw RGBA into ffmpeg (system
  ffmpeg, absence ⇒ 503). PAUSED-only (409
  while running): a running session evicts old records, and the feature is
  for filming a range you already played back. **The frame RENDER, not the
  encode, is the bottleneck** (measured 4-var 1024²: ~410 ms/frame render at
  4K vs 34–109 ms to encode, and the encode already overlaps via the pipe;
  363 ms of the render is the four `imshow` panels). So export renders frames
  across a `ProcessPoolExecutor` (`export_workers`, `WIGNERF_EXPORT_WORKERS`,
  auto = min(cpu, 8)) while this thread feeds the ORDERED frames to one
  ffmpeg — a sliding window of ≤w+2 futures consumed FIFO by `.result()`
  (workers run ahead, memory bounded). Measured ~3× (4K/4-var 2.2 → ~7 fps;
  1080p 3.3 → ~9-10 fps). The pool is **spawn, NOT fork** — the backend
  initializes CUDA and forking after that inherits a broken context; spawn
  workers only touch matplotlib/numpy (never cupy — `xp` imports it lazily).
  A small job (`< max(2·w, POOL_MIN_FRAMES=16)`) renders serially in-process
  to skip the ~1-2 s pool warmup (`_render_serial`; the light path
  unchanged). Encoder via `choose_encoder`/`WIGNERF_EXPORT_ENCODER`
  (auto|cpu|nvenc): auto uses the GPU **`h264_nvenc` encoder** if a one-shot
  runtime probe passes (`_nvenc_ok`, cached — the encoder can be built-in yet
  fail with no driver/GPU, e.g. the VPS), else `libx264 -preset veryfast
  -crf 18` (was `medium`; ~2× faster, file ~7% larger, visually identical for
  this smooth content, and frees cores for the render pool). NB the GPU path
  is the h264_nvenc ENCODER, NOT ffmpeg `-hwaccel` — that is a DECODE flag and
  does nothing for our rawvideo input. Two passes: a scan collects
  the E/ΔX·ΔP/γ series, the per-variant FIXED colour scale (no brightness
  flicker), the fixed marginal amplitudes and the widest window any record
  used, and proves every record is still retained before ffmpeg starts; then
  one figure update per frame. Only VALUE scales are export-wide — the
  SPATIAL axes follow each record's own geometry (`_apply_geom`, which also
  re-captures the blit background since ticks are static art), exactly as
  the SPA follows the painted frame; freezing them at the union rendered
  every frame before an auto-expansion as a stamp in the corner of its
  panel, and the union now only labels the metadata block. The figure is
  built ONCE and BLITTED (static background + ~15 animated artists): 465 →
  ~17-80 ms/frame measured at FHD (~320 ms at 4K, 4 variants), the
  difference between minutes and half an hour for a 1000-frame export.
  Sizes offered: FHD / QHD / 4K UHD. The figure is always 19.2×10.8 in and
  the RESOLUTION RIDES ON THE DPI (`FrameFigure.REF_WIDTH`) — font sizes
  are in points, so a fixed dpi would render every label at half its
  relative size at 4K. The downloaded name is descriptive
  (`wignerf-QN-QR-CN-CR-41rec-3840x2160-20260722-0107.mp4`, via
  `Content-Disposition`) while the on-disk path keeps session+job ids: two
  exports of the same range in one minute must not collide, least of all
  while one is being downloaded. Frame content mirrors the SPA (panels +
  marginals + series with a time cursor, variant colours/dashes from
  `lib/variants.ts`, the shader's symmetric bwr scale) plus a metadata
  block. The SPA carries the same cursor at the PAINTED frame's t
  (`SeriesPlot.vue`, `.wf-tcursor`): a DOM element in uPlot's `over` layer,
  moved by one transform write per frame — a canvas artist would cost a full
  `u.redraw()` (re-pathing every series) at display rate, and `over` clips
  it when a zoom scrolls it out of view.
  The video must READ like the screen: plot titles are copied
  verbatim from `SeriesPlot.vue`/`MarginalsPlot.vue` (γ keeps the UI's
  "purity γ(t) = 2πℏ∬W²dxdp", never an equivalent like Tr ρ²), field labels
  match the Setup panel (ℏ, "batch"), and the series y-window +
  tick decimals reproduce that component's `scales.y.range` rule
  (`render_mpl.series_ylim`); the "grid lines on plots" toggle rides along
  in `ExportSpec.show_grid` and governs EVERY plot in the frame — charts
  get uPlot's grid stroke (`--wf-chart-grid`, so it follows the theme like
  the rest of the chrome; see the Theming bullet), the W panels get
  `GridOverlay.vue`'s theme-INDEPENDENT
  rgba(120,120,120,.28/.55-at-zero) drawn AFTER the image (matplotlib puts
  the axes grid under it, which is why the heatmaps first had none; the
  lines are animated artists ordered behind the images in `_dynamic`) — matplotlib's own autoscale renders a 2e-5
  purity drift as a dramatic dive with a "×10⁻⁵+1" offset where the UI
  shows a flat line at 1.000000, from byte-identical data. Mirror any
  change to those rules on both sides. The block carries
  U(x), parameters, the IC as an analytic expression
  (`core/describe.py`; cat states print ψ(x,0), the compact complete form),
  and any live parameter change inside the range (`session.param_log`) —
  so one frame documents the whole run; the same facts go into the mp4
  `comment` tag as JSON. That block is anchored at `FrameFigure.META_TOP` with
  `va="top"` and grows DOWNWARD, so it had nothing stopping it running off the
  bottom edge: the figure is always 19.2×10.8 in at 16:9 (dpi carries the
  resolution), 8 pt at linespacing 1.6 advances 0.016461 of the height, and
  0.185/0.016461 = **11 lines fit**. `param_lines` grows by one line per live
  parameter change plus the float32 PREVIEW line, and a 4-variant cat run with
  4 live changes measures **10 lines in float64, 11 in float32** — i.e. a
  realistic export sat exactly at the edge and one more change clipped in
  silence. `_meta_fontsize` now shrinks to fit (one size for BOTH columns, so
  they stay matched, derived from `get_figheight()` so a non-16:9 export keeps
  the full 8 pt), and past a 5 pt floor `_meta_fit` elides with "… +N more
  lines — full detail in the mp4 comment tag", which is an honest pointer
  because `describe.config_json` really does carry all of it. Static art, baked
  into the blit background: no per-frame cost. Progress: `export` events on the session WS plus a
  REST poll; the file lives in `WIGNERF_EXPORT_DIR` until downloaded, TTL
  (30 min), session close or shutdown. The header button stays ENABLED
  while computing (a disabled button explained only by a tooltip is how
  this feature first read as broken): the panel states the gate and
  "Pause & render" pauses, waits for the server to confirm and re-seeds an
  untouched range before posting. Rendering continues while the popover is
  CLOSED (the poll and the WS events keep updating), so the button IS the
  notification — "⤓ export 42%" while running, emerald "⤓ export ready"
  (red "failed") when finished, and reopening it collects the file; a
  finished job survives reopening and is dropped only by a new render or a
  session change (a restart deletes the old session's files). The panel re-reads the extent from
  `GET /sessions/{id}` when it opens — the streamed status lags a frame
  burst by up to seconds after a pause, and seeding the range from it
  silently exported half the history.
- **Theming (light/dark)**: a header button (`☀ light` / `☾ dark`) flips the
  whole UI; **light is the DEFAULT** and the choice persists in
  `localStorage.wignerf.theme`, a sibling of `wignerf.layout`/`wignerf.grid`
  and so untouched by "Reset setup to defaults". **CSS is the single source of
  truth**: `frontend/src/style.css` defines every colour once per theme as
  `--wf-*` on `:root` (light) and `.dark`, and a Tailwind 4 `@theme inline`
  block turns them into semantic utilities (`bg-panel`, `text-fg-3`,
  `border-line`, `text-warn`, …). **`inline` is load-bearing** — a plain
  `@theme` copies the VALUE in at build time and a runtime override then
  changes nothing. There are ~15 roles, not 200 literals: the migration
  replaced 217 hard-coded `neutral-*`/accent utilities with them, so
  `grep -rn 'neutral-\|#[0-9a-f]\{6\}' frontend/src` should now only find the
  legitimate remainder (below).
  State lives in `frontend/src/lib/theme.ts` — a module-singleton `ref`, NOT a
  prop like `showGrid`, because the uPlot option builders need it too and
  half a dozen components read it at once. `chartPalette()` reads the
  `--wf-chart-*` properties back off the document, so JS duplicates no value;
  it must run AFTER the root class is applied, hence a function called per
  chart build (one `getComputedStyle`, never per frame) with literal fallbacks
  for the DOM-less unit tests. That ordering is why `setTheme`/`toggleTheme`
  apply the root class SYNCHRONOUSLY as well as from the watch: `chartPalette`
  CACHES per theme name, so one read taken before the class landed would pin
  the wrong palette for the life of the page — not a one-frame glitch. The
  watch alone happens to win that race (a watcher created outside a component
  has no job id, so its pre-flush job sorts ahead of the components'), but
  nothing should rest on a Vue scheduler detail, and `apply` is idempotent.
  **The charts destroy+rebuild** on a theme change
  — uPlot takes axis/grid/series colours at construction only — by widening the
  watch the grid-lines toggle already had (`[() => props.showGrid, theme]` in
  `SeriesPlot`/`MarginalsPlot`, `watch(theme, …)` in `PotentialEditor`; all
  three re-apply their existing data to the new chart rather than re-fetching
  it — `PotentialEditor` from `result.samples`, or the U(x) trace blanks for a
  round-trip on every flip). The
  theme is deliberately NOT in `SimulatorView`'s `plotsKey`, for the reason the
  note there gives: a flip must never remount/blank the W panels.
  `index.html` applies the stored class in a **blocking inline script**,
  because `main.ts` is a deferred module and a dark user would otherwise get a
  white flash on every load; it also declares `color-scheme`, which fixes a
  standing dark-mode bug (native `<select>`/`<input type=range>` widgets were
  rendering in the OS's LIGHT style over the dark UI).
  **What does NOT follow the theme, on purpose**: the bwr heatmap LUT
  (`lib/colormaps.ts`, `Colorbar.vue`, mpl `cmap="bwr"`) — blue-white-red with
  W = 0 at white is the physics convention — and therefore everything drawn ON
  the heatmap rather than on the page: `GridOverlay.vue`'s
  `rgba(120,120,120,.28/.55)` lines and grey glyphs, `Colorbar.vue`'s
  `bg-black/70 text-white` chrome, `WignerPanel`/`PanelGrid`'s `bg-black/75`
  overlay labels, and `ICEditor`'s IC-marker rings. Saturated filled action
  buttons (`bg-sky-700`, `bg-pink-800` Solve, `bg-emerald-700`, `bg-red-800`,
  white text) also stay put — they read correctly on white. What DOES change
  and is easy to miss: **variant curve colours**
  (`lib/variants.ts` `VARIANT_COLORS`, Tailwind `*-400` on dark → `*-600` on
  light, because `#fbbf24` amber on white is unreadable) via `variantColor()`,
  and the Timeline readouts' halo (`--wf-label-shadow` inverts, or it smears
  the text instead of separating it from the bar).
  **The mp4 export follows the UI theme** (`ExportSpec.theme`, defaulted from
  the app every time the Export panel opens and overridable per job — it is
  never persisted, or it would stop tracking). Its schema default is `light`,
  matching the SPA's: the panel always sends the field, so that default is only
  what a direct API call gets, and it should get what a first-time browser
  does. It threads the same path
  `show_grid` does: `ExportPanel` → `protocol.ExportSpec` →
  `videoexport._worker_init`/`_render_serial` → `render_mpl.FrameFigure(theme=)`,
  which resolves `PALETTE[theme]` once into `self.pal` and passes it to
  `_style_axes`. `render_mpl`'s `PALETTE`/`VARIANT_COLORS` MIRROR the `--wf-*`
  values (it cannot read our stylesheet) — change a colour on one side and
  change it on the other. Two module-level style blocks moved into
  `style.css` in the process, `.wf-num` (was inside `ICEditor.vue`, styles every
  input in three components) and `.wf-plot .u-*` (was inside `MarginalsPlot.vue`,
  styled all three charts).
- **Parameter policy**: U, c, mass, hbar_eff, tol, dt_sign, auto_expand
  apply live at the frontier; **ndim**/grid/IC/variant-set and the whole COMPUTE
  group (precision, device, history_mb — the Setup panel's third section)
  require a session restart, because each of them is fixed at worker
  construction (FFT plan dtype, `ArrayBackend` device, `FrameHistory` cap)
  (auto-expand moves the LIVE grid; the Setup panel shows it and
  offers "adopt" to copy it into the form). ndim is the most restart-only of
  them all — it decides the array rank — and switching it in the form rebuilds
  the grid and the IC (`config.setNdim`, which mirrors each component's second
  dimension on its first, i.e. the separable product of what was there) and
  replaces U **only if it was still the default**, since `x^2/2` cannot silently
  become a two-variable expression and a hand-written potential must never be
  discarded. **The BOX follows the same "only if untouched" rule, and for a
  sharper reason**: a still-default box is replaced by the TARGET ndim's default,
  because carrying [-6,6] into 2D reproduces exactly what `DEFAULT_AXES[2]` was
  widened to [-8,8] to avoid — the edge band is max(4, N/32) CELLS, so at N=64
  the 4-cell floor makes it 0.750 a.u. wide, only 4.60σ from the default packet
  at x0=2, and a FRESH 2D default tripped its own boundary warning on the first
  Restart (measured 3.78e-06 band mass against the 1e-6 trigger; analytic tail
  2.15e-06 — real mass and a CORRECT warning, not detector noise; at [-8,8] the
  same band sits at 7.07σ and reads 2.12e-12). A box the user CHOSE still carries
  over untouched: silently widening someone's domain is worse than a warning. N
  is always their own choice capped at the target default (a 1D 256 would be
  4.3e9 cells at ndim=2). Pinned in `config.test.ts`.
  **Every restart-only field goes amber when it disagrees with the session**,
  COMPUTE included — a form reading `cuda:0` over a session on `cuda:1` is the
  same trap as one reading float64 over a float32 run. `precision` gets it via
  `LiveRun`, but `device` and `history_mb` cannot: the form holds a REQUEST
  (`''` = the host's pool, `0` = its ceiling) while `status` reports what was
  GRANTED (a resolved device list, a clamped cap). So `SetupPanel` resolves the
  request the way the server would — `''` against the pool from `/api/device`'s
  `devices` (hence `hostPool`, distinct from `choices`), a bare `cuda` to
  `cuda:0` to match `resolve_devices`, and `history_mb` clamped to
  `history_mb_max` so asking for 999999 on a 110000 host is not a "difference"
  — and only then compares. One amber line names what is running, exactly as
  the RUN section's does, and only while something differs.
  **Each section's summary line covers its OWN fields.** `precision` lives in
  `LiveRun` because that is where `status` carries it, but it is a COMPUTE
  control: it triggers and is named by COMPUTE's line, and it is deliberately
  absent from `runStale` and from the RUN line's text. It used to be in both,
  so a plain float64 → float32 switch raised THREE amber notes instead of two
  — the third announcing "running: interactive (no t₂), Δt rec = 0.05" at a
  user who had changed neither and whose mode and Δt rec did not differ at
  all, which reads as the form having quietly moved something else.
  The steady-state facts are NOT repeated in the panel: the devices and the
  history cap ride the timeline's own `hist 0.1 / 107 GiB · dev: cuda:1, cuda:0`
  readout, which is drawn anyway. A standing "running on …, history cap …"
  paragraph there spent vertical space in a narrow column to say what that
  readout already says for free.
  **The panel's column is 320px (`w-80`), and its grids are shaped to it.**
  PHYSICS is `grid-cols-7`: m + c + ℏ on one row, tol + t dir on the next, with
  c taking 3 of the 7 because it is the only field holding a long number
  (137.035999 truncates at an equal third) and tol taking 4 because its LABEL
  grows by " ≥1e-5" in float32. COMPUTE is `grid-cols-3` — precision, device,
  history in one row — and is the ONE section that puts its label ABOVE the
  control rather than beside it: those three labels are words, and a third of
  320px does not fit "precision" plus a select reading "float64". Stacked it is
  two lines where three full-width rows were three. history keeps its unit to
  the RIGHT of the field; the "(0 = host max N)" that used to sit beside it
  moved into `historyHelp`, the tooltip, which had to name that number anyway.
  Verify layout changes here at the real width — headless Chrome, screenshot
  the `<aside>`, and assert nothing's rect exceeds it (see the UI-debugging
  note above); every field fits today with zero overflow.
  `apply_params` echoes a fresh `status` right after its `params_applied`,
  exactly as play/pause do — the periodic one is only every `STATUS_PERIOD`
  (1 s) and that check sits at the top of the sender's tick, so a field the
  form marks amber against `status` could stay amber for a second or more after
  the "✓ applied" flash had already confirmed the change. Pinned by
  `test_params_applied_is_followed_by_a_fresh_status`; measured end to end in a
  browser at 256² while computing, click → `params_applied` is 12 ms and the
  status follows in 1 ms.
  `apply_params` compares against what is LIVE and drops the fields that
  did not change — no worker command, no `param_log` entry, no
  `params_applied`, and nothing at all if the whole message is a no-op (the
  UI sends complete fields; "Apply live" always carries the U(x) draft, so
  the log used to fill with U changes that never happened and an export's
  "how to reproduce this" block lied about its own frames). Entries carry
  `before` as well as `applied`, so the block renders "ℏ 1 → 2" and
  `describe.state_at` rewinds the header physics to the FIRST exported
  record instead of quoting the values the run ended with. Live changes are
  visible in the UI: the header flashes "✓ applied …", and any live-appliable
  field whose form value differs from `status` renders amber — the numeric
  Physics fields (which apply on blur/Enter) via `pending()`, and the U(x)
  input via `PotentialEditor.isPending`.
  **U(x) is a LIVE-appliable parameter and has exactly ONE button.** It used
  to have two, "Use at restart" beside "Apply live", and both were confusing
  for the same reason: the draft was local to the editor, so the FORM did not
  mean what it showed until you pressed the first one — while every other
  setting in the panel is bound straight to `cfg`. Now the draft auto-commits
  to `cfg.potential` from `compile()`'s success path, gated on the server's
  verdict (a half-typed `x^2/` must never reach `cfg`, which is persisted to
  localStorage and is what a restart computes from) and on the response still
  describing the CURRENT text. So `Use at restart` had nothing left to do, and
  U(x) no longer marks the session restart-dirty at all: it applies live, so
  "restart to apply" was a false claim that no successful Apply live could
  clear.
  **Solve carries the form's U(x).** `SimulatorView.sendCommand` pushes
  `set_params {U: cfg.potential}` before a `play` whose action is `solve`,
  whenever the form's (validated) U differs from `status.potential`. Without it
  the form was authoritative for nothing: measured 369 records computed under
  `x^2/2` behind a form reading `x^2/2 + 5*x^4`, the only signal being an amber
  input in a panel that can be hidden. Playback is excluded — it computes
  nothing. So "Apply live" is gated on `status.computing`, NOT on `live` and
  NOT on `running`: while nothing computes there is no live run to reach and
  Solve does the job, and an enabled button there invites a click that is at
  best redundant and reads as the only way to make the new U count. The button
  exists for the one case Solve cannot serve — a computation already in flight,
  which you would otherwise pause. `computing`, because `running` is true
  during pure PLAYBACK too (`running and not stop_at_frontier`), where the
  button's own tooltip — "push this U(x) into the computation already in
  progress" — would be false; it is the same field batch mode dims the display
  on. Bonus: `useSession`'s optimistic transport flip touches only `running`,
  so the button now stays briefly DISABLED after a Solve click rather than
  briefly enabled, which is the safer direction to be wrong in.
  What is left is one emerald "Apply live", disabled unless the session is
  COMPUTING AND the draft is valid AND it differs from `status.potential`, with
  the reason in its `title` and **no standing paragraph** — the old
  "already the live U(x) — edit it to enable …" line was on screen at every
  page load, because that IS the steady state (see
  [no-mystery-disabled-controls]: marker + tooltip + amber on the transition,
  never permanent prose).
  The setup form gates the transport: while the potential draft is invalid
  for the active variant families or the IC preview errors, Solve (button
  AND Space) is disabled and "Apply live" is greyed — a computation must
  never run behind a visibly broken form.
  **Every saturated action button carries `.wf-solid`** (`style.css`): Solve/
  Play, Restart session, Apply live, Render. It supplies `color: #fff` —
  those buttons never set a text colour, they INHERITED the shell's light
  text, which went invisible ("black on blue") the moment the shell could be
  light — and a disabled state that drops to the neutral raised surface,
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
  value. Batch's live
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
  Setup persists in browser localStorage; "↺ defaults" (IC editor) and
  "Reset setup to defaults" (Setup panel) restore defaults in the form and
  mark the session restart-dirty.
- **Potentials** (`core/potential.py`): tokenize-screen (security boundary)
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

`WIGNERF_DEVICE=auto|cpu|cuda:N|comma list` (config.py) names a device
POOL. `core/xp.resolve_devices` expands it fastest-first (`auto` = all
CUDA devices ranked by SM count; an explicit list like `cuda:1,cuda:0` is
trusted as written) and `core/session.assign_devices` spreads variant
workers over it: costliest variants (relativistic, then quantum) and the
larger share go to the fastest card; each worker owns its own
`ArrayBackend`, so no propagator code is device-aware. `core/xp.py` pins
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match nvidia-smi (RTX 3090 =
cuda:1, the display-driving 2080 Ti = cuda:0 on the main workstation).
GPU deps: `cupy-cuda13x[ctk]` — the `[ctk]` extra is REQUIRED (cupy
JIT-compiles kernels at runtime via NVRTC — never nvcc — and needs the
PyPI CUDA headers/libs; NO system CUDA Toolkit anywhere, only the
driver). Note: CUDA 13 dropped Maxwell/Pascal/Volta — the dev
workstation's GTX 1060 (Pascal) needs `cupy-cuda12x[ctk]` instead.
RTX 3090: ~2400 steps/s at 512², ~550 at 1024², ~134 at 2048²; 2080 Ti:
~390 at 1024²; CPU (pyfftw): ~75 at 512². Measured 4-worker lockstep at
1024²: 135 steps/s all-on-3090 vs 191 split 2+2 across the pair (+41%,
and 2+2 beats 3+1's 181 — the even chunk is right); 2 workers: 270 vs
376 (+39%).
**WHETHER A 2D SESSION STARTS IS DECIDED BY ASKING THE DRIVER, not by a cell
count** (`routers/sessions._fit_error`). **Its refusal describes the POOL, and
the ROOMIEST device decides which of two stories it tells** — it used to name
whichever assigned device sorted first and always close with "pick a device with
more room", which on the real pair said *"cuda:0 has 8.9 GiB free … pick a device
with more room"* for a 128×128×128×64 grid: the small card named, a roomier one
implied, and the 3090's 23.6 GiB unable to hold one 26.0 GiB worker either. The
honest reading of that is "so what, I have cuda:1", and it is wrong. So: if the
per-worker footprint exceeds EVERY device's budget it says *no device in the pool
can hold even one*, names the roomiest with its free/installed figures, and
states that dropping a variant or changing device will not help — because
neither will, and only the grid is left. Otherwise a worker does fit somewhere,
which makes it a DISTRIBUTION problem: it names the over-subscribed device and
points at the one with room, with a count ("set device to cuda:1, which has room
for 2 of them"). Pinned by
`test_the_fit_refusal_describes_the_POOL_not_the_first_device`. `WIGNERF_MAX_CELLS_2D` is only a rail
(see its table row); the operative check runs `assign_devices` to learn which
devices this session's workers land on, counts the workers per device, and
compares `n·cells·BYTES_PER_CELL_2D + CONTEXT_BYTES` (300 MiB of CUDA context +
cuFFT plan cache, per process per device) against `xp.device_free_bytes(dev)` ×
`FIT_MARGIN` (0.9 — free memory is a moving target and, unlike the IC preview, a
session has no CPU fallback to drop to). Free memory comes from the driver
(`mem_info`), or from `MemAvailable` for `cpu`, so whatever else is on the card
— another session, another process — is already counted. Two properties are
load-bearing: **the SMALLER card binds**, which no per-session cell count can
express (4 variants at 2+2 is refused by the 11 GiB half of the pair, not by the
24 GiB half); and **unknown free memory does NOT refuse** — there the rail is the
only guard and guessing would be worse. Skipped at ndim=1, where `WIGNERF_MAX_GRID`
already bounds a 2D array to 4096² ≈ 2.7 GiB/worker. Pinned by
`test_the_device_fit_check_is_the_operative_2d_guard`.
**The IC preview is BOUNDED the same two ways** — by `protocol.grid_limit_error`
(the shared rail: a grid a session would refuse is one the preview must not
allocate either) and, before the CPU fallback, by `preview._cpu_fit_error`, which
asks `xp.device_free_bytes("cpu")` the question `_fit_error` asks of a card. Both
are needed, because the preview builds the FULL state at the requested grid and
fires on every form change, long before anyone presses Restart — while
`_fit_error` runs only at session creation and so never sees it. It had no bound
at all until 2026-07-26, when a form grid of 256⁴ (4.3e9 cells — ONE dims switch
away from the 1D default, before any Restart) found no GPU with room, fell
through to the CPU path below and allocated 34 GiB arrays until the kernel
OOM-killed the server on a 125 GiB host. The GPU path was never the hazard:
`_pick_device` simply declines. The CPU fallback is the one that needed the
check, which is why the rail has to be at the door AND the fallback has to
measure — the rail alone was enough only while it was tight enough to double as
a memory bound, and it is deliberately not that any more.
**The IC preview runs on a GPU too, and hands the VRAM straight
back** (`routers/preview.py`). It used to be CPU-only "to keep the GPU free
for sessions", which was the right instinct and the wrong trade: the preview
is built at the SESSION's grid, so at 8192² it is the same 67M-cell array the
solver evolves — measured **25.9 s on the CPU vs 0.50 s on the 3090**, paid on
every page reload AND every IC edit, while the main W panel showed the
identical array in 1.4 s because a GPU worker built it. (That asymmetry is the
tell if it ever regresses: big panel instant, small IC panel slow, and — since
`preview.py` owns its own float64 CPU backend — identical at either session
precision.) What matters is the transient PEAK, not the steady state:
**88 bytes per grid cell** (0.34 GiB at 2048², 1.38 at 4096², 5.50 at 8192²;
64 for one cat component, plateauing at 88 from three up, since `cat_wigner`
reuses its temporaries per pair). So `_pick_device` takes the CUDA device with
the most FREE VRAM and only if the build fits with 1.4× headroom — free memory
as reported by the driver, which already accounts for running sessions and
other processes — GPU previews are serialized (`_gpu_lock`) so two peaks cannot
stack, and ANY failure (OOM above all, since a session can claim the card
between the check and the build) falls back to the CPU, which is slow but
always correct. The release works only because `_build_frame` keeps every
device array in its own frame, so they die on return before `free_all_blocks()`
— measured back to **0.000 GiB** after each call. Workers release CuPy
pool blocks back to the driver on session close (nvidia-smi "used" while
running is pool recycling, not a leak).

Two things about that release are load-bearing and both were wrong until
2026-07-25. **The preview allocates from its OWN pool** (`_pool`, installed with
`cupy.cuda.using_allocator`, which is thread-local — previews run in starlette's
threadpool, workers own their own threads). `free_all_blocks()` acts on whichever
pool it is handed, and there is no per-backend allocator anywhere, so releasing
the process DEFAULT pool also returned the running workers' cached blocks to the
driver — on every IC keystroke, the exact opposite of what the free-VRAM check
above is for. Isolation is free: a cold 1 GiB allocation measured 3.1 ms against
2.6 ms pool-warm, and the release empties the pool after every preview anyway, so
every preview was already cold. **And the failure path needs a SECOND release,
after the `except` handler has exited.** While an exception propagates its
traceback still references `_build_frame`'s frame and every device array in it,
so the `finally`'s `free_all_blocks()` frees nothing (measured at 128 MiB:
`finally` alone left all of it reserved, `finally` plus the later call left none)
— and a release *inside* the handler is no better, the exception is live for the
whole handler. Untreated, a preview that OOMs at 8192² parks GiB on the card
until the next SUCCESSFUL preview, starving the solver the fallback exists to
protect. Related: that handler logs `traceback.format_exc()` and deliberately
NOT `exc_info=True`, because a LogRecord built with `exc_info` stores the
traceback, and any handler that retains records (pytest's log capture does) then
pins the frame past the release.

## Configuration (environment variables, read by backend/config.py)

| Variable | Default | Meaning |
|---|---|---|
| `WIGNERF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda:N` \| comma list (`cuda:1,cuda:0`). Names the device pool; sessions spread variant workers across it. `auto` = all CUDA devices fastest-first if cupy imports, else CPU; a list's order IS the speed ranking. Indices are PCI order (match nvidia-smi). The **host default and an enforced POLICY**: `SessionCreate.device` (Setup → Compute, restart-only) may narrow it per session but never widen it — a spec outside `xp.devices_allowed(WIGNERF_DEVICE)` (the pool **plus cpu**) is a 422 naming the pool, as is a malformed or absent one. It used to check only that the spec parsed and the card existed, so a host pinned to `cuda:1` (or to `cpu`, to keep its cards free) could be overridden by any client. `GET /api/device` returns BOTH `devices` (the pool) and `choices` — and `choices` IS `devices_allowed`, the same list the validator uses, so the Setup select can never offer a device the API refuses. That endpoint is where every HOST fact the form needs before it can create anything lives: the device lists, `WIGNERF_PRECISION`, and the per-ndim grid ceilings (`max_grid`, `max_cells`, `bytes_per_cell_2d` — see the `WIGNERF_MAX_GRID_2D` row for why those cannot come off `status`). Those three sit OUTSIDE `_probe_backend`'s `lru_cache`, so they follow a monkeypatched `config` and ride the probe's error path too. CPU is always a legal target but never appears in an `auto` pool on a CUDA host, which is why it is appended. `resolve_devices` returns CANONICAL specs (a bare `cuda` → `cuda:0`), without which that membership test would reject a device the host does offer. |
| `WIGNERF_PORT` | `8010` | Backend port (8000 belongs to urantia-library). Used by start.sh; `uvicorn --port` otherwise. |
| `WIGNERF_PRECISION` | `float64` | Default spectral working precision (`float64` \| `float32`); the Setup form's **Compute** section overrides it per session (`SessionCreate.precision`, restart-only). float32 is a PREVIEW mode — measured 3.3-3.8× faster and ~58% of the working set on CUDA, **nothing on CPU** — and it refuses auto-expand and `tol < 1e-5`. See the float64/float32 gotcha for exactly what it costs; do not make this `float32` on a host where anyone might read a result off it (setting it logs a WARNING for that reason, and an unrecognized value falls back to float64 with one too). It reaches sessions through `SessionCreate.precision`, which is `Optional` and **resolved in `_check`, not by a `default_factory`** — a hard-coded literal there once made this var decorative, advertised by `/api/device` and applied by nothing, and a factory made it collide with a gate: 2D defers float32 (M1), so a resolved host default of float32 refused EVERY 2D session over a value the client never sent (the SPA, `curl` and `scripts/ws_smoke.py --ndim 2` alike, and `tests/test_api2d.py` with them). **An omitted precision therefore resolves to the host default at ndim=1 and to `float64` at ndim=2**, and `MSG_F32_2D` fires only for an EXPLICIT float32 — a gate must refuse what was asked for. **The SPA defers rather than guesses** (`lib/config.precisionForPayload`): until the user operates the precision control — or an IMPORT supplies one — the create payload OMITS the field, so the host decides and the answer comes back in `status` (which the form then syncs to). That is what makes the `/device` probe non-load-bearing — it only seeds the displayed default and `resetToDefaults`, so a probe that fails or times out costs a device list, never the wrong precision. Sending the form's placeholder as though it were a decision is precisely how a float32 host got silently overridden. Two exceptions, both deliberate. An **imported** setup document or mp4 marks the precision CHOSEN (`importConfig` → `markPrecisionChosen`): it is the run someone exported, and reproducing it is what the document is for — without it the import silently ran at the host default and left the form showing a float32 that never happened, behind a "restart to apply" no restart could clear (`status.precision` never CHANGES, so the sync watcher never fires). And a form with **auto-expand on**, or at **ndim=2**, sends `float64` explicitly rather than deferring, because both are float64-only, so asking for either IS asking for float64 — deferring on a float32 host asks for a pair the schema refuses and 422s every create. (Belt and braces with the backend resolution above: the backend keeps a non-SPA client working, the SPA keeps the payload saying what the form shows, and the exported setup document then records the precision the run actually had.) |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session (scrub/replay window). 32 GiB ≈ 4000 four-variant records at 1024², ≈ 64000 at 256². On the VPS (32 GB RAM shared with urantia-library, Open WebUI, …) set `16384`. This is the CEILING as well as the default: `SessionCreate.history_mb` (Setup → Compute) may ask for less, never more, and status reports both `history_cap_bytes` and `history_mb_max`. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto (ncores/(2·n_variants), capped at 4). Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download. Under systemd (`PrivateTmp=yes`) the default is a private tmpfs — i.e. RAM, wiped on restart; point it at a disk path for long 1440p exports. Files are removed after download, on session close, at shutdown, or 30 min after finishing. |
| `WIGNERF_EXPORT_ENCODER` | `auto` | mp4 video encoder: `auto` \| `cpu` \| `nvenc`. `auto` = the GPU `h264_nvenc` encoder if a runtime probe succeeds (dedicated encoder block, ~3× faster at 4K, frees CPU for the render pool), else `libx264 -preset veryfast`. `cpu` forces libx264, `nvenc` forces the GPU. The bottleneck is frame RENDERING not encoding, so this only tops up the parallel render pool — and the right GPU path is the h264_nvenc ENCODER, NOT ffmpeg `-hwaccel` (a decode flag, irrelevant to our rawvideo input). The host default; `ExportSpec.encoder` (the Export panel's encoder select) overrides it per JOB, which is the right granularity — the best choice depends on what else is competing for cores at that moment. |
| `WIGNERF_EXPORT_WORKERS` | `0` | Export frame-render processes; `0` = auto (`min(cpu_count, 8)`; scaling flattens past the physical cores). Rendering a frame (matplotlib/Agg) dominates export time, so it is spread over a **spawn** `ProcessPoolExecutor` (spawn, not fork: the backend has CUDA up) while one ffmpeg encodes the ordered stream. One export at a time (`_RENDER_LOCK`) uses all of these; a job below `max(2·workers, 16)` frames renders serially to skip pool warmup. |
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling — enforced at session creation AND for auto-expand doublings; tunable BOTH ways (schema sanity rail: 16384). The UI's Nx/Np selects follow it — from **`GET /api/device`, per ndim**, NOT from `status`; see the `WIGNERF_MAX_GRID_2D` row for why. Lower it on VRAM-constrained hosts (`lib/config.axisSizeOptions` clamps its own 256 floor to the cap, so a host at 128 still gets a usable select). Measured peak per variant worker: 160 MiB at 1024², 672 MiB at 2048², 2.7 GiB at 4096², 10.0 GiB at 8192² (~4× per doubling), plus ~300 MiB of CUDA context + cuFFT plan cache per process per device. Workers spread over the pool, so what matters is the per-card share: 4 variants at 4096² is ~5.4 GiB/card at 2+2 (fits both the 3090 and the 2080 Ti); at 8192² it is ~20 GiB/card, which fits the 3090 and does NOT fit the 2080 Ti — cap by variant count, not just by grid. In a **float32** session those peaks fall to ~58% (measured arena: 448 MiB at 2048², 1.79 GiB at 4096², 7.0 GiB at 8192²), so 4 variants at 8192² is ~12 GiB/card at 2+2 — comfortable on the 3090, still too much for the 2080 Ti. float32 moves that line; it does not remove it. At the cap the session warns and keeps computing (moves still allowed). |
| `WIGNERF_MAX_GRID_2D` | `128` | Per-axis ceiling for **ndim=2** sessions. A sanity rail only — a 4D array grows as N⁴, so a per-axis cap is no guard at all (128⁴ = 268M cells is ~52 GiB per worker while every axis sits inside a 128 rail). What actually binds is the per-device fit check, `routers/sessions._fit_error` — see the GPU section. **The UI's per-axis N selects follow this from `GET /api/device`, which reports every ndim's ceiling, NOT from `status`** — `status.max_grid`/`max_cells`/`bytes_per_cell` are resolved once for the ndim of the session that is RUNNING, while the form must describe the ndim it is SHOWING, and `dims` is restart-only so the two disagree for as long as a switch waits for its restart. Reading them off `status` broke the panel in both directions (measured 2026-07-27): over a live 1D session a 2D form offered N up to 4096 against this 128 ceiling AND rendered no footprint line at all (`bytes_per_cell` is null at ndim=1 — the number that says whether a 2D session can start, missing exactly before the first 2D restart), and over a live 2D session a 1D form's N select collapsed to one option (cap 128, 1D list starting at 256, loop body never entered). `lib/config.axisSizeOptions` is the extracted, unit-tested list — extracted for that reason: both bugs were reachable only through the DOM. **Its 2D floor is 32, not 16**, because `boundary._band_mass` reports nothing below 32 cells per axis (the edge band would cover a quarter of the axis), so a 16⁴ session has no boundary watch and says so nowhere; 16⁴ stays reachable through the API and through an imported config, which the select keeps listed. |
| `WIGNERF_MAX_CELLS_2D` | `2**27` (134M) | **Total-cell** RAIL for ndim=2 — a cheap deterministic stop for absurd values (256⁴ = 4.3e9 cells is one dims switch away from the 1D default), and the only guard on a host where free memory cannot be read. It is deliberately NOT the operative limit: at the default it permits 26.0 GiB per worker, far past any card here. A fixed cell count cannot do that job — it is wrong in both directions, refusing 128×128×64×64 (13.0 GiB, one worker) on a 24 GiB card while permitting 6.5 GiB × 2 workers on an 11 GiB one — so the real check asks the driver (`_fit_error`, GPU section). Measured on an RTX 3090, 2026-07-26, on a real worker footprint (both exponent slots, an `adjust_step` pass, a frame build): **208 B/cell in float64, flat across sizes** — 0.20 GiB at 32⁴, 1.03 at 48⁴, 3.25 at 64⁴, and ~6.5 GiB/card for 4 variants at 64⁴ split 2+2, which fits both the 3090 and the 2080 Ti. **The STATE is only 4% of that** — W is real, so float64 = 8 B/cell, 0.12 GiB at 64⁴; the rest is the step's machinery at full shape, the two exponent slots (64 B/cell) largest among it. `config.BYTES_PER_CELL_2D` carries the measured stage-by-stage breakdown, and it is what M7 and M1 attack. Throughput on the same card: 610 steps/s at 32⁴, 130 at 48⁴, 35.1 at 64⁴, 13.8 at 80⁴ — so **32⁴ is for exploration and 64⁴ is a serious run** (~0.23 s per record at 8 substeps). Both the rail's refusal and the fit check's quote the estimate, and `/api/device`'s `bytes_per_cell_2d` feeds the Setup panel's footprint line so a grid that cannot start says so BEFORE the restart — from `/device` and not from `status`, or the line is absent on the one path that reaches 2D (see the `WIGNERF_MAX_GRID_2D` row). |

## Commands

```sh
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
/usr/bin/google-chrome, flags `--no-sandbox --disable-gpu`). The series
plots expose `window.__wfSeries.<which>()` (poller state) and element
screenshots of `.wf-plot` reveal what uPlot actually painted — this is how
the "flat purity line camouflaged on a gridline" bug was found.
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
   the top). Its deferred follow-ups are the milestone list below, and they
   are committed work, not optional extras.

## 2D follow-up milestones (deferred from the first 2D cut, 2026-07-26)

**These are not optional extras — they are all wanted, and each is out of the
first 2D cut only so the physics core lands verified.** Revisit as soon as 2D
is stable, tested and committed. M1, M3 and M4 are enforced meanwhile by
explicit, message-bearing refusals so a half-feature can never be mistaken for a
working one; do NOT quietly relax a gate without doing the verification it
stands in for. M1 is the one to do next — memory is *the* 2D constraint.

**M2 is DONE (2026-07-27)** and its row is kept below, struck through, because
what it stands for is now a worked example of retiring one of these gates: the
verification the gate named got done and got measured, and the numbers live in
`tests/test_propagator2d.py` and in the relativistic-2D gotcha. M2 was the
cheapest of the four, not the most valuable — the physics was already generic,
so it was a gate removal plus its verification.

| # | Milestone | v1 gate | What it needs |
|---|---|---|---|
| M1 | **float32 in 2D** | `SessionCreate` 422 on `ndim == 2` and an EXPLICIT `precision == "float32"` (an omitted one resolves to float64 there — see the `WIGNERF_PRECISION` row); the Setup panel disables the select and marks it `(1D)` | Highest value of the four: ~3.4× faster and ~58% of the working set, i.e. 2.4 → 1.4 GiB/worker at 64⁴. Needs the mixed-precision construction rules (float64 meshes, single stepping) re-verified for the CORRELATED 2D Bopp shift, and `adjust_step`'s float32 residual floor re-measured at 4D sizes (`TOL_MIN_F32` was measured at 256², and the note in `boundary.py` says the floor grows with grid size). |
| ~~M2~~ | ~~**Relativistic 2D** (`qr`, `cr`)~~ | **DONE 2026-07-27** — gate removed from `SessionCreate` and from `applyNdimInvariants` | Both named risks were measured, not argued away: the mc² cancellation is FLAT between 1D and 2D (abs error 3.6e-12 vs 4.2e-12, set by m²c²·eps and indifferent to how many momentum components enter the sum), and the massless gradient is defined as 0 at the lattice origin, which is bitwise what 1D's `sign(0) == 0` already gave. The relativistic shear diagnostic works in 2D. See the gotcha below. |
| M3 | **Auto-expand in 2D** | `SessionCreate` 422 and live `ParamChange` refusal on `ndim == 2 and auto_expand` | Boundary DETECTION already ships and warns on all four axes. The exact regrid is deferred because in 4D each axis doubling doubles a multi-GiB footprint (against 1D, where the whole grid is a rounding error next to VRAM). `GridState`, `plan_axis`, `support_cells` and `embed_window` are written generically, so what is left is `session._schedule_regrid` orchestration over 4 axes plus a memory guard that refuses a doubling it cannot afford. |
| M4 | **mp4 export of 2D runs** | `POST /sessions/{id}/export` 422 on `ndim == 2`; the Export panel states the gate | `render_mpl.FrameFigure` needs a plane-set-driven panel grid, four marginals, ⟨Lz⟩, and the (2πℏ)² purity title. Axis subscripts there must NOT use mathtext — see the plain-text gotcha below; the SPA gets real subscripts from HTML `<sub>` (`lib/axes.ts labelHtml`) but matplotlib has no cheap equivalent, so plain `px`/`py` is the default and a second smaller text artist is the only alternative worth trying. NB the SETUP-DOCUMENT half of the Export panel (`GET /sessions/{id}/setup`, and import of .json or .mp4) does work for 2D from the first cut — only the video render is gated. |
| M5 | **Cuts / slices** | the wire reserves a per-plane `mode` byte; only `mode=0` (projection) is defined | Projections are EXACT for separable states but average away fringe contrast for entangled ones, which is precisely the interesting 2D regime. A cut at fixed (y, py) keeps the interference. Purely additive to protocol v4 — no version bump, no new reduction cost (a cut is cheaper than a projection). |
| M6 | **FFT fusion** | — | The trailing inverse transform of step *n* and the leading forward transform of step *n+1* are inverses; staying in λ-space across step boundaries and merging the two half-`expT`s removes 4 of the 12 one-dimensional sweeps per 2D step, i.e. **+50%**. Not free: the per-step `real()` projection becomes per-record, which changes numerics (and `test_time_reversal`'s ~1e-9 residue budget), so it needs its own verification pass in both 1D and 2D. |
| M7 | **One exponent slot instead of two** | — | `worker._exponents` pins the full step and keeps a second slot for the substep clamped onto τ_k: 4 complex meshes = 64 B/cell, the largest single item in the 4D working set. Choosing `dt' = rem/ceil(rem/dt)` makes every substep within a record identical, so the straggler and its slot both disappear: −32 B/cell, −22% of the 4D footprint, and 1D benefits too. Changes 1D numerics (uniform substeps instead of n equal plus one short), hence not bundled with the 2D work. |

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
  streamed (x,y) plane against |ψ|² cell by cell. Measured 2026-07-26, relative
  error at t = 1: correct shift **1.45e-4 at dt = 0.02 and 3.43e-5 at 0.01
  (ratio 4.22, i.e. O(dt²))**; independent shift **7.83e-3 and 7.81e-3 (ratio
  1.003, FLAT)**. 228× worse and dt-independent, because a wrong shift is a
  different evolution OPERATOR, not a smaller time step away from the right
  one. So the assertion that matters is the dt RATIO, not a tolerance — do not
  "simplify" it to a single run. And do not extend the sweep to dt = 0.005:
  the ratio falls to 1.9 there because the residual has reached a
  dt-independent ~1.5e-5 floor set by the GRID (the Bopp shift samples U
  outside the box on the discrete θ lattice, which `exp(-iU dt)` never does),
  so the convergence check only means something where splitting error
  dominates.
- **The other 2D anchors, and what each is for.** Separability: for
  U = Ux(x) + Uy(y) and a product IC, W_2D(t) must equal the outer product of
  two 1D runs — a separable U makes dU a SUM, so the exponent factorises
  exactly, which validates the whole 4D pipeline against code the 1D suite
  already trusts (measured 5.9e-11 of the peak after 150 steps, pure
  `exp(a+b)` vs `exp(a)exp(b)` roundoff). ⟨Lz⟩: conserved by a central U to
  5.65e-6 over 300 steps and off by 3.05 (150%) for an anisotropic one — a
  5e5 separation, and the drift is IDENTICAL for the quantum and classical
  variants, which is the tell that it is the square lattice breaking
  rotational symmetry rather than physics. And a reductions-vs-naive test
  over all six planes and four marginals, which is what catches 4-axis
  fftshift bookkeeping — the likeliest porting error after the shift pairing.
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
  3.6e-14) since |dT| grows. M1 must re-measure this in float32, not inherit it.
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
- **MASSLESS (m = 0) relativistic runs lose purity to the |k| KINK, and it is
  the GRID not the step.** m = 0 became reachable in 2D only with M2 (the schema
  requires exclusively relativistic variants there, since non-relativistic
  T = p²/2m diverges). The gradient c·kᵢ/|k| is 0/0 at the origin — which IS a
  lattice point, a symmetric box with even N puts an exact 0.0 on every axis — so
  it is defined as 0 there. That is not a new 2D convention: at ndim=1 it returns
  `c·sign(p)` BITWISE, because `sqrt(k*k) == |k|` exactly for every finite
  lattice value and `sign(0)` was already 0. Pinned by
  `test_the_massless_gradient_reduces_to_the_1d_convention`.
  What massless does cost is purity, because T = c|k| is not smooth at the origin
  and its Bopp difference has slowly-decaying Fourier content in λ that a finite
  lattice truncates. Measured over 100 steps at 32⁴: **m=0 quantum −7.19e-6 at
  dt = 0.01 and −6.99e-6 at dt = 0.005 (dt-INDEPENDENT)**, m=0 classical −4.66e-5
  / −3.43e-5, against m=1 c=1 at −3.27e-9 and m=1 c=137.036 at −4.62e-12. Halving
  dt does not help; refining the MOMENTUM grid does — **7.19e-6 at N=32 falls to
  7.16e-7 at N=48**, ~10× for a 1.5× refinement. So the remedy for a user who
  needs a clean massless run is a finer momentum axis, never a smaller dt. Norm
  stays at machine precision throughout, which is what says the map is still
  exactly unitary and it is the RESOLUTION of the kink that is lossy. Only the
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
    lands in the same 3.3-3.7× band. **On CPU it buys nothing**: pyFFTW through
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
  - **Memory drops to ~58%, not 50%** — `dU_im`/`dT_im`/`H` stay float64 and are
    irreducible (24 B/cell = 1.5 GiB at 8192²). Measured per-worker CuPy arena
    on the 3090: 768 → 448 MiB at 2048², 3072 → 1792 MiB at 4096², 12.0 → 7.0
    GiB at 8192². Note the frame history is NOT affected: it is already uint16
    via `core/quantize.py`, so `WIGNERF_HISTORY_MB` buys the same record count
    either way.
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
    label that costs no line ("auto-expand (f64)", "tol ≥1e-5", and "precision
    (1D)" for the 2D gate below); the full reason in the control's `title`;
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
    applies LIVE — `SimulatorView` watches `[cfg.precision, cfg.grid.ndim]` and
    sends `auto_expand: false` to a running session, since the status→form watcher
    cannot (`status.auto_expand` does not CHANGE, so it never fires) and the
    checkbox is by then disabled, which left a session quietly expanding behind
    an unchecked, unreachable box. **ndim is in that key for the same reason
    precision is**: 2D refuses auto-expand too (M3), `applyNdimInvariants` clears
    the form, and `dims` is restart-only — so the old 1D session can go on
    regridding for as long as the switch waits for its restart, which is longer
    than the float32 case ever lasts.
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
- The worker keeps **two** exponent slots, not a cache (`_exp_main` for the
  full dt, `_exp_odd` for the substep clamped onto τ_k). The 8-entry dict
  this replaced retained seven dead complex128 pairs — two thirds of the
  whole working set — and measurably saved zero rebuilds, because dt is
  re-tuned by `adjust_step` every 20 steps so the old entries were never
  asked for again. Measured 4 workers at 1024²: 1781 → 904 MiB.
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
  does not run on a real unload, and its awaited DELETE would not complete):
  the backend learns only via the WS close, whose `finally` merely *pauses +
  detaches* (`ws_attached=False`, `set_running(False)`, stamp `last_seen`) —
  it never calls `close()`. So the session lingers in `SESSIONS` with
  alive-but-idle workers holding the full `FrameHistory` (RSS) and each
  worker's CuPy pool + cuFFT cache + exponent meshes (VRAM) until the idle
  sweeper reaps it. A reload creates a NEW session at once, so it competes
  with its own just-orphaned twin for the same RAM/VRAM (worker OOM = the
  "denied computation" symptom). THREE-part fix: (1) the frontend fires a
  `keepalive` `DELETE` on `pagehide` (`useSession.beaconDestroy`, registered
  in `SimulatorView.vue`; skips `event.persisted` bfcache) so a genuine
  departure frees resources promptly — this is safe against `recover()`, which
  is a live-tab `sock.onclose`, never a pagehide. The `DELETE` path
  (`delete_session`) drops its own `s` local (`del s`) and calls
  `_collect_closed()` right after `close()` so the history's cyclic garbage is
  gc'd + `malloc_trim`'d promptly instead of waiting up to a sweep cadence; it
  is a sync endpoint so that runs in a threadpool thread, off the event loop.
  **The `del s` and the arm-until-freed logic are not optional** — a closed
  session is a cycle, so `gc.collect()` frees NOTHING while any live reference
  roots it, and at DELETE time there are two: the handler's own `s` local (hence
  `del s`) and, if a client was attached, the `ws_endpoint` streamer coroutine
  still unwinding on the event loop (its teardown is bounded at 3 s and races
  the threadpool DELETE). So `_collect_closed` keeps `_closed_since_sweep`
  ARMED while any `weakref` in `_closed_refs` is still alive, and only clears it
  once the cycle is actually gone. Clearing it unconditionally — the original
  DELETE-path call did — meant a collect that ran a beat before the streamer
  released left the flag down, the 5 s sweeper then no-op'd, and the multi-GB
  history sat resident until a chance gen-2 gc (the observed "RSS stuck at
  13.2 GB, nothing in the logs"). Now it frees at once when nothing else roots
  it, else the sweeper retries within ~5 s. Pinned by
  `test_collect_stays_armed_while_a_closed_session_is_rooted`. VRAM comes back
  at worker-join inside `close()` (a worker finishes its in-flight record before
  seeing the stop flag, so a mid-compute large-grid quit takes a few seconds).
  (2) `WS_IDLE_TTL` is 20 s
  (down from 120), swept every 5 s (down from 15), so the crash/kill fallback
  is bounded at ~20-25 s; 20 s stays well above `recover()`'s ~1.5 s reattach,
  so a transient drop on a live tab still re-shields (`ws_attached=True`)
  before the sweep. (3) `start.sh` PINS `--ws-ping-interval/timeout 20` (these
  MATCH uvicorn's current defaults) so a HALF-OPEN drop (kill -9, laptop
  sleep, network partition — no TCP FIN) is detected by the keepalive and
  closed, running the `finally` (detach) instead of `receive_text()` blocking
  on the dead socket forever; that bounds the case at ~60 s. Explicit only so
  the keepalive can't silently regress (a `--ws` impl swap, a future default
  change). The 20 s grace is pinned by
  `test_detached_session_swept_after_grace_attached_is_shielded`.
- **The BROWSER'S WebSocket receive path is the large-grid wall, and it
  degrades with MESSAGE SIZE — not the server, not painting, not pacing.**
  Measured 2026-07-23, 4096² (32 MiB/record), Chrome + RTX 2080 Ti:
  `__wfPerf` reported 3.5 records/s and 112 MiB/s with `queue_drops: 0` and
  `fanout` 8.7 ms/frame — i.e. the client could paint ~115 fps and was idle,
  waiting on delivery — while the SAME server fed a raw Python client on the
  same machine at 402 MiB/s (14.8 rec/s). Two runs of different length
  reported 110.91 and 112.77 MiB/s: a hard ceiling, not a loop settling.
  32 MiB ÷ 112 MiB/s = 285 ms = the 3.5 fps observed. But it is NOT a fixed
  bandwidth: at 2048² (8 MiB/record) the same browser sustains 60 fps ⇒
  ≥480 MiB/s, 4× better, so the cost is per-message and grows sharply with
  payload size. This is the measurement that makes display-downsampling the
  only real fix for interactive 4096²/8192² (1024² display frames are 2 MiB;
  the same ceiling then allows ~56 rec/s), and it is why no pacing policy can
  help: the pacer targets paint time (8.7 ms), 33× off the real constraint.
  Related, also measured: a full-speed replay makes server RSS hump ~3 GB
  over 120 records at 4096² and then drain back to baseline (the sender
  running ahead into the in-flight send queue plus allocator churn —
  transient, not a leak; backpressure to a genuinely SLOW reader is bounded
  at ~4 records). `pack_frame` costs 28 ms/record at 4096² ON THE EVENT LOOP
  (two full copies: `tobytes()` then `b"".join`), capping replay at ~35 rec/s
  server-side before the transport is even involved.
- **`free_all_blocks()` frees only what is FREE — drop the worker's own arrays
  first.** `_release_gpu_pool` runs in `run()`'s `finally`, where `_run`'s
  locals (W, prop) are gone but ATTRIBUTES are not: the two exponent slots
  still hold 4 complex128 meshes, so the release left exactly that behind —
  256 MiB at 2048², 1.0 GiB at 4096², 4.0 GiB at 8192², per worker. Those
  returned to the pool only when the worker was collected (session↔worker
  cycle ⇒ needs gc) and to the DRIVER only at some LATER worker's
  `free_all_blocks()`, which is why VRAM used to come back on the SECOND
  "Restart session" and not the first. `self._exp_clear()` now runs before
  the GPU guard (on CPU those meshes are host RAM, held just as long), and
  the worker's own cuFFT plan cache (per thread AND device) is cleared in the
  same place. Measured at 2048², one QN worker, gc disabled: release went
  `used 256 → 256 MiB` before, `256 → 0` after; steady-state process VRAM
  1094 → 838 MiB, and the two-restart staircase became one step.
- **Two more things kept a closed session's RAM resident, both found only by
  measuring RSS across a Restart (2026-07-23).** (1) `ttl_sweeper` iterated
  `SESSIONS` inline, and a `for` target outlives its loop — so the sweeper
  held the LAST session it examined across its sweep sleep, and FOREVER once
  SESSIONS emptied, because an empty loop never rebinds the name. 3.2 GB
  survived DELETE + explicit `gc.collect()` at 4096²/100 records; tens of GB
  at 8192². The loop now lives in `_sweep_idle`, whose frame dies on return
  (pinned structurally by `test_ttl_sweeper_never_binds_a_session_in_its_own_
  frame`). (2) glibc's mmap threshold is DYNAMIC — 128 KiB initially,
  ratcheting up to the size of each freed mmap'd block, capped at 32 MiB. A
  4096² record is 32.03 MiB (just over the cap, always mmap'd, self-returning)
  but a 2048² record is 8.02 MiB, so after the ratchet those come from the
  arena and `free()` never lowers RSS: 1459 MiB still held at 2048²/300
  records, 964 MiB of it recovered by `malloc_trim(0)`, which
  `_collect_closed` now calls. **Record size decides which of these you
  see**, so test memory at more than one grid — 4096² looked clean while
  2048² sat at ~9.8 GB after two Restarts.
- **A closed session's history is CYCLIC garbage — freeing it needs the
  collector, not refcounting.** `SimSession.workers` holds each
  `SolverWorker` and `worker.session` holds the session back, so after
  `close()` the pair (and the whole `FrameHistory` hanging off it) is
  unreachable but not refcount-free. On an otherwise idle server a gen-2
  collection may not run for many minutes, so tens of GB stay resident long
  after Restart and look EXACTLY like a leak. `session._collect_closed()`
  makes it deterministic: `close()` sets `_closed_since_sweep` and the TTL
  sweeper does one `gc.collect()` per sweep that had a close (off the event
  loop; collection cost scales with tracked CONTAINERS, not with the bytes
  they point at, so a multi-GB history is cheap to reap). Pinned by
  `test_closed_history_needs_the_cyclic_collector`, which asserts BOTH
  halves — the history survives `close()` + `del`, and dies on
  `_collect_closed()`. If the back-reference is ever removed, that test
  fails loudly rather than silently keeping a now-pointless collect.
- **Do not chase "leaked" objects with `gc.get_referrers` alone — it cannot
  see frame locals.** The 2026-07-23 hunt for stray `SimSession`s (a sweeper
  diagnostic listing live-but-unregistered sessions and their referrer
  types) reported `{'list': 2, 'dict': 1}` and was wrong twice over: the two
  lists were the diagnostic's OWN `live`/`leaked` locals, and the one dict
  was a `SolverWorker.__dict__` — i.e. the ordinary cycle above, still
  uncollected because the diagnostic never ran `gc.collect()` first. Verified
  by reproducing the exact signature with `gc.disable()`; every real
  lifecycle path (create/delete, reconnect churn, delete-while-streaming,
  abandoned-then-closed) leaks nothing once collected. Two traps to remember:
  a referrer snapshot must exclude its own containers, and in CPython 3.12
  `gc.get_referrers` does NOT report an object held by a plain local
  variable (fast locals are invisible unless `f_locals` was materialized) —
  so "no coroutine frame holds it" is a conclusion that instrument can never
  support. Use a `weakref` + explicit `gc.collect()` to decide whether
  something leaked, and thread stacks (`sys._current_frames()`) to find who
  is still running.
- **The boundary warning is ONE SHORT LINE, and the cells it names are
  drawable.** It can clear again within a couple of records as a state drifts
  back out of the band, so the old sentence — which also explained periodicity
  and offered a remedy — was regularly gone before it had been read. What
  survives is only what cannot be got elsewhere: `⚠ W(x,y,px,py,t) has reached
  the px edge — 1.3e-4 of its integral is in the outer 4 cells.` ("integral",
  not "probability": the quantity is ∫W over the band, and W is signed.) The
  reasoning and the remedy moved into the span's `title` (`boundaryTitle`), which
  a hover holds still. The cell COUNT comes from the server (`band` in the
  boundary payload, per axis) rather than being re-derived, because it must be
  the width the mass was actually measured with. **`lib/cells.ts` mirrors
  `boundary.edge_band` separately, for DRAWING only** — a second "cells" toggle
  (`wignerf.cells`, its own key, so "Reset setup to defaults" leaves it alone)
  paints the computed lattice faintly on the W panels and the IC preview with the
  edge-band cells brighter, so the number in that warning points at something
  visible. The mirror exists because the overlay follows the PAINTED frame, which
  during a scrub across an auto-expand boundary is not the live window the
  server's `band` describes. Ticks and cells are INDEPENDENT layers
  (`showTicks`/`showCells`), and the lattice is dropped when more than ~200 of
  its lines would land in the visible window — a COUNT not a pixel test, so
  zooming in brings it back, which is what makes it usable at 1D's 4096
  (measured: N=64 draws 110 lattice + 20 band lines; N=1024 draws 0 + 132).
  **The three toggles share ONE row** (`flex`, natural widths, labels
  "auto-expand", "grid", "cells"), which is what pays for the third control in a
  320px column: verified at the real width, one line and `scrollWidth ===
  clientWidth` at ndim=1, ndim=1/float32 and ndim=2. That budget is also why the
  auto-expand gate marker is `(1D)` / `(f64)` rather than a spelled-out clause —
  it stays permanent and visible, which is the load-bearing half, and the full
  reason is in the tooltip. Cell lines are deliberately NOT in `ExportSpec`: the
  mp4 is a finished artefact, and this is a diagnostic you reach for.
  **And the edge finding is stated ONCE.** The IC-preview check and the runtime
  watch describe the same thing — record 0 IS the IC — so an IC sitting at the
  edge produced two amber messages, a header line and a paragraph under the IC
  plot, with the same number in both. `preview_warnings` now takes an
  `edge_axes` out-list: given one it puts `(axis, band_mass)` pairs there and
  keeps the sentence OUT of `warnings` (given none it still returns the full
  prose, so no other caller changes), `from_spec` returns it, and
  `routers/preview._respond` sends it as **`X-Wignerf-Edge: axis:mass,…`** — its
  own header precisely so the client is not pattern-matching a sentence.
  `ICEditor.edgeNotice` then renders the SET DIFFERENCE against the session's
  tripped axes. Differencing rather than simply deleting the IC-side
  message is the point: it keeps the one case the header cannot cover — an IC you
  have EDITED but not yet restarted into, whose new axes the running session
  knows nothing about. Verified: form == session shows exactly one line; editing
  the IC out to the py edge leaves the header on x and adds "this IC reaches the
  py edge" alone.
  **What it differences against must SURVIVE A DISMISSAL**, hence
  `session.boundary?.axes ?? status.boundary?.axes`: the first is the transient
  event, and the header's own `×` sets it to null — so differencing against it
  alone meant dismissing the header made this line appear instead, i.e. the one
  fact stated twice after all, just sequentially. `status.boundary` is the
  session's standing `boundary_state`, always present and untouched by the ×; the
  event stays first only because it is fresher than the 1 s status cadence.
- **The header's TRANSIENT notices are an absolute overlay, not flow content —
  the W panels must never move because a message arrived.** `restartNeeded`,
  `boundaryText`, `paramFlash` and `regridFlash` come and go while you are
  watching the heatmaps, and the header is `flex-wrap` above a `flex-1` main: as
  inline children, each arrival wrapped the header to a second line and moved
  the panels down 32 px at 1280 px wide, then back up when it cleared (measured
  `HEADER HEIGHTS SEEN: 38,70` / `PANEL TOPS SEEN: 46,78`). They now live in a
  `absolute left-0 right-0 top-full z-30` strip anchored to the `relative`
  header, so they are out of the vertical flow in BOTH layouts — verified with a
  warning toggling 43× in 24 s: `hdr`, `panTop`, `panH` and `plotsTop` each take
  exactly one value, landscape and portrait. The trade is deliberate: the strip
  OVERLAYS the top ~24 px of the columns, i.e. it briefly hides the panel's label
  chip and the first plot's title — chrome, never data — which is much cheaper
  than relaying out the thing being watched, and it is dismissible. Full width so
  no message has to be truncated to fit a header row. **The float32 badge stays
  inline on purpose** (a permanent property of the session, set before there is
  anything to watch), and so does `createError` — but it must be inline INSIDE
  the header (`basis-full`, its own flex-wrap line), not after it. Outside, it
  landed at exactly the y the `top-full` strip is anchored to, and the amber
  "setup changed — restart to apply" was painted straight OVER the error text.
  That pairing is not exotic: **every restart that 422s sets both**, because the
  form still differs from the session it failed to create. Measured on
  128×128×128×64 — the fit refusal was on screen the whole time and unreadable.
  Do not "tidy" these back into the header row, and do not move `createError`
  back out of the header.
- **A FAILED restart leaves the app session-less, and three things used to hide
  that.** `useSession.create` calls `destroy()` BEFORE it posts, so a 422 deletes
  the old session and leaves `info`/`status` null with the form intact. Found on
  a 2D grid of 128×128×128×64: (1) the server's refusal was rendered but painted
  over by the transient strip (above); (2) the header read **"connecting…"**
  forever, a false progress report on a socket that will never open — it is now
  suppressed while `createError` is set; and (3) the transport button stayed pink
  "Solve" and ENABLED, doing nothing on click, because `solveBlocked` only knew
  about an invalid setup — `ControlBar.noSession` (`!props.status`) now disables
  it and says so in the title. Being briefly disabled just after a create is the
  safe direction, the same argument the "Apply live" button's `computing` gate
  makes.
  **And the Setup panel's 2D footprint line now says when a grid cannot fit at
  all.** `GET /api/device` reports each pool device's `total_bytes` (added
  `xp.device_total_bytes`); the panel compares its per-device estimate against
  the SMALLEST of them — the workers spread, so the smaller card binds, the same
  property `_fit_error` rests on — and turns red with "will not fit this host".
  TOTAL, not free, deliberately: total is static, so the panel needs no polling
  and can never contradict the server, whose live-free refusal is a strict
  superset. Before this the line was plain grey at **26.00 GiB/device on a host
  whose largest card is 24**, and `WIGNERF_MAX_CELLS_2D` did not cover it either
  — 128×128×128×64 is 134,217,728 cells, EXACTLY 2²⁷, so the `>` rail is not
  tripped. At the cap and far past the hardware, and reading as fine.
- **A marginal's NEGATIVE part measures the noise floor its edge-band mass is
  read against, and on a coarse grid that floor is ABOVE the trigger.** ρ(x) is
  a probability density, so any negative value in it is pure numerical error —
  which makes `(|ρ|−ρ)/2` summed over the axis a free, self-calibrating error
  bar (elementwise only, so it costs one reduction on numpy and cupy alike; no
  boolean indexing, no extra device sync). It has to be used, because the floor
  is the Nyquist truncation of the state's own Fourier tail,
  `exp(−(πσ_q/dx)²/2)` — measured 2026-07-26 on a coherent state parked at
  |q| = 2, nowhere near an edge: **N=32 predicted 5.17e-5 / measured 5.35e-5;
  N=48 2.27e-10 / 1.66e-10; N=64 7.15e-18 / 2.29e-13** (below ~1e-13 other
  error sources take over). At 32⁴ — a size this project *recommends* for
  exploration — that is 50× the 1e-6 trigger, so the band mass IS noise:
  negative in 96 of 204 readings, sign flipping every record or two. Ungated
  that produced **79 boundary state changes in 201 records, and 243 WS events
  in 25 s in a browser**, each rewriting the header warning — which wraps the
  `flex-wrap` header and so moved the W panels 32 px at 1280 px wide, on and
  off, for the whole run. That is what "the heatmaps jump while computing" was,
  in both interactive and batch mode (compute-linked, hence absent in
  playback), and it was neither the scrollbar nor the time cursor. Fix:
  `EdgeState` carries `noise` and an axis trips only above
  `max(threshold, EDGE_NOISE_MARGIN·noise)`, plus `session._confirm_edge`
  requires `EDGE_CONFIRM` consecutive records in BOTH directions. 8×/4 measured
  zero false announcements against a real approach still announced at record 3;
  see boundary.py for the full sweep. Three properties are load-bearing: the
  gate is **inert wherever the detector already worked** (1D N=256..4096 and 2D
  from N=48 have floors three orders under the trigger, so 1e-6 alone still
  decides — 1D behaviour is untouched); a **ratio-only** gate is not enough,
  because a state genuinely over the edge rings in proportion (band/noise 11
  there against 4.2 for the worst noise blip, which is why the margin is 8 and
  not 12); and the **first reading per slot is exempt**, or an IC that starts at
  the edge — a paused session with exactly one record — would never warn.
- **Secular E drift + slow purity decay = boundary wrap, not a solver
  bug.** The spectral domain is a torus: when a state's orbit + ~5σ tails
  reach the x or p edge, mass wraps through the seam and the run faithfully
  evolves the WRONG (torus) problem. Tells: IC norm deficit >> 1e-6, the
  4σ edge warning, secular (not oscillatory-bounded) drifts. Fix: enlarge
  the domain — or enable auto-expand, which detects the approach (edge-band
  mass of the total sampled W, also checked at IC-preview time — the
  per-component 4σ boxes alone miss interference terms) and regrids
  exactly before mass wraps. Verified: same cat state, [-6,6]x[-7,7] gives E drift 2e-3;
  [-12,12]² gives 4e-6 with purity conserved to 5e-12 — the discrete map
  is exactly unitary for contained states (healthy E behavior is a BOUNDED
  O(dt²) oscillation from Strang splitting, never a drift).
- **Growing ΔX·ΔP in the RELATIVISTIC variants only = anharmonic shear, not
  a bug.** T = c√(p²+m²c²) carries a −p⁴/(8m³c²) term, so ω depends on E
  (δω = −3E/(8c²)) and the ensemble shears at k = t·r²·3/(8c²). The shear is
  symplectic: purity and det C are conserved and the LOWER envelope of ΔX·ΔP
  stays exactly at ħ/2, while the upper one grows ∝ t² (modulated at 2ω).
  Tells that it is physics: halving dt leaves it identical while the E(t)
  splitting oscillation drops 4×, it scales as 1/c⁴, purity stays flat.
  Non-relativistic harmonic H is exactly quadratic ⇒ no shear ⇒ flat.
  Measured: coherent state at (2,0) in x²/2 with c = 137.036 → 2e-5 at
  t = 100 (analytic σ²k²/2 = 1.6e-5). Pinned by
  `test_relativistic_uncertainty_shear` — and since M2 there is a 2D one of the
  same name in `test_propagator2d.py`, which measures the same four tells at
  c = 10 because 1/c⁴ makes c = 137 unaffordable over a 32⁴ grid; see the
  relativistic-2D gotcha above for its numbers.
