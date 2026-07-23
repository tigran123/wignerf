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
  only after the frontier record was SENT. The transport must stay
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
- **Boundary watch / auto-expand** (`core/boundary.py`): detection is
  ALWAYS on — every record, each worker sums the outer edge band of the
  ρ/φ marginals it already computed (host-side, O(Nx+Np), no extra device
  sync) and `session.report_edge` posts a `boundary` WS event on state
  change (band = max(4, N/32) cells/side, trigger 1e-6 — expansion
  prevents wrap, it cannot repair it, so it must fire while edge mass is
  negligible). The `auto_expand` toggle (SessionCreate field AND
  live-appliable via ParamChange) governs only the RESPONSE: an exact
  fixed-lattice regrid. dx/dp and the lattice anchor are FROZEN at session
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
  self-restoring.
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
  match the Setup panel (ℏ, "run-ahead"), and the series y-window +
  tick decimals reproduce that component's `scales.y.range` rule
  (`render_mpl.series_ylim`); the "grid lines on plots" toggle rides along
  in `ExportSpec.show_grid` and governs EVERY plot in the frame — charts
  get uPlot's `#3f3f46`, the W panels get `GridOverlay.vue`'s
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
  `comment` tag as JSON. Progress: `export` events on the session WS plus a
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
- **Parameter policy**: U(x), c, mass, hbar_eff, tol, dt_sign, auto_expand
  apply live at the frontier; grid/IC/variant-set changes require a session
  restart (auto-expand moves the LIVE grid; the Setup panel shows it and
  offers "adopt" to copy it into the form).
  `apply_params` compares against what is LIVE and drops the fields that
  did not change — no worker command, no `param_log` entry, no
  `params_applied`, and nothing at all if the whole message is a no-op (the
  UI sends complete fields; "Apply live" always carries the U(x) draft, so
  the log used to fill with U changes that never happened and an export's
  "how to reproduce this" block lied about its own frames). Entries carry
  `before` as well as `applied`, so the block renders "ℏ 1 → 2" and
  `describe.state_at` rewinds the header physics to the FIRST exported
  record instead of quoting the values the run ended with. Live changes are
  visible in the UI: the header flashes "✓ applied …", a Physics field
  whose form value differs from `status` renders amber (they apply on
  blur/Enter), and "Apply live" is greyed with an inline reason when the
  draft already IS the live U.
  The setup form gates the transport: while the potential draft is invalid
  for the active variant families or the IC preview errors, Solve (button
  AND Space) is disabled and "Use at restart"/"Apply live" are greyed —
  a computation must never run behind a visibly broken form.
- **Sessions always start paused** (both modes): computation begins only on
  the explicit Solve/Play command. The transport button label predicts its
  effect: Solve = will compute, Play = pure history playback, Pause while
  running. Playback-only runs (play pressed behind the frontier, or after a
  finished run-ahead) auto-pause AT the frontier — they never roll into
  computation; only an explicit Solve does (`SessionClock.stop_at_frontier`).
  A run-ahead that REACHES t2 ends the run too (`advance_cursor`, same
  delivery-aware condition): its workers already idle there, and leaving
  `running` set froze the transport on "Pause" forever and locked out
  every paused-only action (pinned by `test_runahead_starts_paused_and_
  stops_at_t2`).
  Setup persists in browser localStorage; "↺ defaults" (IC editor) and
  "Reset setup to defaults" (Setup panel) restore defaults in the form and
  mark the session restart-dirty.
- **Potentials** (`core/potential.py`): tokenize-screen (security boundary)
  → sympy parse → per-family validity. The Bopp arguments are REAL
  (x ∓ ħθ/2, complex dtype only): quantum needs U real+finite on the
  extended range [x1 − πħ/(2dp), x2 + πħ/(2dp)] (Abs is quantum-valid);
  classical needs DiracDelta-free dU/dx (Heaviside steps are quantum-only).
- **ICs** (`core/initial.py`): Gaussian mixtures (independent σx, σp) and
  cat states (analytic pairwise cross-Wigner; σp derived = ħ/(2σx)).

- **Purity** γ = 2πℏ_eff∬W²dxdp (= Tr ρ²) is computed per record and
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
376 (+39%). Previews always run on CPU by design. Workers release CuPy
pool blocks back to the driver on session close (nvidia-smi "used" while
running is pool recycling, not a leak).

## Configuration (environment variables, read by backend/config.py)

| Variable | Default | Meaning |
|---|---|---|
| `WIGNERF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda:N` \| comma list (`cuda:1,cuda:0`). Names the device pool; sessions spread variant workers across it. `auto` = all CUDA devices fastest-first if cupy imports, else CPU; a list's order IS the speed ranking. Indices are PCI order (match nvidia-smi). |
| `WIGNERF_PORT` | `8010` | Backend port (8000 belongs to urantia-library). Used by start.sh; `uvicorn --port` otherwise. |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session (scrub/replay window). 32 GiB ≈ 4000 four-variant records at 1024², ≈ 64000 at 256². On the VPS (32 GB RAM shared with urantia-library, Open WebUI, …) set `16384`. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto (ncores/(2·n_variants), capped at 4). Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download. Under systemd (`PrivateTmp=yes`) the default is a private tmpfs — i.e. RAM, wiped on restart; point it at a disk path for long 1440p exports. Files are removed after download, on session close, at shutdown, or 30 min after finishing. |
| `WIGNERF_EXPORT_ENCODER` | `auto` | mp4 video encoder: `auto` \| `cpu` \| `nvenc`. `auto` = the GPU `h264_nvenc` encoder if a runtime probe succeeds (dedicated encoder block, ~3× faster at 4K, frees CPU for the render pool), else `libx264 -preset veryfast`. `cpu` forces libx264, `nvenc` forces the GPU. The bottleneck is frame RENDERING not encoding, so this only tops up the parallel render pool — and the right GPU path is the h264_nvenc ENCODER, NOT ffmpeg `-hwaccel` (a decode flag, irrelevant to our rawvideo input). |
| `WIGNERF_EXPORT_WORKERS` | `0` | Export frame-render processes; `0` = auto (`min(cpu_count, 8)`; scaling flattens past the physical cores). Rendering a frame (matplotlib/Agg) dominates export time, so it is spread over a **spawn** `ProcessPoolExecutor` (spawn, not fork: the backend has CUDA up) while one ffmpeg encodes the ordered stream. One export at a time (`_RENDER_LOCK`) uses all of these; a job below `max(2·workers, 16)` frames renders serially to skip pool warmup. |
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling — enforced at session creation AND for auto-expand doublings; tunable BOTH ways (schema sanity rail: 16384). The UI's Nx/Np selects follow it (status carries `max_grid`). Lower it on VRAM-constrained hosts. Measured peak per variant worker: 160 MiB at 1024², 672 MiB at 2048², 2.7 GiB at 4096², 10.0 GiB at 8192² (~4× per doubling), plus ~300 MiB of CUDA context + cuFFT plan cache per process per device. Workers spread over the pool, so what matters is the per-card share: 4 variants at 4096² is ~5.4 GiB/card at 2+2 (fits both the 3090 and the 2080 Ti); at 8192² it is ~20 GiB/card, which fits the 3090 and does NOT fit the 2080 Ti — cap by variant count, not just by grid. At the cap the session warns and keeps computing (moves still allowed). |

## Commands

```sh
# backend tests (GPU tests auto-skip without cupy/CUDA)
cd backend && .venv/bin/pytest

# live-server streaming smoke test (no browser)
.venv/bin/uvicorn main:app --port 8010 --ws-per-message-deflate false &
.venv/bin/python scripts/ws_smoke.py

# throughput benchmark
.venv/bin/python scripts/bench.py [cpu] [cuda:1]

# frontend: decoder golden test + typecheck + build
cd frontend && npm run test && npm run build

# mp4 export needs the system ffmpeg (libx264); its tests skip without it
ffmpeg -version

# dev loop: uvicorn (above) + `npm run dev`, open http://localhost:5173
# prod-style: ../start.sh, open http://localhost:8010
```

After changing the binary protocol: bump `VERSION` in BOTH protocol files,
regenerate the fixture (`scripts/gen_fixture.py`), and update the vitest.

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
grids) and the measured refresh interval.

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
   modular for it), multi-D.

## Conventions / gotchas

- Do not reference the old project website domain anywhere in wignerf —
  it expired (old code/comments elsewhere in the repo may keep theirs).
- Nx, Np must be even (shader unshift + fftshift symmetry); powers of 2
  for FFT speed. Grid warns, API schema enforces evenness.
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
  grid or fftshift bookkeeping.
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
- **The solver is float64 and stays float64 — this was measured, not
  assumed.** float32 saves nothing where it looks like it would: the frame
  history (`WIGNERF_HISTORY_MB`, the big RAM number) is already uint16 via
  `core/quantize.py`, so the solver dtype buys zero extra records. float64
  lives only in the per-worker device working set. And complex64 stepping
  costs the diagnostics that this project navigates by: measured over 2000
  steps at 256², Δpurity −2.4e-4 and ΔE +9.4e-4, both SECULAR — i.e. exactly
  the boundary-wrap signature in the gotcha below, from a perfectly contained
  state — with ΔX·ΔP noise of 1.3e-3, 150× the ~7e-6 relativistic shear that
  `test_relativistic_uncertainty_shear` pins. (float64 for comparison:
  +6.7e-13, bounded +4.2e-5, +5.1e-8.) Exponent construction could not be
  float32 even in a mixed scheme: relativistic `dT` built in float32 has max
  abs error 455 against max |dT| = 228 — 200% — because mc² cancels inside a
  difference of ~1.9e4-magnitude terms. If throughput is ever the goal,
  complex64 cuFFT is genuinely ~5× faster and could be an explicit opt-in
  "preview" mode, but it must never be the default and never the setting a
  physics claim is made from.
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
  held the LAST session it examined across its 15 s sleep, and FOREVER once
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
  `test_relativistic_uncertainty_shear`.
