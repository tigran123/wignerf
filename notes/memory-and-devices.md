# Device fit, VRAM and the IC preview — the measurements

Why `_fit_error` asks the driver instead of counting cells, what the IC preview
costs and how it hands VRAM back, and the per-cell footprint tables. Split out
of `CLAUDE.md` on 2026-08-05 (see `notes/precision.md` for why). Read this
before touching `core/fit.py`, `routers/sessions._fit_error`,
`routers/preview.py`'s device choice, or `config.BYTES_PER_CELL_2D`.


## The GPU section, as it stood before the split


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
count** (`routers/sessions._fit_error`) — **and since M3 (2026-08-01) the same
question is asked again whenever auto-expand wants to DOUBLE the grid.** The
arithmetic behind both lives in `core/fit.py` so the two cannot drift; the
messages do not, because the create-time advice ("drop a variant, change
device") is unavailable mid-run. See the 2D-auto-expand gotcha for the
regrid-time inequality and the measured transient factor it carries. **Its refusal describes the POOL, and
the ROOMIEST device decides which of two stories it tells** — it used to name
whichever assigned device sorted first and always close with "pick a device with
more room", which on the real pair said *"cuda:0 has 8.9 GiB free … pick a device
with more room"* for a 128×128×128×64 grid: the small card named, a roomier one
implied, and the 3090's 23.6 GiB unable to hold one 22.0 GiB worker either (its
budget is 20.97). The honest reading of that is "so what, I have cuda:1", and it
is wrong. So: if the per-worker footprint exceeds EVERY device's budget it says
*no device in the pool can hold even one*, names the roomiest with its
free/installed figures, and states that dropping a variant or changing device
will not help — because neither will, and only the grid is left. Otherwise a worker does fit somewhere,
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
`test_the_device_fit_check_is_the_operative_guard`.
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



## The configuration table, as it stood before the 2026-08-05 split

| `WIGNERF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda:N` \| comma list (`cuda:1,cuda:0`). Names the device pool; sessions spread variant workers across it. `auto` = all CUDA devices fastest-first if cupy imports, else CPU; a list's order IS the speed ranking. Indices are PCI order (match nvidia-smi). The **host default and an enforced POLICY**: `SessionCreate.device` (Setup → Compute, restart-only) may narrow it per session but never widen it — a spec outside `xp.devices_allowed(WIGNERF_DEVICE)` (the pool **plus cpu**) is a 422 naming the pool, as is a malformed or absent one. It used to check only that the spec parsed and the card existed, so a host pinned to `cuda:1` (or to `cpu`, to keep its cards free) could be overridden by any client. `GET /api/device` returns BOTH `devices` (the pool) and `choices` — and `choices` IS `devices_allowed`, the same list the validator uses, so the Setup select can never offer a device the API refuses. That endpoint is where every HOST fact the form needs before it can create anything lives: the device lists, `WIGNERF_PRECISION`, and the per-ndim grid ceilings (`max_grid`, `max_cells`, `bytes_per_cell_2d` — see the `WIGNERF_MAX_GRID_2D` row for why those cannot come off `status`). Those three sit OUTSIDE `_probe_backend`'s `lru_cache`, so they follow a monkeypatched `config` and ride the probe's error path too. CPU is always a legal target but never appears in an `auto` pool on a CUDA host, which is why it is appended. `resolve_devices` returns CANONICAL specs (a bare `cuda` → `cuda:0`), without which that membership test would reject a device the host does offer. |
| `WIGNERF_PORT` | `8010` | Backend port (8000 belongs to urantia-library). Used by start.sh; `uvicorn --port` otherwise. |
| `WIGNERF_PRECISION` | `float64` | Default spectral working precision (`float64` \| `float32`); the Setup form's **Compute** section overrides it per session (`SessionCreate.precision`, restart-only). float32 is a PREVIEW mode — measured 3.3-3.8× faster in 1D but only **1.5-2.6× in 2D**, and ~54-55% of the working set on CUDA, **nothing on CPU** — and it refuses auto-expand and `tol < 1e-5`. See the float64/float32 gotcha for exactly what it costs; do not make this `float32` on a host where anyone might read a result off it (setting it logs a WARNING for that reason, and an unrecognized value falls back to float64 with one too). It reaches sessions through `SessionCreate.precision`, which is `Optional` and **resolved in `_check`, not by a `default_factory`** — a hard-coded literal there once made this var decorative, advertised by `/api/device` and applied by nothing, and a factory once collided with the (now retired) M1 gate by refusing 2D sessions over a value the client never sent: a gate must refuse what was ASKED FOR. An omitted precision now resolves to the host default at **every** ndim, so a float32 host gives a 2D session float32, which is where the memory saving actually matters. **The SPA defers rather than guesses** (`lib/config.precisionForPayload`): until the user operates the precision control — or an IMPORT supplies one — the create payload OMITS the field, so the host decides and the answer comes back in `status` (which the form then syncs to). That is what makes the `/device` probe non-load-bearing — it only seeds the displayed default and `resetToDefaults`, so a probe that fails or times out costs a device list, never the wrong precision. Sending the form's placeholder as though it were a decision is precisely how a float32 host got silently overridden. Two exceptions, both deliberate. An **imported** setup document or mp4 marks the precision CHOSEN (`importConfig` → `markPrecisionChosen`): it is the run someone exported, and reproducing it is what the document is for — without it the import silently ran at the host default and left the form showing a float32 that never happened, behind a "restart to apply" no restart could clear (`status.precision` never CHANGES, so the sync watcher never fires). And a form with **auto-expand on** sends `float64` explicitly rather than deferring, because auto-expand is float64-only, so asking for it IS asking for float64 — deferring on a float32 host asks for a pair the schema refuses and 422s every create. (Belt and braces with the backend resolution above: the backend keeps a non-SPA client working, the SPA keeps the payload saying what the form shows, and the exported setup document then records the precision the run actually had.) |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session (scrub/replay window). 32 GiB ≈ 4000 four-variant records at 1024², ≈ 64000 at 256². On the VPS (32 GB RAM shared with urantia-library, Open WebUI, …) set `16384`. This is the CEILING as well as the default: `SessionCreate.history_mb` (Setup → Compute) may ask for less, never more, and status reports both `history_cap_bytes` and `history_mb_max`. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto (ncores/(2·n_variants), capped at 4). Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download. Under systemd (`PrivateTmp=yes`) the default is a private tmpfs — i.e. RAM, wiped on restart; point it at a disk path for long 1440p exports. Files are removed after download, on session close, at shutdown, or 30 min after finishing. |
| `WIGNERF_EXPORT_ENCODER` | `auto` | mp4 video encoder: `auto` \| `cpu` \| `nvenc`. `auto` = the GPU `h264_nvenc` encoder if a runtime probe succeeds (dedicated encoder block, ~3× faster at 4K, frees CPU for the render pool), else `libx264 -preset veryfast`. `cpu` forces libx264, `nvenc` forces the GPU. The bottleneck is frame RENDERING not encoding, so this only tops up the parallel render pool — and the right GPU path is the h264_nvenc ENCODER, NOT ffmpeg `-hwaccel` (a decode flag, irrelevant to our rawvideo input). The host default; `ExportSpec.encoder` (the Export panel's encoder select) overrides it per JOB, which is the right granularity — the best choice depends on what else is competing for cores at that moment. |
| `WIGNERF_EXPORT_WORKERS` | `0` | Export frame-render processes; `0` = auto (`min(cpu_count, 8)`; scaling flattens past the physical cores). Rendering a frame (matplotlib/Agg) dominates export time, so it is spread over a **spawn** `ProcessPoolExecutor` (spawn, not fork: the backend has CUDA up) while one ffmpeg encodes the ordered stream. One export at a time (`_RENDER_LOCK`) uses all of these; a job below `max(2·workers, 16)` frames renders serially to skip pool warmup. |
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling — enforced at session creation AND for auto-expand doublings; tunable BOTH ways (schema sanity rail: 16384). The UI's Nx/Np selects follow it — from **`GET /api/device`, per ndim**, NOT from `status`; see the `WIGNERF_MAX_GRID_2D` row for why. Lower it on VRAM-constrained hosts (`lib/config.axisFloor` clamps the 256 floor to the cap, so a host at 128 still gets a usable select — and `setNdim` asks the same function, so a dims switch cannot land N over a lowered ceiling either). Measured peak per variant worker with the WHOLE-RECORD harness (`bench.py --footprint`, the same instrument as the 2D row): **192 B/cell in float64 and 104 in float32** since M7 dropped the second exponent slot (was 224/120) — 0.19 / 0.75 / 3.00 / 12.00 GiB at 1024² / 2048² / 4096² / 8192², and 0.10 / 0.41 / 1.63 / 6.50 in float32 (~4× per doubling), plus ~300 MiB of CUDA context + cuFFT plan cache per process per device. **These are HIGHER than the step-loop figures this row used to quote** (160 MiB / 672 MiB / 2.7 / 10.0 GiB) and that is not a regression: a step loop misses `adjust_step`'s transient and the frame build, exactly as the 2D row warns, so the old numbers under-reported a real worker. Workers spread over the pool, so what matters is the per-card share: 4 variants at 4096² is ~6.0 GiB/card at 2+2 (fits both the 3090 and the 2080 Ti); at 8192² it is ~24 GiB/card, which does **not** fit even the 3090 — cap by variant count, not just by grid. In a **float32** session those peaks fall to ~54%, so 4 variants at 8192² is ~13 GiB/card at 2+2 — comfortable on the 3090, still too much for the 2080 Ti. float32 moves that line; it does not remove it. At the cap the session warns and keeps computing (moves still allowed). |
| `WIGNERF_MAX_GRID_2D` | `128` | Per-axis ceiling for **ndim=2** sessions. A sanity rail only — a 4D array grows as N⁴, so a per-axis cap is no guard at all (128⁴ = 268M cells is ~44 GiB per worker while every axis sits inside a 128 rail). What actually binds is the per-device fit check, `routers/sessions._fit_error` — see the GPU section. **The UI's per-axis N selects follow this from `GET /api/device`, which reports every ndim's ceiling, NOT from `status`** — `status.max_grid`/`max_cells`/`bytes_per_cell` are resolved once for the ndim of the session that is RUNNING, while the form must describe the ndim it is SHOWING, and `dims` is restart-only so the two disagree for as long as a switch waits for its restart. Reading them off `status` broke the panel in both directions (measured 2026-07-27): over a live 1D session a 2D form offered N up to 4096 against this 128 ceiling AND rendered no footprint line at all (`bytes_per_cell` is null at ndim=1 — the number that says whether a 2D session can start, missing exactly before the first 2D restart), and over a live 2D session a 1D form's N select collapsed to one option (cap 128, 1D list starting at 256, loop body never entered). `lib/config.axisSizeOptions` is the extracted, unit-tested list — extracted for that reason: both bugs were reachable only through the DOM. **The list is FIXED per ndim: powers of two from `AXIS_N_FLOOR[ndim]` to this ceiling** (1D 256…`WIGNERF_MAX_GRID`, 2D 32…`WIGNERF_MAX_GRID_2D`; verified in a headless DOM at 256…8192 and 32…128 on a host whose `wignerf.env` sets 8192). **Its 2D floor is 32, not 16**, because `boundary._band_mass` reports nothing below 32 cells per axis (the edge band would cover a quarter of the axis), so a 16⁴ session has no boundary watch and says so nowhere; 16⁴ stays reachable through the API and through an imported config, which the select keeps listed. **`AXIS_N_FLOOR` is shared with `setNdim`, and that is the whole point of it being a constant**: a dims switch lands N *inside the target's list* — their own choice when it is offerable, else that ndim's DEFAULT (`DEFAULT_AXES`, 1024² in 1D and 64⁴ in 2D), everything clamped by the target's cap so a host at 128 is not pushed over its own ceiling. Capping from ABOVE alone was wrong in the other direction: 1D → 2D → 1D brought the 2D choice back with it, so the select rendered `64, 256, 512, …` **with a hole in it** and a 1D session ran at 64², a resolution the panel does not offer. Falling back to the FLOOR would be its quieter twin — 256² for a user who started at the 1D default and only looked at 2D. What a round trip cannot do, since the two lists do not overlap on a default host, is preserve a 2D 32 across 1D; that needs a per-dimensionality memory in the form. Pinned in `config.test.ts` (`a dims round trip leaves no value off its own list`, the round-trip-lands-on-the-default case, the lowered-cap case, and the small-choice case the `min()` still serves). |
| `WIGNERF_MAX_CELLS_2D` | `2**27` (134M) | **Total-cell** RAIL for ndim=2 — a cheap deterministic stop for absurd values (a dims switch that carried N over would mean 1024⁴ = 1.1e12 cells from the 1D default, which is why `setNdim` caps it at the target's own), and the only guard on a host where free memory cannot be read. **Checked on an auto-expand DOUBLING as well as at create time since M3 (2026-08-01)** — it was stored on the session and consulted by the planner nowhere, which was the accounting that milestone's gate had been hiding. It is deliberately NOT the operative limit: at the default it permits 22.0 GiB per worker, far past any card here. A fixed cell count cannot do that job — it is wrong in both directions, refusing 128×128×64×64 (11.0 GiB, one worker) on a 24 GiB card while permitting 5.5 GiB × 2 workers on an 11 GiB one — so the real check asks the driver (`_fit_error`, GPU section). Measured on an RTX 3090 with `scripts/bench.py --ndim 2 --footprint`, which runs a whole worker record (the exponent slot, an `adjust_step` pass, a frame build) rather than a step loop — a step loop misses half of it, which is why that mode had to be written before M1 could set a number. **176 B/cell in float64 and 96 in float32 (55%), both flat across sizes and both identical for the relativistic variants** — float64 0.17 / 0.87 / 2.75 / 6.71 GiB at 32⁴ / 48⁴ / 64⁴ / 80⁴ against float32's 0.09 / 0.47 / 1.50 / 3.66. So 4 variants at 64⁴ split 2+2 is ~5.5 GiB/card in float64 (fits both the 3090 and the 2080 Ti) and ~3.0 in float32, and **80⁴ is reachable only in float32** (7.3 GiB/card against float64's 13.4, and the 2080 Ti's 10.6 is what binds). **The STATE is only 5% of that** — W is real, so float64 = 8 B/cell, 0.12 GiB at 64⁴; the rest is the step's machinery at full shape, `adjust_step`'s transient (80 B/cell) largest among it since M7 removed the second exponent slot. float32 lands at 55% rather than 50% because `dU_im`/`dT_im` stay float64. `config.BYTES_PER_CELL_2D` carries the measured stage-by-stage breakdown and is **keyed by precision** — `config.bytes_per_cell(ndim, precision)` — because `_fit_error` reads it: a flat float64 figure there would refuse precisely the grids float32 makes affordable. M7 (2026-08-02) took it from 208/112 by dropping the second exponent slot. Throughput on the same card: 610 steps/s at 32⁴, 130 at 48⁴, 35.1 at 64⁴, 13.8 at 80⁴ — so **32⁴ is for exploration and 64⁴ is a serious run** (~0.23 s per record at 8 substeps). Both the rail's refusal and the fit check's quote the estimate, and `/api/device`'s `bytes_per_cell_2d` feeds the Setup panel's footprint line so a grid that cannot start says so BEFORE the restart — from `/device` and not from `status`, or the line is absent on the one path that reaches 2D (see the `WIGNERF_MAX_GRID_2D` row). |
