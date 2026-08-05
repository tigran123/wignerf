# mp4 export — the measurements

Split out of `CLAUDE.md` on 2026-08-05 (see `notes/precision.md` for why).
What stayed there is what changes the code you would write today; this is how
each number behind it was measured. Read this before editing
`core/videoexport.py`, `core/render_mpl.py` or `routers/export.py`.


## The M4 (2D mp4) gotcha, as it stood before the split

- **2D mp4 EXPORT (M4): the frame is a SELECTION, real subscripts are FREE, and
  the blit decides where static art may live.** **Read `notes/2d-milestones.md`
  before editing `core/render_mpl.py`** — it carries the measurements behind all
  of this. The rules:
  **MATHTEXT IS USED, and the note that once forbade it was backwards.** In
  isolation a mathtext artist is 1.83× a plain one, but the figure BLITS, so
  every title and axis label is a STATIC artist baked into the background once
  and in situ it measures as noise. `axes.sub_math` typesets the two-letter axis
  names as `$p_x$` and everything else — ∬, ⨌, γ, ℏ, ρ, φ, ⟨⟩ — stays the same
  Unicode the screen uses, so the two cannot drift and 1D (single-letter names)
  is untouched at the byte level; every title function takes a `math=` flag, the
  counterpart of its frontend mirror's `html=`. **usetex is NOT used**: 12×, plus
  it needs a LaTeX install (the VPS has none), cannot render our Unicode without
  a THIRD spelling of every string, and is global — the metadata block's
  user-supplied U(x) would go through LaTeX too. **And it has to be the WHOLE
  frame** (`geom_line`, `_extents`, the panel list, `diagnostic_label`,
  `describe.ic_expression`), because half the job reads worse than none: the same
  axis then appears twice on one screen in two spellings.
  **The metadata block is WRAPPED PLAIN and typeset afterwards** (`_emit`) —
  `$p_x$` is five characters that draw as two glyphs, so which characters land on
  which line has to be decided by the plain text. **And it must not break a line
  MID-FACT** (`px ∈ [-7, 7]`, `(6 planes × 1 variant = 6)`, the units figure):
  each group is joined with `_NB` and `_emit` restores real spaces after
  wrapping, which has to be a character `textwrap` cannot see as whitespace at
  all — a Unicode NBSP is `\s`, so it does not work. A test for this must check
  **per line**: joining a column with " " is exactly what reconstitutes a group
  that was split (`test_the_metadata_block_never_breaks_a_line_mid_fact`).
  **U(x,y) is typeset by a LEXICAL rewrite of the user's own string**
  (`describe.potential_math`), NOT via `sympy.latex`, which parses cleanly and is
  still wrong twice over: it CANONICALISES (`x^2/2 + 0.3*x^4` → `0.3x^4 + x^2/2`;
  `exp(0.5)` evaluated to 1.64872127070013) where this block's job is "how to
  reproduce this run" and the source string is what you paste back into the U(x)
  box, and it emits `\frac`, which is TALL where the block advances by a fixed
  `META_LINESPACING`. **A typeset line built from USER input cannot be the only
  line of defence** — a mathtext parse error raises at DRAW time and would take
  the export down — so every candidate goes through `render_mpl.mathtext_ok`
  (matplotlib's own parser, per `$…$` span, cached) and falls back to plain per
  line. `describe.config_json` and the setup document stay plain either way.
  **The header readout and the block deliberately DISAGREE**: the header's
  geometry follows the PAINTED record as the SPA does, while the block's is
  labelled "at record k0" and stays there, because a per-frame rewrite would make
  "what this video is" a moving target — the union is quoted on its own line.
  They diverge only across an auto-expand regrid, pinned at 1D by
  `test_the_header_follows_the_painted_record_and_the_block_does_not` and at 2D
  by `test_the_header_and_block_diverge_in_2d_too`.
  **THE BLIT DECIDES WHERE STATIC ART MAY LIVE.** `update()` restores the static
  background and then `draw_artist`s the images on top, so anything static drawn
  INSIDE the axes box is painted over every frame — which is why the panel grid
  lines are in `_dynamic` and ordered after the images, and why the per-panel
  scale caption (used past `CBAR_MAX_CELLS` = 8 panels, where a colorbar is a
  ~9 px strip) lives in the panel's `loc="right"` TITLE, outside the axes box and
  therefore still static and free. Its first version sat in the corner of the
  heatmap and rendered as NOTHING. Pinned by
  `test_a_dense_grid_states_its_scales_in_the_titles_not_over_the_heatmap`.
  **COST: 2D is CHEAPER per frame than 1D, not dearer** — 6.4 fps for the full
  24-panel matrix against 1D's ~9–10 for 4 panels, because a 2D plane is at most
  128×128 where a 1D W is up to 4096², so the `imshow` upsampling that dominates
  a 1D frame barely registers. `POOL_MIN_FRAMES` and the w+2 window needed no
  change, and the metadata block did not overflow (worst realistic case 10 lines
  against the 11 that fit at 8 pt).


## The mp4-export architecture bullet, as it stood before the split

- **mp4 export** (`core/videoexport.py` + `core/render_mpl.py` +
  `routers/export.py`): renders an ALREADY-COMPUTED record range on the
  BACKEND — matplotlib/Agg frames piped as raw RGBA into ffmpeg (system
  ffmpeg, absence ⇒ 503). PAUSED-only (409 while running): a running session
  evicts old records, and the feature is for filming a range you already
  played back.
  **WHAT GOES IN THE FRAME IS A CHOICE** (`ExportSpec.planes` /
  `.diagnostics`, M4, 2026-07-28), because a 2D record carries far more than a
  frame can hold: six planes × four variants is 24 panels, and the diagnostics
  column has NINE plots against 1D's five. The SPA answers that by scrolling
  its column and offering two panel readings; a video frame can do neither.
  So **panels are the CARTESIAN PRODUCT of the selected planes and variants**,
  which makes `PanelGrid`'s two readings the two EDGES of one control rather
  than modes the renderer has to know about — "compare variants" is one plane ×
  every variant, "phase portrait" is every plane × one variant, and the Export
  panel offers both as one-click presets that just set the checkboxes.
  `render_mpl.panel_grid` REFLOWS by count when one dimension is 1 (so 1D and
  "compare variants" keep the 1×1/1×2/2×2 tiling the figure always had, and six
  planes give the 3×2 the phase portrait shows on screen) and otherwise lays out
  the matrix itself, rows = planes. Diagnostics are plot ids shared VERBATIM
  with `frontend/src/lib/plotPrefs.ts` (`marg0..marg3`, `E`, `uncertainty0/1`,
  `purity`, `lz`) — one vocabulary for the hidden-series preferences, the export
  wire and the metadata block — and an EMPTY list is legal (a panels-only
  video). **The 2D default drops the four marginals, and that is a physical
  argument, not a space-saving one**: at ndim=2 the (x,y) and (px,py) PANELS
  already ARE the spatial and momentum densities, so ρ(x) is a further reduction
  of something a panel is showing, and dropping them keeps the frame's shape
  (one 5-row column, panels at `[0.045, 0.60]`) identical to 1D's. Past
  `DIAG_ROWS_MAX` = 7 plots the column splits in two and the panels pay for it
  in width (`diag_layout`); the Export panel states the resulting panel count
  and approximate pixel size and warns when they become thumbnails. Both
  refusals live in `routers/export.py` and not in the schema, for the reason the
  `variants` check does: what is available depends on the session's ndim, which
  the request body does not carry — and both name what IS available, since a 1D
  session has one plane and no ⟨Lz⟩ and neither is guessable from the index that
  failed. `RangeStats` is keyed to match: `scale[(variant, plane)]` and
  `uncert[(variant, dim)]`, `marg_max` per axis, `lo`/`hi` per axis.
  **The frame RENDER, not the encode, is the bottleneck** (measured 4-var 1024²:
  ~410 ms/frame render at 4K vs 34–109 ms to encode, and the encode already
  overlaps via the pipe; 363 ms of the render is the four `imshow` panels). So
  export renders frames across a `ProcessPoolExecutor` (`export_workers`,
  `WIGNERF_EXPORT_WORKERS`, auto = min(cpu, 8)) while this thread feeds the
  ORDERED frames to one ffmpeg — a sliding window of ≤w+2 futures consumed FIFO
  by `.result()` (workers run ahead, memory bounded). Measured ~3× (4K/4-var
  2.2 → ~7 fps; 1080p 3.3 → ~9-10 fps). The pool is **spawn, NOT fork** — the
  backend initializes CUDA and forking after that inherits a broken context;
  spawn workers only touch matplotlib/numpy (never cupy — `xp` imports it
  lazily). A small job (`< max(2·w, POOL_MIN_FRAMES=16)`) renders serially
  in-process to skip the ~1-2 s pool warmup (`_render_serial`). Encoder via
  `choose_encoder`/`WIGNERF_EXPORT_ENCODER` (auto|cpu|nvenc): auto uses the GPU
  **`h264_nvenc` encoder** if a one-shot runtime probe passes (`_nvenc_ok`,
  cached — the encoder can be built-in yet fail with no driver/GPU, e.g. the
  VPS), else `libx264 -preset veryfast -crf 18` (was `medium`; ~2× faster, file
  ~7% larger, visually identical for this smooth content, and frees cores for
  the render pool). NB the GPU path is the h264_nvenc ENCODER, NOT ffmpeg
  `-hwaccel` — that is a DECODE flag and does nothing for our rawvideo input.
  Two passes: a scan collects the E/ΔX·ΔP/γ series, the per-variant FIXED colour
  scale (no brightness flicker), the fixed marginal amplitudes and the widest
  window any record used, and proves every record is still retained before
  ffmpeg starts; then one figure update per frame. Only VALUE scales are
  export-wide — the SPATIAL axes follow each record's own geometry
  (`_apply_geom`, which also re-captures the blit background since ticks are
  static art), exactly as the SPA follows the painted frame; freezing them at
  the union rendered every frame before an auto-expansion as a stamp in the
  corner of its panel, and the union now only labels the metadata block. The
  figure is built ONCE and BLITTED (static background + ~15 animated artists):
  465 → ~17-80 ms/frame measured at FHD (~320 ms at 4K, 4 variants), the
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
  The video must READ like the screen: every plot title comes from
  `core/axes.py`, the same source `lib/axes.ts` mirrors for
  `SeriesPlot.vue`/`MarginalsPlot.vue` (γ keeps the UI's
  "purity γ(t) = 2πℏ∬W²dxdp", never an equivalent like Tr ρ²; ⟨Lz⟩ got the
  `axes.lz_title` the backend had been missing, restoring the three-way
  mirror), field labels match the Setup panel (ℏ, "batch"), and the series
  y-window + tick decimals reproduce that component's `scales.y.range` rule
  (`render_mpl.series_ylim`) — matplotlib's own autoscale renders a 2e-5 purity
  drift as a dramatic dive with a "×10⁻⁵+1" offset where the UI shows a flat
  line at 1.000000, from byte-identical data. The "grid lines on plots" toggle
  rides along in `ExportSpec.show_grid` and governs EVERY plot in the frame —
  charts get uPlot's grid stroke (`--wf-chart-grid`, so it follows the theme
  like the rest of the chrome; see the Theming bullet), the W panels get
  `GridOverlay.vue`'s theme-INDEPENDENT rgba(120,120,120,.28/.55-at-zero) drawn
  AFTER the image (matplotlib puts the axes grid under it, which is why the
  heatmaps first had none; the lines are animated artists ordered behind the
  images in `_dynamic`). Mirror any change to those rules on both sides.
  The block carries U(x), parameters, the IC as an analytic expression
  (`core/describe.py`; cat states print ψ(x,0), the compact complete form),
  and any live parameter change inside the range (`session.param_log`) —
  so one frame documents the whole run; the same facts go into the mp4
  `comment` tag as JSON. It is anchored at `FrameFigure.META_TOP` with
  `va="top"` and grows DOWNWARD, so nothing stopped it running off the bottom
  edge: at 8 pt and linespacing 1.6 a line advances 0.016461 of the height, so
  **11 lines fit**. `param_lines` grows by one line per live parameter change
  plus the float32 PREVIEW line, and a 4-variant cat run with 4 live changes
  measures **10 lines in float64, 11 in float32** — a realistic export sat
  exactly at the edge and one more change clipped in silence. `_meta_fontsize`
  now shrinks to fit (one size for BOTH columns, so they stay matched, derived
  from `get_figheight()` so a non-16:9 export keeps the full 8 pt), and past a
  5 pt floor `_meta_fit` elides with "… +N more lines — full detail in the mp4
  comment tag", an honest pointer because `describe.config_json` really does
  carry all of it. Static art, baked into the blit background: no per-frame cost.
  Progress: `export` events on the session WS plus a REST poll; the file lives
  in `WIGNERF_EXPORT_DIR` until downloaded, TTL (30 min), session close or
  shutdown. The header button stays ENABLED while computing (a disabled button
  explained only by a tooltip is how this feature first read as broken): the
  panel states the gate and "Pause & render" pauses, waits for the server to
  confirm and re-seeds an untouched range before posting. Rendering continues
  while the popover is CLOSED (the poll and the WS events keep updating), so the
  button IS the notification — "⤓ export 42%" while running, emerald "⤓ export
  ready" (red "failed") when finished, and reopening it collects the file; a
  finished job survives reopening and is dropped only by a new render or a
  session change (a restart deletes the old session's files). The panel re-reads
  the extent from `GET /sessions/{id}` when it opens — the streamed status lags
  a frame burst by up to seconds after a pause, and seeding the range from it
  silently exported half the history.


## The render rate the panel shows (2026-08-05)

`ExportJob.render_fps`, beside `done/total` on the progress line. Two rates
exist and they are unrelated: `fps` is the VIDEO's frame rate (what the mp4
plays at, chosen in the form) and `render_fps` is how fast this machine is
producing those frames. Conflating them would make a slow render look like a
slow video.

It is a ROLLING rate while running (the `worker.steps_per_sec` idiom: a ~1 s
window) and the whole run's average once the job finishes. Measured on a
40-frame 1920x1080 export of a 256^2 two-variant session, the parallel path:

    while running   0.14 fps  (pool warmup, first samples)
                    9.66 fps  (steady, 8 workers)
    final average   4.15 fps

That spread is the reason for the rolling window. A cumulative average would
have started at 0.14 and spent the whole render climbing toward 4.15, i.e.
displaying a number that only ever goes UP while nothing about the machine
changes — and displaying it lowest exactly when someone is deciding whether the
export is worth waiting for. The final figure keeps the warmup in, because it
really was part of the wait.

Panel formatting is two decimals below 10 fps: this feature lives in the
0.5-10 range (a 24-panel 2D matrix renders at ~6 fps, a large 1D frame slower),
where rounding to integers would print "6" and "1" for a 40% difference.


## Why exports were slow, and what the encoder had to do with it (2026-08-05)

Reported as "the rendering was so slow that I doubted it used nvenc at all".
Two separate findings, and the suspicion was half right.

### The nvenc probe had never once succeeded

`_probe_nvenc` encoded a **64x64** clip. NVENC's H.264 minimum is **145x49**,
and below it ffmpeg fails with

    InitializeEncoder failed: invalid param (8):
    Frame Dimension less than the minimum supported value

which at the exit code is indistinguishable from "there is no GPU here". So the
probe answered "unavailable" on every machine it ever ran on, including this
workstation with a 3090 and a 2080 Ti idle, and `WIGNERF_EXPORT_ENCODER=nvenc`
could not be reached at all. Measured floor, this ffmpeg (8.0.1):

    64x64   FAIL     145x49  OK
    128x128 FAIL     256x256 OK

The probe clip is now 256x256 — over the floor with room, still instant.
`test_the_nvenc_probe_clip_clears_the_encoder_minimum` asserts the PROPERTY
rather than running the probe, which on a CPU-only host would skip and pass
vacuously exactly where the bug lived.

### ...and fixing it changed nothing, because the encode was never the cost

Same 60-frame 1920x1080 export, 2 variants, one job after the other:

    libx264 -crf 18      4.3 s wall    0.19 MiB
    h264_nvenc -cq 19    4.3 s wall    0.35 MiB

Identical wall clock, 1.8x the bytes. So `auto` stays **libx264** and the GPU
encoder is opt-in per host or per job. The old rationale — "it frees cores for
the render pool" — is precisely what the wall clock declines to show; do not
restore auto=nvenc without re-measuring both columns.

### The real cost is the GRID, and the video resolution is nearly free

ms per frame per worker, min of 5, one plane per variant:

    case                     before   after (pyramid)
    256^2  1080p 2var           80      78
    2048^2 1080p 2var          516      94
    4096^2 1080p 2var         2247      93
    256^2  4K    2var           72      74
    4096^2 4K    2var         2310     341
    4096^2 1080p 4var         4429      92
    4096^2 1080p 4var 5diag   4520     165
    4096^2 4K    4var 5diag   4618     518

Before, cost scaled as N^2 and 1080p vs 4K differed by **3%** — matplotlib's
per-frame cost is set by the ARRAY it is handed, not by the figure. A 4096^2
plane drawn into a ~990 px panel was 16x more data than the video could carry.

`render_mpl.plane_step` now picks the coarsest mip level that still covers the
panel (`FrameFigure._panel_px`, taken once from the axes' own figure-fraction
box). Never below the panel's resolution, so it is not a quality trade — 4K
legitimately costs more than 1080p now, which is the point.

Verified on a cat state with interference fringes at 2048^2, full draw against
pyramid draw of the same frame: **92.6% of pixels bit-identical, max channel
difference 2/255, nothing over 8/255.** Fringes are what a wrong level or an
aliasing subsample would destroy, which is why the check uses them rather than
noise.

Note the 4-variant row: 4429 -> 92 ms. Four panels in one figure are each half
the width, so each needs a quarter of the samples — the pyramid makes panel
COUNT nearly free too, where before it was linear.
