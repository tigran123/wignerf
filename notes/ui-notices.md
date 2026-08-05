# Transient notices, the boundary warning and the header strip — the measurements

Split out of `CLAUDE.md` on 2026-08-05 (see `notes/precision.md` for why). What
stayed there is the RULE; this is every measurement, browser reproduction and
false start behind it. Read this before changing `lib/sticky`, the header's
notice strip, `SimulatorView.boundaryText`/`wOf`, or the IC editor's ⚠ block.


## The failed-restart bullet, as it stood before the split

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
  whose largest card is 24** — the same grid reads 22.00 since M7, still far past
  the 10.6 GiB card that actually binds — and `WIGNERF_MAX_CELLS_2D` did not
  cover it either — 128×128×128×64 is 134,217,728 cells, EXACTLY 2²⁷, so the
  `>` rail is not tripped. At the cap and far past the hardware, and reading as
  fine.


## The header-overlay bullet, as it stood before the split

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


## The transient-notice rule, as it stood before the split

- **EVERY TRANSIENT NOTICE IS AN OVERLAY AND HAS A MINIMUM DWELL. This is a
  RULE, not a fact about the header** — it was written down for the header alone
  and then broken the next time a panel grew a conditional line, which is
  exactly what a rule stated as a special case invites.
  A notice derived from live state comes and goes at the rate that state
  changes, not at the rate a person reads, so:
  **(1) it must not occupy layout.** Anchor it `absolute left-0 right-0 top-full
  z-*` to a `relative` parent, overlaying the top of whatever follows. The IC
  editor's `⚠` block hangs off the preview box for this reason: `edgeNotice` is
  differenced against the RUNNING session's tripped axes, so a packet sloshing
  through the edge band made it appear and vanish every record or two, moving
  every control below it 38 px on and off for the whole computation (measured
  ten controls each taking two positions, 1184/1222 …). A panel you WATCH must
  not relayout because a message arrived; obscuring chrome for a few seconds is
  much the cheaper trade, and the one the user asked for by name.
  **(2) it must stay long enough to be read** — `lib/sticky`, MIN_DWELL_MS =
  5 s, holding the last non-empty value after the condition clears. A
  REPLACEMENT is never delayed (a newer message beats an older one and restarts
  the clock); only the clearing is. Display only, so the transport, the Solve
  gate and every test still see the instantaneous state. Measured on a coarse
  grid with a packet at the edge: the boundary warning flipped on and off
  through the whole run before, and holds one continuous 12 s span after.
  The steady READOUTS are the exception and stay in flow — the norm deficit, the
  normalisation scale — because they change only when the form does, which is
  the user's own doing and already a relayout.
  **Two traps, both of which shipped once:**
  **An `overflow-hidden` ancestor deletes the strip from the screen and leaves
  it in the DOM.** The IC preview needs that class to keep the heatmap inside
  its rounded border, and it also clips any absolutely-positioned child hanging
  BELOW the box — which is where `top-full` puts one. Hence two nested boxes:
  an outer `relative` anchor and an inner clipping one. Do not merge them.
  **"Nothing moved" and "the text is in the DOM" together do NOT prove the
  message is still visible** — that pair passed while the warnings were being
  clipped away, and only a screenshot showed it. Assert that it is PAINTED:
  `document.elementFromPoint` inside the notice's own rect must land on the
  notice. Same discipline as the WebGL-canvas rule above, in the other
  direction: there a screenshot lies, here the DOM does.
  And when several lines share one strip, **the flickering one gets its own
  timer**. Joining them into a single sticky string means that losing one line
  still leaves a non-empty value, which `sticky` correctly reads as a
  REPLACEMENT and applies at once — a dwell that only works when the message is
  alone is not a dwell.



## The boundary-warning bullet, as it stood before the split

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
  **The edge finding is stated ONCE, IN ONE PLACE, AND THE TIME ARGUMENT SAYS
  WHICH STATE IT IS ABOUT** (2026-08-04). The detector runs on the IC preview and
  on the live record, and at record 0 those are the SAME state — so the fact was
  genuinely duplicated. It used to be reported twice in two different sentences
  ("W(x,p,t) has reached the x edge" in the header, "this IC reaches the x edge"
  under the preview) carrying the identical number, with the IC copy suppressed
  by DIFFERENCING the axis sets. That was worse than the duplication it fixed:
  the surviving copy flickered in and out as the live axes changed, so the same
  fact appeared in different places at different times and read as two problems.
  Now the header says it, once, and `wOf` writes **`W(x,p,0)`** while nothing has
  been computed past the Cauchy data and `W(x,p,t)` after — one sentence whose
  argument distinguishes the initial condition from the evolved state.
  `preview_warnings`'s `edge_axes` out-list and the `X-Wignerf-Edge:
  axis:mass,…` header still exist and are still the right shape (structured, so
  the client never pattern-matches a sentence); what changed is that `ICEditor`
  EMITS the finding upward instead of rendering it. The one case the session
  cannot cover — an IC you have edited but not restarted into — is
  `SimulatorView.icEdgeText`, gated on `restartNeeded` so it can never compete
  with the session's own reading, in the same sentence and the same place.
  **The × dismisses BY TEXT.** Clearing `session.boundary` — all it used to do —
  cannot dismiss this: the sentence may come from `icEdgeText` or from the
  standing `status.boundary`, neither of which that touches, and even for the
  transient event `sticky` keeps displaying the last value for its dwell. The
  button was inert on screen while looking perfectly wired in the source.
  Keyed on the text, so dismissing one warning never hides the next, different
  one; the transient event is still cleared alongside.
  **A REQUEST SEQUENCE MUST BE BUMPED ONLY WHEN A REQUEST IS ACTUALLY SENT.**
  `ICEditor.refresh` incremented `seq` on its first line, before the body de-dup
  could return — so every successful commit re-fired the deep watch, landed
  here, declined to repeat the request, and silently invalidated the responses
  still in flight from the one it had just declined to repeat. The ψ/φ traces
  were the casualty: they could be stranded empty with no later edit able to
  refill them, because the body never changed again. The wavefunction call also
  has its OWN counter now — it and the W preview are independent requests, so
  neither may invalidate the other.
  **And a failed ψ sample SAYS SO.** It gates nothing (the W response owns the
  Solve gate) but it must not blank two charts in silence: the W preview can
  succeed while it fails, and two empty plots with no message read as a broken
  component rather than an expression that did not compile. Their titles are
  built from the CUT rather than from the response for the same reason — a title
  that degrades whenever the data does turns a failed request into apparent rot.


## The parameter-policy bullet, as it stood before the split

- **Parameter policy**: U, c, mass, hbar_eff, tol, dt_sign, auto_expand
  apply live at the frontier; **ndim**/grid/IC/variant-set and the whole COMPUTE
  group (precision, device, history_mb — the Setup panel's third section)
  require a session restart, because each is fixed at worker construction (FFT
  plan dtype, `ArrayBackend` device, `FrameHistory` cap). Auto-expand moves the
  LIVE grid; the Setup panel shows it and offers "adopt" to copy it into the
  form. ndim is the most restart-only of them all — it decides the array rank —
  and switching it in the form rebuilds the grid and the IC (`config.setNdim`,
  mirroring each component's second dimension on its first, i.e. the separable
  product of what was there) and replaces U **only if it was still the default**,
  since `x^2/2` cannot silently become a two-variable expression and a
  hand-written potential must never be discarded.
  **The BOX follows the same "only if untouched" rule, and for a sharper
  reason**: a still-default box is replaced by the TARGET ndim's default, because
  carrying [-6,6] into 2D reproduces exactly what `DEFAULT_AXES[2]` was widened
  to [-8,8] to avoid — the edge band is max(4, N/32) CELLS, so at N=64 the 4-cell
  floor makes it 0.750 a.u. wide, only 4.60σ from the default packet at x0=2, and
  a FRESH 2D default tripped its own boundary warning on the first Restart
  (measured 3.78e-06 band mass against the 1e-6 trigger; analytic tail 2.15e-06 —
  real mass and a CORRECT warning, not detector noise; at [-8,8] the same band
  sits at 7.07σ and reads 2.12e-12). A box the user CHOSE still carries over
  untouched: silently widening someone's domain is worse than a warning. **N
  lands inside the TARGET's own select list**: their own choice when it is
  offerable there, capped at that ndim's default (a 1D 1024 would be 1.1e12 cells
  at ndim=2), and the target's DEFAULT when it is not — never the list's floor,
  which would quarter the resolution of anyone who started at the 1D default
  (1024², since 2026-08-01) and merely looked at 2D. See the
  `WIGNERF_MAX_GRID_2D` row for the hole that carrying a 2D 64 into 1D left in
  the select. Pinned in `config.test.ts`.
  **Every restart-only field goes amber when it disagrees with the session**,
  COMPUTE included — a form reading `cuda:0` over a session on `cuda:1` is the
  same trap as one reading float64 over a float32 run. `precision` gets it via
  `LiveRun`, but `device` and `history_mb` cannot: the form holds a REQUEST
  (`''` = the host's pool, `0` = its ceiling) while `status` reports what was
  GRANTED (a resolved device list, a clamped cap). So `SetupPanel` resolves the
  request the way the server would — `''` against the pool from `/api/device`'s
  `devices` (hence `hostPool`, distinct from `choices`), a bare `cuda` to
  `cuda:0` to match `resolve_devices`, and `history_mb` clamped to
  `history_mb_max` so asking for 999999 on a 110000 host is not a "difference" —
  and only then compares. One amber line names what is running, and only while
  something differs.
  **Each section's summary line covers its OWN fields.** `precision` lives in
  `LiveRun` because that is where `status` carries it, but it is a COMPUTE
  control: it triggers and is named by COMPUTE's line, and is deliberately absent
  from `runStale` and from the RUN line's text. It used to be in both, so a plain
  float64 → float32 switch raised THREE amber notes instead of two — the third
  announcing "running: interactive (no t₂), Δt rec = 0.05" at a user who had
  changed neither, which reads as the form having quietly moved something else.
  The steady-state facts are NOT repeated in the panel: the devices and the
  history cap ride the timeline's own `hist 0.1 / 107 GiB · dev: cuda:1, cuda:0`
  readout, which is drawn anyway.
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
  (1 s) and that check sits at the top of the sender's tick, so a field the form
  marks amber against `status` could stay amber for a second or more after the
  "✓ applied" flash had confirmed the change. Pinned by
  `test_params_applied_is_followed_by_a_fresh_status`; measured in a browser at
  256² while computing, click → `params_applied` is 12 ms, status 1 ms later.
  `apply_params` compares against what is LIVE and drops the fields that did not
  change — no worker command, no `param_log` entry, no `params_applied`, and
  nothing at all if the whole message is a no-op (the UI sends complete fields;
  "Apply live" always carries the U(x) draft, so the log used to fill with U
  changes that never happened and an export's "how to reproduce this" block lied
  about its own frames). Entries carry `before` as well as `applied`, so the
  block renders "ℏ 1 → 2" and `describe.state_at` rewinds the header physics to
  the FIRST exported record instead of quoting the values the run ended with.
  Live changes are visible in the UI: the header flashes "✓ applied …", and any
  live-appliable field whose form value differs from `status` renders amber — the
  numeric Physics fields (which apply on blur/Enter) via `pending()`, and the
  U(x) input via `PotentialEditor.isPending`.
  **U(x) is a LIVE-appliable parameter and has exactly ONE button.** It used to
  have two, "Use at restart" beside "Apply live", both confusing for the same
  reason: the draft was local to the editor, so the FORM did not mean what it
  showed until you pressed the first one — while every other setting in the panel
  is bound straight to `cfg`. The draft now auto-commits to `cfg.potential` from
  `compile()`'s success path, gated on the server's verdict (a half-typed `x^2/`
  must never reach `cfg`, which is persisted to localStorage and is what a
  restart computes from) and on the response still describing the CURRENT text.
  So U(x) no longer marks the session restart-dirty at all: it applies live, so
  "restart to apply" was a false claim no successful Apply live could clear.
  **Solve carries the form's U(x).** `SimulatorView.sendCommand` pushes
  `set_params {U: cfg.potential}` before a `play` whose action is `solve`,
  whenever the form's (validated) U differs from `status.potential`. Without it
  the form was authoritative for nothing: measured 369 records computed under
  `x^2/2` behind a form reading `x^2/2 + 5*x^4`, the only signal being an amber
  input in a panel that can be hidden. Playback is excluded — it computes
  nothing. So "Apply live" is gated on `status.computing`, NOT on `live` and NOT
  on `running`: while nothing computes there is no live run to reach and Solve
  does the job, and an enabled button there invites a click that is at best
  redundant and reads as the only way to make the new U count. The button exists
  for the one case Solve cannot serve — a computation already in flight, which
  you would otherwise pause. `computing`, because `running` is true during pure
  PLAYBACK too (`running and not stop_at_frontier`), where the button's own
  tooltip ("push this U(x) into the computation already in progress") would be
  false; it is the same field batch mode dims the display on. Bonus:
  `useSession`'s optimistic transport flip touches only `running`, so the button
  stays briefly DISABLED after a Solve click rather than briefly enabled, the
  safer direction to be wrong in. What is left is one emerald "Apply live",
  disabled unless the session is COMPUTING AND the draft is valid AND it differs
  from `status.potential`, with the reason in its `title` and **no standing
  paragraph** — the old "already the live U(x) — edit it to enable …" line was on
  screen at every page load, because that IS the steady state (see
  [no-mystery-disabled-controls]: marker + tooltip + amber on the transition,
  never permanent prose).
  The setup form gates the transport: while the potential draft is invalid for
  the active variant families or the IC preview errors, Solve (button AND Space)
  is disabled and "Apply live" is greyed — a computation must never run behind a
  visibly broken form.
  **Every saturated action button carries `.wf-solid`** (`style.css`): Solve/
  Play, Restart session, Apply live, Render. It supplies `color: #fff` — those
  buttons never set a text colour, they INHERITED the shell's light text, which
  went invisible ("black on blue") the moment the shell could be light — and a
  disabled state that drops to the neutral raised surface, because
  `disabled:opacity-40` over a saturated fill is pale colour under equally pale
  text, unreadable in either theme.


## The theming bullet, as it stood before the split

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


## The edge-noise-floor bullet, as it stood before the split

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


## The boundary-watch bullet, as it stood before the split

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
