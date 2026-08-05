# Backend display downsampling (protocol v5, 2026-08-05)

Why: a panel is ~870x875 device pixels and a 1D plane at 4096^2 is 32 MiB
quantized, 128 MiB at 8192^2. The browser's WebSocket receive path degrades
with MESSAGE SIZE (see `session-lifecycle.md`), so the fix has to be smaller
messages. The client now tells the server which physical region each panel
shows and at what pixel size, and gets back a crop decimated to match.

Read this before touching `core/pyramid.py`, `core/planeview.py` or the
per-plane wire fields.


## The measurements that decided the design

Host-side, per plane per variant (numpy, uint16):

| N | plane | `pack_frame` copies | strided crop -> <=1024^2 | contiguous level crop | host area-mean -> 1024^2 |
|---|---|---|---|---|---|
| 2048 | 8 MiB | 0.8 ms | 0.47 ms | — | 30 ms |
| 4096 | 32 MiB | 28 ms | 0.59 ms | 0.07 ms | 92 ms |
| 8192 | 128 MiB | 109 ms | 1.14 ms | 0.07–0.90 ms | 250 ms |

Device-side, per plane per variant (2080 Ti — the SLOWER card):

| N | float64 plane | full pyramid | quantize level 0 (already paid) | quantize + unshift |
|---|---|---|---|---|
| 2048 | 32 MiB | 0.30 ms | 0.59 ms | 0.53 ms |
| 4096 | 128 MiB | 1.07 ms | 2.25 ms | 1.97 ms |
| 8192 | 512 MiB | 4.09 ms | 8.96 ms | 7.73 ms |

Three conclusions, and each is a rule in the code:

- **Area-averaging must happen on the DEVICE, once per record.** It is cheaper
  there than the quantize already being paid; on the host it is 250 ms/plane,
  which no streaming path can absorb.
- **The send-time crop must be a contiguous slice of a pre-reduced level.**
  0.07 ms, against 1.1 ms for a strided gather off the base array. This is the
  whole reason the pyramid is built at all rather than decimating per send.
- **The unshift is free on the device**, which is what made one wire
  convention (natural order everywhere outside the propagator) affordable.

Pyramid cost: **+33.3% of the base plane**, bounded whatever the depth — it is
a geometric series in 1/4 — so `PYRAMID_FLOOR` is not there to control size. It
is there because a level below the smallest request a panel can make would
never be read, hence `PYRAMID_FLOOR <= planeview.VIEW_N_MIN`, pinned by
`test_the_pyramid_reaches_every_level_select_can_ask_for`.


## What it bought, measured in a real browser

Headless Chrome (SwiftShader, so the SOFTWARE renderer — the transport figures
are unaffected by that), one `qn` variant on the 3090, an 866x875 panel,
`__wfPerf.snapshot()` over a 12 s live run. The A/B is the same page and the
same session with the client's `view` message suppressed, which reproduces the
pre-v5 behaviour exactly:

| grid | | records/s | MiB/s | upload ms/frame | drops |
|---|---|---|---|---|---|
| 4096^2 | whole planes | 3.1 | 99.2 | 9.13 | 0 |
| 4096^2 | downsampled | **8.0** | **4.2** | 0.08 | 0 |
| 8192^2 | whole planes | **0.0** | 0.0 | — | 0 |
| 8192^2 | downsampled | **0.8** | 0.5 | 0.06 | 0 |

Two things worth reading carefully. At 4096^2 the "before" row sits at 99 MiB/s,
i.e. right at the ~112 MiB/s ceiling `session-lifecycle.md` measured — the
transport was saturated and the client idle. Afterwards it moves 23x fewer
bytes and gets 2.6x the records, and what limits it is now the SOLVER. At
8192^2 the before row is not slow, it is **zero**: not one 128 MiB record
arrived in 12 s. That is the wall, and it is the case the feature exists for.

The reduction the panel reports tracks the zoom as it should — at 8192^2 the
marker reads ↓16x at the full view, then ↓4x, ↓2x, and disappears (full
computed resolution) as you scroll in.

Reproducing it: drive the BUILT SPA with `puppeteer-core` (system Chrome,
`--no-sandbox --disable-gpu`), seed `localStorage['wignerf.cfg']` BEFORE the
first navigation, and read `window.__wfPerf.snapshot()`. Two traps cost real
time here:

- **The config key is `wignerf.cfg`.** Seeding the wrong key looks like it
  worked — the app runs, panels draw, numbers come out — while the session
  silently uses the stored/default grid. A run that "measured 4096^2" was
  measuring 1024^2, and the only tell was a reduction factor that did not
  match the arithmetic.
- **`canvas.getContext('webgl2')` returns NULL when re-queried on a live
  context** (already in CLAUDE.md, and it bites again here), so `readPixels`
  verification is unavailable from outside the renderer. Verify the loop
  through the reduction marker and the recorded `view` messages instead —
  patch `WebSocket.prototype.send` in `evaluateOnNewDocument` to capture them.


## The two blank panels (2026-08-05)

Reported: "Reset to defaults", select the psi(x) IC, tick all four variants,
"Restart session" — and the QR and CR heatmaps are blank while QN and CN paint.
Clicking "link zoom" makes both appear at once.

Nothing was wrong with the pyramid, the crop or the wire. The client had told
the server that nobody was watching those two planes, and could not take it
back.

The chain, in the order it has to be understood:

1. `DEFAULTS.variants` is `['qn','cn']`, so the session before the restart had
   two panels and emitted a two-entry viewport. `SimulatorView` caches the last
   one it sent (`lastView`) because it has to be RE-SENT on reconnect — the
   server drops a departing client's viewport.
2. `restart()` did not clear that cache. The new session's socket opened, the
   `connected` watcher replayed the OLD panel set's viewport into it, and
   `stream.py` took it as authoritative: `s.views` now had entries for vids 1
   and 0 only.
3. `pack_frame` therefore wrote a header-only plane (`na = 0`) for vids 3 (qr)
   and 2 (cr). **The blank set was exactly the vids missing from the default
   pair** — that is the fingerprint, and it is worth checking first in any
   report of this shape.
4. `WignerPanel`'s frame handler returned on `na === 0` **above** the line that
   requests a view. So those panels never asked for the plane they were not
   being sent, and nothing else could ask for them either: `requestView` needed
   a vid and extents that only a PAINTED frame supplied, which made the
   mount-time and resize-time calls dead code. Self-perpetuating, and it
   survived further restarts because the two-entry cache was replayed again.

Two diagnostics made it quick, and both are worth reusing:

- **The missing COLORBAR is the proof of which early return was taken.** The
  chip renders only from `wmin`/`wmax`, assigned immediately after the `na === 0`
  check; the label chip and the axis ticks come from lines ABOVE it. A panel with
  a title and no colorbar has been reached by frames and has not been given a
  plane. (A blank canvas alone cannot distinguish that from a dead GL context.)
- **The marginal curves kept updating for QR and CR throughout**, because
  marginals are always sent in full. Data flowing on one axis of the same
  variant is what rules out the worker, the session and the socket.

Why "link zoom" of all things was the cure: toggling it swaps the `view` PROP
object of every panel, and the watcher on that prop is the one code path that
called `requestView()` without having painted first — the vid had in fact been
recorded (above the return), so all four panels suddenly requested, the server
grew `s.views` to four and re-sent the current record. It was never about zoom
coupling. In the 2D phase portrait the toggle is HIDDEN (`canLink`), so there
the same deadlock had no cure at all.

The fix (all four parts, because each is independently a defect): a panel takes
its vid from its variant KEY and its extents from the session geometry when no
frame has painted, so it can state its viewport at mount; it re-requests on a
header-only plane as a self-heal; `lastView` is cleared by `restart()`; and
`flushViews` MERGES a burst over the live panel set instead of replacing it —
which was a second live bug, since an unlinked zoom on one paused panel emitted
a one-entry viewport and retracted the other three planes.

Reproducing it (the control matters — an earlier version of this script passed
against the BROKEN build): the `psi(x)` IC is load-bearing, not decoration.
`psi_wigner`'s transform makes record 0 slow enough that the client's viewport
reaches the server BEFORE the record is packed, which is what lets the stale
viewport decide the first frame's contents. With a Gaussian IC the record wins
the race, the first frame goes out whole, every panel paints, and the bug is
invisible. Assert per PANEL (scope the colorbar query to the panels — the IC
preview has one too, so a page-wide count is off by one and passes anything).

A side effect worth knowing: with the panels emitting at mount, the FIRST frame
of a fresh session is already a crop (1.4 MiB against 11 MiB on the reported
setup). That is a bonus, not a guarantee — lose the race and it degrades to one
whole plane followed by crops, which is what the `na === 0` re-request covers.


## What the traffic IS after v5, and why a small grid is the loud one

The question this answers, asked 2026-08-05: "why 483 MiB/s on loopback at a
mere 2048^2, when 8192^2 was quiet and the ceiling was supposed to be 120?"

    bytes/record ~= 2 B * n_variants *  sum   min(n_asked, N_a)*min(n_asked, N_b)
                                    planes shown
    bytes/s       = bytes/record * records/s

`n_asked` is `_pow2_at_most(device px)` clamped to `VIEW_N_MAX` = 1024, and
`records/s` is 60 at the delay dial's "0" (one record per display refresh).

**Once both axes are >= 1024 the `min` saturates and the GRID DROPS OUT.** So
1024^2, 2048^2, 4096^2 and 8192^2 all ship the same bytes per record; what
differs is how fast records arrive. That is the whole inversion: pre-v5 traffic
scaled as N^2, and now it scales with the RECORD RATE, so the small fast grid is
the noisy one and 8192^2 is quiet only because the solver gives it ~1 record/s.
Below 1024 per axis the grid still counts (512^2 is a quarter, 256^2 a
sixteenth).

Confirmed on the reported setup — 1D, one plane, HiDPI panels at the 1024 cap,
delay 0, so 2.02 MiB per variant per record:

| variants | predicted | observed |
|---|---|---|
| 1 | 121 MiB/s | ~120 |
| 2 | 242 MiB/s | ~240 |
| 4 | 484 MiB/s | 483 at 2048^2, 487 at 4096^2 |

and independently at dpr=2 headless with a smaller panel (512x1024 samples,
`↓4x`): 4.06 MiB/record predicted, 4.06 measured off the wire.

Two things NOT to conclude from a number like 483:

- **It is not the ceiling being hit.** The ~112 MiB/s figure in
  `session-lifecycle.md` is the browser's limit for 32 MiB MESSAGES; the
  degradation is per-message, and the same run measured >=480 MiB/s at 2048^2.
  `60.0/60.0 fps` with zero drops is the proof — at a real wall painted lags
  received.
- **It is not a downsampling failure.** The feature bounds bytes per record to
  what a panel can draw; it never promised a bytes/s cap. The lever on rate is
  the delay dial, which is a DISPLAY policy: ~100 ms gives 10 rec/s and ~80
  MiB/s with no effect on computation.

2D never appears in this arithmetic: the portrait draws 6 planes, but
`WIGNERF_MAX_GRID_2D` is 128, so `min(1024, 128) = 128` and a record is ~192 KiB.


## Why the window is physical, not fractional

`ViewCmd` carries `a1..a2`/`b1..b2` in phase-space units. A fraction of the
domain would be cheaper to compute and is what the renderer uses internally,
but it does not survive auto-expand: when the domain doubles, the same fraction
means somewhere else, and every retained record would need its own remap. In
physical units the server answers each record against the geometry that record
was computed on, so a scrub across a regrid boundary needs no bookkeeping at
all — `_views_for` resolves per record for exactly this reason.


## The colour range is the full plane's, at every level

`requantize` takes `(wmin, wmax)` from the base plane and every mip uses them.
The server changes level whenever a panel resizes or the zoom crosses a power
of two, and a range that moved with it would repaint the colorbar under a user
who only scrolled. Averaging cannot exceed the values it averages, so nothing
clips. Asserted in `test_a_view_request_shrinks_the_wire_and_still_shows_the_
same_region`, which also checks the served level really is the area mean of the
full plane rather than a subsample — a subsample of a fringed state aliases,
turning unresolved interference into moire that reads as structure.
