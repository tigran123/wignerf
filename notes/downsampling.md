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
