# Session lifecycle, RAM and the streaming wall — the measurements

Every RSS/VRAM figure, forensic dead end and browser measurement behind the
"a closed session must actually free" rules. Split out of `CLAUDE.md` on
2026-08-05 (see `notes/precision.md` for why). Read this before touching
`session.close`, `_collect_closed`, `ttl_sweeper`, `routers/stream.py`'s sender,
or anything that claims a memory number.


## The gc.get_referrers dead end

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


## The two remaining RAM holders

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


## The browser receive ceiling

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


## Prompt session release on a departing browser

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


## The keepalive kill: uvicorn's WS transport has no backpressure (2026-08-05)

**`await ws.send_bytes(...)` does not wait for anything.** uvicorn's
`websockets-sansio` implementation — what `--ws auto` selects once
`websockets` >= 14 is installed, and what this host runs (verified:
`AutoWebSocketsProtocol` is `WebSocketsSansIOProtocol`, websockets 16.1.1) —
builds an `asyncio.Event` called `writable`, `.set()`s it once in `__init__`
(`websockets_sansio_impl.py:96-97`), awaits it in `send()` (`:373`), and
**never clears it**. There is no `pause_writing`/`resume_writing` pair in that
file at all. `wsproto_impl.py:183-193` has them and `websockets_impl.py` awaits
the real drain, but neither is what `auto` picks now. So the streamer's
founding claim — *"Backpressure by design: the sender never queues binary
frames"* — was true when it was written and silently false afterwards. This is
the `--ws` impl swap `start.sh`'s comment was worried about; the pins guarded
the ping VALUES, which turned out not to be the thing that mattered.

**What it does.** A replay pushes the whole history into an unbounded transport
buffer within seconds. The keepalive PING is written at the TAIL of that
buffer, so it cannot be answered inside `--ws-ping-timeout`;
`keepalive_timeout` fires `conn.fail(1011, "keepalive ping timeout")`, sets
`close_sent`, and the SERVER has killed its own socket. Our next send then
raises `RuntimeError: Unexpected ASGI message 'websocket.send', after sending
'websocket.close'`. Two tells that identify this rather than a client
departure: the kill lands at an exact multiple of the ping interval after
accept (the 2026-08-05 journal shows 180 s, 40 s, 240 s, 40 s — all multiples
of 20), and the `code=1006` in the disconnect log was a **hardcoded constant**
in `_guard_send`, not a wire code. That constant sent a whole investigation
after a red herring; it is now `None` and the message no longer says "client
gone".

**Reproduced and fixed, measured on the 3090.** `slowclient.py`-style harness:
batch-compute 100 records at 4096² (32 MiB each), then replay while draining at
the measured browser ceiling (110 MiB/s).

| | rec/s | outcome |
|---|---|---|
| unpaced (before) | 3.20 | **socket killed at frame 84**, `sent 1011 keepalive ping timeout` |
| credit, 24 MiB cap | 2.72 | all 100 delivered, −15% (stop-and-wait: the cap was below one frame) |
| credit, 64 MiB cap | 3.21 | all 100 delivered, **the client's own receive rate** |

So the correctness costs nothing once the window admits two records. The cap is
`routers/stream.INFLIGHT_MAX_BYTES`; its size is set by what must drain before
the queued PING (64 MiB is 0.6 s at the browser ceiling, 6.4 s even at a
pathological 10 MiB/s, against a 20 s deadline). A record LARGER than the cap
is still sent — the check runs with the queue empty, so it can never block the
first frame — which just means one frame in flight, the least any transport can
do. Cutting THAT is display downsampling's job, not the cap's.

**The second half of the data loss: `WS_IDLE_TTL` was measured from the wrong
instant.** It was 20 s, justified as "well above `recover()`'s ~1.5 s
reattach" — which silently assumed the client learns of a close as soon as the
server does. It does not: the close frame goes out at the tail of the same
backlog, so a client with GiB queued sees it tens of seconds later. The journal
correlates exactly — reconnects landing 22 s and 18 s after the server gave up
kept their session; 24 s did not, and was swept, giving a 404, a fresh session
id, and `0 / [0, 0]` with 100 computed records gone. Now 90 s. The grace must
not depend on the backlog being small, even though the credit cap now makes it
so. Three supporting changes: `recover()` probes immediately and backs off
afterwards (the 1500 ms was paid up front for nothing); a reattach racing its
predecessor's ~3 s teardown WAITS instead of returning 4409, which the client
read as a plain close and answered with another reconnect; and the 404 path now
SAYS a session was lost with how many records went with it, instead of swapping
in an empty session in silence.

**And `loop` could not loop from a frontier start.** `advance_cursor`'s wrap
gate tested `loop_from < latest_complete`, but a finished batch run puts
`loop_from` exactly AT the frontier (`set_running` captures `int(cursor)`), so
it fell through to the pause and the checkbox silently did nothing — the
reported "computed 100 records, played back, it stopped at the last record".
It "worked the second time" only because the cursor then sat behind the
frontier. A reconnect reaches the same state unattended, since detaching pauses
the session and `recover()` re-issues `play`. It now falls back to the oldest
retained record: an armed loop always loops. Pinned by
`test_loop_wraps_even_when_the_pass_started_at_the_frontier`, which counts
arrivals at the region START (at the frontier a dead loop reads as two passes).


## The cyclic-garbage bullet, as it stood before the split

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


## The streaming bullet, as it stood before the split

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
