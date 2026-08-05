"""
WebSocket streamer: binary frame bundles out, JSON control in.

Backpressure by design: the sender never queues binary frames. In the live
region (cursor at the lockstep frontier) it always sends the NEWEST complete
record — a slow client skips frames, never buffers them. In the replay
region (cursor behind the frontier) records are sent in exact sequence and
playback slips in wall time instead: `delay` (seconds injected between
played-back frames) paces the display, and its default 0 simply means "as
fast as this client renders". A playback-only run never skips a record —
it must not coalesce to the frontier while sequential records remain
unsent, and it auto-pauses only once the frontier record was actually
delivered. Seek sends the exact requested record, paused or not.
"""

import asyncio
import json
import logging
import time
from contextlib import suppress
from time import monotonic

from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from core import planeview
from core import protocol
from core import session as sessions
from routers.sessions import compile_for

log = logging.getLogger(__name__)

router = APIRouter()

_client_msg = TypeAdapter(protocol.ClientMsg)

STATUS_PERIOD = 1.0
PROGRESS_PERIOD = 0.25   # batch-mode progress cadence (~4 Hz, ~300 bytes each)

# Transport credit (see protocol.AckCmd). A frame is in flight from the send
# until the client acks having PAINTED it; past either bound the sender stops
# and playback slips in wall time, which is what the module docstring below has
# always claimed and what uvicorn's sansio impl silently stopped delivering.
#
# The bound is set by what must still drain before the keepalive PING queued
# behind it: at the measured browser ceiling (~110 MiB/s at 32 MiB messages)
# 64 MiB is 0.6 s, and even at a pathological 10 MiB/s it is 6.4 s — both far
# inside the 20 s pong deadline that killed the socket.
#
# It has to be at least TWO records wide or it degenerates to stop-and-wait,
# and the bubble is measurable. Replaying 100 records at 4096^2 (32 MiB each)
# to a client draining at the measured browser ceiling: unpaced 3.20 rec/s and
# the socket KILLED at 84 frames; a 24 MiB cap admitted one frame at a time and
# finished at 2.72 rec/s (-15%); 64 MiB finished all 100 at 3.21 rec/s, i.e.
# the client's own receive rate and no cost at all for the correctness.
#
# A single record LARGER than the cap is still sent — the check is made with
# the queue empty, so it never blocks the first frame and cannot deadlock; it
# simply means one frame in flight, which is the least any transport can do.
# At 8192^2 x 4 variants that one frame is 512 MiB and its drain time is what
# display downsampling exists to cut, not something this cap can help with.
INFLIGHT_MAX_BYTES = 64 << 20
INFLIGHT_MAX_FRAMES = 3


def _no_credit(s):
    """True when the client owes us acks and must not be sent more."""
    return s.paced and (s.inflight_bytes > INFLIGHT_MAX_BYTES
                        or len(s.inflight) >= INFLIGHT_MAX_FRAMES)


async def _handle(msg, s, ws):
    if msg.type == "play":
        # the frontier at play time decides playback-only vs solving
        s.clock.set_running(True, s.history.latest_complete())
        s.post_msg(s.status())      # echo the flip ahead of any frame burst
    elif msg.type == "pause":
        s.clock.set_running(False)
        s.post_msg(s.status())
    elif msg.type == "delay":
        s.clock.set_delay(msg.seconds)
    elif msg.type == "loop":
        s.clock.set_loop(msg.on)
        s.post_msg(s.status())      # echo, as play/pause do — the toggle is
                                    # otherwise invisible until the next status
    elif msg.type == "seek":
        # move the cursor NOW, not on the next sender tick: a play arriving
        # right behind the seek must classify playback-vs-solve against the
        # seeked position, never the stale cursor
        first, last = s.history.extent()
        if last >= 0:
            k = min(max(msg.record, first), last)
            s.clock.set_cursor(k, s.history.latest_complete())
            s.pending_seek = k
            s.frame_evt.set()
    elif msg.type == "ping":
        s.post_msg({"type": "pong"})
    elif msg.type == "view":
        # What the client is showing, so the sender can crop to it. Not a
        # `set_params`: nothing computed changes, only which samples of an
        # already-computed record go on the wire.
        s.views = {(v.vid, v.a, v.b): v for v in msg.planes}
        # Re-send the current record at the new resolution instead of waiting
        # for the next one — while paused there may not BE a next one, and a
        # zoom that sharpens only after you press play is not a zoom.
        first, last = s.history.extent()
        if last >= 0:
            s.pending_seek = min(max(int(s.clock.cursor), first), last)
        s.frame_evt.set()
    elif msg.type == "ack":
        # Frame credit returned. Wake the sender NOW rather than letting it
        # wait out its frame-event timeout: at 60 fps a 50 ms tick is three
        # frames of latency added to every paint.
        s.note_ack(msg.record)
        s.frame_evt.set()
    elif msg.type == "set_params":
        cp = None
        if msg.params.U is not None or msg.params.hbar_eff is not None:
            # validate against the LIVE window (auto-expand may have moved
            # it; unions in the pre-regrid window while a plan is pending).
            # hbar-only changes are validated too: a larger hbar widens the
            # Bopp range, and letting an invalid one through would surface
            # as a fatal non-finite check when a pending regrid applies
            # (worker rollback cannot help there — lockstep geometry must
            # stay uniform).
            try:
                hbar = msg.params.hbar_eff or s.cfg.hbar_eff
                expr = msg.params.U if msg.params.U is not None \
                    else s.cfg.potential
                probe = await compile_for(s.validation_grid(), expr,
                                          hbar, s.cfg.variants)
                if msg.params.U is not None:
                    cp = probe
            except Exception as e:
                detail = getattr(e, "detail", str(e))
                s.post_msg({"type": "error", "code": "bad_potential",
                            "message": str(detail)})
                return
        s.apply_params(msg.params, cp)


async def _receiver(ws, s):
    while True:
        text = await ws.receive_text()
        try:
            msg = _client_msg.validate_json(text)
        except ValidationError as e:
            s.post_msg({"type": "error", "code": "bad_message",
                        "message": e.errors()[0].get("msg", "invalid message")})
            continue
        await _handle(msg, s, ws)


def _views_for(s, geom):
    """{(vid, a, b): (Window, Window) | None} for this record's geometry.

    Resolved per RECORD, not per request: a scrub can land on a record computed
    before an auto-expand regrid, and the physical window the client asked for
    has to be answered against the geometry that record actually has. A panel
    the client did not mention maps to None — not sent at all.
    """
    if s.views is None:
        return None
    out = {}
    for (vid, a, b), v in s.views.items():
        if a >= len(geom.N) or b >= len(geom.N):
            continue           # a stale view from before an ndim change
        out[(vid, a, b)] = (
            planeview.select(geom.N[a], geom.lo[a], geom.hi[a], v.a1, v.a2,
                             v.na, _max_step(s, a, b)),
            planeview.select(geom.N[b], geom.lo[b], geom.hi[b], v.b1, v.b2,
                             v.nb, _max_step(s, a, b)))
    return out


def _max_step(s, a, b):
    """Coarsest decimation the retained planes can serve. Read off a record
    rather than recomputed, since the pyramid depth is a property of the grid
    the record was BUILT on."""
    rec = s.history.get(s.history.latest_complete())
    if rec is None:
        return 1
    for pl in rec[2][0].planes:
        if (pl.a, pl.b) == (a, b):
            return pl.max_step
    return 1


def _pack_record(s, k, live):
    rec = s.history.get(k)
    if rec is None:
        return None
    t, geom, variants = rec
    # batch mode streams no live preview at all (only interactive coalesces
    # to the frontier), so FLAG_LIVE_PREVIEW is now never set — the constant
    # stays defined for the unchanged binary layout.
    flags = 0 if live else protocol.FLAG_REPLAY
    # geometry comes from the RECORD, never the session's current grid —
    # replay across a regrid boundary must decode with the old geometry
    return protocol.pack_frame(k, t, geom, variants, flags=flags,
                               views=_views_for(s, geom))


def _progress_msg(s, lc):
    """A tiny, throttled progress report for BATCH compute — no frames. The
    heavy live-preview bundle (tens/hundreds of MiB per record) is replaced
    by this ~400-byte JSON: current time, percent toward t2, the frontier
    record, per-variant throughput, and the frontier record's OBSERVABLES.

    The observables ride along because they are free: the worker computed them
    when it emitted the record, history.get returns references (no array
    copies), and this message is already being sent at PROGRESS_PERIOD. Without
    them the control bar's E / ΔX·ΔP / γ readouts sat at "—" for the whole of a
    batch run while the series plots beside them were live from their own REST
    poll — the data was on screen, just not in the one place that shows the
    current value. Built on the sender's event-loop task, never on the worker
    threads."""
    c = s.clock
    t = c.t_of(lc) if lc >= 0 else c.t1
    span = (c.t2 - c.t1) or 1.0
    pct = max(0.0, min(1.0, (t - c.t1) / span)) * 100.0   # sign-agnostic
    rec = s.history.get(lc) if lc >= 0 else None
    obs = {v.vid: v for v in rec[2]} if rec else {}
    out = []
    for w in s.workers:
        e = {"variant": w.key, "steps_per_sec": round(w.steps_per_sec, 2),
             "steps_total": w.steps_total}
        v = obs.get(protocol.variant_id(**w.flavor))
        if v is not None:
            e.update(E=v.E, purity=v.purity, std=list(v.std),
                     mean=list(v.mean), lz=v.lz)
            if v.ndim == 1:
                e.update(x_std=v.x_std, p_std=v.p_std)
        out.append(e)
    return {"type": "progress", "record": lc, "t": t, "t1": c.t1, "t2": c.t2,
            "percent": pct, "per_variant": out}


async def _guard_send(coro):
    """Await a websocket send, treating a transport-closed error as a normal
    disconnect. uvicorn raises RuntimeError('Unexpected ASGI message
    "websocket.send", after sending "websocket.close"') once the socket is
    closed under us. That is the connection ending, not a streamer failure —
    unwind quietly instead of the traceback that used to spam the log and, via
    the frontend's auto-recover, churn reconnects.

    It does NOT say the client went away, and it used to: the same error is
    raised when uvicorn's own keepalive fails the connection because a ping
    went out behind a backlogged transport buffer (protocol.AckCmd), which is
    a self-inflicted kill. Log what actually happened and nothing more. The
    code is None for the same reason — the old hardcoded 1006 read as a wire
    code and sent a whole investigation after a constant.
    """
    try:
        await coro
    except RuntimeError as e:
        log.warning("streamer send failed, socket already closed: %s", e)
        raise WebSocketDisconnect(code=None) from e


async def _sender(ws, s, recv_task):
    # Resume replay from wherever the cursor is, not from record 0. The
    # browser can drop the socket mid-playback at large grids (a 128 MiB
    # frame overflows its WS receive buffer while the main thread is busy
    # painting) and the frontend auto-reconnects — a fresh sender must
    # CONTINUE from the cursor, not restart the whole replay. Attached/live
    # is unaffected: there the cursor sits at the frontier and the live
    # branch coalesces to the newest record regardless of this seed.
    c = s.clock.cursor
    last_sent = int(c) - 1 if c >= 1 else -1
    last_wall = monotonic()
    last_status = 0.0
    last_progress = 0.0
    last_running = s.clock.running
    last_computing = (s.cfg.mode == "batch" and s.clock.running
                      and not s.clock.stop_at_frontier)
    # The frontier the newest progress report described, or None if THIS sender
    # has sent none. While a batch compute is stopped, only a record landing
    # after a report we ourselves sent may trigger another one — so a fresh
    # sender reports nothing unsolicited, whether it attached to a paused run
    # (the client keeps its own last report across a socket drop, and an
    # unsolicited one would flip its readouts off a painted frame onto the
    # frontier) or to an idle one whose record 0 completes a tick later.
    last_progress_rec = None
    last_loop_epoch = s.clock.loop_epoch
    await _guard_send(ws.send_text(json.dumps(s.status())))
    # exit on s.closed too: a DELETE/TTL close() pops the session and stops
    # its workers but this coroutine still holds `s` (hence its whole
    # FrameHistory). Without this check a streamer attached at close() time
    # keeps tens of GB resident until the client disconnects — invisible to
    # the TTL sweeper (the session is already gone). close() wakes us.
    while not recv_task.done() and not s.closed:
        now = monotonic()
        lc = s.history.latest_complete()
        # Delivery, for the playback auto-pause and the loop wrap: what the
        # client has PAINTED once it is acking, else the best we can know.
        # Strictly more correct than last_sent — with no backpressure on this
        # transport, "sent" meant "buffered", so the gate that exists to stop
        # the cursor running past unseen records was reading a number that had
        # already run past them. A tab that stops painting (hidden, throttled)
        # now correctly stops the cursor too.
        delivered = s.acked if s.paced else last_sent
        hist_first, _ = s.history.extent()
        cursor = s.clock.advance_cursor(now - last_wall, lc, delivered,
                                        hist_first)
        last_wall = now

        # The cursor wrapped: rearm the replay walk behind loop_from. Without
        # this `nxt = max(last_sent + 1, first)` still points past the frontier
        # and the loop stalls with the display frozen on the last record.
        if s.clock.loop_epoch != last_loop_epoch:
            last_loop_epoch = s.clock.loop_epoch
            last_sent = s.clock.loop_from - 1

        # Control channel FIRST: play/pause echoes and periodic status must
        # never queue behind a burst of binary frame sends — the transport
        # button's state depends on them arriving promptly.
        while s.msgs:
            await _guard_send(ws.send_text(json.dumps(s.msgs.popleft())))
        if s.history.take_evicted_flag():
            first, last = s.history.extent()
            await _guard_send(ws.send_text(json.dumps({"type": "eviction",
                                                       "new_extent": [first, last]})))
        # push status immediately on a running-state flip (auto-pause at the
        # frontier, play/pause echo) — the 1 s cadence covers the rest
        if s.clock.running != last_running or now - last_status > STATUS_PERIOD:
            last_running = s.clock.running
            last_status = now
            await _guard_send(ws.send_text(json.dumps(s.status())))
        # batch compute streams no frames — send a throttled progress report
        # instead. Only while actually computing new records (running and not
        # a playback-only replay): batch PLAYBACK takes the replay branch below
        # and streams frames exactly like interactive.
        batch_computing = (s.cfg.mode == "batch" and s.clock.running
                           and not s.clock.stop_at_frontier)
        if batch_computing and now - last_progress > PROGRESS_PERIOD:
            last_progress = now
            last_progress_rec = lc
            await _guard_send(ws.send_text(json.dumps(_progress_msg(s, lc))))
        # ...and a batch compute that STOPS leaves a FINAL report behind. The
        # control bar keeps displaying the newest one it received (batch paints
        # no frames, so nothing else says where the run stopped), and the
        # periodic report above is up to PROGRESS_PERIOD old — with in-flight
        # records still landing after the pause. So: once on the compute→stop
        # flip (a pause, or arriving at t2 inside advance_cursor), and again for
        # each record that lands afterwards. Self-quiescing — the frontier stops
        # moving within a record or two of the pause.
        # It must never fire when a batch PLAYBACK pauses: `batch_computing` was
        # already False through the replay and `lc` does not move, so neither
        # condition holds and the readouts stay on the browsed frame.
        elif s.cfg.mode == "batch" and not batch_computing \
                and (last_computing
                     or (last_progress_rec is not None
                         and lc > last_progress_rec)):
            last_progress = now
            last_progress_rec = lc
            await _guard_send(ws.send_text(json.dumps(_progress_msg(s, lc))))
        last_computing = batch_computing

        k = None
        live = True
        seek = getattr(s, "pending_seek", None)
        if seek is not None:
            s.pending_seek = None
            k = seek                    # already clamped by the handler
            live = k >= lc
            last_sent = -1              # force resend even of the same index
        elif lc >= 0:
            target = int(cursor)
            # A playback-only run must deliver EVERY record: while
            # sequential records remain unsent, stay in the replay branch
            # even when a send blocked long enough for the wall clock to
            # lump the cursor past the frontier — coalescing over that gap
            # is what used to teleport playback straight to the end.
            gap = s.clock.stop_at_frontier and last_sent < lc
            if target >= lc and not gap:
                # live: coalesce to newest — but batch NEVER streams a live
                # frame here (computing OR paused-at-frontier): its only
                # feedback is the progress report above, and the display is
                # reviewed via explicit playback. Only interactive keeps the
                # live preview pinned to the frontier. Frames still reach a
                # batch client through seek and sequential replay (below);
                # playback already delivered the frontier record there, so
                # suppressing this coalesce never drops it.
                if s.cfg.mode != "batch":
                    k = lc
            else:
                # Replay: exact sequential records from history, paced by
                # the cursor. Batch the sends (the loop ticks at ~20 Hz;
                # one record per tick would cap replay at 20 records/s) —
                # but under a WALL-CLOCK budget with preemption: to a slow
                # client each send can block for seconds on backpressure,
                # and an unbounded batch would starve the control channel
                # and keep streaming frames long after a pause arrived.
                # If the client can't keep up, pull the cursor back so
                # playback slips in wall time rather than skipping records.
                first, _ = s.history.extent()
                nxt = max(last_sent + 1, first)
                t0 = monotonic()
                while nxt <= min(target, lc) and not _no_credit(s):
                    payload = _pack_record(s, nxt, live=False)
                    if payload is None:
                        break
                    await _guard_send(ws.send_bytes(payload))
                    s.note_sent(nxt, len(payload))
                    last_sent = nxt
                    nxt += 1
                    if not s.clock.running or s.pending_seek is not None \
                       or monotonic() - t0 > 0.2:
                        break
                if s.pending_seek is None and last_sent < target:
                    s.clock.set_cursor(last_sent, lc)

        # A seek is honoured even with no credit outstanding acks would deny:
        # it is a direct answer to a click, it is one frame, and the client is
        # by definition still painting if it just asked for something.
        if k is not None and k != last_sent and (seek is not None
                                                 or not _no_credit(s)):
            payload = _pack_record(s, k, live)
            if payload is not None:
                await _guard_send(ws.send_bytes(payload))
                s.note_sent(k, len(payload))
                last_sent = k

        s.frame_evt.clear()
        try:
            await asyncio.wait_for(s.frame_evt.wait(),
                                   timeout=0.05 if s.clock.running else 0.2)
        except asyncio.TimeoutError:
            pass
    # normal loop exit (recv_task done, or the session closed under us) —
    # exceptions from a send propagate to ws_endpoint instead. last_sent
    # says where playback had reached, which pins down a mid-replay drop.
    log.info("streamer %s: sender stopped (closed=%s recv_done=%s last_sent=%d)",
             s.id, s.closed, recv_task.done(), last_sent)


@router.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    s = sessions.get_session(sid)
    if s is None:
        await ws.close(code=4404)
        return
    if s.ws_attached:
        # Give the PREVIOUS attachment a moment to finish unwinding before
        # refusing. A reconnect races its own predecessor's teardown, which is
        # bounded at ~3 s (the `finally` awaits both tasks), and a 4409 reads to
        # the client as a plain close — so it re-entered recover(), waited, and
        # tried again, turning one dropped socket into a churn loop. Polling is
        # enough: the flag is cleared by that same event loop.
        for _ in range(40):
            await asyncio.sleep(0.1)
            if not s.ws_attached or s.closed:
                break
        if s.ws_attached:
            await ws.accept()
            await ws.close(code=4409)
            return
    if s.closed:
        await ws.close(code=4404)
        return
    # claim the session BEFORE the first await — two near-simultaneous
    # connects must not both pass the check above
    s.ws_attached = True
    s.pending_seek = None
    # A fresh socket has nothing in flight and has issued no credit yet: a
    # reattach must not inherit the dead socket's debt (which would stall the
    # new sender outright) nor its armed pacing (the new client might not ack).
    s.reset_inflight()
    recv_task = None
    send_task = None
    try:
        await ws.accept()
        recv_task = asyncio.create_task(_receiver(ws, s))
        # run the sender as a task so close() can CANCEL it — a sender blocked
        # inside a large backpressured send_bytes cannot poll self.closed, and
        # a session deletion must not wait on it (that stranded the history).
        send_task = asyncio.create_task(_sender(ws, s, recv_task))
        s.stream_task = send_task
        # When the CLIENT drops (recv_task ends), tear the sender down at once
        # instead of waiting for it to notice at its loop top: a sender blocked
        # in a backpressured 128 MiB send never reaches that check, so without
        # this the coroutine (and the whole session it closes over) is
        # stranded until TCP finally errors the send — the reconnect-churn
        # leak. Cancelling a finished task is a harmless no-op.
        recv_task.add_done_callback(lambda _: send_task.cancel())
        await send_task
    except WebSocketDisconnect as e:
        # A code only when one really came off the wire — _guard_send passes
        # None precisely so a closed transport cannot masquerade as one.
        code = getattr(e, "code", None)
        log.warning("streamer %s: disconnected%s", s.id,
                    "" if code is None else " (code=%s)" % code)
    except asyncio.CancelledError:
        # close() cancelled OUR sender (session deleted) — normal teardown.
        # If instead ws_endpoint itself was cancelled (server shutdown),
        # send_task is still running: propagate so shutdown isn't swallowed.
        if send_task is None or not send_task.cancelled():
            raise
        log.info("streamer %s: torn down by close()", s.id)
    except Exception:
        log.exception("streamer for session %s failed", s.id)
    finally:
        s.stream_task = None
        # Close the socket FIRST. A _receiver blocked in ws.receive_text()
        # does not always respond to task.cancel() (the leak-check found
        # ws_endpoint hung here on `await recv_task`, pinning the whole
        # session); closing the transport makes that pending receive raise so
        # the task can actually finish.
        with suppress(Exception):
            await ws.close()
        tasks = [t for t in (send_task, recv_task) if t is not None]
        for t in tasks:
            t.cancel()
        # Await them so their exceptions are retrieved, but NEVER hang the
        # teardown on a task that refuses to die — a stranded receive must not
        # keep this coroutine (and its session) alive.
        if tasks:
            with suppress(Exception):
                await asyncio.wait(tasks, timeout=3.0)
        s.ws_attached = False
        s.clock.set_running(False)     # pause on disconnect; TTL takes over
        s.last_seen = time.monotonic()
