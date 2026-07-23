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

from core import protocol
from core import session as sessions
from routers.sessions import compile_for

log = logging.getLogger(__name__)

router = APIRouter()

_client_msg = TypeAdapter(protocol.ClientMsg)

STATUS_PERIOD = 1.0


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


def _pack_record(s, k, live):
    rec = s.history.get(k)
    if rec is None:
        return None
    t, geom, variants = rec
    flags = 0 if live else protocol.FLAG_REPLAY
    if s.cfg.mode == "runahead" and live:
        flags |= protocol.FLAG_LIVE_PREVIEW
    # geometry comes from the RECORD, never the session's current grid —
    # replay across a regrid boundary must decode with the old geometry
    return protocol.pack_frame(k, t, geom, variants, flags=flags)


async def _guard_send(coro):
    """Await a websocket send, treating a transport-closed error as a normal
    disconnect. uvicorn raises RuntimeError('Unexpected ASGI message
    "websocket.send", after sending "websocket.close"') when a send races the
    client going away (a plain disconnect, or a reconnect superseding this
    socket). That is the connection ending, not a streamer failure — unwind
    quietly instead of the traceback that used to spam the log and, via the
    frontend's auto-recover, churn reconnects."""
    try:
        await coro
    except RuntimeError as e:
        log.warning("streamer send failed (client gone / transport closed): %s", e)
        raise WebSocketDisconnect(code=1006) from e


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
    last_running = s.clock.running
    await _guard_send(ws.send_text(json.dumps(s.status())))
    # exit on s.closed too: a DELETE/TTL close() pops the session and stops
    # its workers but this coroutine still holds `s` (hence its whole
    # FrameHistory). Without this check a streamer attached at close() time
    # keeps tens of GB resident until the client disconnects — invisible to
    # the TTL sweeper (the session is already gone). close() wakes us.
    while not recv_task.done() and not s.closed:
        now = monotonic()
        lc = s.history.latest_complete()
        cursor = s.clock.advance_cursor(now - last_wall, lc, last_sent)
        last_wall = now

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
                k = lc                   # live: coalesce to newest
                # (runahead keeps the cursor pinned here until the user
                # seeks, so the newest frame previews while computing;
                # after a seek both modes replay from history identically)
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
                while nxt <= min(target, lc):
                    payload = _pack_record(s, nxt, live=False)
                    if payload is None:
                        break
                    await _guard_send(ws.send_bytes(payload))
                    last_sent = nxt
                    nxt += 1
                    if not s.clock.running or s.pending_seek is not None \
                       or monotonic() - t0 > 0.2:
                        break
                if s.pending_seek is None and last_sent < target:
                    s.clock.set_cursor(last_sent, lc)

        if k is not None and k != last_sent:
            payload = _pack_record(s, k, live)
            if payload is not None:
                await _guard_send(ws.send_bytes(payload))
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
        await ws.accept()
        await ws.close(code=4409)
        return
    # claim the session BEFORE the first await — two near-simultaneous
    # connects must not both pass the check above
    s.ws_attached = True
    s.pending_seek = None
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
        log.warning("streamer %s: disconnected (code=%s)", s.id,
                    getattr(e, "code", "?"))
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
