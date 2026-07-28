"""
mp4 export of a computed record range: create a job, watch it (REST poll or
the session's WebSocket 'export' events), download the file, delete it.

Export is PAUSED-only: a running session evicts old records once the history
cap is reached, and it is also the interaction the feature is for — you film
a range you have already played back and judged interesting.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import config
from core import axes as ax
from core import render_mpl
from core import session as sessions
from core import videoexport
from core.protocol import ExportSpec

router = APIRouter()


@router.post("/sessions/{sid}/export", status_code=202)
def create_export(sid: str, spec: ExportSpec):
    s = sessions.get_session(sid)
    if s is None:
        raise HTTPException(404, "no such session")
    # WHAT was asked for is checked first, before anything about the server or
    # the session's state: a spec naming a plane this run does not have is
    # wrong however much history exists and whether or not ffmpeg is
    # installed, and "no computed records to export" would send you looking in
    # the wrong place. Same reasoning the removed ndim gate carried.
    #
    # These live here rather than in the schema because what is available
    # depends on the session's ndim, which the request body does not carry.
    # Both refusals name what IS available, not just what was wrong — a 1D
    # session has one plane and no <Lz>, and neither is guessable from the
    # index that failed.
    unknown = set(spec.variants or ()) - set(s.cfg.variants)
    if unknown:
        raise HTTPException(422, "variants not in this session: %s"
                            % ", ".join(sorted(unknown)))
    planes = ax.planes(s.ndim)
    bad = [i for i in (spec.planes or ()) if i >= len(planes)]
    if bad:
        raise HTTPException(422, "no such plane%s for a %dD run: %s — this "
                            "session has %d (%s)"
                            % ("s" if len(bad) > 1 else "", s.ndim,
                               ", ".join(str(i) for i in bad), len(planes),
                               ", ".join(ax.plane_label(s.ndim, p)
                                         for p in planes)))
    available = render_mpl.diagnostics_available(s.ndim)
    bad = [d for d in (spec.diagnostics or ()) if d not in available]
    if bad:
        raise HTTPException(422, "unknown diagnostics plot%s for a %dD run: "
                            "%s — available: %s"
                            % ("s" if len(bad) > 1 else "", s.ndim,
                               ", ".join(bad), ", ".join(available)))
    if videoexport.ffmpeg_path() is None:
        raise HTTPException(503, "ffmpeg is not installed on the server")
    if s.clock.running:
        raise HTTPException(409, "pause the session before exporting "
                                 "(export renders already-computed records)")
    if videoexport.active_for(sid) is not None:
        raise HTTPException(409, "an export is already running for this session")
    first, last = s.history.extent()
    if last < 0:
        raise HTTPException(422, "no computed records to export")
    k0 = first if spec.k0 is None else max(spec.k0, first)
    k1 = last if spec.k1 is None else min(spec.k1, last)
    if k1 < k0:
        raise HTTPException(422, "empty record range after clamping to the "
                                 "retained history [%d, %d]" % (first, last))
    job = videoexport.start(s, spec, k0, k1, config.EXPORT_DIR)
    return job.status()


@router.get("/exports/{jid}")
def export_status(jid: str):
    job = videoexport.get(jid)
    if job is None:
        raise HTTPException(404, "no such export job")
    return job.status()


@router.get("/exports/{jid}/file")
def export_file(jid: str):
    job = videoexport.get(jid)
    if job is None:
        raise HTTPException(404, "no such export job")
    if job.state != "done" or not os.path.exists(job.path):
        raise HTTPException(409, "export is %s" % job.state)
    return FileResponse(job.path, media_type="video/mp4",
                        filename=job.download_name)


@router.delete("/exports/{jid}")
def export_delete(jid: str):
    if videoexport.drop(jid) is None:
        raise HTTPException(404, "no such export job")
    return {"ok": True}
