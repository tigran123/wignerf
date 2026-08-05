"""
mp4 export of an already-computed record range: a background thread reads
records out of the session's FrameHistory, renders each one with
core.render_mpl and pipes the raw RGBA Agg buffer into
`ffmpeg -c:v libx264`.

Deliberately NOT a live recorder: export is a PAUSED-only action on history
that already exists. That is not just a scope decision — a running session
evicts its oldest records once the byte cap is reached, and an export
reading behind the frontier would lose them mid-file.

Two passes over the range:
  1. scan  — collects the E/ΔQ·ΔK/γ/⟨Lz⟩ series, the fixed colour scale per
             (variant, plane), the fixed marginal amplitude per axis and the
             widest window any record used (all cheap scalars already stored
             in the records), and proves every record is still retained
             BEFORE ffmpeg is spawned. Only VALUE scales are export-wide: the
             spatial axes follow each record's own geometry
             (render_mpl._apply_geom), so a frame from before an
             auto-expansion still fills its panel;
  2. render — one figure update + one stdin write per frame.

WHAT the frame shows is `spec.planes` x `spec.variants` panels plus
`spec.diagnostics` plots — see render_mpl's docstring. A 2D record carries six
planes and nine diagnostics, which is more than a frame can hold, so the job
carries a selection and the metadata block records it.

This module must not import core.session (the session imports it back for
cleanup); the session object is duck-typed here.
"""

import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from time import monotonic

import numpy

import config

from . import axes as ax
from . import describe, pyramid, render_mpl

log = logging.getLogger(__name__)

# how long a finished file stays downloadable before the sweeper unlinks it
FILE_TTL = 30*60.0
PROGRESS_PERIOD = 0.5
# below this many frames an export renders serially (the ~1-2 s pool warmup —
# spawn + a FrameFigure per worker — is not worth it for a tiny job)
POOL_MIN_FRAMES = 16

_JOBS = {}
_LOCK = threading.Lock()
# Matplotlib guarantees nothing about two figures rendering in parallel
# threads (shared font manager and image caches), and two exports would
# fight for the same cores anyway — so renders are serialized process-wide.
# A job waiting here honestly reports "queued".
_RENDER_LOCK = threading.Lock()


def ffmpeg_path():
    return shutil.which("ffmpeg")


# The frame RENDER (matplotlib/Agg) dominates export time, not the encode, so
# the encoder choice is a top-up: nvenc frees the CPU for the render pool and
# is ~3x faster at 4K, libx264 veryfast is the portable fallback. NB the right
# GPU path is the h264_nvenc ENCODER, not ffmpeg's -hwaccel (that is a decode
# flag and does nothing for our rawvideo input).
_NVENC_OK = None


# The probe clip's size. NVENC REFUSES ANYTHING SMALLER THAN 145x49 (H.264):
# it reports "Frame Dimension less than the minimum supported value", which is
# indistinguishable at the exit code from "there is no GPU here". This probe
# used 64x64 and therefore answered "unavailable" on every machine it was ever
# run on, including a workstation with two idle NVIDIA cards — measured
# 2026-08-05, and the reason every export here had quietly been libx264. Small
# enough to stay instant, comfortably over the floor so a future NVENC
# generation raising it does not silently reintroduce the same false negative.
_PROBE_SIZE = "256x256"


def _nvenc_ok():
    """Whether h264_nvenc actually WORKS here — cached. The encoder can be
    built into ffmpeg yet fail at runtime without a driver/GPU (the CPU-only
    VPS), so grepping -encoders is not enough: we run a tiny encode once.

    It has to be a REAL encode at a REAL size — see _PROBE_SIZE."""
    global _NVENC_OK
    if _NVENC_OK is None:
        _NVENC_OK = _probe_nvenc()
        log.info("export: h264_nvenc %s", "available" if _NVENC_OK else
                 "unavailable (falling back to libx264)")
    return _NVENC_OK


def _probe_nvenc():
    exe = ffmpeg_path()
    if exe is None:
        return False
    try:
        return subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "nullsrc=s=%s:d=0.1" % _PROBE_SIZE,
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose_encoder(mode=None):
    """ffmpeg `-c:v …` args for the configured encoder (WIGNERF_EXPORT_ENCODER
    = auto | cpu | nvenc).

    **auto is libx264, NOT the GPU, and that is a measured choice.** The GPU
    encoder is reachable — `nvenc`, or ExportSpec.encoder per job — and
    `_nvenc_ok` no longer lies about it. It simply buys nothing here: the same
    60-frame 1920x1080 export takes 4.3 s either way and h264_nvenc -cq 19
    writes a 1.8x LARGER file than libx264 -crf 18 (0.35 vs 0.19 MiB), because
    the ENCODE is a rounding error against the frame RENDER at any grid worth
    exporting (2.2 s/frame/worker at 4096^2 against milliseconds of encode).
    Preferring the GPU by default would trade file size for nothing.

    Do not "restore" auto to nvenc without re-measuring both numbers; the
    reasoning that once made it look right — "it frees cores for the render
    pool" — is exactly what the wall clock refuses to show.
    """
    mode = (mode or config.EXPORT_ENCODER or "auto").lower()
    if mode == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"]
    # veryfast (was medium): ~2x faster encode, file ~7% larger, visually
    # identical for this smooth content — and it frees cores for the render
    # pool, which is where the time actually goes.
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-threads", "0"]


def export_workers():
    """How many frame-render processes an export uses (WIGNERF_EXPORT_WORKERS,
    0 = auto). Scaling flattens past the physical cores, hence the cap."""
    return config.EXPORT_WORKERS or min(os.cpu_count() or 4, 8)


# ---------------------------------------------------------------------------
# render-pool workers (run in spawn subprocesses — no CUDA, matplotlib only)
# ---------------------------------------------------------------------------

_WORKER = {}

# Stands in for a plane this job does not show. Its POSITION has to survive
# (render_mpl._cells indexes planes canonically); its bytes do not.
_EMPTY = numpy.zeros((0, 0), dtype=numpy.uint16)


def _worker_init(variants, stats, meta, width, height, show_grid, theme,
                 planes, diagnostics):
    """One persistent FrameFigure per worker process, reused across records
    (it re-applies the window itself on the rare auto-expand regrid). Building
    it here — once per worker, not once per frame — is what the pool is for."""
    _WORKER["fig"] = render_mpl.FrameFigure(
        variants, stats, meta, width=width, height=height,
        show_grid=show_grid, theme=theme, planes=planes,
        diagnostics=diagnostics)


def _worker_render(args):
    """Render one record; return its RGBA bytes copied out of the Agg buffer
    (the next update() overwrites the buffer, and the bytes are pickled back
    to the parent)."""
    k, t, geom, vframes, k0, k1 = args
    buf = _WORKER["fig"].update(k, t, geom, vframes, k0, k1)
    return bytes(memoryview(buf).cast("B"))


class ExportJob(threading.Thread):
    def __init__(self, session, spec, k0, k1, outdir):
        super().__init__(daemon=True, name="wignerf-export-%s" % session.id)
        self.id = uuid.uuid4().hex[:12]
        self.session = session
        self.spec = spec
        self.k0, self.k1 = int(k0), int(k1)
        self.records = list(range(self.k0, self.k1 + 1, spec.stride))
        self.variants = list(spec.variants or session.cfg.variants)
        # WHAT the frame shows. None = the whole record for panels, and
        # render_mpl's per-ndim default for the diagnostics column (which at
        # ndim=2 is the five series — see its docstring).
        self.ndim = session.cfg.ndim
        all_planes = ax.planes(self.ndim)
        self.planes = ([all_planes[i] for i in spec.planes]
                       if spec.planes is not None else list(all_planes))
        self.diagnostics = (list(spec.diagnostics)
                            if spec.diagnostics is not None
                            else render_mpl.diagnostics_default(self.ndim))
        # On-disk name stays collision-proof (session + job id); the name the
        # BROWSER saves is the readable one below — two exports of the same
        # range in the same minute must not overwrite each other's file
        # while one of them is being downloaded. The plane marker is part of
        # that: at ndim=2 the same range is routinely exported twice with
        # different planes, and those are different videos.
        self.path = os.path.join(outdir, "wignerf-%s-%s.mp4"
                                 % (session.id, self.id))
        self.download_name = "wignerf-%s%s-%drec%s-%dx%d-%s.mp4" % (
            "-".join(v.upper() for v in self.variants),
            self._plane_marker(),
            len(self.records),
            "" if spec.stride == 1 else "-every%d" % spec.stride,
            spec.width, spec.height,
            time.strftime("%Y%m%d-%H%M"))
        self.state = "queued"      # queued|running|done|error|cancelled
        self.done = 0
        self.total = len(self.records)
        # Frames actually rendered per second — the number the export panel
        # shows, and the one that says whether a long render is worth waiting
        # for. Rolling over ~1 s while running (the same idiom as
        # worker.steps_per_sec, and for the same reason: a cumulative average
        # spends its first seconds climbing out of the pool warmup and reads as
        # a slowdown that is not happening), then replaced by the run's overall
        # average once the job finishes, which is the figure worth quoting.
        self.render_fps = 0.0
        self._render_t0 = None
        self._rate_mark = (0, 0.0)
        # Detail bound for what crosses to a render worker (see _trim). The
        # figure's own width: a panel cannot be wider than its figure, so this
        # can never trim below what the renderer draws.
        self._px_bound = int(spec.width)
        self.error = None
        self.finished_at = None
        self.cancel_evt = threading.Event()

    def _plane_marker(self):
        """The plane part of the download name. Empty at ndim=1, where there
        is only ever one plane and every existing name would otherwise grow a
        redundant "-xp"; "-6pl" rather than all six spelled out, which is 24
        characters of nothing."""
        if self.ndim == 1:
            return ""
        if len(self.planes) == len(ax.planes(self.ndim)):
            return "-%dpl" % len(self.planes)
        return "-" + "+".join(ax.plane_label(self.ndim, p).replace(",", "")
                              for p in self.planes)

    # -- status -------------------------------------------------------------

    def status(self):
        return {"job_id": self.id, "session_id": self.session.id,
                "state": self.state, "done": self.done, "total": self.total,
                "bytes": (os.path.getsize(self.path)
                          if self.state == "done" and os.path.exists(self.path)
                          else 0),
                "error": self.error,
                "filename": self.download_name,
                # NB two different rates: `fps` is the VIDEO's frame rate (what
                # the mp4 plays at), `render_fps` is how fast this machine is
                # producing those frames. They are unrelated, and conflating
                # them would make a slow render look like a slow video.
                "fps": self.spec.fps,
                "render_fps": round(self.render_fps, 2),
                "duration_s": self.total/float(self.spec.fps)}

    def _post(self):
        d = dict(self.status())
        d["type"] = "export"
        self.session.post_msg(d)

    def cancel(self):
        self.cancel_evt.set()

    # -- thread body --------------------------------------------------------

    def run(self):
        with _RENDER_LOCK:
            self._run()

    def _run(self):
        if self.cancel_evt.is_set():        # cancelled while queued
            self.state = "cancelled"
            self.finished_at = time.monotonic()
            self._post()
            return
        self.state = "running"
        self._post()
        fig = None
        proc = None
        executor = None
        try:
            stats, geom0 = self._scan()
            meta = render_mpl.meta_columns(
                self.session.cfg, geom0, stats, self.variants, self.k0,
                self.k1, self.total, self.spec.fps, self.session.param_log,
                planes=self.planes, diagnostics=self.diagnostics)
            proc = self._spawn_ffmpeg()
            self._last_post = 0.0
            # the render clock starts HERE, past the scan and the ffmpeg spawn:
            # this is the rate of the thing the progress bar is counting
            self._render_t0 = monotonic()
            self._rate_mark = (0, self._render_t0)
            # Rendering a frame (matplotlib/Agg) dominates export time, so it
            # is spread over a pool of processes while this thread feeds the
            # ordered frames to one ffmpeg. A small job renders serially: the
            # ~1-2 s pool warmup (spawn + a FrameFigure per worker) is not
            # worth it, and it keeps the light path unchanged.
            w = export_workers()
            if w <= 1 or len(self.records) < max(2*w, POOL_MIN_FRAMES):
                fig = self._render_serial(proc, stats, meta)
            else:
                # spawn, NOT fork: the backend initializes CUDA, and forking
                # after that inherits a broken context. spawn starts clean
                # Python; these workers only touch matplotlib/numpy.
                executor = ProcessPoolExecutor(
                    max_workers=w,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_worker_init,
                    initargs=(self.variants, stats, meta, self.spec.width,
                              self.spec.height, self.spec.show_grid,
                              self.spec.theme, self.planes,
                              self.diagnostics))
                self._render_parallel(proc, executor, w)
            proc.stdin.close()
            rc = proc.wait(timeout=120)
            proc = None
            if rc != 0:
                raise ValueError("ffmpeg exited with code %d" % rc)
            self.state = "done"
        except _Cancelled:
            self.state = "cancelled"
            self._unlink()
        except Exception as e:
            log.exception("export job %s failed", self.id)
            self.state = "error"
            self.error = str(e)
            self._unlink()
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if proc is not None:
                _kill(proc)
            if fig is not None:
                fig.close()
            self.finished_at = time.monotonic()
            # The run is over, so the rolling window is a stale sample of its
            # last second; report what the whole render actually averaged.
            if self._render_t0 is not None and self.done:
                dt = monotonic() - self._render_t0
                if dt > 0:
                    self.render_fps = self.done/dt
            self._post()

    def _emit(self, proc, buf):
        """Write one rendered frame to ffmpeg + progress bookkeeping."""
        try:
            proc.stdin.write(buf)
        except BrokenPipeError:
            # ffmpeg died mid-stream (its diagnostics went to the server
            # log); report that, not "broken pipe"
            raise ValueError("ffmpeg exited early with code %s"
                             % proc.wait(timeout=10)) from None
        self.done += 1
        now = monotonic()
        n, mark = self._rate_mark
        if now - mark > 1.0:
            self.render_fps = (self.done - n)/(now - mark)
            self._rate_mark = (self.done, now)
        if now - self._last_post > PROGRESS_PERIOD:
            self._last_post = now
            self._post()

    def _read_record(self, k):
        """Fetch one record's (t, geom, ordered-vframes) from history — always
        in this thread (history is in-process; a paused session never evicts,
        but the guard stays)."""
        rec = self.session.history.get(k)
        if rec is None:
            raise ValueError("record %d is no longer retained "
                             "(history evicted)" % k)
        t, geom, vframes = rec
        return t, geom, self._order(vframes)

    def _render_serial(self, proc, stats, meta):
        fig = render_mpl.FrameFigure(self.variants, stats, meta,
                                     width=self.spec.width,
                                     height=self.spec.height,
                                     show_grid=self.spec.show_grid,
                                     theme=self.spec.theme,
                                     planes=self.planes,
                                     diagnostics=self.diagnostics)
        for k in self.records:
            if self.cancel_evt.is_set():
                raise _Cancelled()
            t, geom, vframes = self._read_record(k)
            self._emit(proc, fig.update(k, t, geom, vframes, self.k0, self.k1))
        return fig

    def _trim(self, vframes):
        """Strip what a render worker cannot draw, BEFORE it is pickled.

        A record crossing a process boundary is copied twice (pickle here,
        unpickle there) and that cost is the whole frame budget at a large
        grid: measured at 8192^2 one record is 170.8 MiB and takes 70 ms just
        to SERIALIZE, before a byte moves through the pipe — ~312 ms/frame all
        told, which is the 3.2 fps a 1D 8192^2 export was stuck at whatever the
        video size. The renderer only ever draws ~767 px per panel, i.e. 2 MiB.

        Two cuts, both of which leave the drawn frame BIT-IDENTICAL:

        - planes this job does not show lose their payload entirely (a phase
          portrait of one plane shipped all six), and
        - the rest keep only the pyramid levels at or under `_px_bound`.

        The bound is the FIGURE's full width, not the panel's. A panel cannot
        be wider than the figure it is in, so the bound can never trim below
        what the worker needs — no layout arithmetic is mirrored here, and
        nothing silently under-resolves if the layout changes. The worker then
        picks its own exact level out of what survives, exactly as it would
        have from the full pyramid, because every retained level keeps its
        absolute decimation.

        Positions are preserved: `render_mpl._cells` indexes planes by their
        CANONICAL index, so a dropped plane has to stay in the list.
        """
        keep = {ax.plane_index(self.ndim, pl) for pl in self.planes}
        out = []
        for vf in vframes:
            planes = []
            for i, pf in enumerate(vf.planes):
                if i not in keep:
                    planes.append(replace(pf, wq=_EMPTY, mips=()))
                    continue
                step = render_mpl.plane_step(pf, self._px_bound)
                j = pyramid.level_for(step)
                planes.append(pf if step == 1 else
                              replace(pf, wq=pf.mips[j], mips=pf.mips[j + 1:]))
            out.append(replace(vf, planes=tuple(planes)))
        return out

    def _render_parallel(self, proc, executor, w):
        """Frames render out of order in the pool but reach ffmpeg in order:
        a sliding window of at most w+2 outstanding futures, consumed FIFO by
        .result() (so workers run ahead while this thread waits on the head),
        which also bounds memory to that many in-flight frames."""
        window = deque()
        pending = iter(self.records)

        def submit_next():
            for k in pending:
                t, geom, vframes = self._read_record(k)
                window.append(executor.submit(
                    _worker_render,
                    (k, t, geom, self._trim(vframes), self.k0, self.k1)))
                return True
            return False

        for _ in range(w + 2):
            if self.cancel_evt.is_set():
                raise _Cancelled()
            if not submit_next():
                break
        while window:
            if self.cancel_evt.is_set():
                raise _Cancelled()
            buf = window.popleft().result()
            self._emit(proc, buf)
            submit_next()

    def _order(self, vframes):
        """Records carry every session variant in bundle order; an export of
        a subset picks its own, keeping the requested order."""
        by_key = {render_mpl.key_of_vid(vf.vid): vf for vf in vframes}
        return [by_key[k] for k in self.variants]

    def _scan(self):
        """Pass 1: series + fixed colour scales + the widest window (quoted
        in the metadata block; the plots follow each record).

        Collected for EVERY plane and every diagnostic the record carries,
        not just the selected ones: it is all in the record already, it costs
        a few floats, and it keeps RangeStats a statement about the RANGE
        rather than about one job's selection."""
        nd = self.ndim
        naxes = ax.n_axes(nd)
        st = render_mpl.RangeStats(ndim=nd)
        st.marg_max = [0.0]*naxes
        for key in self.variants:
            st.E[key], st.purity[key], st.lz[key] = [], [], []
            for d in range(nd):
                st.uncert[(key, d)] = []
            for plane in ax.planes(nd):
                st.scale[(key, plane)] = 0.0
        lo = [float("inf")]*naxes
        hi = [float("-inf")]*naxes
        geom0 = None
        for k in self.records:
            if self.cancel_evt.is_set():
                raise _Cancelled()
            rec = self.session.history.get(k)
            if rec is None:
                raise ValueError("record %d is not retained (evicted, or the "
                                 "range is outside the computed history)" % k)
            t, geom, vframes = rec
            if geom0 is None:
                geom0 = geom
            st.t.append(t)
            for a in range(naxes):
                lo[a], hi[a] = min(lo[a], geom.lo[a]), max(hi[a], geom.hi[a])
            for key, vf in zip(self.variants, self._order(vframes)):
                st.E[key].append(vf.E)
                st.purity[key].append(vf.purity)
                st.lz[key].append(vf.lz)
                # index-matched: the dual of q_d is k_d on array axis ndim+d
                # (axes.conjugate). Pairing these wrong is silent.
                for d in range(nd):
                    st.uncert[(key, d)].append(vf.std[d]*vf.std[nd + d])
                for pf in vf.planes:
                    kp = (key, (pf.a, pf.b))
                    st.scale[kp] = max(st.scale[kp], pf.wmax, -pf.wmin)
                for a in range(naxes):
                    st.marg_max[a] = max(st.marg_max[a], float(vf.marg[a].max()))
        if geom0 is None:
            raise ValueError("no records in the requested range")
        for kp, v in st.scale.items():
            if v <= 0.0:
                st.scale[kp] = 1e-30
        st.lo, st.hi = tuple(lo), tuple(hi)
        return st, geom0

    def _spawn_ffmpeg(self):
        cfg = self.session.cfg
        enc = choose_encoder(self.spec.encoder)
        comment = describe.config_json(
            cfg, self.session.param_log, at_record=self.k0,
            export={"records": [self.k0, self.k1], "stride": self.spec.stride,
                    "fps": self.spec.fps, "frames": self.total,
                    "variants": self.variants, "encoder": enc[1],
                    # what the frame SHOWS, so a re-imported mp4 records the
                    # subset it was rendered from and not just the run
                    "planes": [list(p) for p in self.planes],
                    "diagnostics": self.diagnostics})
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgba",
               "-video_size", "%dx%d" % (self.spec.width, self.spec.height),
               "-framerate", str(self.spec.fps), "-i", "pipe:0", "-an"] + enc + [
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-metadata", "title=wignerf W(%s,t) records %d-%d"
               % (",".join(ax.labels(self.ndim)), self.k0, self.k1),
               "-metadata", "comment=%s" % comment,
               self.path]
        log.info("export %s: %d frames @ %s -> %s",
                 self.id, self.total, enc[1], self.path)
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _unlink(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def cleanup(self):
        self.cancel()
        self._unlink()


class _Cancelled(Exception):
    pass


def _kill(proc):
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def start(session, spec, k0, k1, outdir):
    os.makedirs(outdir, exist_ok=True)
    job = ExportJob(session, spec, k0, k1, outdir)
    with _LOCK:
        _JOBS[job.id] = job
    job.start()
    return job


def get(job_id):
    with _LOCK:
        return _JOBS.get(job_id)


def active_for(session_id):
    """The session's unfinished job, if any (one export at a time)."""
    with _LOCK:
        for j in _JOBS.values():
            if j.session.id == session_id and j.state in ("queued", "running"):
                return j
    return None


def drop(job_id):
    with _LOCK:
        job = _JOBS.pop(job_id, None)
    if job is not None:
        job.cleanup()
    return job


def close_session(session_id):
    """Cancel and clean every job of a session that is going away."""
    with _LOCK:
        ids = [j.id for j in _JOBS.values() if j.session.id == session_id]
    for jid in ids:
        drop(jid)


def sweep(now=None):
    """Drop finished jobs whose file has outlived FILE_TTL (called from the
    session TTL sweeper)."""
    now = time.monotonic() if now is None else now
    with _LOCK:
        stale = [j.id for j in _JOBS.values()
                 if j.finished_at is not None and now - j.finished_at > FILE_TTL]
    for jid in stale:
        drop(jid)


def close_all():
    with _LOCK:
        ids = list(_JOBS)
    for jid in ids:
        drop(jid)


def probe_json(path):
    """ffprobe helper (tests/diagnostics): stream AND format info of an
    exported file. The format half carries the `comment` tag, which is the
    setup document lib/mp4meta.ts reads back on import."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    out = subprocess.run([exe, "-v", "error", "-print_format", "json",
                          "-show_streams", "-show_format", path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)
