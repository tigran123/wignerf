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
  1. scan  — collects the E/ΔX·ΔP/γ series, the per-variant fixed colour
             scale, the fixed marginal amplitudes and the widest window any
             record used (all cheap scalars already stored in the records),
             and proves every record is still retained BEFORE ffmpeg is
             spawned. Only VALUE scales are export-wide: the spatial axes
             follow each record's own geometry (render_mpl._apply_geom), so
             a frame from before an auto-expansion still fills its panel;
  2. render — one figure update + one stdin write per frame.

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
from time import monotonic

import config

from . import describe, render_mpl

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


def _nvenc_ok():
    """Whether h264_nvenc actually WORKS here — cached. The encoder can be
    built into ffmpeg yet fail at runtime without a driver/GPU (the CPU-only
    VPS), so grepping -encoders is not enough: we run a tiny encode once."""
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
             "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose_encoder(mode=None):
    """ffmpeg `-c:v …` args for the configured encoder (WIGNERF_EXPORT_ENCODER
    = auto | cpu | nvenc). auto = nvenc if it works, else libx264."""
    mode = (mode or config.EXPORT_ENCODER or "auto").lower()
    if mode == "nvenc" or (mode == "auto" and _nvenc_ok()):
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"]
    # veryfast (was medium): ~2x faster encode, file ~7% larger, visually
    # identical for this smooth content — and it frees cores for the render
    # pool where nvenc is unavailable.
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


def _worker_init(variants, stats, meta, width, height, show_grid):
    """One persistent FrameFigure per worker process, reused across records
    (it re-applies the window itself on the rare auto-expand regrid). Building
    it here — once per worker, not once per frame — is what the pool is for."""
    _WORKER["fig"] = render_mpl.FrameFigure(
        variants, stats, meta, width=width, height=height, show_grid=show_grid)


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
        # On-disk name stays collision-proof (session + job id); the name the
        # BROWSER saves is the readable one below — two exports of the same
        # range in the same minute must not overwrite each other's file
        # while one of them is being downloaded.
        self.path = os.path.join(outdir, "wignerf-%s-%s.mp4"
                                 % (session.id, self.id))
        self.download_name = "wignerf-%s-%drec%s-%dx%d-%s.mp4" % (
            "-".join(v.upper() for v in self.variants),
            len(self.records),
            "" if spec.stride == 1 else "-every%d" % spec.stride,
            spec.width, spec.height,
            time.strftime("%Y%m%d-%H%M"))
        self.state = "queued"      # queued|running|done|error|cancelled
        self.done = 0
        self.total = len(self.records)
        self.error = None
        self.finished_at = None
        self.cancel_evt = threading.Event()

    # -- status -------------------------------------------------------------

    def status(self):
        return {"job_id": self.id, "session_id": self.session.id,
                "state": self.state, "done": self.done, "total": self.total,
                "bytes": (os.path.getsize(self.path)
                          if self.state == "done" and os.path.exists(self.path)
                          else 0),
                "error": self.error,
                "filename": self.download_name,
                "fps": self.spec.fps,
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
                self.k1, self.total, self.spec.fps, self.session.param_log)
            proc = self._spawn_ffmpeg()
            self._last_post = 0.0
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
                              self.spec.height, self.spec.show_grid))
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
                                     show_grid=self.spec.show_grid)
        for k in self.records:
            if self.cancel_evt.is_set():
                raise _Cancelled()
            t, geom, vframes = self._read_record(k)
            self._emit(proc, fig.update(k, t, geom, vframes, self.k0, self.k1))
        return fig

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
                    (k, t, geom, vframes, self.k0, self.k1)))
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
        in the metadata block; the plots follow each record)."""
        st = render_mpl.RangeStats()
        for key in self.variants:
            st.E[key], st.uncert[key], st.purity[key] = [], [], []
            st.scale[key] = 0.0
        x1 = p1 = float("inf")
        x2 = p2 = float("-inf")
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
            x1, x2 = min(x1, geom.x1), max(x2, geom.x2)
            p1, p2 = min(p1, geom.p1), max(p2, geom.p2)
            for key, vf in zip(self.variants, self._order(vframes)):
                st.E[key].append(vf.E)
                st.uncert[key].append(vf.x_std*vf.p_std)
                st.purity[key].append(vf.purity)
                st.scale[key] = max(st.scale[key], vf.wmax, -vf.wmin)
                st.rho_max = max(st.rho_max, float(vf.rho.max()))
                st.phi_max = max(st.phi_max, float(vf.phi.max()))
        if geom0 is None:
            raise ValueError("no records in the requested range")
        for key in self.variants:
            if st.scale[key] <= 0.0:
                st.scale[key] = 1e-30
        st.x1, st.x2, st.p1, st.p2 = x1, x2, p1, p2
        return st, geom0

    def _spawn_ffmpeg(self):
        cfg = self.session.cfg
        enc = choose_encoder()
        comment = describe.config_json(
            cfg, self.session.param_log, at_record=self.k0,
            export={"records": [self.k0, self.k1], "stride": self.spec.stride,
                    "fps": self.spec.fps, "frames": self.total,
                    "variants": self.variants, "encoder": enc[1]})
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgba",
               "-video_size", "%dx%d" % (self.spec.width, self.spec.height),
               "-framerate", str(self.spec.fps), "-i", "pipe:0", "-an"] + enc + [
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-metadata", "title=wignerf W(x,p,t) records %d-%d"
               % (self.k0, self.k1),
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
    """ffprobe helper (tests/diagnostics): stream info of an exported file."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    out = subprocess.run([exe, "-v", "error", "-print_format", "json",
                          "-show_streams", path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)
