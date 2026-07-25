"""
Environment-driven configuration (same convention as urantia-library:
per-machine values come from the environment, code holds the defaults).

WIGNERF_DEVICE       auto | cpu | cuda:N | comma list ("cuda:1,cuda:0").
                     Names a device POOL: sessions spread variant workers
                     across it, costliest variants to the fastest device.
                     auto = all CUDA devices fastest-first, else cpu; an
                     explicit list is trusted as written (order = speed).
WIGNERF_PORT         backend port; 8010 because urantia-library owns 8000
                     on the dev machine
WIGNERF_PRECISION    default spectral working precision: float64 | float32.
                     Sessions may choose per run (SessionCreate.precision);
                     this only sets what the form starts from. float32 is a
                     PREVIEW mode — ~3.3-3.7x faster and half the working set
                     on CUDA (no CPU speedup at all), at the cost of the
                     diagnostics: secular purity/energy drift ~1e-4 and
                     uncertainty noise 150x the relativistic shear.
WIGNERF_HISTORY_MB   in-RAM frame history cap per session (default 32 GiB:
                     ~4000 four-variant records at 1024², plenty at smaller
                     grids; set lower on RAM-constrained hosts like the VPS).
                     This is the CEILING as well as the default: a session may
                     ask for less, never for more.
WIGNERF_FFT_THREADS  threads per FFT; 0 = auto (ncores // (2*n_variants),
                     capped at 4; decided at session start)
WIGNERF_MAX_GRID     per-axis Nx/Np ceiling for auto-expand doublings
                     (default 4096 — the schema maximum; lower it on
                     VRAM-constrained hosts: a 4096x4096 complex working
                     set is ~1.3 GiB per variant worker)
WIGNERF_EXPORT_DIR   where mp4 exports are written before being downloaded
                     (default <tempdir>/wignerf-exports; files are deleted
                     after the download TTL, on session close and at exit)
WIGNERF_EXPORT_ENCODER
                     mp4 video encoder: auto | cpu | nvenc. auto = the GPU
                     h264_nvenc if a probe succeeds (dedicated encoder block,
                     ~3x faster at 4K), else libx264. cpu forces libx264,
                     nvenc forces the GPU encoder. The bottleneck is frame
                     RENDERING, not encoding — this only tops up the parallel
                     render pool (and is the right GPU path: h264_nvenc, NOT
                     ffmpeg -hwaccel, which is for decoding).
WIGNERF_EXPORT_WORKERS
                     export frame-render processes; 0 = auto
                     (min(cpu_count, 8)). Rendering a frame (matplotlib/Agg)
                     dominates export time, so it is spread over a spawn
                     ProcessPoolExecutor while one ffmpeg encodes the ordered
                     stream. One export at a time uses all of these.
"""

import logging
import os
import tempfile

log = logging.getLogger(__name__)

PRECISIONS = ("float64", "float32")


def _precision():
    """WIGNERF_PRECISION, validated. An unrecognized value falls back to
    float64 and says so: falling back to the SAFE setting is never dangerous,
    whereas letting a typo through would advertise a precision to every client
    and (before this was validated) hand `flaot32` straight to SessionCreate's
    default. A deliberate host-wide float32 is logged too — making every
    session on a host preview-grade by default is precisely the thing this
    project requires to be explicit, so it belongs in the journal."""
    v = os.environ.get("WIGNERF_PRECISION", "float64")
    if v not in PRECISIONS:
        log.warning("WIGNERF_PRECISION=%r is not one of %s — using float64",
                    v, "/".join(PRECISIONS))
        return "float64"
    if v == "float32":
        log.warning("WIGNERF_PRECISION=float32: sessions default to SINGLE "
                    "precision (a preview mode — purity/energy drift ~1e-4). "
                    "Results from this host are not physics-grade unless the "
                    "session asks for float64.")
    return v


DEVICE = os.environ.get("WIGNERF_DEVICE", "auto")
PORT = int(os.environ.get("WIGNERF_PORT", "8010"))
PRECISION = _precision()
HISTORY_MB = int(os.environ.get("WIGNERF_HISTORY_MB", "32768"))
FFT_THREADS = int(os.environ.get("WIGNERF_FFT_THREADS", "0"))
MAX_GRID = int(os.environ.get("WIGNERF_MAX_GRID", "4096"))
EXPORT_DIR = os.environ.get(
    "WIGNERF_EXPORT_DIR",
    os.path.join(tempfile.gettempdir(), "wignerf-exports"))
EXPORT_ENCODER = os.environ.get("WIGNERF_EXPORT_ENCODER", "auto")
EXPORT_WORKERS = int(os.environ.get("WIGNERF_EXPORT_WORKERS", "0"))
