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
                     PREVIEW mode — 3.3-3.8x faster in 1D but only 1.5-2.6x in
                     2D (the multi-axis transform gains far less), and ~54-58%
                     of the working set on CUDA (no CPU speedup at all), at the
                     cost of the diagnostics: secular purity/energy drift ~1e-4
                     in 1D and ~1e-5 per 50 steps in 2D, and uncertainty noise
                     150x the relativistic shear.
WIGNERF_HISTORY_MB   in-RAM frame history cap per session (default 32 GiB:
                     ~4000 four-variant records at 1024², plenty at smaller
                     grids; set lower on RAM-constrained hosts like the VPS).
                     This is the CEILING as well as the default: a session may
                     ask for less, never for more.
WIGNERF_FFT_THREADS  threads per FFT; 0 = auto (ncores // (2*n_variants),
                     capped at 4; decided at session start)
WIGNERF_MAX_GRID     per-axis ceiling for 1D (ndim=1) sessions and for
                     auto-expand doublings (default 4096 — the schema
                     maximum; lower it on VRAM-constrained hosts: a
                     4096x4096 complex working set is ~1.3 GiB per variant
                     worker)
WIGNERF_MAX_GRID_2D  per-axis ceiling for 2D (ndim=2) sessions (default 128).
                     A rail only — see below for the operative one.
WIGNERF_MAX_CELLS_2D total-cell RAIL for 2D sessions (default 2**27 = 134M,
                     i.e. 22 GiB/worker — past any single card here). A rail,
                     NOT the operative guard: the real check is per-device,
                     asking the driver how much is actually free and comparing
                     it against the workers assigned to that device
                     (routers/sessions._fit_error). A fixed cell count cannot
                     do that job — it is wrong in both directions, refusing
                     128x128x64x64 (11.0 GiB, one worker) on a 24 GiB card
                     while permitting 5.5 GiB x 2 workers on an 11 GiB one.
                     This rail exists to stop absurd values (256^4 = 4.3e9
                     cells) cheaply and deterministically, and to be the only
                     guard on a host where free memory cannot be read. See
                     BYTES_PER_CELL_2D below for where the bytes go (the state
                     is 5% of them) — and note it is per PRECISION, so the same
                     cell count is 22.0 GiB/worker in float64 and 12.0 in
                     float32, which is why this rail is a rail.
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
MAX_GRID_2D = int(os.environ.get("WIGNERF_MAX_GRID_2D", "128"))
MAX_CELLS_2D = int(os.environ.get("WIGNERF_MAX_CELLS_2D", str(2**27)))

# Device bytes ONE variant worker holds per grid cell, per spectral precision.
# Measured on an RTX 3090 with `scripts/bench.py --ndim 2 --footprint`, which
# runs a whole worker record (worker._advance then _emit) rather than a step
# loop — the distinction matters, because a step loop misses adjust_step's
# transients and frame.build's reductions and so reports well under this.
# Reproduce it rather than trusting it.
#
#   float64   176.0 B/cell   0.17 / 0.87 / 2.75 / 6.71 GiB at 32^4 48^4 64^4 80^4
#   float32    96.0 B/cell   0.09 / 0.47 / 1.50 / 3.66 GiB   (54.55% of float64)
#
# Both are FLAT across sizes, and both are identical for the relativistic
# variants (measured 2026-07-27, `--relativistic`): a sqrt over meshes that
# already exist costs nothing.
#
# THE STATE IS 5% OF IT, which is the thing everyone gets wrong: W is REAL
# (solve_spectral returns B.real), so it is float64 = 8 B/cell — 0.12 GiB at
# 64^4 — and the other 168 B/cell is the machinery of the step, all at full
# shape. Pool high-water by stage, measured at float64:
#
#   Propagator rebuild (dU_im + dT_im at 16, plus the two Bopp
#     evaluations U(q -+ i*hbar*theta/2) at complex argument)      +80
#   W, the state                                                  + 0  (pooled)
#   the exponent slot = (expU, expT), 2x complex128               + 0  (pooled)
#   one Strang step: complex working arrays + cuFFT work area      +16
#   adjust_step: W1 and W2, plus a 2nd exponent pair for the halves +80
#   frame.build: 6 plane reductions + the int W^2 pass            + 0  (pooled)
#
# adjust_step's transient is now the largest single item. It used to be the
# EXPONENT SLOTS, of which there were two — the full step plus one for the
# straggler clamped onto tau_k — and M7 (2026-08-02) removed the second by
# making every committed substep inside a record the same size, measured 208 -> 176 and
# 112 -> 96, i.e. exactly the 32 B/cell that slot booked. (The M7 row predicted
# "-22%"; the truth is -15.4%, because that row long predated M1's measurement
# of the 208 it was a fraction OF.) Note the surviving slot books +0: the pool
# is already holding blocks of that size class from Propagator.rebuild, which
# is why removing its twin was worth 32 rather than 64.
#
# float32 (M1) halves the complex arrays and the state while dU_im/dT_im stay
# float64, which is why it lands at 55% rather than 50% — and slightly higher
# than the pre-M7 53.85%, since the slot that went was pure complex. Note the
# float32 saving is NOT cancelled by exponents()' cast: that builds the phase
# in complex128 and rounds down, so its transient peak is a mesh higher than
# float64's per call, but the pool high-water is still 96 — measured, which is
# the only reason this is known.
#
# Not a display detail: at 4 variants split 2+2 over a card pair this is what
# decides whether a session starts, so the create-time refusal quotes it and
# status() reports it for the Setup panel's footprint line.
BYTES_PER_CELL_2D = {"float64": 176, "float32": 96}


def max_grid(ndim):
    """Per-axis ceiling for this dimensionality."""
    return MAX_GRID if ndim == 1 else MAX_GRID_2D


def max_cells(ndim):
    """Total-cell ceiling. Unbounded at ndim=1 (MAX_GRID already bounds a 2D
    array to 4096^2 = 16.8M cells); the operative limit past that, because
    N^4 outruns any per-axis rail."""
    return None if ndim == 1 else MAX_CELLS_2D


def bytes_per_cell(ndim, precision="float64"):
    """Per-cell worker footprint, or None where nothing needs one.

    None at ndim=1 on purpose: MAX_GRID already bounds a 2D array to 4096^2, so
    no 1D session needs a memory estimate to decide whether it can start, and a
    number offered there would be one nobody had measured."""
    if ndim == 1:
        return None
    return BYTES_PER_CELL_2D.get(precision, BYTES_PER_CELL_2D["float64"])


EXPORT_DIR = os.environ.get(
    "WIGNERF_EXPORT_DIR",
    os.path.join(tempfile.gettempdir(), "wignerf-exports"))
EXPORT_ENCODER = os.environ.get("WIGNERF_EXPORT_ENCODER", "auto")
EXPORT_WORKERS = int(os.environ.get("WIGNERF_EXPORT_WORKERS", "0"))
