"""
Binary frame-bundle format shared by the WebSocket stream and the
/api/preview/wigner response. Mirrored by frontend/src/lib/protocol.ts;
scripts/gen_fixture.py dumps a golden bundle consumed by the frontend
decoder's vitest, so the two implementations are cross-checked.

All little-endian; sections are 4-byte aligned so the JS decoder can create
zero-copy TypedArray views.

Header (64 bytes):
  u8  magic 0x57 ('W') | u8 version | u8 msg_type | u8 n_variants
  u32 record index | f64 t | u32 Nx | u32 Np | u32 flags | u32 reserved
  f64 x1 | f64 x2 | f64 p1 | f64 p2

Grid geometry is a PER-RECORD fact (auto-expand may move/double the domain
mid-run), so every frame carries its own Nx/Np AND extents — a replayed
record must decode with the geometry it was computed on, never the
session's current one.

Per variant (contiguous, n_variants times):
  u8 variant id (bit0 quantum, bit1 relativistic), 3 pad bytes,
  f32 Wmin, Wmax, E, x_mean, x_std, p_mean, p_std, purity, dt  (40 bytes)
  u16[Nx*Np] W  quantized, row-major [ix][ip], fftshifted order
  f32[Nx] rho, f32[Np] phi                                     (natural order)

This module also holds the pydantic JSON schemas: grid/IC/session-config
(shared by the REST routers) and the client->server WebSocket control
messages. Server->client JSON (status/error/...) is constructed as plain
dicts in routers/stream.py.
"""

import struct
from dataclasses import dataclass
from math import pi
from typing import Annotated, Literal, Optional, Union

import numpy
from pydantic import BaseModel, Field, model_validator

import config
from .xp import C_AU

MAGIC = 0x57
VERSION = 3          # v3: per-record grid geometry (f64 x1,x2,p1,p2 in header)
MSG_FRAME = 1

FLAG_LIVE_PREVIEW = 1 << 0
FLAG_REPLAY = 1 << 1

_HDR = struct.Struct("<BBBBIdIIIIdddd")
_VHDR = struct.Struct("<B3x9f")


def variant_id(quantum, relativistic):
    return (1 if quantum else 0) | (2 if relativistic else 0)


# Variant keys as used by the four UI toggles (and the old bin/ scripts):
# q/c = quantum/classical, n/r = non-relativistic/relativistic.
VARIANTS = {
    "qn": dict(quantum=True, relativistic=False),
    "qr": dict(quantum=True, relativistic=True),
    "cn": dict(quantum=False, relativistic=False),
    "cr": dict(quantum=False, relativistic=True),
}


# ---------------------------------------------------------------------------
# Pydantic schemas (REST + WS control)
# ---------------------------------------------------------------------------

class GridSpec(BaseModel):
    # le is only a sanity rail (a 16384² uint16 frame is already 512 MiB);
    # the OPERATIVE per-axis ceiling is WIGNERF_MAX_GRID (default 4096),
    # enforced at session creation and for auto-expand doublings
    x1: float
    x2: float
    Nx: int = Field(ge=4, le=16384)
    p1: float
    p2: float
    Np: int = Field(ge=4, le=16384)

    @model_validator(mode="after")
    def _check(self):
        if not (self.x2 > self.x1 and self.p2 > self.p1):
            raise ValueError("require x2 > x1 and p2 > p1")
        if self.Nx % 2 or self.Np % 2:
            raise ValueError("Nx and Np must be even")
        return self

    def theta_half_range(self, hbar_eff):
        """hbar*theta_amp/2: how far beyond [x1, x2] the quantum propagator
        evaluates U."""
        dp = (self.p2 - self.p1)/self.Np
        return hbar_eff*(pi/dp)/2.

    def x_extended(self, hbar_eff):
        h = self.theta_half_range(hbar_eff)
        return (self.x1 - h, self.x2 + h)


class ICComponent(BaseModel):
    x0: float
    p0: float
    sigma_x: float = Field(gt=0)
    sigma_p: Optional[float] = Field(default=None, gt=0)  # ignored/derived for cat
    weight: float = Field(default=1.0, gt=0)
    phase: float = 0.0


class ICSpec(BaseModel):
    type: Literal["mixture", "cat"]
    components: list[ICComponent] = Field(min_length=1, max_length=8)


# adjust_step shrinks |dt| until one full step and two half steps agree to a
# RELATIVE tol. In complex64 that difference has a roundoff floor near 1e-7, so
# a smaller tol is unreachable: the controller would burn all 15 shrink attempts
# every 20 steps, drive dt to nothing, and hand back a run slower than the
# float64 one it was chosen to beat. Refuse the combination instead of silently
# stalling — the fix is either a larger tol or float64, and the user must pick.
TOL_MIN_F32 = 1e-5
MSG_TOL_F32 = ("tol = %g is below the float32 roundoff floor: the adaptive step "
               "controller compares a full step against two half steps, which "
               "in single precision agree only to ~7e-7 (measured at 256², and "
               "larger at larger grids), so it would shrink dt without ever "
               "converging. Use tol >= 1e-5, or precision float64.")

# Auto-expand needs measurements a float32 run cannot supply. Its 1e-6 edge
# trigger and its 1e-8 support scan both sit BELOW the spectral noise of a
# perfectly contained float32 state (measured 2026-07-25 at 256²: edge mass
# 1.6e-6 and a support scan covering the whole axis by step 600, against
# 1.5e-15 and an exact [43, 214) in float64 — see core/boundary.py). The
# detector's own warning survives on a raised threshold; the exact regrid does
# not, because it would size the new window from noise. Refuse rather than
# expand a domain for no reason.
MSG_EXPAND_F32 = (
    "auto-expand is not available in float32: it fires on edge-band mass at "
    "1e-6 and sizes the new domain from a 1e-8 support scan, and a contained "
    "float32 state's own spectral noise exceeds both within a few hundred "
    "steps, so the domain would double for no physical reason. Boundary "
    "DETECTION still runs (at a raised threshold) and will warn you. Use "
    "precision float64 to auto-expand, or size the domain by hand.")


class SessionCreate(BaseModel):
    grid: GridSpec
    potential: str
    ic: ICSpec
    variants: list[Literal["qn", "qr", "cn", "cr"]] = Field(min_length=1, max_length=4)
    mass: float = Field(default=1.0, ge=0)
    c: float = Field(default=C_AU, gt=0)
    hbar_eff: float = Field(default=1.0, gt=0)
    tol: float = Field(default=1e-2, gt=0, lt=1)
    t1: float = 0.0
    record_dt: float = Field(default=0.05, gt=0)
    mode: Literal["interactive", "batch"] = "interactive"
    t2: Optional[float] = None
    # boundary watch response policy: detection always runs; when True the
    # session auto-regrids (exact fixed-lattice move/double) at the frontier
    auto_expand: bool = False
    delay: float = Field(default=0.0, ge=0)  # seconds injected between played-back
                                             # frames; 0 = as fast as the client renders
    # Spectral working precision. float32 is an explicit PREVIEW mode: ~3.3-3.8x
    # faster and ~58% of the working set on CUDA, at the cost of the diagnostics
    # this project navigates by (secular purity/energy drift ~1e-4, uncertainty
    # noise 150x the relativistic shear). Restart-only — it decides the FFT plan
    # dtype at worker construction — and never inferred, always chosen.
    #
    # The default is the HOST's WIGNERF_PRECISION, not a literal: a hard-coded
    # "float64" here made that env var decorative — `/api/device` advertised it
    # while every session that omitted the field silently ran float64.
    # default_factory (not a module-level constant) so a test or an embedder
    # that patches config.PRECISION is honoured, and validate_default so a bad
    # value is caught here as well as in config._precision().
    precision: Literal["float64", "float32"] = Field(
        default_factory=lambda: config.PRECISION, validate_default=True)
    # Per-session overrides of the host defaults. None = use the host's
    # (WIGNERF_DEVICE / WIGNERF_HISTORY_MB). A session may NARROW the host's
    # policy, never widen it: history_mb is clamped to the host ceiling, and an
    # unavailable device is a 422 (see routers/sessions.py).
    device: Optional[str] = Field(default=None, max_length=200)
    history_mb: Optional[int] = Field(default=None, ge=64)

    @model_validator(mode="after")
    def _check(self):
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("duplicate variants")
        if self.mode == "batch" and self.t2 is None:
            raise ValueError("batch mode requires t2")
        # mass >= 0 exists for massless RELATIVISTIC runs (T = c|p|); the
        # non-relativistic T = p^2/2m diverges and would stream NaN frames
        if self.mass == 0.0 and any(not VARIANTS[v]["relativistic"]
                                    for v in self.variants):
            raise ValueError("mass = 0 requires exclusively relativistic "
                             "variants (non-relativistic T = p²/2m "
                             "diverges)")
        if self.precision == "float32" and self.tol < TOL_MIN_F32:
            raise ValueError(MSG_TOL_F32 % self.tol)
        if self.precision == "float32" and self.auto_expand:
            raise ValueError(MSG_EXPAND_F32)
        return self


class ExportSpec(BaseModel):
    """mp4 export of an already-computed record range (routers/export.py).
    k0/k1 default to the whole retained history; the range is clamped to it
    server-side. Even width/height: libx264 with yuv420p requires it."""
    k0: Optional[int] = Field(default=None, ge=0)
    k1: Optional[int] = Field(default=None, ge=0)
    stride: int = Field(default=1, ge=1, le=1000)
    fps: int = Field(default=30, ge=1, le=120)
    width: int = Field(default=1920, ge=320, le=3840)
    height: int = Field(default=1080, ge=240, le=2160)
    variants: Optional[list[Literal["qn", "qr", "cn", "cr"]]] = \
        Field(default=None, min_length=1, max_length=4)
    # mirrors the SPA's "grid lines on plots" toggle (localStorage
    # wignerf.grid, default on) — one setting for every plot in the frame,
    # W heatmaps included
    show_grid: bool = True
    # None = the host's WIGNERF_EXPORT_ENCODER. Per-JOB, not per-session: the
    # right choice depends on what else is running (nvenc frees cores for the
    # render pool, which is the actual bottleneck).
    encoder: Optional[Literal["auto", "cpu", "nvenc"]] = None

    @model_validator(mode="after")
    def _check(self):
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even (yuv420p)")
        if self.k0 is not None and self.k1 is not None and self.k1 < self.k0:
            raise ValueError("require k1 >= k0")
        if self.variants is not None and \
           len(set(self.variants)) != len(self.variants):
            raise ValueError("duplicate variants")
        return self


class ParamChange(BaseModel):
    U: Optional[str] = None
    c: Optional[float] = Field(default=None, gt=0)
    mass: Optional[float] = Field(default=None, ge=0)
    hbar_eff: Optional[float] = Field(default=None, gt=0)
    tol: Optional[float] = Field(default=None, gt=0, lt=1)
    dt_sign: Optional[Literal[1, -1]] = None
    auto_expand: Optional[bool] = None   # session-level policy, applies live


class PlayCmd(BaseModel):
    type: Literal["play"]


class PauseCmd(BaseModel):
    type: Literal["pause"]


class DelayCmd(BaseModel):
    type: Literal["delay"]
    seconds: float = Field(ge=0)


class SeekCmd(BaseModel):
    type: Literal["seek"]
    record: int = Field(ge=0)


class SetParamsCmd(BaseModel):
    type: Literal["set_params"]
    params: ParamChange


class PingCmd(BaseModel):
    type: Literal["ping"]


ClientMsg = Annotated[
    Union[PlayCmd, PauseCmd, DelayCmd, SeekCmd, SetParamsCmd, PingCmd],
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class RecordGeom:
    """Grid geometry of one record — travels in every frame header."""
    Nx: int
    Np: int
    x1: float
    x2: float
    p1: float
    p2: float


@dataclass
class VariantFrame:
    vid: int
    wq: numpy.ndarray      # uint16 (Nx, Np), fftshifted order
    wmin: float
    wmax: float
    E: float
    x_mean: float
    x_std: float
    p_mean: float
    p_std: float
    purity: float          # gamma = 2*pi*hbar_eff * int W^2 dx dp
    dt: float
    rho: numpy.ndarray     # float32/float64 (Nx,), natural order
    phi: numpy.ndarray     # float32/float64 (Np,), natural order


@dataclass
class DecodedFrame:
    record: int
    t: float
    geom: RecordGeom
    flags: int
    variants: list


def pack_frame(record, t, geom, variants, flags=0):
    parts = [_HDR.pack(MAGIC, VERSION, MSG_FRAME, len(variants),
                       record, t, geom.Nx, geom.Np, flags, 0,
                       geom.x1, geom.x2, geom.p1, geom.p2)]
    for v in variants:
        parts.append(_VHDR.pack(v.vid, v.wmin, v.wmax, v.E,
                                v.x_mean, v.x_std, v.p_mean, v.p_std,
                                v.purity, v.dt))
        parts.append(numpy.ascontiguousarray(v.wq, dtype="<u2").tobytes())
        parts.append(numpy.ascontiguousarray(v.rho, dtype="<f4").tobytes())
        parts.append(numpy.ascontiguousarray(v.phi, dtype="<f4").tobytes())
    return b"".join(parts)


def unpack_frame(buf):
    """Host-side decoder (tests, ws_smoke.py). Returns a DecodedFrame."""
    (magic, version, msg, nv, record, t, Nx, Np, flags, _,
     x1, x2, p1, p2) = _HDR.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("bad magic 0x%02x" % magic)
    if version != VERSION:
        raise ValueError("protocol version %d != %d" % (version, VERSION))
    if msg != MSG_FRAME:
        raise ValueError("unexpected msg_type %d" % msg)
    off = _HDR.size
    variants = []
    for _ in range(nv):
        vid, wmin, wmax, E, xm, xs, pm, ps, pur, dt = _VHDR.unpack_from(buf, off)
        off += _VHDR.size
        wq = numpy.frombuffer(buf, dtype="<u2", count=Nx*Np, offset=off).reshape(Nx, Np)
        off += 2*Nx*Np
        rho = numpy.frombuffer(buf, dtype="<f4", count=Nx, offset=off)
        off += 4*Nx
        phi = numpy.frombuffer(buf, dtype="<f4", count=Np, offset=off)
        off += 4*Np
        variants.append(VariantFrame(vid, wq, wmin, wmax, E, xm, xs, pm, ps,
                                     pur, dt, rho, phi))
    if off != len(buf):
        raise ValueError("trailing bytes: %d != %d" % (off, len(buf)))
    return DecodedFrame(record, t, RecordGeom(Nx, Np, x1, x2, p1, p2),
                        flags, variants)
