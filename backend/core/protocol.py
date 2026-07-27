"""
Binary frame-bundle format shared by the WebSocket stream and the
/api/preview/wigner response. Mirrored by frontend/src/lib/protocol.ts;
scripts/gen_fixture.py dumps a golden bundle consumed by the frontend
decoder's vitest, so the two implementations are cross-checked.

All little-endian; every section length is a multiple of 4 and the header
length a multiple of 8, so the JS decoder can create zero-copy TypedArray
views at any of them (Uint16Array needs 2-byte offsets, Float32Array 4).

A record carries a LIST OF 2D PLANES, not the state: W(x,y,px,py) can be
neither drawn nor sent, so a 2D session streams the six pairwise projections
(core.axes.PLANES) instead. At ndim=1 the list holds exactly one plane whose
complement is empty — i.e. W itself — so 1D is the general case, not a
special one.

Header (2*ndim axes; 64 bytes at ndim=1, 104 at ndim=2):
  u8  magic 0x57 ('W') | u8 version | u8 msg_type | u8 n_variants
  u32 record index | f64 t | u32 flags
  u8  ndim | u8 n_planes | u8 n_marginals | u8 reserved       (24 bytes)
  u32 N[a]                    for each of the 2*ndim axes
  f64 lo[a] | f64 hi[a]       for each of the 2*ndim axes

Grid geometry is a PER-RECORD fact (auto-expand may move/double the domain
mid-run), so every frame carries its own axis counts AND extents — a replayed
record must decode with the geometry it was computed on, never the
session's current one.

Per variant (contiguous, n_variants times):
  u8 variant id (bit0 quantum, bit1 relativistic) | u8 n_planes | u16 pad
  f32 dt | f32 E | f32 purity | f32 lz                        (20 bytes)
  f32 mean[a], f32 std[a]     for each of the 2*ndim axes
  per plane:
    u8 axis_a | u8 axis_b | u8 mode | u8 pad | f32 Wmin | f32 Wmax
    u16[N[a]*N[b]] quantized plane, row-major [ia][ib], fftshifted order
  per axis:
    f32[N[a]] marginal density                               (natural order)

The plane dimensions are implied by (axis_a, axis_b) and the header's N, so
they are not repeated. `mode` is the reduction kind — only projection (0) is
defined; cuts are milestone M5 and need no version bump. `lz` is <x*py - y*px>,
present always and 0.0 at ndim=1, because one fixed layout per ndim is easier
to keep right on both sides than a conditional field.

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
from . import axes as axes_mod
from .xp import C_AU

MAGIC = 0x57
VERSION = 4          # v4: per-record ndim, a list of 2D planes and 2*ndim
                     # marginals (1D is one plane = W); v3 was 2 fixed axes
MSG_FRAME = 1

FLAG_LIVE_PREVIEW = 1 << 0
FLAG_REPLAY = 1 << 1

_HDR = struct.Struct("<BBBBIdIBBBB")     # 24 bytes, then N[] then lo/hi[]
_VHDR = struct.Struct("<BBH4f")          # 20 bytes, then mean/std[]
_PHDR = struct.Struct("<BBBB2f")         # 12 bytes, then the u16 plane


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

class AxisSpec(BaseModel):
    # `le` is only a sanity rail (a 16384² uint16 frame is already 512 MiB);
    # the OPERATIVE ceilings are WIGNERF_MAX_GRID / WIGNERF_MAX_GRID_2D and
    # WIGNERF_MAX_CELLS_2D, enforced at session creation and for auto-expand
    # doublings
    lo: float
    hi: float
    N: int = Field(ge=4, le=16384)


class GridSpec(BaseModel):
    """The phase-space box, `2*ndim` axes in core.axes order (all spatial, then
    all momentum). The legacy {x1, x2, Nx, p1, p2, Np} shape is accepted and
    converted, so a stored browser config, an exported setup document and an
    mp4 comment tag from before 2D existed all still load."""
    ndim: Literal[1, 2] = 1
    axes: list[AxisSpec] = Field(min_length=2, max_length=4)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy(cls, v):
        if isinstance(v, dict) and "axes" not in v and "x1" in v:
            try:
                v = dict(ndim=1, axes=[
                    dict(lo=v["x1"], hi=v["x2"], N=v["Nx"]),
                    dict(lo=v["p1"], hi=v["p2"], N=v["Np"])])
            except KeyError as e:
                raise ValueError("a 1D grid needs x1, x2, Nx, p1, p2, Np "
                                 "(missing %s)" % e) from None
        return v

    @model_validator(mode="after")
    def _check(self):
        if len(self.axes) != 2*self.ndim:
            raise ValueError("a %dD grid needs %d axes (got %d)"
                             % (self.ndim, 2*self.ndim, len(self.axes)))
        names = axes_mod.labels(self.ndim)
        for i, a in enumerate(self.axes):
            if not a.hi > a.lo:
                raise ValueError("axis %s: require hi > lo" % names[i])
            if a.N % 2:
                raise ValueError("N%s must be even" % names[i])
        return self

    @property
    def d(self):
        return tuple((a.hi - a.lo)/a.N for a in self.axes)

    @property
    def cells(self):
        """Total grid cells — the operative memory scale, not the per-axis N."""
        n = 1
        for a in self.axes:
            n *= a.N
        return n

    def extended(self, hbar_eff):
        """Per-axis real range the quantum propagator evaluates U (spatial
        axes) and T (momentum axes) on: half-width hbar*theta_amp/2, set by
        the CONJUGATE axis's cell size."""
        d = self.d
        out = []
        for a in range(2*self.ndim):
            h = hbar_eff*(pi/d[axes_mod.conjugate(self.ndim, a)])/2.
            out.append((self.axes[a].lo - h, self.axes[a].hi + h))
        return tuple(out)

    def spatial_extended(self, hbar_eff):
        """Just the spatial half — what compile_potential takes."""
        return self.extended(hbar_eff)[:self.ndim]

    def spatial_ranges(self):
        return tuple((a.lo, a.hi) for a in self.axes[:self.ndim])

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("GridSpec.%s is a 1D-only spelling; this "
                                 "grid has ndim=%d" % (name, self.ndim))

    @property
    def x1(self):
        self._only1d("x1"); return self.axes[0].lo

    @property
    def x2(self):
        self._only1d("x2"); return self.axes[0].hi

    @property
    def Nx(self):
        self._only1d("Nx"); return self.axes[0].N

    @property
    def p1(self):
        self._only1d("p1"); return self.axes[1].lo

    @property
    def p2(self):
        self._only1d("p2"); return self.axes[1].hi

    @property
    def Np(self):
        self._only1d("Np"); return self.axes[1].N

    def theta_half_range(self, hbar_eff):
        """hbar*theta_amp/2: how far beyond [x1, x2] the quantum propagator
        evaluates U."""
        self._only1d("theta_half_range")
        return hbar_eff*(pi/self.d[1])/2.

    def x_extended(self, hbar_eff):
        self._only1d("x_extended")
        return self.extended(hbar_eff)[0]


def grid_limit_error(grid, n_variants=1, precision="float64"):
    """The host's ceilings for this grid, as a message — or None if it fits.

    SHARED by session creation and the IC preview, and that sharing is the
    point. The preview builds the FULL state at the session's grid, so a grid
    a session would refuse is one the preview must not try to allocate either.
    While it did not check, a form grid of 256⁴ (4.3e9 cells) — one dims switch
    away from the 1D default, with no Restart pressed — sent the preview to its
    CPU fallback, where it allocated 34 GiB arrays until the kernel OOM-killed
    the server. The create-time refusal the user would eventually have seen came
    far too late; nothing had ever bounded the preview.

    `precision` only scales the GiB figure the ndim=2 message quotes, never the
    cell rail itself — the rail is a cell count by design.

    BOTH callers must pass the session's precision, including the IC preview.
    It is tempting to let the preview keep the float64 default on the grounds
    that it builds on its own float64 backend — but the figure this message
    quotes is "per variant WORKER", i.e. the session the grid would create, not
    the preview's own allocation (that is PREVIEW_BYTES_PER_CELL, which really is
    precision-blind and stays so). Conflating the two put "~52.0 GiB per variant
    worker" under the IC plot while the Setup panel two columns away showed
    28.00 GiB for the same 128⁴ grid — the same quantity, 1.9x apart, both on
    screen at once.
    """
    nd = grid.ndim
    cap = config.max_grid(nd)
    names = axes_mod.labels(nd)
    over = [names[i] for i, a in enumerate(grid.axes) if a.N > cap]
    if over:
        return ("N%s may not exceed %s=%d on this host"
                % ("/N".join(over),
                   "WIGNERF_MAX_GRID" if nd == 1 else "WIGNERF_MAX_GRID_2D",
                   cap))
    cells = config.max_cells(nd)
    if cells is not None and grid.cells > cells:
        per = grid.cells*config.bytes_per_cell(nd, precision)/1024**3
        # N⁴ when every axis matches, the product otherwise. The user reads this
        # on every keystroke past the cap, so it states the numbers and the
        # remedy and nothing else — the reasoning behind the cap lives in
        # config.py and CLAUDE.md, not in a message shown a hundred times.
        ns = [a.N for a in grid.axes]
        shape = ("%d⁴" % ns[0] if len(set(ns)) == 1
                 else "×".join(str(n) for n in ns))
        total = ("" if n_variants == 1
                 else ", %.1f GiB for %d variants" % (per*n_variants, n_variants))
        return ("the grid has %s = %s cells, over WIGNERF_MAX_CELLS_2D=%s — "
                "~%.1f GiB per variant worker%s. Reduce an axis."
                % (shape, format(grid.cells, ","), format(cells, ","),
                   per, total))
    return None


class ICComponent(BaseModel):
    """One Gaussian packet: ndim values per geometry field. The legacy
    {x0, p0, sigma_x, sigma_p} shape is accepted and converted."""
    q0: list[float] = Field(min_length=1, max_length=2)
    k0: list[float] = Field(min_length=1, max_length=2)
    sigma_q: list[float] = Field(min_length=1, max_length=2)
    # ignored/derived per dimension for cat states
    sigma_k: Optional[list[float]] = Field(default=None, min_length=1,
                                           max_length=2)
    weight: float = Field(default=1.0, gt=0)
    phase: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy(cls, v):
        if isinstance(v, dict) and "q0" not in v and "x0" in v:
            sp_ = v.get("sigma_p")
            v = dict(q0=[v["x0"]], k0=[v["p0"]], sigma_q=[v["sigma_x"]],
                     sigma_k=None if sp_ is None else [sp_],
                     weight=v.get("weight", 1.0), phase=v.get("phase", 0.0))
        return v

    @model_validator(mode="after")
    def _check(self):
        n = len(self.q0)
        for name in ("k0", "sigma_q", "sigma_k"):
            f = getattr(self, name)
            if f is not None and len(f) != n:
                raise ValueError("component %s has %d values, q0 has %d"
                                 % (name, len(f), n))
        if any(s <= 0 for s in self.sigma_q):
            raise ValueError("sigma_q must be positive")
        if self.sigma_k is not None and any(s <= 0 for s in self.sigma_k):
            raise ValueError("sigma_k must be positive")
        return self

    @property
    def ndim(self):
        return len(self.q0)

    # -- 1D-only spellings -------------------------------------------------

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("ICComponent.%s is a 1D-only spelling; this "
                                 "component is %dD" % (name, self.ndim))

    @property
    def x0(self):
        self._only1d("x0"); return self.q0[0]

    @property
    def p0(self):
        self._only1d("p0"); return self.k0[0]

    @property
    def sigma_x(self):
        self._only1d("sigma_x"); return self.sigma_q[0]

    @property
    def sigma_p(self):
        self._only1d("sigma_p")
        return None if self.sigma_k is None else self.sigma_k[0]


class ICSpec(BaseModel):
    type: Literal["mixture", "cat"]
    components: list[ICComponent] = Field(min_length=1, max_length=8)


# adjust_step shrinks |dt| until one full step and two half steps agree to a
# RELATIVE tol. In complex64 that difference has a roundoff floor near 1e-6, so
# a smaller tol is unreachable: the controller would burn all 15 shrink attempts
# every 20 steps, drive dt to nothing, and hand back a run slower than the
# float64 one it was chosen to beat. Refuse the combination instead of silently
# stalling — the fix is either a larger tol or float64, and the user must pick.
#
# ONE constant covers every grid, and that is measured rather than assumed. M1
# re-measured the floor across both dimensionalities (2026-07-27, RTX 3090, the
# residual read straight off adjust_step's own expression at dt down to 0.0025,
# where float64 is still at 5e-9 and nowhere near its own floor):
#
#   1D   256^2  9.2e-7     1024^2  1.2e-6     4096^2  1.2e-6
#   2D    32^4  2.5e-6      48^4   2.0e-6      64^4   2.2e-6
#
# It SATURATES rather than growing with cell count — 1024^2 and 4096^2 agree
# across a 16x change, as do 32^4 and 64^4 — so the earlier "larger at larger
# grids" reading was only the 256^2 -> 1024^2 step. 2D sits ~2x above 1D because
# a 4-axis Strang step does more arithmetic per element. Worst case 2.5e-6, so
# 1e-5 keeps 4x margin everywhere and did not have to move for 2D.
TOL_MIN_F32 = 1e-5
MSG_TOL_F32 = ("tol = %g is below the float32 roundoff floor: the adaptive step "
               "controller compares a full step against two half steps, which "
               "in single precision agree only to ~1e-6 (measured 9e-7 in 1D and "
               "2.5e-6 in 2D, flat in the grid size), so it would shrink dt "
               "without ever converging. Use tol >= 1e-5, or precision float64.")

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


# 2D still ships without auto-expand and mp4 export — milestones M3 and M4 in
# CLAUDE.md, both wanted and both out only until the work each names is done.
# Each is refused explicitly here rather than half-working: a gate that says what
# it stands in for cannot be mistaken for settled scope, and cannot be relaxed by
# accident.
#
# TWO gates that were here are gone, and both left their verification behind
# rather than being waved through. M2 (relativistic qr/cr, 2026-07-27): the mc^2
# cancellation inside the 4D kinetic difference and the massless gradient at the
# lattice origin, measured and pinned in tests/test_propagator2d.py. M1 (float32,
# 2026-07-27): the mixed-precision rules re-verified BITWISE against the
# correlated 2D Bopp shift for all four variants, the adjust_step residual floor
# re-measured at 4D sizes (it saturates at ~2.5e-6, so TOL_MIN_F32 = 1e-5 keeps
# 4x margin and did not move), and the footprint measured at 112 B/cell against
# float64's 208 — see tests/test_precision.py and config.BYTES_PER_CELL_2D.
MSG_EXPAND_2D = (
    "auto-expand is not available for 2D runs yet (milestone M3): in 4D every "
    "axis doubling doubles a multi-GiB working set, so the planner needs a "
    "memory guard it does not have. Boundary DETECTION still runs on all four "
    "axes and will warn you; size the domain by hand.")
MSG_EXPORT_2D = (
    "mp4 export is not available for 2D runs yet (milestone M4): the export "
    "figure needs a plane-set panel grid, four marginals, ⟨Lz⟩ and the (2πℏ)² "
    "purity scale. The run's SETUP document (GET /sessions/{id}/setup) does "
    "work, and re-importing it reproduces the run.")


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
    # Spectral working precision. float32 is an explicit PREVIEW mode: 3.3-3.8x
    # faster in 1D but only 1.5-2.6x in 2D, and ~54-58% of the working set on
    # CUDA, at the cost of the diagnostics this project navigates by (secular
    # purity/energy drift ~1e-4, uncertainty noise 150x the relativistic shear).
    # Restart-only — it decides the FFT plan dtype at worker construction — and
    # never inferred, always chosen.
    #
    # None = "whatever the host is configured for" (WIGNERF_PRECISION), which
    # is the SPA's normal state until the user operates the control, and it
    # resolves the same way at every ndim now that M1 has landed. It is resolved
    # in _check rather than by a default_factory, which a hard-coded "float64"
    # here cannot be: that made the env var decorative once — `/api/device`
    # advertised float32 while every session that omitted the field silently ran
    # float64.
    precision: Optional[Literal["float64", "float32"]] = None
    # Per-session overrides of the host defaults. None = use the host's
    # (WIGNERF_DEVICE / WIGNERF_HISTORY_MB). A session may NARROW the host's
    # policy, never widen it: history_mb is clamped to the host ceiling, and an
    # unavailable device is a 422 (see routers/sessions.py).
    device: Optional[str] = Field(default=None, max_length=200)
    history_mb: Optional[int] = Field(default=None, ge=64)

    @property
    def ndim(self):
        return self.grid.ndim

    @model_validator(mode="after")
    def _check(self):
        # Resolve the host default FIRST, so every check below reads a concrete
        # value. One rule at every ndim since M1: the ndim=2 special case existed
        # only because float32 was refused there, and a gate must refuse what was
        # ASKED FOR rather than collide with a default the client never sent.
        # Assignment here does not re-enter validation (validate_assignment off).
        if self.precision is None:
            self.precision = config.PRECISION
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
        nd = self.grid.ndim
        bad = [c for c in self.ic.components if c.ndim != nd]
        if bad:
            raise ValueError("the grid is %dD but %d initial-condition "
                             "component(s) carry %d coordinate(s) each"
                             % (nd, len(bad), bad[0].ndim))
        if nd > 1 and self.auto_expand:
            raise ValueError(MSG_EXPAND_2D)
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
    # the SPA's light/dark theme (localStorage wignerf.theme): the video has
    # to read like the screen, so the Export panel defaults this to whatever
    # the app is showing and lets a job override it. "light" matches the SPA's
    # own default — the panel always sends the field explicitly, so this is
    # only what a direct API call gets, and it should get the same look a
    # first-time browser does.
    theme: Literal["dark", "light"] = "light"
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


class LoopCmd(BaseModel):
    """Repeat a playback-only run instead of stopping at the frontier.

    A DISPLAY policy, like `delay`, not a physics one: it changes only what the
    cursor does when it arrives, never what is computed. Live-settable so it can
    be armed mid-playback, and mid-run in time to catch the current pass.
    """
    type: Literal["loop"]
    on: bool


class SetParamsCmd(BaseModel):
    type: Literal["set_params"]
    params: ParamChange


class PingCmd(BaseModel):
    type: Literal["ping"]


ClientMsg = Annotated[
    Union[PlayCmd, PauseCmd, DelayCmd, SeekCmd, SetParamsCmd, PingCmd, LoopCmd],
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class RecordGeom:
    """Grid geometry of one record — travels in every frame header. Frozen
    tuples, so history.py's cross-variant geometry-equality invariant (all
    variants of a record share one grid) still works by value."""
    ndim: int
    N: tuple
    lo: tuple
    hi: tuple

    @classmethod
    def from_1d(cls, Nx, Np, x1, x2, p1, p2):
        return cls(1, (Nx, Np), (x1, p1), (x2, p2))

    def extent(self, a):
        return (self.lo[a], self.hi[a])

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("RecordGeom.%s is a 1D-only spelling; this "
                                 "record is %dD" % (name, self.ndim))

    @property
    def Nx(self):
        self._only1d("Nx"); return self.N[0]

    @property
    def Np(self):
        self._only1d("Np"); return self.N[1]

    @property
    def x1(self):
        self._only1d("x1"); return self.lo[0]

    @property
    def x2(self):
        self._only1d("x2"); return self.hi[0]

    @property
    def p1(self):
        self._only1d("p1"); return self.lo[1]

    @property
    def p2(self):
        self._only1d("p2"); return self.hi[1]


@dataclass
class PlaneFrame:
    """One quantized 2D reduction of a state: the (a, b) plane, in the surviving
    axes' fftshifted order, with its OWN range — the reductions of one state
    differ in scale by orders of magnitude, so one shared colour range would
    render most panels blank."""
    a: int
    b: int
    mode: int              # axes.MODE_PROJECTION (cuts are milestone M5)
    wq: numpy.ndarray      # uint16 (N[a], N[b])
    wmin: float
    wmax: float


@dataclass
class VariantFrame:
    vid: int
    dt: float
    E: float
    purity: float          # gamma = (2*pi*hbar_eff)^ndim * int W^2
    lz: float              # <x*py - y*px>; 0.0 at ndim=1
    mean: tuple            # per phase-space axis
    std: tuple
    planes: tuple          # PlaneFrame, in axes.PLANES order
    # float64 (N[a],) per axis, natural order — in BOTH precision modes: the
    # observable reductions accumulate and leave in double whatever the solver
    # works in (see core/observables.py), which is what keeps history.py's byte
    # accounting and the <f4 wire codec identical either way.
    marg: tuple

    @property
    def ndim(self):
        return len(self.marg)//2

    # -- 1D-only spellings (see grid.Grid._only1d) -------------------------

    def _only1d(self, name):
        if self.ndim != 1:
            raise AttributeError("VariantFrame.%s is a 1D-only spelling; this "
                                 "frame is %dD — use planes/marg/mean/std"
                                 % (name, self.ndim))

    @property
    def wq(self):
        self._only1d("wq"); return self.planes[0].wq

    @property
    def wmin(self):
        self._only1d("wmin"); return self.planes[0].wmin

    @property
    def wmax(self):
        self._only1d("wmax"); return self.planes[0].wmax

    @property
    def x_mean(self):
        self._only1d("x_mean"); return self.mean[0]

    @property
    def x_std(self):
        self._only1d("x_std"); return self.std[0]

    @property
    def p_mean(self):
        self._only1d("p_mean"); return self.mean[1]

    @property
    def p_std(self):
        self._only1d("p_std"); return self.std[1]

    @property
    def rho(self):
        self._only1d("rho"); return self.marg[0]

    @property
    def phi(self):
        self._only1d("phi"); return self.marg[1]


@dataclass
class DecodedFrame:
    record: int
    t: float
    geom: RecordGeom
    flags: int
    variants: list


def pack_frame(record, t, geom, variants, flags=0):
    nax = 2*geom.ndim
    nplanes = len(variants[0].planes) if variants else 0
    parts = [_HDR.pack(MAGIC, VERSION, MSG_FRAME, len(variants),
                       record, t, flags, geom.ndim, nplanes, nax, 0),
             numpy.asarray(geom.N, dtype="<u4").tobytes(),
             numpy.asarray([v for a in range(nax)
                            for v in (geom.lo[a], geom.hi[a])],
                           dtype="<f8").tobytes()]
    for v in variants:
        parts.append(_VHDR.pack(v.vid, len(v.planes), 0,
                                v.dt, v.E, v.purity, v.lz))
        parts.append(numpy.asarray(
            [x for a in range(nax) for x in (v.mean[a], v.std[a])],
            dtype="<f4").tobytes())
        for pl in v.planes:
            parts.append(_PHDR.pack(pl.a, pl.b, pl.mode, 0, pl.wmin, pl.wmax))
            parts.append(numpy.ascontiguousarray(pl.wq, dtype="<u2").tobytes())
        for m in v.marg:
            parts.append(numpy.ascontiguousarray(m, dtype="<f4").tobytes())
    return b"".join(parts)


def unpack_frame(buf):
    """Host-side decoder (tests, ws_smoke.py). Returns a DecodedFrame."""
    (magic, version, msg, nv, record, t, flags,
     ndim, nplanes, nmarg, _) = _HDR.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("bad magic 0x%02x" % magic)
    if version != VERSION:
        raise ValueError("protocol version %d != %d" % (version, VERSION))
    if msg != MSG_FRAME:
        raise ValueError("unexpected msg_type %d" % msg)
    nax = 2*ndim
    if nmarg != nax:
        raise ValueError("ndim %d wants %d marginals, header says %d"
                         % (ndim, nax, nmarg))
    off = _HDR.size
    N = tuple(int(n) for n in numpy.frombuffer(
        buf, dtype="<u4", count=nax, offset=off))
    off += 4*nax
    ext = numpy.frombuffer(buf, dtype="<f8", count=2*nax, offset=off)
    off += 8*2*nax
    geom = RecordGeom(ndim, N, tuple(ext[0::2]), tuple(ext[1::2]))

    variants = []
    for _ in range(nv):
        vid, npl, _pad, dt, E, purity, lz = _VHDR.unpack_from(buf, off)
        off += _VHDR.size
        ms = numpy.frombuffer(buf, dtype="<f4", count=2*nax, offset=off)
        off += 4*2*nax
        planes = []
        for _ in range(npl):
            a, b, mode, _p, wmin, wmax = _PHDR.unpack_from(buf, off)
            off += _PHDR.size
            n = N[a]*N[b]
            wq = numpy.frombuffer(buf, dtype="<u2", count=n,
                                  offset=off).reshape(N[a], N[b])
            off += 2*n
            planes.append(PlaneFrame(a, b, mode, wq, wmin, wmax))
        marg = []
        for a in range(nax):
            marg.append(numpy.frombuffer(buf, dtype="<f4", count=N[a],
                                         offset=off))
            off += 4*N[a]
        variants.append(VariantFrame(
            vid=vid, dt=dt, E=E, purity=purity, lz=lz,
            mean=tuple(float(x) for x in ms[0::2]),
            std=tuple(float(x) for x in ms[1::2]),
            planes=tuple(planes), marg=tuple(marg)))
    if off != len(buf):
        raise ValueError("trailing bytes: %d != %d" % (off, len(buf)))
    return DecodedFrame(record, t, geom, flags, variants)
