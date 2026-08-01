"""Per-device memory budgeting for 2D sessions — will this actually fit?

Two callers ask the same arithmetic a different question:

- `routers/sessions` asks whether a session can START. Nothing is allocated
  yet, so the CUDA context has to be paid for out of the budget.
- `core/session` asks, mid-run, whether the auto-expand planner can afford a
  DOUBLING. There the context is already paid, our own workers already hold the
  old footprint, and the switch peaks a little above the new one. A DOUBLING and
  nothing else: a plan that does not grow the window is refused by nothing here,
  and `regrid_shortfall` says why at the point it returns.

Only the arithmetic lives here. The MESSAGES deliberately do not: at create time
the remedy is "drop a variant, reduce an axis, change device", and mid-run none
of those is available to the user — the run is already going — so the two
callers write their own prose over one set of numbers.

WHY A DOUBLING IS NOT SIMPLY "THE NEW FOOTPRINT". worker._apply_regrid holds the
old state while embed_window builds the new one, and Propagator.rebuild()
allocates the new full-shape meshes. Measured on an RTX 3090 with
`scripts/bench.py --ndim 2 --regrid`, as a factor on the NEW footprint and flat
across 32^4 / 48^4 / 64^4:

    before release-before-allocate      1.269   (0.46 of the old came back)
    after                               1.038   (0.92) float64
                                        1.071   (0.86) float32

REGRID_PEAK is the rounded ceiling of the second row. The improvement is what
makes the `+ what we already hold` term below legitimate rather than hopeful:
Propagator.set_grid drops the old meshes and returns the blocks to the driver
BEFORE building the new ones, so the memory really is there to be reused.

WHAT THIS CANNOT SEE. `xp.device_free_bytes` asks the driver, so everything any
process has already ALLOCATED is in the answer — but three things are not, and
all three are accepted deliberately rather than modelled:

1. Another PROCESS, or this one's IC preview. A second backend, an unrelated CUDA
   job, or routers/preview.py's own private pool (which allocates on the roomiest
   card on every IC keystroke) can take memory between this reading and the
   allocation it licenses. Unmodellable in principle, which is what
   xp.device_free_bytes' own docstring says asking the system is FOR.
2. Another SESSION in this process whose growing RegridPlan is committed but has
   not landed. `k_star` is the frontier + 2 records, so the window is tens of ms
   at 32^4 and up to ~1 s at 64^4, and inside it two sessions on one card can
   each be told the same bytes are free. A registry walk could narrow that to the
   microseconds between the reading and the commit, but not close it, and it
   would buy that at the price of a claim that can outlive its plan: apply_params
   commits a plan on a PAUSED session (see session.py, and
   test_toggle_while_paused_schedules_immediately), whose frontier never reaches
   k_star, so the claim would refuse every OTHER session's expansion silently and
   for good. That is the shape of bug `_no_room_posted` already had once (see
   report_edge) with a wider blast radius and no all-clear to reset it.
3. The un-guarded transients elsewhere in a worker's life, which are LARGER than
   the one this module budgets: `set_physics` -> `rebuild()` runs qd()'s ~80
   B/cell Bopp construction with both exponent slots still resident (~168 B/cell
   in float64 at ndim=2), on any live parameter change, at any instant, at either
   dimensionality — and a 1D doubling is unguarded outright (per_worker returns
   None), including 4096^2, the same cell count as 64^4.

FIT_MARGIN does not cover (1) or (2) and should not be read as if it did: a tenth
of free memory against a sibling's whole doubling is an order of magnitude short.
What makes the exposure acceptable is the FAILURE MODE. Losing the race means a
cupy OOM inside worker._apply_regrid, which is fatal by design: the worker's
`run()` posts an error with the traceback and pauses the session, so one session
stops loudly and the others carry on. Every mechanism that would narrow it trades
that for a quiet refusal — and a wrong answer beats nothing, in this project's
order of preferences, exactly never. Both refusal and ACCEPT log the reading and
the ask (session._schedule_regrid), so an OOM at k_star is diagnosable from the
journal rather than being a surprise.
"""

from math import prod

import config

from . import xp

# One CUDA context + cuFFT plan cache per process per device, on top of the
# per-worker arrays (CLAUDE.md's GPU section measures it at ~300 MiB).
CONTEXT_BYTES = 300*1024**2
# Leave a tenth of the card. Free memory is a moving target — another process
# can claim some between this check and the first allocation — and unlike the
# IC preview a session has no CPU fallback to drop to.
FIT_MARGIN = 0.9
# Transient headroom a regrid needs over the new footprint. See the module
# docstring: measured 1.038 (float64) and 1.071 (float32); one rounded constant
# rather than a precision-keyed pair, because 3-6% sits far inside FIT_MARGIN
# and a second table to keep in step with a measurement is not worth that.
REGRID_PEAK = 1.10


def per_worker(cells, ndim, precision):
    """Device bytes one variant worker holds at this grid, or None at ndim=1
    (where WIGNERF_MAX_GRID already bounds a 2D array and no estimate is
    needed — see config.bytes_per_cell)."""
    b = config.bytes_per_cell(ndim, precision)
    return None if b is None else cells*b


def worker_counts(devices):
    """{device: how many of our workers sit on it} from an iterable of device
    specs (assign_devices' values at create time, worker.device mid-run)."""
    counts = {}
    for dev in devices:
        counts[dev] = counts.get(dev, 0) + 1
    return counts


def budgets(counts, context=True):
    """{device: (free, spendable)} for the devices we would put workers on.

    A device whose free memory cannot be read is DROPPED rather than guessed
    at: there the WIGNERF_MAX_CELLS_2D rail is the only guard, and guessing
    would be worse than not checking. `context=False` mid-run, where the CUDA
    context is already allocated and so already excluded from `free`.
    """
    out = {}
    for dev in counts:
        free = xp.device_free_bytes(dev)
        if free is None:
            continue
        ctx = 0 if (dev == "cpu" or not context) else CONTEXT_BYTES
        out[dev] = (free, free*FIT_MARGIN - ctx)
    return out


def roomiest(budget):
    """(device, free, spendable) for the device with the most room."""
    dev, (free, spend) = max(budget.items(), key=lambda kv: kv[1][1])
    return dev, free, spend


def regrid_shortfall(cells_old, cells_new, ndim, precision, counts,
                     budget=None):
    """Devices that cannot afford the new window: [(device, n, need, have)].

    What the device must hold for us afterwards is
    `n * per_new * REGRID_PEAK`; what is available is the driver's free memory
    PLUS the `n * per_old` our own workers are holding right now, because the
    regrid releases it before it allocates. Empty list = every assigned device
    has room, or none of them could be measured.

    `budget` is a `budgets(counts, context=False)` reading taken by the caller,
    so one regrid ATTEMPT asks the driver once and the greedy loop that walks
    back doublings compares every candidate against the same numbers. Omitted,
    it takes its own.
    """
    per_new = per_worker(cells_new, ndim, precision)
    per_old = per_worker(cells_old, ndim, precision)
    if per_new is None:
        return []                       # ndim=1: not this check's business
    # A WINDOW THAT DOES NOT GROW IS NOT THIS CHECK'S BUSINESS EITHER, and this
    # is not an optimization — the inequality below is simply wrong there. With
    # per_new == per_old it reduces to F >= (REGRID_PEAK/FIT_MARGIN - 1)*n*per_old
    # = 0.222*n*per_old, i.e. it refuses a whole-cell window SHIFT on a card that
    # is merely holding the session the shift belongs to (1.44 GiB free needed at
    # 64^4 float64 with 2 workers on it). The `+ n*per_old` term is wrong for a
    # move too: Propagator.set_grid takes its `else` branch on an unchanged shape
    # and releases nothing.
    #
    # AND A MOVE COSTS THE CARD NOTHING, measured rather than argued —
    # `scripts/bench.py --ndim 2 --regrid move` on a 3090, 32^4/48^4/64^4:
    # peak/steady 1.000 and driver +0 MiB in FLOAT64, which is the only precision
    # a move happens in (auto-expand is float64-only, MSG_EXPAND_F32). Its peak
    # stage is rebuild()'s ~80 B/cell Bopp construction, and every block that
    # needs is a size class the 208 B/cell arena is already holding from
    # adjust_step. So there is nothing here to guard, and a guard would be
    # incoherent besides: the same rebuild runs ~168 B/cell on any live parameter
    # change (set_physics, both exponent slots still resident), unguarded, at
    # either dimensionality. Refusing a move would only trade a REPORTED OOM for
    # mass wrapping through the seam — the failure auto-expand exists to prevent.
    # Returning early also keeps the driver query off the path entirely, which is
    # what lets the planner keep retrying a move every record the way the
    # max_grid cap always has.
    if per_new <= per_old:
        return []
    out = []
    if budget is None:
        budget = budgets(counts, context=False)
    for dev, (free, _spend) in budget.items():
        n = counts[dev]
        need = n*per_new*REGRID_PEAK
        have = (free + n*per_old)*FIT_MARGIN
        if need > have:
            out.append((dev, n, need, have))
    return sorted(out, key=lambda t: t[2] - t[3], reverse=True)


def gib(b):
    return b/1024.0**3


def cells_of(N):
    return prod(N)
