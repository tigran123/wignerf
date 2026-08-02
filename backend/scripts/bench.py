"""
Propagator throughput benchmark: steps/second per backend, dimensionality,
grid size and spectral precision.

    .venv/bin/python scripts/bench.py [cpu] [cuda:1] ...
    .venv/bin/python scripts/bench.py --precision both cuda:1
    .venv/bin/python scripts/bench.py -N 1024,2048,4096 --precision both cuda:1
    .venv/bin/python scripts/bench.py --ndim 2 -N 32,48,64 cuda:1

No device arguments: benchmarks cpu and every detected CUDA device.
`--precision both` runs each grid at float64 and float32 and prints the
speedup — this is the measurement the float32 preview mode is sold on, so it
must be reproducible from the repo rather than quoted from a session log.

`--ndim 2` benchmarks 4D phase space, where -N is the size of EVERY axis (so
N=64 is 64^4 = 16.8M cells, four times a 2048^2 run) and the reported working
set is what decides whether a session fits at all: WIGNERF_MAX_CELLS_2D exists
because N^4 outruns any per-axis cap. On a CUDA device the peak pool usage is
measured, not estimated.

`--footprint` measures config.BYTES_PER_CELL_2D instead of throughput, and it
exists because the throughput loop CANNOT: that loop calls solve_spectral and
nothing else, so it never allocates adjust_step's W1/W2 and second exponent pair
(+80 B/cell) or frame.build's reductions — i.e. it measures well under what a
real worker holds. This mode runs one whole worker record instead, in
worker._advance/_emit's own order, and prints
B/cell so the number in config.py can be reproduced rather than trusted.
`--relativistic` covers the quantum relativistic construction (the mc^2
cancellation inside qd(T, ...)), which the throughput path never builds either.
It applies to every mode, `--regrid` included.

`--regrid [double|move|both]` measures an auto-expand switch rather than a
steady state: what it peaks at over the footprint it lands on, and — for a
doubling — how much of the old footprint the driver gets back. Those are the two
constants core/fit.py budgets a 2D regrid with. A `move` is the plan the guard
deliberately never refuses, so what it measures instead is whether the switch is
served out of the pool without the card seeing it at all.
"""

import argparse
import sys
from math import ceil, prod
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import frame
from core.grid import Axis, Grid, GridState, embed_window
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.xp import ArrayBackend, device_free_bytes, resolve_devices


def _problem(ndim):
    """(U, gradU, IC component) for an isotropic harmonic well at any ndim."""
    if ndim == 1:
        return (lambda x: x**2/2., (lambda x: x,),
                GaussianComponent(2.0, 0.0, .707, .707))
    return (lambda x, y: (x**2 + y**2)/2.,
            (lambda x, y: x + 0.*y, lambda x, y: y + 0.*x),
            GaussianComponent((2.0, 0.0), (0.0, 0.0), (.707, .707),
                              (.707, .707)))


def _grid(b, ndim, N):
    if ndim == 1:
        return Grid.from_1d(-6.0, 6.0, N, -7.0, 7.0, N, b)
    return Grid((Axis(-6.0, 6.0, N), Axis(-6.0, 6.0, N),
                 Axis(-7.0, 7.0, N), Axis(-7.0, 7.0, N)), b)


def _backend(device, precision):
    return ArrayBackend(device=device, fft_threads=4 if device == "cpu" else 1,
                        precision=precision)


def bench(device, N, ndim=1, precision="float64", nsteps=200,
          relativistic=False):
    """Returns (backend name, seconds per step, peak device bytes or None)."""
    b = _backend(device, precision)
    with b.device():
        if b.is_gpu:
            b.xp.get_default_memory_pool().free_all_blocks()
        U, gradU, comp = _problem(ndim)
        g = _grid(b, ndim, N)
        prop = Propagator(g, quantum=True, U=U, gradU=gradU,
                          relativistic=relativistic)
        W = g.shift(mixture_wigner(g, [comp])).astype(b.real_dtype, copy=False)
        expU, expT = prop.exponents(0.01)
        for _ in range(10):                      # warm up plans/pool
            W = prop.solve_spectral(W, expU, expT)
        b.synchronize()
        t0 = perf_counter()
        for _ in range(nsteps):
            W = prop.solve_spectral(W, expU, expT)
        b.synchronize()
        dt = (perf_counter() - t0)/nsteps
        # the arena, not the live set: this is what a card has to hold
        peak = (b.xp.get_default_memory_pool().total_bytes()
                if b.is_gpu else None)
    return b.name, dt, peak


def footprint(device, N, ndim=1, precision="float64", relativistic=False):
    """Device bytes ONE variant worker holds, measured over a whole record.

    Returns (backend name, peak bytes or None, bytes per cell or None).

    The order below is worker._advance followed by worker._emit, and it is the
    order that matters rather than the individual calls: a pending boundary
    probe overlaps its temporary pairs with the cached slot from the preceding
    record, then drops both before the new record's one production pair is
    built. A harness that only measures either phase would under-report the
    arena a worker actually needs.
    """
    b = _backend(device, precision)
    with b.device():
        if b.is_gpu:
            b.xp.get_default_memory_pool().free_all_blocks()
        U, gradU, comp = _problem(ndim)
        g = _grid(b, ndim, N)
        prop = Propagator(g, quantum=True, U=U, gradU=gradU,
                          relativistic=relativistic)
        # Pass the IC inline: a caller-local W would pin that old state for
        # the entire helper call and overstate the worker arena by one state.
        W, slots = _record(
            prop, g,
            g.shift(mixture_wigner(g, [comp])).astype(b.real_dtype, copy=False))
        b.synchronize()
        peak = (b.xp.get_default_memory_pool().total_bytes()
                if b.is_gpu else None)
        # keep the current production slot alive to here: see the docstring
        del slots
    per = None if peak is None else peak/float(prod(g.N))
    return b.name, peak, per


def _record(prop, g, W, dt=0.01, record_dt=0.05):
    """One whole worker record, in worker._advance/_emit's order — the same
    sequence footprint() measures, factored out so the regrid mode can hold a
    worker's real steady state on both sides of the switch.

    Model a periodic adjustment at a boundary: the old production slot remains
    live while adjust_step probes a larger dt, but its returned state and pairs
    are dropped. The whole new record is then advanced at one quotient of the
    selected cap, retaining only that production slot."""
    old_exp = prop.exponents(dt)
    for _ in range(11):                    # warm plans + the cuFFT work area
        W = prop.solve_spectral(W, *old_exp)
    trial, dt, eU, eT = prop.adjust_step(dt/0.7, W)
    del trial, eU, eT, old_exp
    n = ceil(record_dt/abs(dt))
    dts = (record_dt/n) if dt > 0 else -(record_dt/n)
    exp = prop.exponents(dts)
    for _ in range(n):
        W = prop.solve_spectral(W, *exp)
    frame.build(W, g, prop.hbar_eff, prop=prop, dt=dt)
    return W, (exp,)                       # the slot stays alive: footprint()


def regrid(device, N, ndim=2, precision="float64", axis=0,
           relativistic=False, kind="double"):
    """Device bytes ACROSS an auto-expand regrid: what the switch peaks at,
    against what the new grid needs, and how much of the old comes back.

    routers/sessions._fit_error asks whether a session can START. The M3 regrid
    guard has to ask whether it can DOUBLE, and the new footprint alone is not
    the answer: worker._apply_regrid holds the old state and the old propagator
    while embed_window and Propagator.rebuild() allocate the new ones. So the
    guard needs two measured numbers, and this is where they come from:

      peak/new     the transient factor (1 + epsilon) a doubling must reserve
      recovered    the fraction of the OLD footprint the driver gets back,
                   which is what says whether the budget may count it as
                   available (the `r` term)

    `kind="move"` measures the OTHER plan, a whole-cell window shift at
    unchanged N, which the guard deliberately never refuses (core/fit.py says
    why). There per_old == per_new, so the same row reads as a pure transient
    factor: what a move costs over the steady state it starts and ends at. It is
    a different code path in Propagator.set_grid — no new FFT plans, no pool
    release — and the only way to know what it really costs is to measure it.

    The order below mirrors worker._apply_regrid; the release-before-allocate
    work it measures lives in Propagator.set_grid, which is real shared code
    called here rather than reimplemented.
    """
    b = _backend(device, precision)
    if not b.is_gpu:
        return b.name, None
    xp = b.xp
    with b.device():
        pool = xp.get_default_memory_pool()
        pool.free_all_blocks()
        U, gradU, comp = _problem(ndim)
        nax = 2*ndim
        g0 = _grid(b, ndim, N)
        old = GridState(anchor=tuple(g0.lo), d=tuple(g0.d),
                        offset=(0,)*nax, N=tuple(g0.N))
        new = (old.moved(axis, -(N//2), 2*N)         # double, support centred
               if kind == "double" else
               old.moved(axis, N//4, N))             # slide by a quarter axis

        # (1) a worker's steady state on the OLD grid
        g = old.make_grid(b)
        prop = Propagator(g, quantum=True, U=U, gradU=gradU,
                          relativistic=relativistic)
        # The IC is built in float64 (initial.py is precision-independent) and
        # is dropped at the first solve_spectral, here and in footprint() alike
        # — whether it is passed inline or bound to a local, the name is rebound
        # by the first step either way. Which is what makes the two harnesses
        # comparable, and they are: this mode prints B/cell for exactly that
        # check, and both report 176.0 / 96.0 at 32^4 and 64^4 on the 3090,
        # i.e. config.BYTES_PER_CELL_2D.
        W, slots = _record(
            prop, g,
            g.shift(mixture_wigner(g, [comp])).astype(b.real_dtype, copy=False))
        b.synchronize()
        old_arena = pool.total_bytes()
        free_before = device_free_bytes(device)

        # (2) the switch itself, in worker._apply_regrid's order: slots first,
        #     then the state dropped as it is transformed, then the propagator
        #     (which releases its own old meshes inside set_grid)
        del slots                                    # _exp_clear()
        Wnat = prop.grid.unshift(W)
        W = None
        Wnew = embed_window(Wnat, old, new, xp)
        del Wnat
        g2 = new.make_grid(b)
        prop.set_grid(g2)
        W = g2.shift(Wnew)
        del Wnew
        # (3) and a whole record on the NEW grid, so the peak covers what the
        #     worker needs AFTERWARDS, not just the switch
        W, slots = _record(prop, g2, W)
        b.synchronize()
        peak = pool.total_bytes()
        # what the DRIVER sees with the new state live and the pool untouched —
        # the pool holds freed blocks, so this is ~0 recovery unless the regrid
        # path returned them, which is exactly what the restructure adds
        free_after = device_free_bytes(device)

        # (4) from a clean pool, what the new grid needs ON ITS OWN
        del W, slots, prop, g, g2
        pool.free_all_blocks()
        g3 = new.make_grid(b)
        prop3 = Propagator(g3, quantum=True, U=U, gradU=gradU,
                           relativistic=relativistic)
        W3, slots3 = _record(
            prop3, g3,
            g3.shift(mixture_wigner(g3, [comp])).astype(b.real_dtype, copy=False))
        b.synchronize()
        new_arena = pool.total_bytes()
        del W3, slots3, prop3, g3
        pool.free_all_blocks()

    per_old, per_new = float(old_arena), float(new_arena)
    # r: the fraction of the OLD footprint the driver actually gets back.
    #   used_after = (total - other) - free_after,  and free_before pins
    #   (total - other) = free_before + per_old.
    recovered, net = None, None
    if free_before is not None and free_after is not None:
        used_after = free_before + per_old - free_after
        recovered = 1.0 - (used_after - per_new)/per_old
        # what the driver holds beyond the steady state we started from. The
        # number that matters for a MOVE, where per_new == per_old and the claim
        # under test is that the switch is served out of the pool: 0 means the
        # card never saw the transient at all.
        net = used_after - per_old
    return b.name, dict(
        kind=kind, cells_old=prod(old.N), cells_new=prod(new.N),
        per_old=per_old, per_new=per_new, peak=float(peak),
        eps=peak/per_new - 1.0, recovered=recovered, net=net,
        b_old=per_old/prod(old.N), b_new=per_new/prod(new.N))


def _label(ndim, N):
    return "x".join([str(N)]*(2*ndim))


def _line(name, ndim, N, precision, dt, peak):
    mem = "" if peak is None else "  %7.2f GiB" % (peak/1024**3)
    print("%-18s %-19s %-8s %9.1f steps/s  (%9.3f ms/step)%s"
          % (name, _label(ndim, N), precision, 1./dt, 1e3*dt, mem))


def _fline(name, ndim, N, precision, peak, per):
    if peak is None:
        print("%-18s %-19s %-8s   (host memory: not measured)"
              % (name, _label(ndim, N), precision))
        return
    print("%-18s %-19s %-8s %7.1f B/cell  %7.2f GiB"
          % (name, _label(ndim, N), precision, per, peak/1024**3))


def _rline(name, ndim, N, precision, r):
    if r is None:
        print("%-18s %-19s %-8s   (regrid: CUDA only)"
              % (name, _label(ndim, N), precision))
        return
    if r["kind"] == "move":
        # per_old == per_new here, so "recovered" has nothing to be a fraction
        # OF; what is being measured is whether the card ever sees the switch.
        net = "  n/a" if r["net"] is None else "%+.0f MiB" % (r["net"]/1024**2)
        print("%-18s %-19s %-8s  move   steady %6.2f  peak %6.2f GiB   "
              "peak/steady %5.3f   driver %s   (%.0f B/cell)"
              % (name, _label(ndim, N), precision, r["per_new"]/1024**3,
                 r["peak"]/1024**3, r["peak"]/r["per_new"], net, r["b_new"]))
        return
    rec = "  n/a" if r["recovered"] is None else "%5.2f" % r["recovered"]
    print("%-18s %-19s %-8s  old %6.2f  new %6.2f  peak %6.2f GiB   "
          "peak/new %5.3f   recovered %s   (%.0f/%.0f B/cell)"
          % (name, _label(ndim, N), precision, r["per_old"]/1024**3,
             r["per_new"]/1024**3, r["peak"]/1024**3, r["peak"]/r["per_new"],
             rec, r["b_old"], r["b_new"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("devices", nargs="*", metavar="DEVICE",
                    help="cpu, cuda:N (default: cpu + every CUDA device)")
    ap.add_argument("--precision", default="float64",
                    choices=["float64", "float32", "both"])
    ap.add_argument("--ndim", type=int, default=1, choices=[1, 2],
                    help="spatial dimensions (2 = 4D phase space)")
    ap.add_argument("-N", "--grids", default=None,
                    help="comma-separated per-axis sizes "
                         "(default 256,512,1024 at ndim=1, 32,48,64 at 2)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--footprint", action="store_true",
                    help="measure what one worker HOLDS (B/cell) over a whole "
                         "record, instead of throughput — this is the "
                         "config.BYTES_PER_CELL_2D measurement")
    ap.add_argument("--relativistic", action="store_true",
                    help="build the relativistic kinetic exponent (the mc^2 "
                         "cancellation the default path never exercises)")
    ap.add_argument("--regrid", nargs="?", const="double", default=None,
                    choices=["double", "move", "both"],
                    help="measure an auto-expand regrid. 'double' (the "
                         "default): the transient peak as a factor on the new "
                         "footprint and how much of the old one the driver gets "
                         "back — the two numbers the M3 regrid memory guard "
                         "budgets with. 'move': the whole-cell window shift the "
                         "guard never refuses, where the question is instead "
                         "whether the switch is served out of the pool")
    args = ap.parse_args()

    devices = args.devices or ["cpu"] + [d for d in resolve_devices("auto")
                                         if d != "cpu"]
    precisions = (["float64", "float32"] if args.precision == "both"
                  else [args.precision])
    default_grids = "256,512,1024" if args.ndim == 1 else "32,48,64"
    grids = [int(n) for n in (args.grids or default_grids).split(",")
             if n.strip()]

    for dev in devices:
        for N in grids:
            got = {}
            try:
                for p in precisions:
                    if args.regrid:
                        kinds = (["double", "move"] if args.regrid == "both"
                                 else [args.regrid])
                        for kind in kinds:
                            name, r = regrid(dev, N, args.ndim, p,
                                             relativistic=args.relativistic,
                                             kind=kind)
                            _rline(name, args.ndim, N, p, r)
                    elif args.footprint:
                        name, peak, per = footprint(dev, N, args.ndim, p,
                                                    args.relativistic)
                        _fline(name, args.ndim, N, p, peak, per)
                        if per is not None:
                            got[p] = per
                    else:
                        name, dt, peak = bench(dev, N, args.ndim, p,
                                               args.steps, args.relativistic)
                        got[p] = dt
                        _line(name, args.ndim, N, p, dt, peak)
            except Exception as e:
                print("%-18s %-19s unavailable: %s"
                      % (dev, _label(args.ndim, N), e))
                break
            if len(got) == 2:
                # footprint: SMALLER is better, so report the fraction kept
                # rather than a "faster" factor that would read backwards
                print("%-18s %-19s %-8s %9.2f%s"
                      % ("", _label(args.ndim, N),
                         "of f64" if args.footprint else "speedup",
                         (got["float32"]/got["float64"]*100 if args.footprint
                          else got["float64"]/got["float32"]),
                         "% of float64" if args.footprint else "x faster"))


if __name__ == "__main__":
    main()
