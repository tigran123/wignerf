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
exists because the throughput loop CANNOT: that loop holds one exponent pair and
calls solve_spectral, so it never allocates adjust_step's W1/W2 and second
exponent pair (+80 B/cell), the _exp_odd slot (+32) or frame.build's reductions
— i.e. it measures roughly half of what a real worker holds. This mode runs one
whole worker record instead, in worker._advance/_emit's own order, and prints
B/cell so the number in config.py can be reproduced rather than trusted.
`--relativistic` covers the quantum relativistic construction (the mc^2
cancellation inside qd(T, ...)), which the throughput path never builds either.
"""

import argparse
import sys
from math import prod
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import frame
from core.grid import Axis, Grid
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.xp import ArrayBackend, resolve_devices


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
        W = g.shift(mixture_wigner(g, [comp]))
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
    order that matters rather than the individual calls: the pool high-water is
    reached with the two exponent slots and W already resident, so a harness
    that measured adjust_step on its own would under-report exactly the overlap
    this is for. Both slots are held to the end for the same reason — a worker
    holds them across the whole record, and dropping either here would return
    32 B/cell to the pool before the frame build asks for its reductions.
    """
    b = _backend(device, precision)
    with b.device():
        if b.is_gpu:
            b.xp.get_default_memory_pool().free_all_blocks()
        U, gradU, comp = _problem(ndim)
        g = _grid(b, ndim, N)
        prop = Propagator(g, quantum=True, U=U, gradU=gradU,
                          relativistic=relativistic)
        W = g.shift(mixture_wigner(g, [comp]))

        # both exponent slots, as worker._exponents fills them: the full step
        # and the shorter one clamped onto tau_k
        dt = 0.01
        exp_main = prop.exponents(dt)
        exp_odd = prop.exponents(0.7*dt)
        for _ in range(10):                    # warm plans + the cuFFT work area
            W = prop.solve_spectral(W, *exp_main)
        W = prop.solve_spectral(W, *exp_odd)
        # every 20 steps, and the largest transient in the record
        W, _, eU, eT = prop.adjust_step(dt, W)
        # ...and the record itself: six plane reductions plus the int W^2 pass
        frame.build(W, g, prop.hbar_eff, prop=prop, dt=dt)
        b.synchronize()
        peak = (b.xp.get_default_memory_pool().total_bytes()
                if b.is_gpu else None)
        # keep the slots alive to here: see the docstring
        del exp_main, exp_odd, eU, eT
    per = None if peak is None else peak/float(prod(g.N))
    return b.name, peak, per


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
                    if args.footprint:
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
