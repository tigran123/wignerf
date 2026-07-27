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
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def bench(device, N, ndim=1, precision="float64", nsteps=200):
    """Returns (backend name, seconds per step, peak device bytes or None)."""
    b = ArrayBackend(device=device, fft_threads=4 if device == "cpu" else 1,
                     precision=precision)
    with b.device():
        if b.is_gpu:
            b.xp.get_default_memory_pool().free_all_blocks()
        U, gradU, comp = _problem(ndim)
        g = _grid(b, ndim, N)
        prop = Propagator(g, quantum=True, U=U, gradU=gradU)
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


def _label(ndim, N):
    return "x".join([str(N)]*(2*ndim))


def _line(name, ndim, N, precision, dt, peak):
    mem = "" if peak is None else "  %7.2f GiB" % (peak/1024**3)
    print("%-18s %-19s %-8s %9.1f steps/s  (%9.3f ms/step)%s"
          % (name, _label(ndim, N), precision, 1./dt, 1e3*dt, mem))


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
            times = {}
            try:
                for p in precisions:
                    name, dt, peak = bench(dev, N, args.ndim, p, args.steps)
                    times[p] = dt
                    _line(name, args.ndim, N, p, dt, peak)
            except Exception as e:
                print("%-18s %-19s unavailable: %s"
                      % (dev, _label(args.ndim, N), e))
                break
            if len(times) == 2:
                print("%-18s %-19s %-8s %9.2fx faster"
                      % ("", _label(args.ndim, N), "speedup",
                         times["float64"]/times["float32"]))


if __name__ == "__main__":
    main()
