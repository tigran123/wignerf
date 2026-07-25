"""
Propagator throughput benchmark: steps/second per backend, grid size and
spectral precision.

    .venv/bin/python scripts/bench.py [cpu] [cuda:1] ...
    .venv/bin/python scripts/bench.py --precision both cuda:1
    .venv/bin/python scripts/bench.py -N 1024,2048,4096 --precision both cuda:1

No device arguments: benchmarks cpu and every detected CUDA device.
`--precision both` runs each grid at float64 and float32 and prints the
speedup — this is the measurement the float32 preview mode is sold on, so it
must be reproducible from the repo rather than quoted from a session log.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.grid import Grid
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.xp import ArrayBackend, resolve_devices

HARMONIC = dict(U=lambda x: x**2/2., dUdx=lambda x: x)


def bench(device, N, precision="float64", nsteps=200):
    """Returns (backend name, seconds per step)."""
    b = ArrayBackend(device=device, fft_threads=4 if device == "cpu" else 1,
                     precision=precision)
    with b.device():
        g = Grid(-6.0, 6.0, N, -7.0, 7.0, N, b)
        prop = Propagator(g, quantum=True, **HARMONIC)
        W = g.shift2d(mixture_wigner(g, [GaussianComponent(2, 0, .707, .707)]))
        expU, expT = prop.exponents(0.01)
        for _ in range(10):                      # warm up plans/pool
            W = prop.solve_spectral(W, expU, expT)
        b.synchronize()
        t0 = perf_counter()
        for _ in range(nsteps):
            W = prop.solve_spectral(W, expU, expT)
        b.synchronize()
        dt = (perf_counter() - t0)/nsteps
    return b.name, dt


def _line(name, N, precision, dt):
    print("%-18s %5dx%-5d %-8s %9.1f steps/s  (%8.3f ms/step)"
          % (name, N, N, precision, 1./dt, 1e3*dt))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("devices", nargs="*", metavar="DEVICE",
                    help="cpu, cuda:N (default: cpu + every CUDA device)")
    ap.add_argument("--precision", default="float64",
                    choices=["float64", "float32", "both"])
    ap.add_argument("-N", "--grids", default="256,512,1024",
                    help="comma-separated per-axis sizes")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    devices = args.devices or ["cpu"] + [d for d in resolve_devices("auto")
                                         if d != "cpu"]
    precisions = (["float64", "float32"] if args.precision == "both"
                  else [args.precision])
    grids = [int(n) for n in args.grids.split(",") if n.strip()]

    for dev in devices:
        for N in grids:
            times = {}
            try:
                for p in precisions:
                    name, dt = bench(dev, N, p, args.steps)
                    times[p] = dt
                    _line(name, N, p, dt)
            except Exception as e:
                print("%-18s %5dx%-5d unavailable: %s" % (dev, N, N, e))
                break
            if len(times) == 2:
                print("%-18s %5dx%-5d %-8s %9.2fx faster"
                      % ("", N, N, "speedup", times["float64"]/times["float32"]))


if __name__ == "__main__":
    main()
