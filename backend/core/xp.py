"""
Array/FFT backend selection: CuPy on a CUDA device or NumPy on the CPU.

Each SolverWorker owns its own ArrayBackend instance, so FFT plans and
buffers are never shared between threads (pyFFTW plan execution is not
re-entrant; CuPy device context is per-thread).

Device selection (config.py reads WIGNERF_DEVICE). The spec names a POOL:
resolve_devices() expands it to an ordered list of concrete devices,
fastest first, and each session spreads its variant workers over that list
(core/session.py assign_devices). ArrayBackend itself always binds ONE
concrete device.
  "auto"          - all CUDA devices fastest-first if cupy imports, else CPU.
  "cpu"           - NumPy; FFT provider chain: pyfftw -> scipy.fft -> numpy.fft.
  "cuda:N"        - CuPy on CUDA device N.
  "cuda:1,cuda:0" - explicit pool; the written order IS the speed ranking.

A backend also carries the SPECTRAL working precision ("float64" default,
"float32" for the opt-in preview mode). It governs complex_dtype and the FFT
plan dtype and nothing else: every array the propagator's exponents are BUILT
from stays float64 either way.
"""

from contextlib import nullcontext
import logging
import os

import numpy

# CUDA's default enumeration is fastest-first, which disagrees with
# nvidia-smi's PCI order (and would make "cuda:1" mean different cards in
# different tools). Pin PCI order BEFORE cupy is first imported so device
# indices always match nvidia-smi.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

log = logging.getLogger(__name__)

# Speed of light in Hartree atomic units (1/alpha, CODATA 2018).
C_AU = 137.035999084


def _import_cupy():
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception as e:
        log.info("cupy unavailable: %s", e)
    return None


def _gpu_speed_key(cupy, i):
    """Crude compute ranking: SM count first (82 on the 3090 vs 68 on the
    2080 Ti), total memory as the tiebreak. clockRate is deprecated (reads
    0 under CUDA 13) so it cannot participate."""
    props = cupy.cuda.runtime.getDeviceProperties(i)
    return (props["multiProcessorCount"], props["totalGlobalMem"])


def resolve_devices(spec):
    """Expand a WIGNERF_DEVICE spec into an ordered list of CANONICAL device
    strings ("cuda:N" / "cpu"), fastest first. An explicit comma list is
    trusted as written — its order is the speed ranking.

    Canonical means a bare "cuda" comes back as "cuda:0". It used to be
    returned verbatim, which worked only because ArrayBackend defaults the
    index the same way — and would silently fail a set-membership test against
    a pool of "cuda:N" strings (see devices_allowed)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty device spec %r" % spec)
    if parts == ["auto"]:
        cupy = _import_cupy()
        if cupy is None:
            return ["cpu"]
        n = cupy.cuda.runtime.getDeviceCount()
        order = sorted(range(n), key=lambda i: _gpu_speed_key(cupy, i),
                       reverse=True)
        return ["cuda:%d" % i for i in order]
    if len(set(parts)) != len(parts):
        raise ValueError("duplicate devices in %r" % spec)
    out = []
    for p in parts:
        if p == "cpu":
            out.append(p)
            continue
        if p == "auto":
            raise ValueError("'auto' cannot appear in a device list")
        if not p.startswith("cuda"):
            raise ValueError("unknown device spec %r" % p)
        cupy = _import_cupy()
        if cupy is None:
            raise RuntimeError("%r requested but cupy/CUDA is not available" % p)
        idx = int(p.split(":")[1]) if ":" in p else 0
        if idx >= cupy.cuda.runtime.getDeviceCount():
            raise RuntimeError("CUDA device %d does not exist" % idx)
        out.append("cuda:%d" % idx)
    if len(set(out)) != len(out):
        # "cuda,cuda:0" survives the check above but names one device twice
        raise ValueError("duplicate devices in %r" % spec)
    return out


def devices_allowed(pool_spec):
    """The device specs a session may ASK for on this host: the pool, plus cpu.

    cpu is always a legal target — a float64 sanity run, or keeping a session
    off a card you need for something else — but `resolve_devices("auto")`
    returns GPUs only on a CUDA host, so it would never appear in a list built
    from the pool alone. routers/meta.py builds /api/device's `choices` from
    this and routers/sessions.py validates against it, so what the Setup panel
    offers and what the API accepts cannot drift apart."""
    pool = resolve_devices(pool_spec)
    return pool if "cpu" in pool else pool + ["cpu"]


def device_free_bytes(spec):
    """Memory currently available on a device spec, or None when it cannot be
    determined (no cupy, an unreadable /proc, a vanished card).

    Asking the system rather than tracking our own sessions is deliberate, and
    it is the same question routers/preview.py's _pick_device asks: whatever
    else is on the card — another session, another process entirely — is
    already reflected in the answer. On the CPU that means MemAvailable, the
    kernel's own estimate of what can be allocated without swapping, not
    MemFree.
    """
    if spec == "cpu":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1])*1024
        except OSError:
            pass
        return None
    try:
        cupy = _import_cupy()
        if cupy is None:
            return None
        with cupy.cuda.Device(int(spec.split(":")[1]) if ":" in spec else 0) as d:
            return int(d.mem_info[0])
    except Exception:
        return None


def device_total_bytes(spec):
    """Total memory installed on a device spec, or None when it cannot be
    determined. The sibling of device_free_bytes, and deliberately a different
    question: FREE is what routers/sessions._fit_error asks, because it decides
    whether a session can start RIGHT NOW; TOTAL is a static host fact, which is
    what lets the Setup panel warn about a grid that could never fit on any
    hardware here without re-polling a number that moves under it.
    """
    if spec == "cpu":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1])*1024
        except OSError:
            pass
        return None
    try:
        cupy = _import_cupy()
        if cupy is None:
            return None
        with cupy.cuda.Device(int(spec.split(":")[1]) if ":" in spec else 0) as d:
            return int(d.mem_info[1])
    except Exception:
        return None


PRECISIONS = ("float64", "float32")


class ArrayBackend:
    def __init__(self, device="auto", fft_threads=1, precision="float64"):
        self.fft_threads = max(1, int(fft_threads))
        if precision not in PRECISIONS:
            raise ValueError("unknown precision %r (expected one of %s)"
                             % (precision, ", ".join(PRECISIONS)))
        self.precision = precision
        # The SPECTRAL working dtype only. Everything that CONSTRUCTS the
        # propagator's exponents (the grid meshes, the Bopp/qd evaluation,
        # the dU_im/dT_im rate meshes, H) stays float64 in both modes — see
        # the Propagator docstring for why that is not negotiable and why it
        # is also free.
        self.complex_dtype = (numpy.complex64 if precision == "float32"
                              else numpy.complex128)
        self.real_dtype = (numpy.float32 if precision == "float32"
                           else numpy.float64)
        self.xp = numpy
        self.is_gpu = False
        self.device_index = None
        self.fft_provider = None

        if device == "auto":
            device = resolve_devices("auto")[0]   # fastest device in the pool
        if device == "cpu":
            self._use_cpu()
        elif device.startswith("cuda"):
            cupy = _import_cupy()
            if cupy is None:
                raise RuntimeError("WIGNERF_DEVICE=%s but cupy/CUDA is not available" % device)
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx >= cupy.cuda.runtime.getDeviceCount():
                raise RuntimeError("CUDA device %d does not exist" % idx)
            self._use_gpu(cupy, idx)
        else:
            raise ValueError("unknown device spec %r" % device)

    def _use_gpu(self, cupy, index):
        self.xp = cupy
        self.is_gpu = True
        self.device_index = index
        self.fft_provider = "cupy"
        with cupy.cuda.Device(index):
            name = cupy.cuda.runtime.getDeviceProperties(index)["name"].decode()
        self.name = "cuda:%d (%s)" % (index, name)
        log.info("backend: %s", self.name)

    def _use_cpu(self):
        # Provider chain in the spirit of dynamics/solve.py: pyfftw is the
        # fastest, numpy the slowest.
        try:
            import pyfftw  # noqa: F401
            self.fft_provider = "pyfftw"
        except ImportError:
            try:
                import scipy.fft  # noqa: F401
                self.fft_provider = "scipy"
            except ImportError:
                self.fft_provider = "numpy"
        self.name = "cpu (%s)" % self.fft_provider
        log.info("backend: %s", self.name)

    # -- device/context helpers ------------------------------------------

    def device(self):
        """Context manager binding this backend's CUDA device to the current
        thread (no-op on CPU). Workers wrap their whole run loop in it."""
        if self.is_gpu:
            return self.xp.cuda.Device(self.device_index)
        return nullcontext()

    def synchronize(self):
        if self.is_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def asnumpy(self, a):
        return self.xp.asnumpy(a) if self.is_gpu else a

    def fftshift(self, a, axes=None):
        return self.xp.fft.fftshift(a, axes=axes)

    def ifftshift(self, a, axes=None):
        return self.xp.fft.ifftshift(a, axes=axes)

    # -- FFT plan factory --------------------------------------------------

    def fft_pair(self, shape, axes, dtype=None):
        """Return (fft, ifft) callables for complex arrays of the given shape
        along the given axes (a tuple), in this backend's complex_dtype unless
        `dtype` overrides it. pyFFTW plans/buffers are owned by this backend
        instance and must not be called from two threads at once.

        A SINGLE axis takes the one-dimensional entry points, not the n-D ones
        with a length-1 `axes`. That is not tidiness: no provider promises
        fftn(a, axes=(0,)) is bit-identical to fft(a, axis=0), and the 1D
        physics suite's 1e-12 bounds should not have to absorb the difference.
        ndim=1 sessions therefore transform through exactly the code they
        always did.

        The dtype is NOT cosmetic. A pyFFTW builder is planned for one dtype:
        hand a complex64 array to a complex128 plan and auto_align_input
        silently copies it up, returning a correct complex128 result with the
        whole point of single precision quietly gone. The other three providers
        dispatch on the INPUT dtype and hold no plan to get wrong: cupy, scipy
        and — since numpy 2.0 added native single-precision transforms —
        numpy.fft all return complex64 for complex64 (verified against the
        pinned numpy 2.5.1), so they ignore the argument by construction."""
        dtype = numpy.dtype(dtype or self.complex_dtype)
        axes = tuple(axes)
        one = len(axes) == 1
        axis = axes[0] if one else None
        if self.fft_provider == "cupy":
            xp = self.xp
            if one:
                return (lambda a: xp.fft.fft(a, axis=axis),
                        lambda a: xp.fft.ifft(a, axis=axis))
            return (lambda a: xp.fft.fftn(a, axes=axes),
                    lambda a: xp.fft.ifftn(a, axes=axes))
        if self.fft_provider == "pyfftw":
            import pyfftw
            kw = dict(threads=self.fft_threads,
                      planner_effort="FFTW_ESTIMATE",
                      overwrite_input=False, auto_align_input=True,
                      auto_contiguous=True, avoid_copy=False)
            a = pyfftw.empty_aligned(shape, dtype=dtype)
            b = pyfftw.empty_aligned(shape, dtype=dtype)
            if one:
                return (pyfftw.builders.fft(a, axis=axis, **kw),
                        pyfftw.builders.ifft(b, axis=axis, **kw))
            return (pyfftw.builders.fftn(a, axes=axes, **kw),
                    pyfftw.builders.ifftn(b, axes=axes, **kw))
        if self.fft_provider == "scipy":
            import scipy.fft as sfft
            workers = self.fft_threads
            if one:
                return (lambda a: sfft.fft(a, axis=axis, workers=workers),
                        lambda a: sfft.ifft(a, axis=axis, workers=workers))
            return (lambda a: sfft.fftn(a, axes=axes, workers=workers),
                    lambda a: sfft.ifftn(a, axes=axes, workers=workers))
        if one:
            return (lambda a: numpy.fft.fft(a, axis=axis),
                    lambda a: numpy.fft.ifft(a, axis=axis))
        return (lambda a: numpy.fft.fftn(a, axes=axes),
                lambda a: numpy.fft.ifftn(a, axes=axes))
