"""
Frame quantization: real W -> uint16 + (Wmin, Wmax). Runs on the device
(the cast happens before the PCIe download on GPU backends). Dtype-agnostic:
65535 levels need 16 bits, which float32 carries exactly, so a float32
session quantizes to the same frame a float64 one would.
"""


def quantize(W, backend):
    wmin = float(W.min())
    wmax = float(W.max())
    return requantize(W, backend, wmin, wmax), wmin, wmax


def requantize(W, backend, wmin, wmax):
    """Quantize against a range taken from something ELSE.

    The pyramid's coarser levels use the FULL plane's (wmin, wmax): the server
    switches level when a panel resizes or the zoom crosses a power of two, and
    a range that moved with it would repaint the colorbar under a user who only
    scrolled. A level's own extrema are also strictly inside the full plane's
    (averaging cannot exceed the values it averages), so nothing clips.
    """
    xp = backend.xp
    span = wmax - wmin
    if span < 1e-300:
        span = 1.0
    q = xp.clip(xp.rint((W - wmin)*(65535.0/span)), 0, 65535).astype(xp.uint16)
    return backend.asnumpy(q)


def dequantize(q, wmin, wmax):
    """Host-side inverse (tests/ws_smoke)."""
    return wmin + q.astype("float64")*((wmax - wmin)/65535.0)
