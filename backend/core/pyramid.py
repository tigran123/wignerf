"""Mip pyramid of a plane reduction: what makes display downsampling cheap.

A panel is ~600x400 device pixels. A 1D plane at 8192^2 is 128 MiB quantized,
which the browser's per-message receive path cannot absorb (see the
browser-receive-ceiling gotcha in CLAUDE.md) — so the server sends a CROP of
the zoom window, decimated to about the panel's pixel size, and full detail
comes back by zooming in.

The reduction has to be an AREA MEAN, and it has to happen HERE rather than at
send time, for two measured reasons:

- On the device an exact 2x2 mean costs less than the quantize already paid
  (4.1 ms for the whole pyramid at 8192^2 against 9.0 ms for one quantize).
  On the host the same reduction is 250 ms per plane, which is not a budget any
  streaming path has.
- Because every level is a power-of-two decimation of a power-of-two axis, a
  crop INSIDE a level is a plain contiguous slice — 0.07 ms — where a strided
  gather off the base array is 1.1 ms and grows with the base. The send path
  stays trivial precisely because the arithmetic was done once, per record.

An area mean is what W on a coarser grid genuinely looks like, which is the
point: it never aliases, and it is what the shader's bilinear already
approximates between samples. What it does lose is fringe contrast finer than
one output cell, so a reduced panel says so on screen and zooming recovers it.

Cost in bytes is bounded at +33.3% of the base plane whatever the depth — it is
a geometric series in 1/4 — so the floor is not there to control size. It is
there because a level below the smallest request a panel can make would never
be read. At ndim=2 that means one level (a 128^2 plane gains a 64^2), which is
8 KiB on a ~50 KiB record and is what lets a six-panel phase portrait drop to
the resolution its small panels actually have.
"""

# No panel can ask for fewer samples per axis than this, so a level below it
# would never be read. MUST NOT EXCEED planeview.VIEW_N_MIN, or `select` can
# choose a decimation the pyramid does not have; pinned by
# test_the_pyramid_reaches_every_level_select_can_ask_for.
PYRAMID_FLOOR = 64


def levels(plane, backend):
    """Successive exact 2x2 area means of `plane`, coarsest last.

    Level j is a decimation by 2^(j+1); the base plane is not included. Stops
    when either axis would fall below PYRAMID_FLOOR or become odd, so no level
    is ever a partial average — a half-covered edge cell would be a different
    quantity from its neighbours and would show as a bright rim.
    """
    xp = backend.xp
    out = []
    a = plane
    while True:
        n0, n1 = a.shape
        if n0 % 2 or n1 % 2 or n0 < 2*PYRAMID_FLOOR or n1 < 2*PYRAMID_FLOOR:
            return out
        # float64 accumulation regardless of the solver's working precision,
        # for the same reason observables reduce in float64: these are means
        # over up to 16.7M cells and the result is what gets quantized.
        a = a.reshape(n0//2, 2, n1//2, 2).mean(axis=(1, 3), dtype=xp.float64)
        out.append(a)


def level_for(step):
    """Index into `levels()` for a decimation of `step` (a power of two).

    -1 means the base plane. Kept here so the send path and the builder cannot
    disagree about what level 0 of the list means.
    """
    return step.bit_length() - 2
