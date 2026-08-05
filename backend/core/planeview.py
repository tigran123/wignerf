"""What slice of a plane to send: turn a client's viewport into a window.

The client says what PHYSICAL region a panel is showing and how many pixels it
has; this picks the pyramid level and the contiguous window that answers it.
Physical, not fractional, because that is what survives an auto-expand regrid
with no bookkeeping — the same region of phase space keeps meaning the same
thing when the domain doubles under it, which a fraction does not.

Everything is integer arithmetic against the RECORD's own geometry, which is
also what makes it safe to run per send: the record a scrub lands on may have
been computed on a different grid from the one the request was framed against.
"""

from dataclasses import dataclass

# What a panel may ask for per axis. The ceiling is a display bound, not a
# transport one: past ~1024 samples a panel is drawing more samples than it has
# pixels on any realistic screen, and the request is the one place to say so
# once. The floor keeps a tiny panel from asking for a level so coarse that the
# pyramid does not go there.
VIEW_N_MIN = 64
VIEW_N_MAX = 1024


@dataclass(frozen=True)
class Window:
    """One axis of a served plane: `n` samples of `step` base cells from `off`."""
    n: int
    off: int
    step: int


def full(N):
    """The whole axis at full resolution — what a client that asked for nothing
    gets, and byte-for-byte the pre-v5 plane."""
    return Window(int(N), 0, 1)


def _pow2_at_most(v):
    return 1 << max(0, int(v).bit_length() - 1)


def select(N, lo, hi, a1, a2, want, max_step):
    """Window for one axis of a plane.

    `N`/`lo`/`hi` are the RECORD's geometry for that axis, `a1..a2` the physical
    span the panel is showing, `want` its pixel count, `max_step` the coarsest
    decimation the pyramid can serve.

    `n` comes back EXACTLY as asked (clamped to the axis and to the pyramid, and
    to a power of two so the window is a whole number of coarse cells). That is
    deliberate: `texStorage2D` is immutable and the renderer keys its texture on
    the size, so a size that drifted with the zoom would reallocate the texture
    on every wheel notch.
    """
    N = int(N)
    d = (hi - lo)/N
    # The window in base cells, at least one cell wide however far in the zoom
    # has gone.
    cells = max(1, min(N, int(round((a2 - a1)/d))))
    n = _pow2_at_most(max(VIEW_N_MIN, min(VIEW_N_MAX, int(want))))
    n = min(n, N)
    # Decimate only as far as the samples actually asked for, and never further
    # than the pyramid goes. The residual (cells/n not a power of two) is under
    # 2x and the shader's bilinear absorbs it — the standard mip choice, and it
    # cannot alias because every level is an exact area mean.
    step = min(_pow2_at_most(max(1, cells//n)), max_step, N//n)
    step = max(1, step)
    # Snap the offset DOWN to a whole coarse cell: sample boundaries have to sit
    # on the level's own lattice or the "mean of base cells [off, off+step)"
    # contract is a lie and the crop is half a cell off.
    off = int((a1 - lo)//d)
    off -= off % step
    off = max(0, min(off, N - n*step))
    return Window(n, off, step)
