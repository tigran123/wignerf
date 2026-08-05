"""Display downsampling: the pyramid and the window arithmetic.

These are the pieces that decide WHICH samples of a plane go on the wire. They
are unit-tested apart from the stream because the failures they have are silent
ones — a window off by half a coarse cell, or a level the pyramid does not
have — which a "the payload got smaller" assertion happily accepts.
"""

import numpy as np
import pytest

from core import planeview as pv
from core import pyramid
from core.xp import ArrayBackend


@pytest.fixture(scope="module")
def cpu():
    return ArrayBackend("cpu")


def test_the_pyramid_reaches_every_level_select_can_ask_for(cpu):
    """PYRAMID_FLOOR must not exceed VIEW_N_MIN.

    `select` clamps its sample count to VIEW_N_MIN and then picks a decimation
    for it; if the pyramid stopped above that, the level simply would not exist
    and `PlaneFrame.level` would IndexError inside a send — a crash on a zoom,
    which no payload-size assertion can see. Stated as the property rather than
    as the two numbers so it survives either being retuned.
    """
    assert pyramid.PYRAMID_FLOOR <= pv.VIEW_N_MIN
    for N in (256, 512, 1024, 2048):
        plane = np.zeros((N, N))
        mx = 1 << len(pyramid.levels(plane, cpu))
        # the coarsest thing any request can produce, at any zoom
        w = pv.select(N, -1.0, 1.0, -1.0, 1.0, pv.VIEW_N_MIN, mx)
        assert w.step <= mx, (N, w, mx)
        assert w.n*w.step <= N


def test_a_level_is_the_exact_area_mean(cpu):
    """Not a subsample. A subsample of a fringed state aliases — it turns
    unresolved interference into a moire pattern that looks like structure —
    while an area mean is exactly what W on a coarser grid looks like."""
    rng = np.random.default_rng(0)
    plane = rng.standard_normal((256, 256))
    lv = pyramid.levels(plane, cpu)
    assert [a.shape for a in lv] == [(128, 128), (64, 64)]
    ref2 = plane.reshape(128, 2, 128, 2).mean(axis=(1, 3))
    ref4 = plane.reshape(64, 4, 64, 4).mean(axis=(1, 3))
    assert np.allclose(lv[0], ref2, rtol=0, atol=1e-15)
    assert np.allclose(lv[1], ref4, rtol=0, atol=1e-15)


def test_no_level_is_ever_a_partial_average(cpu):
    """An ODD axis stops the pyramid rather than averaging a half-covered edge
    cell, which would be a different quantity from its neighbours and would
    show as a bright or dark rim along one side.

    Not every axis here is a power of two — the API only requires even — so the
    rule is "stop when a halving would not be exact", not "stop at a size".
    """
    # 65 is odd: nothing at all
    assert pyramid.levels(np.zeros((65, 256)), cpu) == []
    # 130 halves exactly ONCE (to 65), and 65 stops it
    assert [a.shape for a in pyramid.levels(np.zeros((130, 256)), cpu)] \
        == [(65, 128)]
    # and the floor binds the other way
    assert [a.shape for a in pyramid.levels(np.zeros((256, 128)), cpu)] \
        == [(128, 64)]


def test_the_window_covers_what_was_asked_for():
    """The served window must CONTAIN the requested region.

    An off/step that snapped the wrong way still shrinks the payload perfectly
    and still decodes, so the only thing that catches it is checking the
    physical span — the panel would just be showing somewhere slightly else.
    """
    N, lo, hi = 1024, -6.0, 6.0
    d = (hi - lo)/N
    for a1, a2 in [(-6.0, 6.0), (0.0, 3.0), (-0.05, 0.05), (5.9, 6.0),
                   (-6.0, -5.9), (-1.0, 1.0)]:
        w = pv.select(N, lo, hi, a1, a2, 256, 8)
        left = lo + w.off*d
        right = left + w.n*w.step*d
        assert left <= a1 + 1e-12 and right >= a2 - 1e-12, (a1, a2, w)
        assert 0 <= w.off and w.off + w.n*w.step <= N, w
        assert w.off % w.step == 0, "window not aligned to its own level"


def test_the_sample_count_is_exactly_what_was_asked_for():
    """`n` may not drift with the zoom. texStorage2D is immutable and the
    renderer keys its texture on the size, so a count that changed per wheel
    notch would reallocate the texture on every frame of a zoom."""
    N = 2048
    sizes = {pv.select(N, -6.0, 6.0, -3.0*f, 3.0*f, 256, 8).n
             for f in (1.0, 0.9, 0.5, 0.31, 0.12, 0.03)}
    assert sizes == {256}
    # ...and it is clamped to the axis rather than over-reading it
    assert pv.select(128, -6.0, 6.0, -6.0, 6.0, 1024, 4).n == 128


def test_a_request_never_asks_for_a_level_that_does_not_exist():
    """max_step is a hard cap: a plane with no pyramid serves step 1, whatever
    the panel asks for. That is what keeps 2D (and any small grid) correct."""
    w = pv.select(1024, -6.0, 6.0, -6.0, 6.0, 64, 1)
    assert w.step == 1 and w.n == 64
    # it then covers only 64 of the 1024 cells, which is honest: the server
    # cannot invent a level, and the alternative is refusing to answer at all
    assert w.off == 0
