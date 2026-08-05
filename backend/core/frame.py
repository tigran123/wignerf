"""
Record construction: one propagator state -> the quantized 2D planes, the 1D
marginals and the scalars that make up a wire record.

This is the module that makes 4D phase space displayable and streamable at all.
W(x,y,px,py) can be neither drawn nor sent, so the state is reduced ON THE
DEVICE to the canonical plane set (core.axes.PLANES) and never leaves the GPU:
a 64^4 record drops from 33 MiB per variant to ~50 KiB, which is why the
browser-receive ceiling and WIGNERF_HISTORY_MB both stop being 2D constraints.
At ndim=1 the single plane's complement is empty, so the plane IS W and this
produces exactly the record it always did, byte for byte.

Shared by worker._emit and routers/preview.py so a session frame and an IC
preview frame are built by one code path — the preview is the same bundle the
WebSocket streams, decoded by the same frontend code.

Every plane keeps its own quantization range, hence its own colorbar. That is
not laziness: the reductions of one state differ in scale by orders of
magnitude (a spatial density against a signed reduced Wigner function), and one
shared range would render most panels blank.

Planes leave here in NATURAL (unshifted) order, as the marginals always have.
See the ifftshift below for why that is a precondition for cropping.
"""

from . import axes as ax
from . import observables
from . import pyramid
from .protocol import PlaneFrame, VariantFrame
from .quantize import quantize, requantize


def build(W, grid, hbar_eff=1.0, prop=None, dt=0.0, vid=0):
    """(VariantFrame, Observables) for the state W (propagator/shifted order).

    `prop` supplies the energy meshes and rest energy; without it E = 0, which
    is what the IC preview wants (it has no potential attached). The plane
    reductions happen ONCE and feed both the frame and the observables.
    """
    planes = observables.reduce_planes(W, grid)
    obs = (observables.compute(W, prop, planes) if prop is not None
           else observables.compute_basic(W, grid, hbar_eff, planes))
    pf = []
    b = grid.backend
    for plane in ax.planes(grid.ndim):
        # NATURAL ORDER LEAVES THIS FUNCTION. The propagator works fftshifted
        # and so do the reductions, but a CROP of a shifted array straddles the
        # seam — the low and high halves are at opposite ends — so cropping
        # would need either a third convention or two client paths. Unshifting
        # once here is measured free on the device (it rides the quantize's
        # copy) and it is what lets the shader drop its half-period offset and
        # its modulo wrap. The marginals were already natural order, so after
        # this NOTHING outside the propagator is shifted.
        nat = b.ifftshift(planes[plane], axes=(0, 1))
        wq, wmin, wmax = quantize(nat, b)
        # Coarser levels for the display downsampler, quantized against the
        # FULL plane's range so a level change never repaints the colorbar.
        # `quantize` recomputes min/max per array, so pass them in.
        mips = tuple(requantize(lv, b, wmin, wmax)
                     for lv in pyramid.levels(nat, b))
        pf.append(PlaneFrame(a=plane[0], b=plane[1], mode=ax.MODE_PROJECTION,
                             wq=wq, wmin=wmin, wmax=wmax, mips=mips))
    vf = VariantFrame(vid=vid, dt=float(dt), E=obs.E, purity=obs.purity,
                      lz=obs.lz, mean=obs.mean, std=obs.std,
                      planes=tuple(pf), marg=obs.marg)
    return vf, obs
