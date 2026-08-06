import { describe, expect, it } from 'vitest'
import { cutAlongX, cutAlongY, cutAt, cutLabel, isLattice, nearestZero }
  from './potentialCuts'

/** U(x, y) = x + 10*y on an ANISOTROPIC lattice, so a swapped abscissa or a
 *  transposed index cannot pass: x runs [-8..8] in 5 samples, y [-2..2] in 3. */
const XS = [-8, -4, 0, 4, 8]
const YS = [-2, 0, 2]
const SM = {
  x: XS,
  y: YS,
  U: XS.map((x) => YS.map((y) => x + 10*y)),
}

describe('the 2D axis cuts', () => {
  it('gives the y cut its OWN abscissa, not the x one', () => {
    // THE bug: [sm.x, alongY] drew U(0, y) — three y samples — at the first
    // three X positions, i.e. rescaled by (x2-x1)/(y2-y1) = 4. Invisible on the
    // isotropic default box, which is exactly where it was looked at.
    const [ax, vx] = cutAlongX(SM)
    const [ay, vy] = cutAlongY(SM)
    expect(ax).toEqual(XS)
    expect(ay).toEqual(YS)
    expect(ay).not.toEqual(ax)
    // U(x, 0) = x
    expect(vx).toEqual([-8, -4, 0, 4, 8])
    // U(0, y) = 10y — and one value per Y sample, not per X sample
    expect(vy).toEqual([-20, 0, 20])
    expect(vy.length).toBe(YS.length)
  })

  it('reports which coordinate each cut was taken at', () => {
    expect(cutAt(SM)).toEqual({ x: 0, y: 0 })
    // a window zoomed away from the origin has no sample there
    const off = { ...SM, x: [1, 2, 3, 4, 5] }
    expect(cutAt(off)!.x).toBe(1)
    expect(cutLabel(1, off.x)).toBe('1')          // named, not silently "0"
    expect(cutLabel(0, XS)).toBe('0')
    // within half a step of the axis still reads as 0
    expect(cutLabel(0.4, [0, 4, 8])).toBe('0')
    expect(cutLabel(2.5, [0, 4, 8])).toBe('2.5')
  })

  it('names the y coordinate too when the Y window is zoomed off-origin', () => {
    // The y cut zooms independently of the x one, so an off-origin window is
    // reachable on EITHER axis and each moves where the OTHER's cut is read:
    // a y window of [1, 3] leaves no y = 0 to cut U(x, ·) at, and the x chart's
    // title must say 1 rather than claim a U(x, 0) it never sampled.
    const offY = { ...SM, y: [1, 2, 3], U: XS.map((x) => [1, 2, 3].map((y) => x + 10*y)) }
    expect(cutAt(offY)).toEqual({ x: 0, y: 1 })
    expect(cutLabel(1, offY.y)).toBe('1')
    // and the cut really is the row at that y: U(x, 1) = x + 10
    expect(cutAlongX(offY)[1]).toEqual([2, 6, 10, 14, 18])
    // the y cut keeps its own abscissa, zoomed or not
    expect(cutAlongY(offY)[0]).toEqual([1, 2, 3])
  })

  it('leaves 1D alone: one trace against its own x', () => {
    const one = { x: XS, U: [0, 1, 2, 3, 4] }
    expect(isLattice(one)).toBe(false)
    expect(cutAlongX(one)).toEqual([XS, [0, 1, 2, 3, 4]])
    expect(cutAt(one)).toBeNull()
  })

  it('nearestZero picks the sample closest to the origin', () => {
    expect(nearestZero([-5, -1, 3])).toBe(1)
    expect(nearestZero([2])).toBe(0)
  })
})
