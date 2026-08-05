import { describe, expect, it } from 'vitest'
import { mergeViews, planeKey, viewChanged, viewKey, viewsChanged,
         type PlaneViewReq } from './planeView'

/** A request for one panel; `na` doubles as a marker for which one it is. */
function req(vid: number, na = 512): PlaneViewReq {
  return { vid, a: 0, b: 1, a1: -6, a2: 6, b1: -7, b2: 7, na, nb: na }
}

function mapOf(...rs: PlaneViewReq[]): Map<string, PlaneViewReq> {
  return new Map(rs.map((r) => [viewKey(r), r]))
}

const QN = 1, QR = 3, CN = 0, CR = 2

describe('viewKey / planeKey', () => {
  it('agree, so a panel with nothing to request yet keys the same', () => {
    const r = req(QR)
    expect(planeKey(r.vid, r.a, r.b)).toBe(viewKey(r))
  })

  it('separates variants of one plane and planes of one variant', () => {
    expect(planeKey(QN, 0, 1)).not.toBe(planeKey(CN, 0, 1))
    expect(planeKey(QN, 0, 1)).not.toBe(planeKey(QN, 0, 3))
  })
})

describe('mergeViews', () => {
  const live = new Set([QN, QR, CN, CR].map((v) => planeKey(v, 0, 1)))

  it('keeps the panels a partial burst says nothing about', () => {
    // the bug this exists for: an unlinked zoom on ONE paused panel emitted a
    // 1-entry view, and the server stopped sending the other three planes
    const sent = mapOf(req(QN), req(QR), req(CN), req(CR))
    const next = mergeViews(sent, mapOf(req(QN, 256)), live)
    expect([...next.keys()].sort()).toEqual([...sent.keys()].sort())
    expect(next.get(planeKey(QN, 0, 1))!.na).toBe(256)   // the burst
    expect(next.get(planeKey(CR, 0, 1))!.na).toBe(512)   // carried over
  })

  it('admits a panel that has never been sent', () => {
    // a restart that adds variants: two panels are new to the map
    const sent = mapOf(req(QN), req(CN))
    const next = mergeViews(sent, mapOf(req(QR), req(CR)), live)
    expect(next.size).toBe(4)
  })

  it('drops a panel that has gone away', () => {
    const sent = mapOf(req(QN), req(QR), req(CN), req(CR))
    const twoLeft = new Set([QN, CN].map((v) => planeKey(v, 0, 1)))
    const next = mergeViews(sent, new Map(), twoLeft)
    expect([...next.keys()]).toEqual([...twoLeft])
  })

  it('is the identity on an empty burst', () => {
    const sent = mapOf(req(QN), req(CN))
    const twoLive = new Set([QN, CN].map((v) => planeKey(v, 0, 1)))
    expect(mergeViews(sent, new Map(), twoLive)).toEqual(sent)
  })

  it('reports nothing for a panel neither side knows', () => {
    const next = mergeViews(new Map(), new Map(), live)
    expect(next.size).toBe(0)
  })
})

describe('viewsChanged', () => {
  it('is false when the burst restates what was sent', () => {
    const sent = mapOf(req(QN), req(CN))
    expect(viewsChanged(sent, mapOf(req(QN), req(CN)))).toBe(false)
  })

  it('is true when a panel arrives or leaves', () => {
    const sent = mapOf(req(QN), req(CN))
    expect(viewsChanged(sent, mapOf(req(QN), req(CN), req(QR)))).toBe(true)
    expect(viewsChanged(sent, mapOf(req(QN)))).toBe(true)
  })

  it('is true when one panel moved, and follows viewChanged for that', () => {
    const sent = mapOf(req(QN), req(CN))
    const moved = { ...req(CN), a1: -3, a2: 3 }
    expect(viewChanged(sent.get(viewKey(moved)), moved)).toBe(true)
    expect(viewsChanged(sent, mapOf(req(QN), moved))).toBe(true)
  })

  it('ignores a sub-pixel resize, so a pan does not become a request stream', () => {
    const sent = mapOf(req(QN, 512))
    expect(viewsChanged(sent, mapOf(req(QN, 513)))).toBe(false)
  })

  it('is true when one panel is swapped for another at the same count', () => {
    expect(viewsChanged(mapOf(req(QN)), mapOf(req(QR)))).toBe(true)
  })
})
