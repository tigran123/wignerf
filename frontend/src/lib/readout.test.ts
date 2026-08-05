import { describe, expect, it } from 'vitest'

import { pickReadout, products } from './readout'
import type { Frame } from './protocol'
import type { ProgressEvent } from '../composables/useSession'

/** Only the fields pickReadout reads; the rest of a decoded Frame is irrelevant
 *  here (and is covered by protocol.test.ts against the binary fixtures). */
function frame(record: number, t: number, ndim = 1): Frame {
  const nax = 2*ndim
  return {
    record, t, ndim,
    N: Array(nax).fill(64), lo: Array(nax).fill(-6), hi: Array(nax).fill(6),
    flags: 0,
    variants: [{
      vid: 1, dt: 0.01, E: 0.5, purity: 0.999999, lz: 0,
      mean: Array(nax).fill(0),
      std: ndim === 1 ? [0.7, 0.71] : [0.7, 0.8, 0.71, 0.81],
      planes: [], marg: [],
    }],
  } as Frame
}

function progress(record: number, t: number,
                  obs: Partial<ProgressEvent['per_variant'][0]> = {},
                  ): ProgressEvent {
  return {
    type: 'progress', record, t, t1: 0, t2: 10, percent: t*10,
    per_variant: [{ variant: 'qn', steps_per_sec: 500, steps_total: 1000,
                    ...obs }],
  }
}

const OBS = { E: 1.25, purity: 0.987654, std: [0.9, 0.6] }

describe('the readouts hold across a batch pause', () => {
  it('keeps the progress report as the source when no frame ever arrived', () => {
    // batch mode: computing streams no frames at all, and the pause that ends
    // the run leaves the report as the ONLY record of where it stopped. Gating
    // this on status.computing blanked t, E, ΔX·ΔP and γ together.
    const r = pickReadout('progress', null, progress(120, 3.5, OBS))
    expect(r.t).toBe(3.5)
    expect(r.E).toBe(1.25)
    expect(r.purity).toBeCloseTo(0.987654, 9)
    expect(r.uncert).toEqual([0.54])
    expect(r.pct).toBe(35)
  })

  it('lets a browsed frame win over a retained report at a HIGHER record', () => {
    // Scrubbing back into a finished batch run: the frame is at record 50, the
    // retained final report at the frontier 150. Arrival order must decide —
    // "highest record wins" would print the frontier's t over a panel painted
    // at record 50.
    const r = pickReadout('frame', frame(50, 1.0), progress(150, 9.99, OBS))
    expect(r.t).toBe(1.0)
    expect(r.E).toBe(0.5)
    expect(r.pct).toBeNull()      // and the "batch NN%" badge goes away with it
  })

  it('reads a 2D run as two uncertainty products', () => {
    expect(pickReadout('frame', frame(3, 0.3, 2), null).ndim).toBe(2)
    expect(pickReadout('frame', frame(3, 0.3, 2), null).uncert)
      .toEqual(products([0.7, 0.8, 0.71, 0.81]))
    expect(products([0.7, 0.8, 0.71, 0.81])).toEqual([0.7*0.71, 0.8*0.81])
    // a 2D progress report carries the same flat std array
    const r = pickReadout('progress', null,
                          progress(3, 0.3, { ...OBS, std: [1, 2, 3, 4] }))
    expect(r.ndim).toBe(2)
    expect(r.uncert).toEqual([3, 8])
  })

  it('shows t from a report that has no observables yet', () => {
    // record -1: the run is armed but nothing has been computed, so the report
    // carries t (= t1) and percent and no scalars. t must still print.
    const r = pickReadout('progress', null, progress(-1, 0, {}))
    expect(r.t).toBe(0)
    expect(r.pct).toBe(0)
    expect(r.E).toBeNull()
    expect(r.uncert).toBeNull()
    expect(r.purity).toBeNull()
    expect(r.ndim).toBe(1)
  })

  it('falls back to whichever source exists, and is empty with neither', () => {
    // nothing has arrived yet: the source is null and either value may be there
    expect(pickReadout(null, frame(0, 0.0), null).t).toBe(0.0)
    expect(pickReadout(null, null, progress(0, 2.0, OBS)).E).toBe(1.25)
    const empty = pickReadout(null, null, null)
    expect([empty.t, empty.E, empty.uncert, empty.purity, empty.pct])
      .toEqual([null, null, null, null, null])
    expect(empty.ndim).toBe(1)
  })
})
