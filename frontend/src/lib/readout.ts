/**
 * The control bar's live numeric readouts, normalized across the TWO sources
 * they can come from: the newest PAINTED frame, or a batch-mode `progress`
 * report. Extracted from ControlBar so the choice can be unit-tested — the
 * same reason lib/transport.ts exists.
 *
 * WHICH SOURCE WINS IS DECIDED BY ARRIVAL ORDER, NEVER BY RECORD INDEX.
 * Batch mode streams no frames while it computes, so the readouts ride the
 * progress report — and they must SURVIVE the pause that ends the run: the
 * report is then the only record of where it stopped (`lastFrame` is null for
 * a whole batch run), and gating on `status.computing` blanked the entire
 * bottom line the moment the run stopped, t included.
 *
 * "Highest record wins" is the tempting alternative and it is wrong: scrubbing
 * to record 50 in a FINISHED batch run leaves the frame at 50 behind a retained
 * progress report at the frontier, and the line would print t2 over a panel
 * painted at record 50. Arrival order gets that right for free, and leaves
 * interactive mode untouched — no progress message is ever sent there, so the
 * source is always the frame.
 */

import type { Frame } from './protocol'
import type { ProgressEvent } from '../composables/useSession'

export type ReadoutSource = 'frame' | 'progress'

export interface Readout {
  t: number | null
  E: number | null
  /** The uncertainty PRODUCT per spatial dimension: std[i]*std[ndim+i]. One
   *  number in 1D, two in 2D — a single "ΔX·ΔP" would silently report only the
   *  x dimension of a run where y is the interesting one. */
  uncert: number[] | null
  purity: number | null
  ndim: number
  /** Percent toward t2 — a progress report only, so it doubles as "these
   *  values came from the batch report" for the compact badge. */
  pct: number | null
}

const EMPTY: Readout =
  { t: null, E: null, uncert: null, purity: null, ndim: 1, pct: null }

export function products(std: number[]): number[] {
  const nd = std.length / 2
  return Array.from({ length: nd }, (_, i) => std[i]! * std[nd + i]!)
}

function fromFrame(f: Frame | null): Readout | null {
  if (!f) return null
  const v = f.variants[0]
  return {
    t: f.t, ndim: f.ndim, pct: null,
    E: v ? v.E : null,
    uncert: v ? products(v.std) : null,
    purity: v ? v.purity : null,
  }
}

function fromProgress(p: ProgressEvent | null): Readout | null {
  if (!p) return null
  // Before the first record exists the report carries t (= t1) and percent but
  // no observables at all, so the fields are nullable one by one rather than
  // the whole source being absent.
  const v = p.per_variant[0]
  const std = v?.std
  return {
    t: p.t, pct: p.percent,
    ndim: std ? std.length / 2 : 1,
    E: v?.E ?? null,
    uncert: std ? products(std) : null,
    purity: v?.purity ?? null,
  }
}

/** Observables of the FIRST active variant, from whichever source spoke last.
 *  Falls back to the other one when the named source is absent, which is what
 *  makes a null `source` (nothing received yet) harmless. */
export function pickReadout(
  source: ReadoutSource | null,
  frame: Frame | null,
  progress: ProgressEvent | null,
): Readout {
  const f = fromFrame(frame)
  const p = fromProgress(progress)
  return (source === 'progress' ? (p ?? f) : (f ?? p)) ?? EMPTY
}
