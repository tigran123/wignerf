/**
 * The ψ/φ traces `POST /api/preview/wavefunction` returns, and the titles they
 * are drawn under.
 *
 * Extracted from the component for the reason lib/potentialCuts.ts was, and it
 * is the same failure mode: a cut is drawn against the wrong abscissa, or a
 * title claims a coordinate the cut was not taken at, and both look perfectly
 * fine on the isotropic default box. vitest has no DOM, so this is the only
 * place either can be pinned.
 */
import { labelHtml } from './axes'
import { cutLabel } from './potentialCuts'

/** One trace: a complex function sampled along ONE axis of the phase space. */
export interface WaveCut {
  /** phase-space axis index — never a label; see the endpoint's own note */
  axis: number
  /** the coordinate the OTHER axis was held at, or null when there was none */
  at: number | null
  v: number[]
  re: number[]
  im: number[]
}

export function abs2(re: number[], im: number[]): number[] {
  return re.map((r, i) => r*r + (im[i] ?? 0)*(im[i] ?? 0))
}

/**
 * uPlot AlignedData for one trace: the shared abscissa plus Re, Im and |·|².
 *
 * The arity matters — a LineChart fed fewer arrays than it has series throws,
 * and the half-built chart collapses to min-content width, which renders as a
 * title typeset one word per line.
 */
export function waveSeries(c: WaveCut): [number[], number[], number[], number[]] {
  return [c.v, c.re, c.im, abs2(c.re, c.im)]
}

/**
 * The chart title, naming the coordinate the cut was ACTUALLY taken at — a grid
 * with no sample on the origin gets "ψ(x, 0.4)", never a false "ψ(x, 0)".
 * cutLabel is potentialCuts' rule, reused rather than rewritten.
 *
 * NO SNAPPING TOLERANCE IS PASSED, and none is wanted. cutLabel's second
 * argument exists for the potential editor, whose cut coordinate is a sample of
 * a ZOOM window and so lands near the origin without being on it. `at` here is
 * a LATTICE value the backend already chose with nearestZero, so it is exactly
 * 0.0 whenever the axis has a point there — the overwhelmingly common
 * symmetric/even-N case — and a genuine coordinate otherwise. Both print
 * correctly with the strict comparison. (The parameter used to be declared here
 * and no caller ever passed it, so the snapping the docstring promised had
 * never once run.)
 */
export function waveTitleHtml(sym: 'ψ' | 'φ', ndim: number, c: WaveCut): string {
  const name = labelHtml(ndim, c.axis)
  if (ndim < 2 || c.at === null) return `${sym}(${name})`
  return `${sym}(${name}, ${cutLabel(c.at, undefined)})`
}
