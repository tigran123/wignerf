/**
 * The two axis cuts a 2D U(x, y) preview is read as, and the abscissa each one
 * belongs to.
 *
 * Extracted from PotentialEditor.vue so the mapping can be pinned by a unit
 * test rather than eyeballed on a canvas: the bug this closes was invisible
 * exactly where it was most likely to be looked at. `POST /preview/potential`
 * returns U on an n x n lattice over (xs, ys), and the two cuts through the
 * origin are
 *
 *   U(x, y0) = rows[i][j0]  for every i   -> indexed by X
 *   U(x0, y) = rows[i0][j]  for every j   -> indexed by Y
 *
 * which are DIFFERENT abscissae: xs is the editor's (possibly zoomed) window
 * and ys is the grid's own y extent. uPlot's AlignedData carries one shared x
 * per chart, so these need two charts — drawn as two traces on one, the y cut
 * was plotted at the x sample positions and silently rescaled by
 * (x2-x1)/(y2-y1). On the isotropic default box the two windows coincide and
 * it looked perfect, which is why it survived.
 */

export interface PotentialSamples {
  x: number[]
  y?: number[]
  /** 1D: one trace. 2D: rows[i][j] = U(x[i], y[j]). */
  U: (number | null)[] | (number | null)[][]
}

/** Is this a 2D response (an n x n lattice) rather than a 1D trace? */
export function isLattice(sm: PotentialSamples): boolean {
  return Array.isArray(sm.U[0])
}

/** Index of the sample closest to the origin. */
export function nearestZero(v: number[]): number {
  let best = 0
  for (let i = 1; i < v.length; i++)
    if (Math.abs(v[i]!) < Math.abs(v[best]!)) best = i
  return best
}

/** The x chart's data: U(x) in 1D, U(x, y≈0) in 2D — indexed by x either way. */
export function cutAlongX(sm: PotentialSamples): [number[], (number | null)[]] {
  if (!isLattice(sm)) return [sm.x, sm.U as (number | null)[]]
  const rows = sm.U as (number | null)[][]
  const j = nearestZero(sm.y ?? [])
  return [sm.x, rows.map((row) => row[j] ?? null)]
}

/** The y chart's data: U(x≈0, y) — indexed by y, hence against sm.y. */
export function cutAlongY(sm: PotentialSamples): [number[], (number | null)[]] {
  const rows = sm.U as (number | null)[][]
  return [sm.y ?? [], rows[nearestZero(sm.x)] ?? []]
}

/** The coordinates the two cuts were actually TAKEN at, or null in 1D. */
export function cutAt(sm: PotentialSamples): { x: number; y: number } | null {
  if (!isLattice(sm)) return null
  const ys = sm.y ?? []
  return { x: sm.x[nearestZero(sm.x)] ?? 0, y: ys[nearestZero(ys)] ?? 0 }
}

/**
 * "0" when the cut really is on the axis (within half a sample step), else the
 * coordinate it was taken at. A window zoomed away from the origin has no
 * sample there, and a title claiming U(0, y) of a cut taken at x = 3.2 is the
 * same class of silent mislabel as drawing it on the wrong axis.
 */
export function cutLabel(v: number | undefined, axis: number[] | undefined): string {
  if (v == null) return '0'
  const step = axis && axis.length > 1 ? Math.abs(axis[1]! - axis[0]!) : 0
  return Math.abs(v) <= step/2 ? '0' : String(+v.toPrecision(3))
}
