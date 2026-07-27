/**
 * The lattice a phase-space panel is actually computed on, for drawing.
 *
 * `edgeBand` MIRRORS `backend/core/boundary.edge_band` — move both together, in
 * the spirit of TOL_MIN_F32. It is mirrored rather than read off the wire
 * because the overlay follows the PAINTED frame's geometry, which during a scrub
 * across an auto-expand boundary is not the live window the server's
 * `boundary.band` describes. (The warning TEXT does use the server's number:
 * that one has to be what the mass was actually measured with.)
 */
export function edgeBand(n: number): number {
  return Math.max(4, Math.floor(n/32))
}

export interface AxisLattice {
  lo: number
  hi: number
  n: number
}

/**
 * Cell-boundary positions of one axis that fall inside the visible window
 * [v1, v2], split into the plain lattice and the outer edge band the boundary
 * watch sums. Returns null for the lattice when too many lines would land in
 * the window to read as lines at all — the caller then draws only the band.
 *
 * `maxLines` is a COUNT, not a pixel test, so it behaves the same in a small IC
 * preview and a full panel; zooming in reduces the count and the lattice
 * reappears, which is the useful behaviour at 1D's 4096.
 */
export function cellLines(ax: AxisLattice, v1: number, v2: number,
                          maxLines = 200) {
  const d = (ax.hi - ax.lo)/ax.n
  if (!(d > 0) || !(v2 > v1)) return { lattice: [] as number[], band: [] as number[] }
  const band = edgeBand(ax.n)
  // index range the window covers, clamped to the axis
  const i0 = Math.max(0, Math.floor((v1 - ax.lo)/d))
  const i1 = Math.min(ax.n, Math.ceil((v2 - ax.lo)/d))
  const inBand = (i: number) => i <= band || i >= ax.n - band
  const bandLines: number[] = []
  const lattice: number[] = []
  const tooMany = i1 - i0 > maxLines
  for (let i = i0; i <= i1; i++) {
    const v = ax.lo + i*d
    if (inBand(i)) bandLines.push(v)
    else if (!tooMany) lattice.push(v)
  }
  return { lattice, band: bandLines }
}
