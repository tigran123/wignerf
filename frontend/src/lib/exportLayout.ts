/**
 * What an export's content selection produces — the frontend mirror of
 * core/render_mpl.py's `panel_grid`, `diag_layout`, `diagnostics_available`
 * and `diagnostics_default`.
 *
 * Extracted from ExportPanel.vue for the reason lib/config.axisSizeOptions
 * was: the interesting cases here (a 24-panel grid at FHD, a 1D form over a
 * 2D session) are otherwise reachable only through the DOM, which is how two
 * of them shipped broken last time. Change a rule here and change it in
 * render_mpl.py — the panel promises what the video will do, and a promise
 * that drifts is worse than no promise.
 */
import { labels, isMomentum, marginalTitle, purityTitle, uncertaintyTitle,
         lzTitle, ndimOf } from './axes'

/** Plot ids, shared verbatim with lib/plotPrefs.ts and the export wire. */
export type DiagId = string

/** Every diagnostics plot at this ndim, in PlotsColumn.vue's display order. */
export function diagnosticsAvailable(ndim: number): DiagId[] {
  const nd = ndimOf(ndim)
  const ids: DiagId[] = []
  for (let a = 0; a < 2 * nd; a++) ids.push(`marg${a}`)
  ids.push('E')
  for (let d = 0; d < nd; d++) ids.push(`uncertainty${d}`)
  ids.push('purity')
  if (nd > 1) ids.push('lz')
  return ids
}

/**
 * What an export renders when the user has not chosen. 1D: all five, i.e. the
 * frame it always had. 2D: the SERIES only — there the x,y and px,py PANELS
 * already ARE the spatial and momentum densities, so a 1D marginal is a
 * further reduction of something a panel is showing, and dropping them keeps
 * the frame's shape (one five-row column) identical to 1D's.
 */
export function defaultDiagnostics(ndim: number): DiagId[] {
  const ids = diagnosticsAvailable(ndim)
  return ndimOf(ndim) === 1 ? ids : ids.filter((i) => !i.startsWith('marg'))
}

/** The full title, for a tooltip. */
export function diagnosticTitle(ndim: number, id: DiagId): string {
  if (id.startsWith('marg')) return marginalTitle(ndim, Number(id.slice(4)))
  if (id === 'E') return 'E(t)'
  if (id.startsWith('uncertainty'))
    return uncertaintyTitle(ndim, Number(id.slice(11)))
  if (id === 'purity') return purityTitle(ndim)
  if (id === 'lz') return lzTitle()
  return id
}

/** A chip-sized name, for the checkbox itself. */
export function diagnosticLabel(ndim: number, id: DiagId): string {
  if (id.startsWith('marg')) {
    const a = Number(id.slice(4))
    return `${isMomentum(ndim, a) ? 'φ' : 'ρ'}(${labels(ndim)[a]})`
  }
  if (id === 'E') return 'E'
  if (id.startsWith('uncertainty'))
    return uncertaintyTitle(ndim, Number(id.slice(11))).slice(0, -3)
  if (id === 'purity') return 'γ'
  if (id === 'lz') return '⟨Lz⟩'
  return id
}

/**
 * (rows, cols) of the W-panel block — render_mpl.panel_grid.
 *
 * When one of the two dimensions is 1 the cells REFLOW by count, reproducing
 * PanelGrid.vue's own rule, so "compare variants" and every 1D export keep the
 * 1x1 / 1x2 / 2x2 tiling. When both vary the grid is the matrix itself: one
 * row per plane, one column per variant.
 */
export function panelGridOf(nPlanes: number, nVariants: number):
    { rows: number; cols: number; cells: number } {
  const cells = Math.max(0, nPlanes) * Math.max(0, nVariants)
  let rows: number, cols: number
  if (nPlanes === 1 || nVariants === 1) {
    if (cells <= 1) [rows, cols] = [1, 1]
    else if (cells === 2) [rows, cols] = [1, 2]
    else if (cells <= 4) [rows, cols] = [2, 2]
    else [rows, cols] = [2, 3]
  } else {
    [rows, cols] = [nPlanes, nVariants]
  }
  return { rows, cols, cells }
}

// The figure's own block rectangle, in figure fractions (render_mpl's
// PANEL_LEFT / PLOT_BOTTOM / PLOT_TOP and diag_layout's panel_right).
const PANEL_LEFT = 0.045
const PLOT_H = 0.935 - 0.235
const DIAG_ROWS_MAX = 7

/** Right edge of the panel block, given how many diagnostics were chosen. */
export function panelRight(nDiag: number): number {
  if (nDiag <= 0) return 0.965
  return Math.ceil(nDiag / DIAG_ROWS_MAX) === 1 ? 0.60 : 0.42
}

/**
 * Roughly how many pixels each W panel gets. Approximate on purpose — it is
 * there to answer "will I be able to see anything?", and matplotlib's own
 * margins make an exact figure meaningless anyway.
 */
export function panelPixels(rows: number, cols: number, nDiag: number,
                            width: number, height: number):
    { w: number; h: number } {
  if (!rows || !cols) return { w: 0, h: 0 }
  const blockW = (panelRight(nDiag) - PANEL_LEFT) * width
  const blockH = PLOT_H * height
  // each cell spends roughly a quarter of its width and a third of its height
  // on ticks, labels and (below the colorbar threshold) its colorbar
  return { w: Math.round((blockW / cols) * 0.72),
           h: Math.round((blockH / rows) * 0.66) }
}

/** Below this a panel is a thumbnail, and the dialog says so. */
export const PANEL_PX_MIN = 200
