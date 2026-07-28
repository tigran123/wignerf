import { describe, expect, it } from 'vitest'
import {
  defaultDiagnostics, diagnosticLabel, diagnosticTitle, diagnosticsAvailable,
  panelGridOf, panelPixels, panelRight, PANEL_PX_MIN,
} from './exportLayout'

/**
 * These mirror core/render_mpl.py — panel_grid, diag_layout,
 * diagnostics_available and diagnostics_default. The export dialog promises
 * what the video will do, so a rule that drifts here is a promise the backend
 * quietly breaks.
 */
describe('diagnostics vocabulary', () => {
  it('lists 1D five plots and 2D nine, in PlotsColumn order', () => {
    expect(diagnosticsAvailable(1)).toEqual(
      ['marg0', 'marg1', 'E', 'uncertainty0', 'purity'])
    expect(diagnosticsAvailable(2)).toEqual(
      ['marg0', 'marg1', 'marg2', 'marg3', 'E',
       'uncertainty0', 'uncertainty1', 'purity', 'lz'])
  })

  it('drops the marginals by default at 2D only', () => {
    // 1D defaults to the frame it always had
    expect(defaultDiagnostics(1)).toEqual(diagnosticsAvailable(1))
    // at 2D the x,y and px,py PANELS already are those densities
    expect(defaultDiagnostics(2)).toEqual(
      ['E', 'uncertainty0', 'uncertainty1', 'purity', 'lz'])
    expect(defaultDiagnostics(2)).toHaveLength(5)
  })

  it('names each plot as the SPA does', () => {
    expect(diagnosticTitle(2, 'marg2')).toBe('φ(px) = ∫W dx dy dpy')
    expect(diagnosticTitle(2, 'purity')).toBe(
      'purity γ(t) = (2πℏ)²⨌W²dxdydpxdpy')
    expect(diagnosticTitle(2, 'lz')).toBe('⟨Lz⟩(t) = ⟨x·py − y·px⟩')
    expect(diagnosticTitle(1, 'uncertainty0')).toBe('ΔX·ΔP(t)')
    // chips are short enough for a 26rem popover
    expect(diagnosticLabel(2, 'marg0')).toBe('ρ(x)')
    expect(diagnosticLabel(2, 'marg3')).toBe('φ(py)')
    expect(diagnosticLabel(2, 'uncertainty1')).toBe('ΔY·ΔPy')
    expect(diagnosticLabel(2, 'purity')).toBe('γ')
    expect(diagnosticLabel(2, 'lz')).toBe('⟨Lz⟩')
    for (const d of diagnosticsAvailable(2))
      expect(diagnosticLabel(2, d).length).toBeLessThanOrEqual(7)
  })
})

describe('panel tiling', () => {
  it('reflows when one dimension is 1, matching PanelGrid.gridClass', () => {
    expect(panelGridOf(1, 1)).toMatchObject({ rows: 1, cols: 1 })
    expect(panelGridOf(1, 2)).toMatchObject({ rows: 1, cols: 2 })
    expect(panelGridOf(1, 4)).toMatchObject({ rows: 2, cols: 2 })
    // the phase portrait: six planes, 3 across and 2 down
    expect(panelGridOf(6, 1)).toMatchObject({ rows: 2, cols: 3, cells: 6 })
  })

  it('is the matrix itself when both vary', () => {
    expect(panelGridOf(3, 2)).toMatchObject({ rows: 3, cols: 2, cells: 6 })
    expect(panelGridOf(6, 4)).toMatchObject({ rows: 6, cols: 4, cells: 24 })
  })
})

describe('block geometry', () => {
  it('keeps the historic 1D width up to seven plots', () => {
    for (const n of [1, 5, 7]) expect(panelRight(n)).toBe(0.60)
    // past that the diagnostics column splits and the panels pay for it
    expect(panelRight(9)).toBe(0.42)
    // nothing selected: the panels take the whole frame
    expect(panelRight(0)).toBe(0.965)
  })

  it('warns exactly when the panels become thumbnails', () => {
    // the classic 1D export: four big panels at FHD
    const one = panelGridOf(1, 4)
    const px = panelPixels(one.rows, one.cols, 5, 1920, 1080)
    expect(Math.min(px.w, px.h)).toBeGreaterThan(PANEL_PX_MIN)

    // 24 panels at FHD is a thumbnail grid, and 4K does not rescue it
    const many = panelGridOf(6, 4)
    for (const [w, h] of [[1920, 1080], [3840, 2160]] as const) {
      const p = panelPixels(many.rows, many.cols, 5, w, h)
      expect(Math.min(p.w, p.h)).toBeLessThan(PANEL_PX_MIN)
    }

    // a 4K phase portrait is comfortable, which is the point of offering it
    const six = panelGridOf(6, 1)
    const p4k = panelPixels(six.rows, six.cols, 5, 3840, 2160)
    expect(Math.min(p4k.w, p4k.h)).toBeGreaterThan(PANEL_PX_MIN)
  })

  it('gives every panel more room when the diagnostics are dropped', () => {
    const g = panelGridOf(6, 1)
    const withDiag = panelPixels(g.rows, g.cols, 9, 1920, 1080)
    const without = panelPixels(g.rows, g.cols, 0, 1920, 1080)
    expect(without.w).toBeGreaterThan(withDiag.w)
    expect(without.h).toBe(withDiag.h)     // only the WIDTH is traded
  })
})
