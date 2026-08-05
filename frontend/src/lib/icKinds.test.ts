import { describe, expect, it } from 'vitest'

import { labels } from './axes'
import { conjugate } from './axes'
import { DEFAULT_IC_EXPR, exprVariables, icTabLabel, icTabTitle, IC_KINDS,
         isExprKind, isICKind } from './icKinds'
import { abs2, waveSeries, waveTitleHtml, type WaveCut } from './wavefn'

describe('IC kinds', () => {
  it('classifies the four kinds', () => {
    expect(IC_KINDS).toEqual(['mixture', 'cat', 'wexpr', 'psi'])
    expect(IC_KINDS.filter(isExprKind)).toEqual(['wexpr', 'psi'])
    expect(isICKind('psi')).toBe(true)
    expect(isICKind('nonsense')).toBe(false)
  })

  it('derives each kind variables from lib/axes, so a label change moves them', () => {
    // The point of this test is the DERIVATION, not the literals: if axes.ts
    // ever renames an axis, these must follow rather than becoming a fourth
    // copy of the naming (core/axes.py, lib/axes.ts, render_mpl.py).
    expect(exprVariables('wexpr', 1)).toEqual(labels(1))
    expect(exprVariables('wexpr', 2)).toEqual(labels(2))
    // psi lives on configuration space only — psi(x, p) would be nonsense
    expect(exprVariables('psi', 1)).toEqual(['x'])
    expect(exprVariables('psi', 2)).toEqual(['x', 'y'])
  })

  it('spells the variables in 1D labels and elides them in 2D', () => {
    expect(icTabLabel('wexpr', 1)).toBe('W(x,p)')
    expect(icTabLabel('psi', 1)).toBe('ψ(x)')
    expect(icTabLabel('wexpr', 2)).toBe('W(…)')
    expect(icTabLabel('psi', 2)).toBe('ψ(…)')
    expect(icTabLabel('mixture', 2)).toBe('mixture')
  })

  it('names every variable in the tooltip, so the elision is never a mystery', () => {
    for (const nd of [1, 2] as const)
      for (const k of ['wexpr', 'psi'] as const)
        for (const v of exprVariables(k, nd))
          expect(icTabTitle(k, nd)).toContain(v)
  })

  it('keeps every default expression inside its own variable set', () => {
    // A 2D default containing `p`, or a 1D one containing `px`, is refused by
    // the backend's free-symbol check — i.e. the tab would be broken on
    // arrival. Checking the reverse direction too: every variable it does use
    // must be one this ndim offers.
    for (const nd of [1, 2] as const)
      for (const k of ['wexpr', 'psi'] as const) {
        const allowed = new Set(exprVariables(k, nd))
        const used = DEFAULT_IC_EXPR[nd][k].match(/\b(px|py|x|y|p)\b/g) ?? []
        expect([...new Set(used)].filter((v) => !allowed.has(v))).toEqual([])
        expect(used.length).toBeGreaterThan(0)
      }
  })
})

describe('the wavefunction traces', () => {
  const cut = (over: Partial<WaveCut> = {}): WaveCut => ({
    axis: 0, at: null, v: [-1, 0, 1], re: [1, 2, 3], im: [0, 1, 0], ...over,
  })

  it('squares the modulus', () => {
    expect(abs2([3], [4])).toEqual([25])
    expect(abs2([1, 2], [])).toEqual([1, 4])     // a real trace has no Im
  })

  it('emits exactly four arrays, matching the three series', () => {
    // uPlot throws when fed fewer arrays than series, and the half-built chart
    // collapses to min-content width — which shows up as a title typeset one
    // word per line rather than as an error.
    expect(waveSeries(cut())).toHaveLength(4)
  })

  it('titles a 1D trace with its own axis and no cut coordinate', () => {
    expect(waveTitleHtml('ψ', 1, cut())).toBe('ψ(x)')
    expect(waveTitleHtml('φ', 1, cut({ axis: 1 }))).toBe('φ(p)')
  })

  it('subscripts a two-letter momentum name rather than running it together', () => {
    const t = waveTitleHtml('φ', 2, cut({ axis: 2, at: 0 }))
    expect(t).toContain('<sub>x</sub>')
    expect(t).not.toContain('φ(px')
  })

  it('names the coordinate the cut was ACTUALLY taken at', () => {
    expect(waveTitleHtml('ψ', 2, cut({ at: 0 }))).toBe('ψ(x, 0)')
    expect(waveTitleHtml('ψ', 2, cut({ at: 0.4 }))).toBe('ψ(x, 0.4)')
  })

  it('pairs a cut axis with its conjugate, at every ndim', () => {
    // The endpoint forces this rather than accepting two independent choices,
    // which is what makes the pairing unmistakable. Assert the map itself.
    expect(conjugate(1, 0)).toBe(1)
    expect(conjugate(2, 0)).toBe(2)
    expect(conjugate(2, 1)).toBe(3)
    expect(conjugate(2, 3)).toBe(1)
  })
})

describe('the WIRE shape of an initial condition', () => {
  it('sends each kind ONLY its own shape', async () => {
    const { icPayload, defaultIC } = await import('./config')
    const ic = defaultIC(1)
    ic.expr.wexpr = 'exp(-x^2-p^2)'
    ic.expr.psi = 'exp(-x^2/2)'

    // Gaussian kinds: components, no expr. Sending a dead `expr` here made it
    // round-trip through describe.setup_document into the mp4 comment tag, so
    // an import showed an expression that had no part in the run.
    for (const k of ['mixture', 'cat'] as const) {
      ic.type = k
      const p = icPayload(ic)
      expect(p.type).toBe(k)
      expect(p.components).toBe(ic.components)
      expect('expr' in p).toBe(false)
    }

    // Expression kinds: ONE expression as a plain string, no components. The
    // form keeps both drafts and the components so switching tabs is
    // non-destructive; none of that is run state. Sending stale components on a
    // psi got SessionCreate to refuse it with a sentence about components a psi
    // does not use.
    ic.type = 'psi'
    expect(icPayload(ic)).toEqual({ type: 'psi', expr: 'exp(-x^2/2)' })
    ic.type = 'wexpr'
    expect(icPayload(ic)).toEqual({ type: 'wexpr', expr: 'exp(-x^2-p^2)' })
  })
})

describe('waveSeries', () => {
  const cut: WaveCut = { axis: 0, at: null, v: [0, 1], re: [3, 4], im: [0, 5] }

  it('is ORDERED v, Re, Im, |·|² — WaveFnPlots.SERIES is positional', () => {
    // Arity alone cannot see a swap, and a swap is silent on screen: Re and Im
    // are both plausible curves. SERIES maps index -> palette role and width,
    // so getting this wrong repaints Re with Im's thin dash and vice versa.
    expect(waveSeries(cut)).toEqual([[0, 1], [3, 4], [0, 5], [9, 41]])
  })

  it('computes |·|² — the SQUARE — from BOTH parts', () => {
    expect(abs2([3], [4])).toEqual([25])
  })
})
