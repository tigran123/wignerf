/**
 * The four initial-condition kinds, and everything about them that is a pure
 * function of (kind, ndim).
 *
 * Extracted from ICEditor.vue for the reason lib/potentialCuts.ts was: vitest
 * here runs with no DOM, so anything that lives in a .vue file cannot be tested
 * at all. The tab labels, the variables each kind accepts and the per-ndim
 * default expressions are exactly the things a dimensionality switch gets
 * wrong, and they are all decidable without a browser.
 */
import { labels } from './axes'

export const IC_KINDS = ['mixture', 'cat', 'wexpr', 'psi'] as const
export type ICKind = (typeof IC_KINDS)[number]
export type ICExprKind = Extract<ICKind, 'wexpr' | 'psi'>

export function isExprKind(t: string): t is ICExprKind {
  return t === 'wexpr' || t === 'psi'
}

export function isICKind(t: unknown): t is ICKind {
  return typeof t === 'string' && (IC_KINDS as readonly string[]).includes(t)
}

/**
 * The variables an expression of this kind may use — DERIVED from lib/axes.ts,
 * so a label change there moves them and cannot leave a fourth copy behind.
 * A W lives on the whole phase space; a ψ lives on configuration space only,
 * because ψ(x, p) would be nonsense.
 */
export function exprVariables(kind: ICExprKind, ndim: number): readonly string[] {
  const all = labels(ndim)
  return kind === 'wexpr' ? all : all.slice(0, ndim)
}

/**
 * The tab button's text.
 *
 * SHORTENED AT ndim=2 — but not because it would overflow. Measured at the real
 * width, the four full labels come to ~185 px against the 301 px the landscape
 * aside actually gives its content, so they fit. The reason is that a tab names
 * a MODE, and `W(x,y,px,py)` as a button label is noise. The full signature
 * belongs where there is room to read it: the tooltip, the input's placeholder
 * and the chart titles — the same trade CLAUDE.md records for the `(1D)` and
 * `(f64)` markers.
 */
export function icTabLabel(kind: ICKind, ndim: number): string {
  if (kind === 'wexpr') return ndim > 1 ? 'W(…)' : `W(${exprVariables('wexpr', 1).join(',')})`
  if (kind === 'psi') return ndim > 1 ? 'ψ(…)' : `ψ(${exprVariables('psi', 1).join(',')})`
  return kind
}

/** The tooltip. It must NAME every variable, which is what keeps the elided
 *  label from being a mystery — a marker plus the reason, never prose in the
 *  panel itself. */
export function icTabTitle(kind: ICKind, ndim: number): string {
  switch (kind) {
    case 'mixture':
      return 'statistical mixture: W >= 0, independent σₓ, σₚ'
    case 'cat':
      return 'coherent superposition (cat): interference fringes, σₚ derived'
    case 'wexpr':
      return `an arbitrary analytic W(${exprVariables('wexpr', ndim).join(', ')})`
        + ' — normalised automatically by its own integral over the grid, so'
        + ' the overall amplitude does not matter'
    default:
      return `an arbitrary analytic wavefunction ψ(${exprVariables('psi', ndim).join(', ')})`
        + ', complex (use I), from which W is built by the Wigner transform'
        + ' — normalised automatically'
  }
}

export function exprPlaceholder(kind: ICExprKind, ndim: number): string {
  if (kind === 'wexpr') {
    return ndim > 1
      ? '(2*(x^2 + px^2) - 1)*exp(-x^2 - px^2 - y^2 - py^2)'
      : '(2*(x^2 + p^2) - 1)*exp(-x^2 - p^2)'
  }
  return ndim > 1
    ? 'exp(-(x^2 + y^2)/2 + I*x)'
    : 'hermite(1,x)*exp(-x^2/2)'
}

/**
 * The default expression per kind per ndim.
 *
 * The W default is the n=1 Fock state, deliberately: it has a NEGATIVE core, so
 * the first thing anyone sees on that tab is the one property that makes a
 * Wigner function not a probability density. Its integral is +π, so
 * auto-normalisation never has to flip a sign.
 *
 * The ψ default is a real, even cat — Im ψ is identically zero, which makes the
 * flat Im trace teach what the three curves are. The phase-carrying form lives
 * in the placeholder.
 */
export const DEFAULT_IC_EXPR: Record<1 | 2, Record<ICExprKind, string>> = {
  1: {
    wexpr: '(2*(x^2 + p^2) - 1)*exp(-x^2 - p^2)',
    psi: 'exp(-(x-2)^2/2) + exp(-(x+2)^2/2)',
  },
  2: {
    wexpr: '(2*(x^2 + px^2) - 1)*exp(-x^2 - px^2 - y^2 - py^2)',
    psi: 'exp(-((x-2)^2 + y^2)/2) + exp(-((x+2)^2 + y^2)/2)',
  },
}
