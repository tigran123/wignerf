/**
 * Axis bookkeeping — the mirror of backend/core/axes.py. Read its docstring
 * for the array-order and conjugation conventions; this file exists so the
 * frontend never hard-codes "x" and "p" again, and so the plot titles here and
 * on the exported video (core/render_mpl.py) say the same words.
 *
 * Change a label or title string in one of the three and change it in all
 * three.
 */

export type Ndim = 1 | 2

/** Axis labels in array order: all spatial, then all momentum. */
export const LABELS: Record<Ndim, readonly string[]> = {
  1: ['x', 'p'],
  2: ['x', 'y', 'px', 'py'],
}

/** Canonical plane set per ndim, in display order (see core/axes.py PLANES). */
export const PLANES: Record<Ndim, readonly (readonly [number, number])[]> = {
  1: [[0, 1]],
  2: [[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]],
}

export function ndimOf(n: number): Ndim {
  return n === 2 ? 2 : 1
}

export function labels(ndim: number): readonly string[] {
  return LABELS[ndimOf(ndim)]
}

export function label(ndim: number, axis: number): string {
  return labels(ndim)[axis] ?? `q${axis}`
}

/**
 * An axis name with its trailing letter as a real subscript: 'px' -> 'p<sub>x</sub>'.
 *
 * HTML rather than Unicode because Unicode cannot do it: the subscript block
 * has ₓ (U+2093) but NO subscript y, so 'pₓ' beside 'py' would look worse than
 * plain text. Every axis name is one letter plus an optional one-letter
 * subscript, which is the whole rule.
 *
 * The strings this produces are ours, never user input, so the v-html bindings
 * that consume them carry no injection surface. Anywhere markup cannot go —
 * an <option>, which renders text only — callers keep the plain form.
 */
export function labelHtml(ndim: number, axis: number): string {
  return subHtml(label(ndim, axis))
}

export function subHtml(name: string): string {
  return name.length === 2 ? `${name[0]}<sub>${name[1]}</sub>` : name
}

export function planes(ndim: number): readonly (readonly [number, number])[] {
  return PLANES[ndimOf(ndim)]
}

export function nAxes(ndim: number): number {
  return 2 * ndim
}

export function isMomentum(ndim: number, axis: number): boolean {
  return axis >= ndim
}

/**
 * The Fourier-conjugate partner of an axis: q_i <-> k_i, index-matched.
 *
 * Mirrors core/axes.conjugate, which states the pairing once for the whole
 * program. It is here because the wavefunction preview is the first frontend
 * code that needs "the momentum axis dual to spatial axis i" — and a wrong
 * pairing is the classic multi-D error precisely because nothing complains.
 */
export function conjugate(ndim: number, axis: number): number {
  return axis >= ndim ? axis - ndim : axis + ndim
}

/** The axes a plane is reduced OVER — empty at ndim=1. */
export function complement(ndim: number, plane: readonly [number, number]): number[] {
  const out: number[] = []
  for (let a = 0; a < 2 * ndim; a++) if (!plane.includes(a)) out.push(a)
  return out
}

/** Short chip for a panel corner: "x,p" / "x,py". */
export function planeLabel(ndim: number, plane: readonly [number, number]): string {
  return `${label(ndim, plane[0])},${label(ndim, plane[1])}`
}

export function planeLabelHtml(ndim: number, plane: readonly [number, number]): string {
  return `${labelHtml(ndim, plane[0])},${labelHtml(ndim, plane[1])}`
}

function measure(ndim: number, over: number[], html = false): string {
  return over.map((a) => 'd' + (html ? labelHtml(ndim, a) : label(ndim, a)))
    .join(' ')
}

/**
 * What the plane IS. Empty at ndim=1: the single plane is W itself, and the
 * panel's title there is the variant name, exactly as it always was.
 */
export function planeTitle(ndim: number, plane: readonly [number, number],
                           html = false): string {
  const over = complement(ndim, plane)
  if (!over.length) return ''
  return `${over.length === 2 ? '∬' : '∫'}W ${measure(ndim, over, html)}`
}

/**
 * rho(x) = int W dp   /   rho(py) = int W dx dy dpx
 *
 * EVERY marginal is rho, momentum ones included — see core/axes.marginal_title
 * for why: phi now belongs to the psi tab's wavefunction chart, which is a
 * probability AMPLITUDE where this is a DENSITY.
 */
export function marginalTitle(ndim: number, axis: number, html = false): string {
  const sym = 'ρ'
  const over: number[] = []
  for (let a = 0; a < 2 * ndim; a++) if (a !== axis) over.push(a)
  const name = html ? labelHtml(ndim, axis) : label(ndim, axis)
  return `${sym}(${name}) = ∫W ${measure(ndim, over, html)}`
}

export function purityExpr(ndim: number, html = false): string {
  const pre = ndim === 1 ? '2πℏ' : '(2πℏ)²'
  const sign = ndim === 1 ? '∬' : '⨌'
  const ds = labels(ndim).map((l, a) => 'd' + (html ? labelHtml(ndim, a) : l))
  return `${pre}${sign}W²${ds.join('')}`
}

export function purityTitle(ndim: number, html = false): string {
  return `purity γ(t) = ${purityExpr(ndim, html)}`
}

/** ΔX·ΔP(t) at ndim=1; ΔX·ΔPx(t) / ΔY·ΔPy(t) at 2. */
export function uncertaintyTitle(ndim: number, dim: number,
                                 html = false): string {
  const q = label(ndim, dim).toUpperCase()
  const k = label(ndim, ndim + dim)
  const kk = `${k.charAt(0).toUpperCase()}${k.slice(1)}`
  return `Δ${q}·Δ${html ? subHtml(kk) : kk}(t)`
}

/** ⟨Lz⟩(t) — the 2D-only angular-momentum series. */
export function lzTitle(html = false): string {
  const lz = html ? 'L<sub>z</sub>' : 'Lz'
  const px = html ? 'p<sub>x</sub>' : 'px'
  const py = html ? 'p<sub>y</sub>' : 'py'
  return `⟨${lz}⟩(t) = ⟨x·${py} − y·${px}⟩`
}

/** The plane index within PLANES(ndim), or -1. */
export function planeIndex(ndim: number, plane: readonly [number, number]): number {
  return planes(ndim).findIndex((p) => p[0] === plane[0] && p[1] === plane[1])
}
