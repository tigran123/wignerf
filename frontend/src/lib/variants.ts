/** Variant keys (the four toggles) and their display metadata. */
import { theme, type ThemeName } from './theme'

export type VariantKey = 'qn' | 'qr' | 'cn' | 'cr'

export const ALL_VARIANTS: VariantKey[] = ['qn', 'qr', 'cn', 'cr']

// Distinct dash patterns: variants frequently produce IDENTICAL curves
// (harmonic: quantum == classical exactly; c = 137: rel ~ nonrel), and the
// last-drawn series would otherwise hide the rest. Later series are drawn
// on top, so their gaps let the earlier (longer-dashed/solid) show through.
// The dashes are the anti-occlusion device, so they are theme-independent;
// only the hues below change.
export const VARIANT_META: Record<VariantKey,
  { label: string; dash: number[] }> = {
  qn: { label: 'Quantum, non-relativistic', dash: [] },      // solid
  qr: { label: 'Quantum, relativistic', dash: [12, 7] },
  cn: { label: 'Classical, non-relativistic', dash: [6, 6] },
  cr: { label: 'Classical, relativistic', dash: [2, 6] },
}

// Two hues per variant, because a curve colour has to hold its own against
// the PAGE. Tailwind's *-400 shades are tuned for the dark theme and wash out
// on white (amber #fbbf24 worst); the *-600 shades are the light-theme
// counterparts. Mirrored by backend/core/render_mpl.py's VARIANT_STYLE.
export const VARIANT_COLORS: Record<ThemeName, Record<VariantKey, string>> = {
  // sky / violet / amber / emerald
  dark: { qn: '#38bdf8', qr: '#a78bfa', cn: '#fbbf24', cr: '#34d399' },
  light: { qn: '#0284c7', qr: '#7c3aed', cn: '#d97706', cr: '#059669' },
}

/** The variant's curve colour in the ACTIVE theme. Reactive: read it inside a
 *  template or a computed and it follows the toggle. */
export function variantColor(k: VariantKey): string {
  return VARIANT_COLORS[theme.value][k]
}

/** vid bitfield (bit0 quantum, bit1 relativistic) -> key. */
export function keyOfVid(vid: number): VariantKey {
  return ((vid & 1 ? 'q' : 'c') + (vid & 2 ? 'r' : 'n')) as VariantKey
}
