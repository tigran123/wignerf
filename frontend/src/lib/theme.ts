/**
 * Light/dark theme. LIGHT is the default; the preference persists in
 * localStorage under `wignerf.theme`, a sibling of the other display-only
 * keys (`wignerf.layout`, `wignerf.grid`) and therefore untouched by "Reset
 * setup to defaults".
 *
 * This is a module singleton rather than a prop like `showGrid`, because the
 * theme is needed by non-component code too (the uPlot option builders) and
 * by half a dozen components at once — prop-drilling it through the layout
 * would buy nothing.
 *
 * The VALUES live in style.css, once per theme. Nothing here duplicates them:
 * `chartPalette()` reads the custom properties back off the document, so a
 * colour is changed in exactly one place. (backend/core/render_mpl.py keeps
 * its own copy for the mp4 export — it cannot read our stylesheet.)
 */
import { ref, watch } from 'vue'

export type ThemeName = 'light' | 'dark'

const KEY = 'wignerf.theme'

function stored(): ThemeName {
  try {
    return localStorage.getItem(KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export const theme = ref<ThemeName>(stored())

/** True while the dark theme is active — for the odd `v-if` on an icon. */
export function isDark(): boolean {
  return theme.value === 'dark'
}

/**
 * Both setters apply the root class SYNCHRONOUSLY, and that is not
 * redundant with the watcher below. `chartPalette()` reads the --wf-chart-*
 * properties off the document and CACHES the answer per theme, so a single
 * read taken before the class lands would pin the wrong palette for the rest
 * of the page's life — not a one-frame glitch. Today the watcher happens to
 * win the race (a watcher created outside a component has no job id, so its
 * pre-flush job sorts ahead of the components' chart rebuilds), but that is
 * an ordering detail of Vue's scheduler and nothing here should depend on it.
 * `apply` is idempotent, so paying it twice costs a classList.toggle.
 */
export function setTheme(v: ThemeName): void {
  theme.value = v
  apply(v)
}

export function toggleTheme(): void {
  setTheme(theme.value === 'dark' ? 'light' : 'dark')
}

/** Apply to <html>. index.html does this inline for the FIRST paint (a
 *  deferred module runs too late and a dark user would see a white flash);
 *  this keeps it in step afterwards. Both are idempotent.
 *  The DOM guard is not decoration: variants.ts imports this module, so it is
 *  reachable from pure-logic code that the unit tests load without a DOM. */
function apply(v: ThemeName): void {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle('dark', v === 'dark')
}

apply(theme.value)
watch(theme, (v) => {
  apply(v)
  try {
    localStorage.setItem(KEY, v)
  } catch { /* private mode / quota — the theme still applies for this page */ }
})

/** Colours the uPlot charts need at CONSTRUCTION time (axis/grid strokes are
 *  not settable afterwards, which is why the charts destroy+rebuild on a
 *  theme change, exactly as they already do for the grid-lines toggle). */
export interface ChartPalette {
  axis: string
  tick: string
  grid: string
  gridSoft: string
  text: string
  cursor: string
}

const FALLBACK: Record<ThemeName, ChartPalette> = {
  light: { axis: '#525252', tick: '#a3a3a3', grid: '#d4d4d8',
           gridSoft: '#f4f4f5', text: '#404040', cursor: '#db2777' },
  dark: { axis: '#a3a3a3', tick: '#525252', grid: '#3f3f46',
          gridSoft: '#26262666', text: '#d4d4d4', cursor: '#f472b6' },
}

const cache = new Map<ThemeName, ChartPalette>()

/**
 * The current theme's chart colours, read from the CSS custom properties.
 * Must be called AFTER the root class is applied — hence a function called
 * per chart build, never a module constant. One getComputedStyle per rebuild,
 * never per frame. Falls back to literals where the properties are missing
 * (jsdom returns '' for anything a real stylesheet would have supplied).
 */
export function chartPalette(): ChartPalette {
  const t = theme.value
  const hit = cache.get(t)
  if (hit) return hit
  const fb = FALLBACK[t]
  let p = fb
  try {
    if (typeof document === 'undefined') throw new Error('no DOM')
    const cs = getComputedStyle(document.documentElement)
    const v = (name: string, dflt: string) =>
      cs.getPropertyValue(name).trim() || dflt
    p = {
      axis: v('--wf-chart-axis', fb.axis),
      tick: v('--wf-chart-tick', fb.tick),
      grid: v('--wf-chart-grid', fb.grid),
      gridSoft: v('--wf-chart-grid-soft', fb.gridSoft),
      text: v('--wf-chart-text', fb.text),
      cursor: v('--wf-cursor', fb.cursor),
    }
  } catch { /* no DOM (unit tests) — the literals above are correct */ }
  cache.set(t, p)
  return p
}
