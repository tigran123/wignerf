import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

/**
 * The theme preference. Three things are load-bearing and none of them is
 * visible from the rendered page: LIGHT is the default (the user's general
 * preference, and index.html only adds a class for the other one), a stored
 * choice survives a reload, and the root class follows the ref — that class is
 * what every --wf-* token and therefore every colour in the app hangs off.
 *
 * Module-level state (the ref, the palette cache) means each case needs a
 * fresh module instance, hence resetModules + dynamic import.
 */
function stubEnv(stored: string | null) {
  const store = new Map<string, string>()
  if (stored !== null) store.set('wignerf.theme', stored)
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => { store.set(k, v) },
    removeItem: (k: string) => { store.delete(k) },
    clear: () => { store.clear() },
  })
  const classes = new Set<string>()
  vi.stubGlobal('document', {
    documentElement: {
      classList: {
        toggle: (c: string, on: boolean) => {
          if (on) classes.add(c)
          else classes.delete(c)
        },
      },
    },
  })
  return { store, classes }
}

async function load(stored: string | null) {
  vi.resetModules()
  const env = stubEnv(stored)
  return { m: await import('./theme'), ...env }
}

beforeEach(() => vi.unstubAllGlobals())

describe('theme', () => {
  it('defaults to light with nothing stored', async () => {
    const { m, classes } = await load(null)
    expect(m.theme.value).toBe('light')
    expect(classes.has('dark')).toBe(false)
  })

  it('honours a stored dark preference', async () => {
    const { m, classes } = await load('dark')
    expect(m.theme.value).toBe('dark')
    expect(classes.has('dark')).toBe(true)
  })

  it('ignores a stored value it does not recognise', async () => {
    const { m } = await load('solarized')
    expect(m.theme.value).toBe('light')
  })

  it('persists the toggle and flips the root class', async () => {
    const { m, store, classes } = await load(null)
    m.toggleTheme()
    await nextTick()
    expect(m.theme.value).toBe('dark')
    expect(store.get('wignerf.theme')).toBe('dark')
    expect(classes.has('dark')).toBe(true)
    m.toggleTheme()
    await nextTick()
    expect(store.get('wignerf.theme')).toBe('light')
    expect(classes.has('dark')).toBe(false)
  })

  it('serves a distinct chart palette per theme, with no DOM to read', async () => {
    const { m } = await load(null)
    const light = m.chartPalette()
    m.setTheme('dark')
    const dark = m.chartPalette()
    for (const k of ['axis', 'tick', 'grid', 'gridSoft', 'text', 'cursor'] as const)
      expect(dark[k], k).not.toBe(light[k])
  })

  it('gives every variant its own colour in both themes', async () => {
    vi.resetModules()
    stubEnv(null)
    const { VARIANT_COLORS } = await import('./variants')
    for (const t of ['light', 'dark'] as const) {
      const hues = Object.values(VARIANT_COLORS[t])
      expect(hues).toHaveLength(4)
      expect(new Set(hues).size).toBe(4)
    }
    // the two sets must actually differ, or the light theme is wearing the
    // dark theme's washed-out *-400 shades
    for (const k of ['qn', 'qr', 'cn', 'cr'] as const)
      expect(VARIANT_COLORS.light[k], k).not.toBe(VARIANT_COLORS.dark[k])
  })
})
