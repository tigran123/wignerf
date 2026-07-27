import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The precision-provenance rules. These decide whether the create payload
 * carries `precision` at all, and that is the only thing standing between a
 * host configured WIGNERF_PRECISION=float32 and a SPA that silently overrides
 * it with a hard-coded float64 — which is exactly what happened while the
 * form treated its placeholder as a decision.
 *
 * Module-level state (the host default, the chosen flag) means each case needs
 * a fresh module instance, hence resetModules + dynamic import.
 */
function stubStorage(initial: string | null) {
  const store = new Map<string, string>()
  if (initial !== null) store.set('wignerf.cfg', initial)
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => { store.set(k, v) },
    removeItem: (k: string) => { store.delete(k) },
    clear: () => { store.clear() },
  })
}

async function load(stored: unknown) {
  vi.resetModules()
  stubStorage(stored === undefined ? null : JSON.stringify(stored))
  const m = await import('./config')
  return { m, cfg: m.loadConfig() }
}

const BASE = {
  grid: { x1: -6, x2: 6, Nx: 64, p1: -7, p2: 7, Np: 64 },
  potential: 'x^2/2',
  ic: { type: 'mixture', components: [{ x0: 2, p0: 0, sigma_x: 0.7, sigma_p: 0.8 }] },
  variants: ['qn'], mode: 'interactive', record_dt: 0.05,
}

beforeEach(() => { vi.unstubAllGlobals() })

describe('precision provenance', () => {
  it('is unchosen on a first-ever load, so the host decides', async () => {
    const { m } = await load(undefined)
    expect(m.precisionIsUserChosen()).toBe(false)
  })

  it('is unchosen for a setup stored before the field existed', async () => {
    const { m } = await load(BASE)
    expect(m.precisionIsUserChosen()).toBe(false)
  })

  it('is chosen when the stored setup carries one', async () => {
    const { m, cfg } = await load({ ...BASE, precision: 'float64' })
    expect(m.precisionIsUserChosen()).toBe(true)
    expect(cfg.precision).toBe('float64')
  })

  it('becomes chosen when the user operates the control', async () => {
    const { m } = await load(undefined)
    m.markPrecisionChosen()
    expect(m.precisionIsUserChosen()).toBe(true)
  })

  it('reset-to-defaults un-chooses and restores the HOST default', async () => {
    const { m, cfg } = await load({ ...BASE, precision: 'float32' })
    expect(m.precisionIsUserChosen()).toBe(true)
    m.setHostPrecision('float32')          // as the /device probe would
    m.resetToDefaults(cfg)
    expect(m.precisionIsUserChosen()).toBe(false)
    expect(cfg.precision).toBe('float32')  // the host's, not a literal float64
  })

  it('setHostPrecision ignores junk rather than propagating it', async () => {
    const { m, cfg } = await load(undefined)
    m.setHostPrecision('flaot32')
    m.setHostPrecision(undefined)
    m.resetToDefaults(cfg)
    expect(cfg.precision).toBe('float64')
  })

  it('an IMPORTED precision counts as a choice, so it is sent', async () => {
    // Otherwise the payload omits it, the session runs at the host default,
    // and the form shows a float32 that never happened behind a "restart to
    // apply" no restart can clear (status.precision never CHANGES, so the
    // sync watcher never fires). This is the setup-document round trip.
    const { m, cfg } = await load(undefined)
    expect(m.precisionIsUserChosen()).toBe(false)
    m.importConfig(cfg, { config: { ...BASE, precision: 'float32' } })
    expect(cfg.precision).toBe('float32')
    expect(m.precisionIsUserChosen()).toBe(true)
    expect(m.precisionForPayload(cfg)).toBe('float32')
  })

  it('and still counts after "reset to defaults" un-chose it', async () => {
    const { m, cfg } = await load({ ...BASE, precision: 'float64' })
    m.resetToDefaults(cfg)                      // un-chooses, by design
    m.importConfig(cfg, { config: { ...BASE, precision: 'float32' } })
    expect(m.precisionForPayload(cfg)).toBe('float32')
  })

  it('an OLD file without the key leaves the host deciding', async () => {
    const { m, cfg } = await load(undefined)
    m.importConfig(cfg, { config: BASE })
    expect(m.precisionIsUserChosen()).toBe(false)
    expect(m.precisionForPayload(cfg)).toBeNull()
  })
})

describe('reset keeps the dimensionality', () => {
  const BASE_2D = {
    grid: {
      ndim: 2,
      axes: [{ lo: -8, hi: 8, N: 32 }, { lo: -8, hi: 8, N: 32 },
             { lo: -9, hi: 9, N: 32 }, { lo: -9, hi: 9, N: 32 }],
    },
    potential: 'x^2*y',
    ic: { type: 'cat', components: [
      { q0: [3, 1], k0: [0, 0], sigma_q: [0.4, 0.4], sigma_k: null }] },
    variants: ['qn'], mode: 'interactive', record_dt: 0.05,
  }

  it('resets a 2D setup to the 2D defaults, not to 1D', async () => {
    const { m, cfg } = await load(BASE_2D)
    expect(cfg.grid.ndim).toBe(2)
    m.resetToDefaults(cfg)
    // ndim is the choice of PROBLEM: a reset replaces the setup WITHIN it
    expect(cfg.grid.ndim).toBe(2)
    expect(cfg.grid.axes.length).toBe(4)
    expect(cfg.grid.axes).toEqual(m.DEFAULT_AXES[2])
    expect(cfg.potential).toBe(m.DEFAULT_POTENTIAL[2])
    // ...and the IC is the 2D one, with two coordinates per field
    expect(cfg.ic.type).toBe('mixture')
    expect(cfg.ic.components.length).toBe(1)
    const c = cfg.ic.components[0]!
    expect(c.q0.length).toBe(2)
    expect(c.k0.length).toBe(2)
    expect(c.sigma_q.length).toBe(2)
    expect(c.sigma_k?.length).toBe(2)
    // the 2D default moves in BOTH dimensions, so <Lz> is a nonzero constant
    expect(c.k0[1]).not.toBe(0)
  })

  it('still resets a 1D setup to the 1D defaults', async () => {
    const { m, cfg } = await load(BASE)
    m.resetToDefaults(cfg)
    expect(cfg.grid.ndim).toBe(1)
    expect(cfg.grid.axes).toEqual(m.DEFAULT_AXES[1])
    expect(cfg.potential).toBe(m.DEFAULT_POTENTIAL[1])
    expect(cfg.ic.components[0]!.q0.length).toBe(1)
  })

  it('leaves a reset 2D setup in a state the API accepts', async () => {
    // the host default may be float32, which ndim=2 refuses (M1) — a reset
    // must not park the form in the one combination Restart would 422 on
    const { m, cfg } = await load(BASE_2D)
    m.setHostPrecision('float32')
    m.resetToDefaults(cfg)
    expect(cfg.precision).toBe('float64')
    expect(cfg.auto_expand).toBe(false)
    expect(cfg.variants.every((v) => v === 'qn' || v === 'cn')).toBe(true)
  })

  it('gives the IC editor the right default for its dimensionality', async () => {
    const { m } = await load(undefined)
    expect(m.defaultIC(1).components[0]!.q0.length).toBe(1)
    expect(m.defaultIC(2).components[0]!.q0.length).toBe(2)
    // and a fresh copy each time — the form mutates these in place
    const a = m.defaultIC(2)
    a.components[0]!.q0[0] = 99
    expect(m.defaultIC(2).components[0]!.q0[0]).not.toBe(99)
  })
})

describe('precisionForPayload', () => {
  it('omits an unchosen precision so the host decides', async () => {
    const { m, cfg } = await load(undefined)
    expect(m.precisionForPayload(cfg)).toBeNull()
  })

  it('sends a chosen one', async () => {
    const { m, cfg } = await load({ ...BASE, precision: 'float32' })
    expect(m.precisionForPayload(cfg)).toBe('float32')
  })

  it('sends float64 explicitly when auto-expand is on but unchosen', async () => {
    // The float32-host + failed-/device-probe trap: deferring the precision
    // while asking for auto-expand requests a pair the schema refuses, and
    // every session create 422s. auto-expand IS a request for float64.
    const { m, cfg } = await load({ ...BASE, auto_expand: true })
    expect(m.precisionIsUserChosen()).toBe(false)
    expect(cfg.auto_expand).toBe(true)
    expect(m.precisionForPayload(cfg)).toBe('float64')
  })

  it('goes back to deferring once auto-expand is off', async () => {
    const { m, cfg } = await load({ ...BASE, auto_expand: true })
    cfg.auto_expand = false
    expect(m.precisionForPayload(cfg)).toBeNull()
  })

  it('sends float64 explicitly at ndim=2, chosen or not', async () => {
    // Same trap as auto-expand, and it took the whole 2D mode down with it: 2D
    // defers float32 (M1), so deferring the field on a float32 host asks for a
    // pair the schema refuses and EVERY 2D restart 422s — behind a form that
    // reads float64, because applyNdimInvariants has already forced it there.
    const { m, cfg } = await load(undefined)
    m.setNdim(cfg, 2)
    expect(m.precisionIsUserChosen()).toBe(false)
    expect(cfg.precision).toBe('float64')
    expect(m.precisionForPayload(cfg)).toBe('float64')
    // ...and it does not depend on the invariants having run: a config that
    // reached ndim=2 with a stale float32 still sends float64
    cfg.precision = 'float32'
    expect(m.precisionForPayload(cfg)).toBe('float64')
  })

  it('still defers at ndim=1 after a round trip through 2D', async () => {
    const { m, cfg } = await load(undefined)
    m.setNdim(cfg, 2)
    m.setNdim(cfg, 1)
    expect(m.precisionForPayload(cfg)).toBeNull()
  })
})

describe('float32 invariants', () => {
  it('clears auto_expand — the backend refuses the pair', async () => {
    const { m, cfg } = await load({ ...BASE, precision: 'float32', auto_expand: true })
    expect(cfg.precision).toBe('float32')
    expect(cfg.auto_expand).toBe(false)
  })

  it('holds when precision is adopted AFTER the merge', async () => {
    // the host-default path: mergeConfig already ran under float64
    const { m, cfg } = await load({ ...BASE, auto_expand: true })
    expect(cfg.auto_expand).toBe(true)
    cfg.precision = 'float32'
    m.applyPrecisionInvariants(cfg)
    expect(cfg.auto_expand).toBe(false)
  })

  it('leaves auto_expand alone in float64', async () => {
    const { m, cfg } = await load({ ...BASE, precision: 'float64', auto_expand: true })
    m.applyPrecisionInvariants(cfg)
    expect(cfg.auto_expand).toBe(true)
  })

  it('raises a sub-floor tol — the other combination the backend refuses', async () => {
    // A stored tol of 1e-8 is perfectly good in float64 and unreachable in
    // float32 (adjust_step's full-vs-two-half-steps residual has a ~7e-7
    // roundoff floor there), so it 422s at create and is popped on the live
    // path. The form must not be able to hold it.
    const { m, cfg } = await load({ ...BASE, precision: 'float32', tol: 1e-8 })
    expect(cfg.tol).toBe(m.TOL_MIN_F32)
  })

  it('leaves a tol at or above the floor untouched', async () => {
    const { cfg } = await load({ ...BASE, precision: 'float32', tol: 0.01 })
    expect(cfg.tol).toBe(0.01)
  })

  it('leaves a sub-floor tol alone in float64', async () => {
    const { cfg } = await load({ ...BASE, precision: 'float64', tol: 1e-8 })
    expect(cfg.tol).toBe(1e-8)
  })
})

/**
 * setNdim's BOX rule. Switching dims must not hand back a default that trips
 * the boundary watch on its first Restart — which is what carrying the 1D box
 * into 2D did, because the edge band is max(4, N/32) CELLS and at N=64 the
 * 4-cell floor is 0.750 a.u. wide, i.e. 4.60σ from the default packet at x0=2
 * (measured band mass 3.78e-06 against a 1e-6 trigger). A box the user chose
 * still carries over: widening someone's domain silently is worse than warning.
 */
describe('setNdim box handling', () => {
  async function mod() {
    vi.resetModules()
    stubStorage(null)
    return await import('./config')
  }

  it('adopts the target ndim default box when the box was untouched', async () => {
    const { defaultConfig, setNdim, DEFAULT_AXES } = await mod()
    const c = defaultConfig(1)
    setNdim(c, 2)
    expect(c.grid.ndim).toBe(2)
    expect(c.grid.axes.map((a) => [a.lo, a.hi]))
      .toEqual(DEFAULT_AXES[2].map((a) => [a.lo, a.hi]))
    // and back again
    setNdim(c, 1)
    expect(c.grid.axes.map((a) => [a.lo, a.hi]))
      .toEqual(DEFAULT_AXES[1].map((a) => [a.lo, a.hi]))
  })

  it('carries a user-chosen box over instead, mirroring x onto y', async () => {
    const { defaultConfig, setNdim } = await mod()
    const c = defaultConfig(1)
    c.grid.axes[0]!.lo = -20
    c.grid.axes[0]!.hi = 20
    setNdim(c, 2)
    expect(c.grid.axes.map((a) => [a.lo, a.hi]))
      .toEqual([[-20, 20], [-20, 20], [-7, 7], [-7, 7]])
  })

  it('keeps N as the user chose it, capped at the target default', async () => {
    const { defaultConfig, setNdim, DEFAULT_AXES } = await mod()
    const c = defaultConfig(1)
    setNdim(c, 2)
    // 1D's 256 must not become 256^4 = 4.3e9 cells
    expect(c.grid.axes.every((a) => a.N === DEFAULT_AXES[2][0]!.N)).toBe(true)
    const small = defaultConfig(1)
    small.grid.axes.forEach((a) => { a.N = 32 })
    setNdim(small, 2)
    expect(small.grid.axes.every((a) => a.N === 32)).toBe(true)
  })
})

/**
 * The per-axis N choices. Extracted from SetupPanel precisely so this test can
 * exist: both of its bugs were reachable only through the DOM, and both were
 * caused by feeding it the ceiling of the ndim that is RUNNING rather than of the
 * ndim the form is SHOWING (they differ for as long as a restart-only `dims`
 * switch waits for its restart).
 */
describe('axisSizeOptions', () => {
  async function mod() {
    vi.resetModules()
    stubStorage(null)
    return await import('./config')
  }

  it('offers the 1D powers of two up to the host cap', async () => {
    const { axisSizeOptions } = await mod()
    expect(axisSizeOptions(1, 4096, [256, 256]))
      .toEqual([256, 512, 1024, 2048, 4096])
  })

  it('starts 2D at 32, never 16', async () => {
    // boundary._band_mass reports nothing below 32 cells per axis (the edge band
    // would cover a quarter of the axis), so a 16^4 session has no boundary
    // watch and says so nowhere. Every grid on offer must have a working one.
    const { axisSizeOptions } = await mod()
    expect(axisSizeOptions(2, 128, [64, 64, 64, 64])).toEqual([32, 64, 128])
  })

  it('never offers an N the API would refuse for THIS ndim', async () => {
    // The 2D ceiling is WIGNERF_MAX_GRID_2D = 128. Handing this the running 1D
    // session's 4096 offered N up to 4096 and POST /sessions 422'd on a value
    // the form had put in the select.
    const { axisSizeOptions } = await mod()
    expect(Math.max(...axisSizeOptions(2, 128, [64]))).toBe(128)
  })

  it('never returns an empty list when the cap is below the start', async () => {
    // Two ways to get here: a form back at 1D while a 2D session runs (cap 128,
    // 1D start 256 — the loop never ran and the select collapsed to its current
    // value alone), and a host that lowered WIGNERF_MAX_GRID, which CLAUDE.md
    // recommends on VRAM-constrained hosts.
    const { axisSizeOptions } = await mod()
    expect(axisSizeOptions(1, 128, [])).toEqual([128])
    expect(axisSizeOptions(1, 128, [64])).toEqual([64, 128])
    expect(axisSizeOptions(1, 0, [])).toEqual([16])
  })

  it('keeps the form current values listed and sorted even over cap', async () => {
    // an imported oversized setup must render its own value, not a blank select;
    // the API refuses it at Restart with a message naming the ceiling
    const { axisSizeOptions } = await mod()
    expect(axisSizeOptions(2, 128, [256, 48])).toEqual([32, 48, 64, 128, 256])
  })
})
