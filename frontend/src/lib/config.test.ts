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
