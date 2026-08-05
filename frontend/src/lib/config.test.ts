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
    // A reset must not park the form in a combination Restart would 422 on.
    // The host default being float32 is no longer one of them (M1 landed
    // 2026-07-27), so a reset now ADOPTS it in 2D exactly as it does in 1D —
    // and auto-expand, the one gate left, must still come back off.
    const { m, cfg } = await load(BASE_2D)
    m.setHostPrecision('float32')
    m.resetToDefaults(cfg)
    expect(cfg.precision).toBe('float32')
    expect(cfg.auto_expand).toBe(false)
  })

  it('leaves every 2D setting alone — there are no ndim gates left', async () => {
    // applyNdimInvariants is GONE (M3, 2026-08-01), and this is the test that
    // notices if anything grows back into its place. It used to strip qr/cr,
    // force float64 and clear auto_expand, to match backend gates retired by
    // M2/M1 (2026-07-27) and M3. If any of them starts again, a 2D user silently
    // loses what they picked and the form stops describing what will be
    // computed — worst for precision, where payload() would put float64 back
    // behind an amber "restart to apply" that no restart could ever clear.
    // FROM THE 1D BASE, and that is load-bearing: setNdim returns at its first
    // line when the config is already at the target ndim, so running it against
    // BASE_2D would assert nothing but that the three assignments below still
    // hold — passing against a function whose body never executed.
    const { m, cfg } = await load(BASE)
    cfg.variants.splice(0, cfg.variants.length, 'qn', 'qr', 'cn', 'cr')
    cfg.precision = 'float64'
    cfg.auto_expand = true
    m.setNdim(cfg, 2)                 // the one path that used to apply them
    expect(cfg.grid.ndim).toBe(2)     // ...and it really ran
    expect(cfg.grid.axes.length).toBe(4)
    expect(cfg.variants).toEqual(['qn', 'qr', 'cn', 'cr'])
    expect(cfg.precision).toBe('float64')
    expect(cfg.auto_expand).toBe(true)
    expect(m.precisionForPayload(cfg)).toBe('float64')   // asking to expand
  })                                                     // IS asking for f64

  it('still clears auto_expand in 2D float32 — a precision gate, not an ndim one',
     async () => {
       // The one refusal M3 did not retire, checked at ndim=2 because that is
       // where the removed gate used to mask it.
       const { m, cfg } = await load(BASE_2D)
       cfg.precision = 'float32'
       cfg.auto_expand = true
       m.applyPrecisionInvariants(cfg)
       expect(cfg.auto_expand).toBe(false)
       expect(cfg.tol).toBeGreaterThanOrEqual(1e-5)
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

  it('defers at ndim=2 exactly as at ndim=1 (M1 landed 2026-07-27)', async () => {
    // This used to send float64 explicitly, because 2D deferred float32 and
    // deferring the field on a float32 host asked for a pair the schema refused
    // — EVERY 2D restart 422'd, behind a form reading float64. With the gate
    // gone, deferring is what lets a float32 host give a 2D session float32, and
    // an explicit float64 here would silently override that host exactly the way
    // a hard-coded default once did in 1D.
    const { m, cfg } = await load(undefined)
    m.setNdim(cfg, 2)
    expect(m.precisionIsUserChosen()).toBe(false)
    expect(m.precisionForPayload(cfg)).toBeNull()
    // ...and a CHOICE is still sent, at ndim=2 as at ndim=1
    cfg.precision = 'float32'
    m.markPrecisionChosen()
    expect(m.precisionForPayload(cfg)).toBe('float32')
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

  it('a dims round trip leaves no value off its own list', async () => {
    // THE BUG: setNdim capped N from above and not from below, so 1D -> 2D -> 1D
    // brought the 2D choice back with it — 64, in a 1D list that starts at 256 —
    // and the select rendered `64, 256, 512, …` with a hole in it while the
    // session ran at 64², a resolution nothing had asked for.
    const { axisSizeOptions, axisFloor, setNdim, defaultConfig } = await mod()
    const cfg = defaultConfig()
    setNdim(cfg, 2, 128)
    cfg.grid.axes.forEach((a) => { a.N = 64 })
    setNdim(cfg, 1, 4096)
    // the 1D DEFAULT, not the 1D floor: 64 is unrepresentable in this list, and
    // snapping to its smallest member would quarter the resolution of anyone who
    // started at the default and merely looked at 2D
    expect(cfg.grid.axes.map((a) => a.N)).toEqual([1024, 1024])
    // ...and the list is the fixed one, with nothing appended to it
    const opts = axisSizeOptions(1, 4096, cfg.grid.axes.map((a) => a.N))
    expect(opts).toEqual([256, 512, 1024, 2048, 4096])
    // both directions land INSIDE the list, which is the property that matters
    for (const [nd, cap] of [[2, 128], [1, 4096]] as const) {
      setNdim(cfg, nd, cap)
      const list = axisSizeOptions(nd, cap)
      for (const a of cfg.grid.axes) expect(list).toContain(a.N)
      expect(Math.min(...cfg.grid.axes.map((a) => a.N)))
        .toBeGreaterThanOrEqual(axisFloor(nd, cap))
    }
  })

  it('a round trip lands on the target default, and that is the ceiling of what it can do', async () => {
    // What a round trip CANNOT do, stated so it is not mistaken for a bug: on a
    // default host the two lists do not overlap (2D tops out at 128, 1D starts
    // at 256), so no 2D choice is representable in 1D and a detour through it
    // lands on the target's own default, both ways. Preserving 32⁴ across 1D
    // would need a per-dimensionality memory in the form — a different feature.
    const { setNdim, defaultConfig, DEFAULT_AXES } = await mod()
    const cfg = defaultConfig()
    setNdim(cfg, 2, 128)
    cfg.grid.axes.forEach((a) => { a.N = 32 })
    setNdim(cfg, 1, 4096)
    expect(cfg.grid.axes.map((a) => a.N))
      .toEqual(DEFAULT_AXES[1].map((a) => a.N))
    setNdim(cfg, 2, 128)
    expect(cfg.grid.axes.map((a) => a.N))
      .toEqual(DEFAULT_AXES[2].map((a) => a.N))
  })

  it('resetting to defaults is 1024² in 1D and 64⁴ in 2D', async () => {
    // 256² was the SMALLEST N the panel offers, which is a strange thing for a
    // default to be — 2D's 64⁴ is mid-list. Changed 2026-08-01.
    const { defaultConfig, setNdim } = await mod()
    const cfg = defaultConfig()
    expect(cfg.grid.axes.map((a) => a.N)).toEqual([1024, 1024])
    setNdim(cfg, 2, 128)
    expect(cfg.grid.axes.map((a) => a.N)).toEqual([64, 64, 64, 64])
  })

  it('the floor is not a flat reset where a small choice IS offerable', async () => {
    // min() is still doing its job in the direction that has room for it: a 1D
    // grid at 32 (reachable by import, not from the select) is a legal 2D choice
    // and must arrive intact rather than being pulled up to the 2D default.
    const { setNdim, defaultConfig } = await mod()
    const cfg = defaultConfig()
    cfg.grid.axes.forEach((a) => { a.N = 32 })
    setNdim(cfg, 2, 128)
    expect(cfg.grid.axes.map((a) => a.N)).toEqual([32, 32, 32, 32])
  })

  it('the floor never pushes N past a lowered WIGNERF_MAX_GRID', async () => {
    // A host at 128 offers exactly [128] in 1D. Landing on the unclamped 256
    // floor there would put the form over the ceiling the same select is drawn
    // from, and POST /sessions would refuse a value nobody typed.
    const { axisSizeOptions, setNdim, defaultConfig } = await mod()
    const cfg = defaultConfig()
    setNdim(cfg, 2, 128)
    cfg.grid.axes.forEach((a) => { a.N = 64 })
    setNdim(cfg, 1, 128)
    expect(cfg.grid.axes.map((a) => a.N)).toEqual([128, 128])
    expect(axisSizeOptions(1, 128)).toEqual([128])
  })
})

/**
 * The two expression IC kinds. Everything here is decidable without a DOM,
 * which is the whole reason it lives in lib/ — vitest runs with `environment:
 * node` and there is no component-test harness at all.
 */
const BASE_2D_GRID = {
  ndim: 2,
  axes: [{ lo: -8, hi: 8, N: 64 }, { lo: -8, hi: 8, N: 64 },
         { lo: -7, hi: 7, N: 64 }, { lo: -7, hi: 7, N: 64 }],
}

describe('expression initial conditions', () => {
  it('gives a fresh copy of both drafts, so the form cannot mutate the defaults', async () => {
    const { m } = await load(undefined)
    const a = m.defaultIC(2)
    a.expr.wexpr = 'MUTATED'
    expect(m.defaultIC(2).expr.wexpr).toBe(m.DEFAULT_IC_EXPR[2].wexpr)
  })

  it('loads an old stored config, which has no ic.expr at all', async () => {
    const { m, cfg } = await load(BASE)
    expect(cfg.ic.type).toBe('mixture')
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[1].wexpr)
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[1].psi)
  })

  it('gives an old stored 2D config the 2D expressions, not the 1D ones', async () => {
    // loadConfig builds defaultConfig() at ndim=1 ALWAYS and merges the stored
    // grid over it, so without conformICExprToNdim every existing user's first
    // 2D page load would sit on a W(x,p) in a four-variable world.
    const { m, cfg } = await load({
      ...BASE, grid: BASE_2D_GRID,
      ic: { type: 'mixture',
            components: [{ q0: [2, 0], k0: [0, 1], sigma_q: [0.7, 0.7],
                           sigma_k: [0.7, 0.7], weight: 1, phase: 0 }] },
    })
    expect(cfg.grid.ndim).toBe(2)
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[2].wexpr)
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[2].psi)
  })

  it('merges a type and an expr that arrive without components', async () => {
    // `type` used to sit INSIDE the components guard, so this document merged
    // as a mixture and its expression was dropped on the floor.
    const { cfg } = await load({ ...BASE, ic: { type: 'psi', expr: { psi: 'exp(-x^2)' } } })
    expect(cfg.ic.type).toBe('psi')
    expect(cfg.ic.expr.psi).toBe('exp(-x^2)')
  })

  it('merges one draft without blanking the other', async () => {
    const { m, cfg } = await load({ ...BASE, ic: { type: 'psi', expr: { psi: 'x' } } })
    expect(cfg.ic.expr.psi).toBe('x')
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[1].wexpr)
  })

  it('falls back to mixture for an unknown stored type', async () => {
    const { cfg } = await load({ ...BASE, ic: { ...BASE.ic, type: 'nonsense' } })
    expect(cfg.ic.type).toBe('mixture')
  })
})

describe('setNdim and the expression drafts', () => {
  it('adopts the target ndim expressions when both are untouched', async () => {
    const { m, cfg } = await load(undefined)
    m.setNdim(cfg, 2)
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[2].wexpr)
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[2].psi)
    m.setNdim(cfg, 1)
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[1].wexpr)
  })

  it('carries a hand-written expression over instead', async () => {
    const { m, cfg } = await load(undefined)
    cfg.ic.expr.psi = 'exp(-x^2/2)*cos(3*x)'
    m.setNdim(cfg, 2)
    expect(cfg.ic.expr.psi).toBe('exp(-x^2/2)*cos(3*x)')
  })

  it('swaps the INACTIVE draft too', async () => {
    // A 1D-only W(x,p) left behind an unselected tab in a 2D form is
    // valid-looking and dead — the failure a "swap only the active one"
    // shortcut produces.
    const { m, cfg } = await load(undefined)
    cfg.ic.type = 'psi'
    cfg.ic.expr.psi = 'MINE'
    m.setNdim(cfg, 2)
    expect(cfg.ic.expr.psi).toBe('MINE')
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[2].wexpr)
  })
})

describe('importing an expression setup document', () => {
  it('imports one that has no components at all', async () => {
    const { m, cfg } = await load(undefined)
    m.importConfig(cfg, { config: { ...BASE, ic: { type: 'wexpr',
                                                   expr: { wexpr: 'exp(-x^2-p^2)' } } } })
    expect(cfg.ic.type).toBe('wexpr')
    expect(cfg.ic.expr.wexpr).toBe('exp(-x^2-p^2)')
    expect(cfg.ic.components.length).toBeGreaterThan(0)   // defaults stay usable
  })

  it('refuses an expression kind whose own expression is missing', async () => {
    const { m, cfg } = await load(undefined)
    expect(() => m.importConfig(cfg, { config: { ...BASE, ic: { type: 'psi' } } }))
      .toThrow(/ic\.expr\.psi is missing/)
  })

  it('reshapes an expression IC components rather than refusing them', async () => {
    // They are inert for this kind, so a dimensionality mismatch in them is no
    // reason to reject an otherwise perfect document...
    const { m, cfg } = await load(undefined)
    m.importConfig(cfg, { config: {
      ...BASE, grid: BASE_2D_GRID,
      ic: { type: 'psi', expr: { psi: 'exp(-x^2-y^2)' },
            components: [{ x0: 1, p0: 0, sigma_x: 0.5, sigma_p: 0.5 }] } } })
    expect(cfg.ic.components[0]!.q0).toHaveLength(2)
  })

  it('...but still refuses a MIXTURE whose components are the wrong ndim', async () => {
    const { m, cfg } = await load(undefined)
    expect(() => m.importConfig(cfg, { config: {
      ...BASE, grid: BASE_2D_GRID,
      ic: { type: 'mixture',
            components: [{ x0: 1, p0: 0, sigma_x: 0.5, sigma_p: 0.5 }] } } }))
      .toThrow(/grid is 2D but the IC components are 1D/)
  })

  it('still refuses an unknown type', async () => {
    const { m, cfg } = await load(undefined)
    expect(() => m.importConfig(cfg, { config: { ...BASE, ic: { type: 'zzz' } } }))
      .toThrow(/unknown IC type/)
  })
})

describe('reset restores both expressions at the current dimensionality', () => {
  it('resets a 2D setup to the 2D expressions', async () => {
    const { m, cfg } = await load(undefined)
    m.setNdim(cfg, 2)
    cfg.ic.type = 'wexpr'
    cfg.ic.expr.wexpr = 'junk'
    cfg.ic.expr.psi = 'more junk'
    m.resetToDefaults(cfg)
    expect(cfg.ic.type).toBe('mixture')
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[2].wexpr)
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[2].psi)
  })
})

describe('an expression a document actually supplied is never repointed', () => {
  /**
   * conformICExprToNdim's "still at the OTHER ndim's default" is a proxy for
   * "untouched", and it is a good one ONLY on an explicit dimensionality
   * switch. Everywhere else it is wrong, because DEFAULT_IC_EXPR[1] is itself a
   * legal 2D expression — its free symbols are a subset of {x, y} — so the test
   * cannot tell "you never touched this" from "you typed exactly this".
   */
  it('a stored 2D config holding the 1D default psi keeps it across a reload',
     async () => {
    // read the constant before load()'s resetModules; the string is the same
    const one = (await import('./icKinds')).DEFAULT_IC_EXPR[1].psi
    const { m, cfg } = await load({
      grid: { ndim: 2, axes: [{ lo: -8, hi: 8, N: 32 }, { lo: -8, hi: 8, N: 32 },
                              { lo: -9, hi: 9, N: 32 }, { lo: -9, hi: 9, N: 32 }] },
      potential: '(x^2+y^2)/2',
      ic: { type: 'psi', components: [], expr: { psi: one, wexpr: 'exp(-x^2)' } },
      variants: ['qn'], mode: 'interactive', record_dt: 0.05,
    })
    expect(cfg.grid.ndim).toBe(2)
    expect(cfg.ic.expr.psi).toBe(one)          // NOT rewritten to the 2D default
    expect(cfg.ic.expr.wexpr).toBe('exp(-x^2)')
  })

  it('an IMPORT keeps the expression it carried, in the bare-string spelling',
     async () => {
    // The bare string is the spelling a setup document and an mp4 comment tag
    // use (icPayload sends one expression, not both), so it is the spelling an
    // import actually takes — and it was the one with no test.
    const { m, cfg } = await load(undefined)
    const one = m.DEFAULT_IC_EXPR[1].psi
    m.importConfig(cfg, { config: {
      grid: { ndim: 2, axes: [{ lo: -8, hi: 8, N: 32 }, { lo: -8, hi: 8, N: 32 },
                              { lo: -9, hi: 9, N: 32 }, { lo: -9, hi: 9, N: 32 }] },
      potential: '(x^2+y^2)/2',
      ic: { type: 'psi', expr: one },
      variants: ['qn'], mode: 'interactive', record_dt: 0.05 } })
    expect(cfg.ic.type).toBe('psi')
    expect(cfg.ic.expr.psi).toBe(one)
  })

  it('...but a draft the document did NOT carry is still repointed', async () => {
    // The convenience the rule exists for survives: a document with no expr at
    // all (everything before this feature) must not leave a 2D form holding a
    // W(x,p) behind a tab the user has not opened.
    const { m, cfg } = await load(undefined)
    m.importConfig(cfg, { config: {
      grid: { ndim: 2, axes: [{ lo: -8, hi: 8, N: 32 }, { lo: -8, hi: 8, N: 32 },
                              { lo: -9, hi: 9, N: 32 }, { lo: -9, hi: 9, N: 32 }] },
      potential: '(x^2+y^2)/2',
      ic: { type: 'mixture', components: [
        { q0: [1, 0], k0: [0, 0], sigma_q: [0.5, 0.5], sigma_k: [0.5, 0.5] }] },
      variants: ['qn'], mode: 'interactive', record_dt: 0.05 } })
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[2].psi)
    expect(cfg.ic.expr.wexpr).toBe(m.DEFAULT_IC_EXPR[2].wexpr)
  })

  it('an explicit setNdim still repoints an untouched default', async () => {
    // The one place the proxy IS right: the user just asked for the switch.
    const { m, cfg } = await load(undefined)
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[1].psi)
    m.setNdim(cfg, 2)
    expect(cfg.ic.expr.psi).toBe(m.DEFAULT_IC_EXPR[2].psi)
  })
})
