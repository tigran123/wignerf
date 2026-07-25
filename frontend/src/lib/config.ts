/** Session configuration shared by the setup panels and the view. */

import { C_AU } from './units'
import type { VariantKey } from './variants'

export interface GridCfg {
  x1: number
  x2: number
  Nx: number
  p1: number
  p2: number
  Np: number
}

export interface ICComponentCfg {
  x0: number
  p0: number
  sigma_x: number
  sigma_p: number | null
  weight: number
  phase: number
}

export interface ICCfg {
  type: 'mixture' | 'cat'
  components: ICComponentCfg[]
}

export interface SimConfig {
  grid: GridCfg
  potential: string
  ic: ICCfg
  variants: VariantKey[]
  mass: number
  c: number
  hbar_eff: number
  tol: number
  record_dt: number
  delay: number
  mode: 'interactive' | 'batch'
  t2: number
  // boundary watch response: detection always runs server-side; this only
  // decides whether the domain auto-moves/doubles (live-toggleable)
  auto_expand: boolean
  // Spectral working precision. float32 is an explicit PREVIEW mode: ~3.3-3.8x
  // faster and ~58% of the working set on CUDA, at the cost of the diagnostics
  // (purity/energy drift ~1e-4). Restart-only, and refused together with
  // auto_expand — see the backend's protocol.MSG_EXPAND_F32.
  precision: 'float64' | 'float32'
  // Per-session overrides of the host's defaults. '' / 0 mean "use the host's"
  // (WIGNERF_DEVICE / WIGNERF_HISTORY_MB); payload() strips them.
  device: string
  history_mb: number
}

/** The subset of SimConfig that applies LIVE (status reports it back): the
 *  setup form marks a field that differs from it as edited-but-not-applied. */
export type LivePhysics = Pick<SimConfig,
  'potential' | 'mass' | 'c' | 'hbar_eff' | 'tol'>

/** The RUN settings of the SESSION that is actually running, from `status`.
 *  These are SessionCreate-only (absent from ParamChange), so a form value
 *  that differs from these is not pending — it is inert until a restart, and
 *  the panel marks it amber to say so. */
export type LiveRun = Pick<SimConfig, 'mode' | 'record_dt' | 'precision'>
  // t2 is NULL in interactive mode — not merely absent. A form t2 sitting
  // beside a live null is the exact mismatch this type exists to expose, so
  // it must not be narrowed to SimConfig's `number`.
  & { t2: number | null }
  // `precision` is here for the same reason as `mode`, with a worse payload.
  // The 2026-07-23 incident was a run believed to be "batch, t₂=100" that was
  // really the previous interactive session; a form reading float64 over a
  // session actually computing in float32 is that trap carrying a physics
  // claim — the E/ΔX·ΔP/purity curves on screen would be preview-grade and
  // nothing would say so.

const STORAGE_KEY = 'wignerf.cfg'
const ALL_KEYS = ['qn', 'qr', 'cn', 'cr'] as const

/**
 * The HOST's default precision (WIGNERF_PRECISION, reported by GET /device).
 * The form cannot know it synchronously — `loadConfig()` runs at module setup,
 * long before any request — so it starts at the safe value and the view
 * installs the real one when the probe returns. `defaultConfig()` reads it, so
 * "Reset setup to defaults" restores the HOST's default rather than a literal
 * that would silently disagree with the server on a float32 host.
 */
let hostPrecision: SimConfig['precision'] = 'float64'
let precisionWasStored = false

export function setHostPrecision(p: unknown) {
  if (p === 'float64' || p === 'float32') hostPrecision = p
}

/**
 * Whether the precision in the form is the USER's choice rather than a
 * placeholder. False on a first-ever load (and for a setup stored before the
 * field existed), true once they pick one — or once a stored config carries
 * it, so someone who deliberately chose float64 on a float32 host is not
 * overridden on every reload.
 *
 * This is what decides whether `precision` is SENT at all. An unchosen form
 * omits it, which is the only way the host's WIGNERF_PRECISION can win when
 * the SPA could not read it: the probe can fail, and a hard-coded float64 sent
 * as though it were a decision would silently override a float32 host.
 */
export function precisionIsUserChosen() {
  return precisionWasStored
}

/** Call when the user actually operates the precision control — NOT for the
 *  programmatic adoptions (host probe, status sync), which must stay
 *  overridable by the server. */
export function markPrecisionChosen() {
  precisionWasStored = true
}

/**
 * Merge a loosely-typed config (localStorage, an imported setup file, an
 * mp4's metadata) into `target`, IN PLACE — the view holds the config in a
 * long-lived reactive object, so nested objects/arrays must be mutated, not
 * replaced, for existing bindings to keep working. Unknown keys and fields
 * of the wrong shape are ignored; `t2: null` (an interactive session's wire
 * form) leaves the form's value alone.
 */
export function mergeConfig(target: SimConfig, s: unknown) {
  if (!s || typeof s !== 'object') return
  const src = s as Record<string, any>
  if (src.grid && typeof src.grid === 'object') Object.assign(target.grid, src.grid)
  if (Array.isArray(src.ic?.components) && src.ic.components.length) {
    target.ic.type = src.ic.type === 'cat' ? 'cat' : 'mixture'
    target.ic.components.splice(0, target.ic.components.length,
                                ...src.ic.components.map(
                                  (c: Record<string, unknown>) => ({ ...c })))
  }
  // An older setup file or mp4 has no `precision` key; absent keys are
  // skipped, so an import of one lands on float64 — the safe direction.
  for (const k of ['potential', 'mass', 'c', 'hbar_eff', 'tol',
                   'record_dt', 'delay', 'mode', 't2',
                   'auto_expand', 'precision', 'device',
                   'history_mb'] as const) {
    if (k in src && src[k] != null)
      (target as unknown as Record<string, unknown>)[k] = src[k]
  }
  if (Array.isArray(src.variants)) {
    const v = src.variants.filter((x: string) =>
      (ALL_KEYS as readonly string[]).includes(x))
    if (v.length) target.variants.splice(0, target.variants.length, ...v)
  }
  // 'runahead' was renamed to 'batch' (2026-07-24); migrate a persisted or
  // imported setup so it is not rejected by the backend's mode literal.
  if ((target as unknown as Record<string, unknown>).mode === 'runahead')
    target.mode = 'batch'
  applyPrecisionInvariants(target)
}

/**
 * The backend refuses float32 + auto-expand (in single precision the edge
 * detector's noise floor is above its own trigger, and the support scan it
 * would size the new domain from is worse still). Reaching that combination
 * from a stale localStorage entry, an imported setup, or a host default
 * adopted AFTER the merge would leave Restart failing with a 422 the user
 * never chose — so drop the response and keep the precision.
 *
 * It lives here rather than only in the Setup panel's watcher because it is a
 * property of the config, not of a mounted component: the panel can be hidden,
 * and the adoption path in SimulatorView.probeHost() sets precision from
 * outside it.
 */
export function applyPrecisionInvariants(c: SimConfig) {
  if (c.precision === 'float32') c.auto_expand = false
}

/** Load the persisted setup (merged over defaults) — a hard reload must
 *  not silently reset mode/t2/grid/IC, or the user ends up running a
 *  different simulation than they configured. */
export function loadConfig(): SimConfig {
  const d = defaultConfig()
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    precisionWasStored = !!raw && typeof raw === 'object' && raw.precision != null
    mergeConfig(d, raw)
  } catch { /* corrupted storage -> defaults */ }
  return d
}

/**
 * Apply an imported setup document to the live form. Accepts what
 * `GET /api/sessions/{id}/setup` writes ({format, version, config}), the
 * same blob as carried in an exported mp4's `comment` tag ({generator,
 * config, param_log, export}), and a bare config object. Throws an Error
 * whose message is meant to be shown to the user.
 */
export function importConfig(target: SimConfig, doc: unknown) {
  if (!doc || typeof doc !== 'object')
    throw new Error('not a wignerf setup file')
  const d = doc as Record<string, any>
  const cfg = (d.config && typeof d.config === 'object') ? d.config : d
  if (!cfg.grid || typeof cfg.grid !== 'object' || typeof cfg.potential !== 'string'
      || !cfg.ic || !Array.isArray(cfg.ic.components))
    throw new Error('not a wignerf setup file (no grid/potential/IC)')
  for (const k of ['x1', 'x2', 'p1', 'p2', 'Nx', 'Np'] as const)
    if (typeof cfg.grid[k] !== 'number' || !Number.isFinite(cfg.grid[k]))
      throw new Error(`grid.${k} is missing or not a number`)
  // the API enforces this too, but a clear message here beats a 422 after
  // the user presses Restart
  for (const k of ['Nx', 'Np'] as const)
    if (cfg.grid[k] % 2 !== 0) throw new Error(`grid.${k} must be even`)
  if (!cfg.ic.components.length) throw new Error('the IC has no components')
  if (cfg.ic.type !== 'mixture' && cfg.ic.type !== 'cat')
    throw new Error(`unknown IC type "${cfg.ic.type}"`)
  if (Array.isArray(cfg.variants)
      && !cfg.variants.some((v: string) => (ALL_KEYS as readonly string[]).includes(v)))
    throw new Error('no known variants in the file')
  mergeConfig(target, cfg)
}

export function saveConfig(c: SimConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(c))
}

/** Restore the whole setup to defaults IN PLACE — the view holds the
 *  config in a long-lived reactive object, so nested objects/arrays must
 *  be mutated, not replaced, for existing bindings to keep working (the
 *  deep watcher then persists the defaults to localStorage). */
export function resetToDefaults(c: SimConfig) {
  const d = defaultConfig()
  Object.assign(c.grid, d.grid)
  c.ic.type = d.ic.type
  c.ic.components.splice(0, c.ic.components.length, ...d.ic.components)
  c.variants.splice(0, c.variants.length, ...d.variants)
  c.potential = d.potential
  c.mass = d.mass
  c.c = d.c
  c.hbar_eff = d.hbar_eff
  c.tol = d.tol
  c.record_dt = d.record_dt
  c.delay = d.delay
  c.mode = d.mode
  c.t2 = d.t2
  c.auto_expand = d.auto_expand
  c.precision = d.precision
  c.device = d.device
  c.history_mb = d.history_mb
  // "reset to defaults" un-chooses: the form goes back to deferring to the
  // host, which is what the default IS.
  precisionWasStored = false
}

export function defaultConfig(): SimConfig {
  return {
    grid: { x1: -6.0, x2: 6.0, Nx: 256, p1: -7.0, p2: 7.0, Np: 256 },
    potential: 'x^2/2',
    ic: {
      type: 'mixture',
      // sigma = 0.70711 (not 0.707): sigma_x*sigma_p must be >= hbar/2 = 0.5
      // or the default state is (marginally) sub-Heisenberg and the purity
      // warning fires on first load. 0.70711^2 = 0.5000045.
      components: [
        { x0: 2.0, p0: 0.0, sigma_x: 0.70711, sigma_p: 0.70711, weight: 1, phase: 0 },
      ],
    },
    variants: ['qn', 'cn'],
    mass: 1.0,
    c: C_AU,
    hbar_eff: 1.0,
    tol: 0.01,
    record_dt: 0.05,
    delay: 0.0,   // seconds injected between played-back frames (0 = max speed)
    mode: 'interactive',
    t2: 20.0,
    auto_expand: false,
    precision: hostPrecision,   // the host's WIGNERF_PRECISION once probed
    device: '',             // '' = the host's WIGNERF_DEVICE pool
    history_mb: 0,          // 0 = the host's WIGNERF_HISTORY_MB
  }
}
