/** Session configuration shared by the setup panels and the view. */

import { C_AU } from './units'
import { labels as axisLabels, nAxes } from './axes'
import { ALL_VARIANTS, type VariantKey } from './variants'

export type Ndim = 1 | 2

export interface AxisCfg {
  lo: number
  hi: number
  N: number
}

/**
 * The FORM's grid: 2*ndim axes in core/axes.py order (all spatial, then all
 * momentum). The legacy flat {x1, x2, Nx, p1, p2, Np} spelling is still
 * accepted on the way IN (see normalizeGrid) so every stored browser config,
 * exported setup document and mp4 comment tag from before 2D keeps loading —
 * the backend's GridSpec makes the same bargain.
 */
export interface GridCfg {
  ndim: Ndim
  axes: AxisCfg[]
}

/**
 * A LIVE geometry, as `status` and every decoded frame report it. Distinct from
 * GridCfg because it is a fact about what is running, per record, not a form
 * value: auto-expand moves it and the panels follow the PAINTED frame.
 */
export interface GeomCfg {
  ndim: number
  lo: number[]
  hi: number[]
  N: number[]
}

export interface ICComponentCfg {
  q0: number[]
  k0: number[]
  sigma_q: number[]
  /** null for cat states, where it is derived per dimension as hbar/(2 sigma_q) */
  sigma_k: number[] | null
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
const ALL_KEYS = ALL_VARIANTS

/** Default axes per dimensionality. 2D starts at 64^4 = 16.8M cells, which
 *  measures 3.25 GiB per variant worker and ~35 steps/s on an RTX 3090 — a
 *  serious run rather than a toy, and comfortably inside the default
 *  WIGNERF_MAX_CELLS_2D. Drop to 32 per axis for quick exploration. */
export const DEFAULT_AXES: Record<Ndim, AxisCfg[]> = {
  1: [{ lo: -6.0, hi: 6.0, N: 256 }, { lo: -7.0, hi: 7.0, N: 256 }],
  // The 2D spatial box is WIDER than the 1D one at the same extents, and that
  // is not arbitrary. boundary.edge_band is max(4, N/32) CELLS, so its physical
  // width is max(4, N/32)*L/N: at N=256 the N/32 term wins (0.375 a.u.) but at
  // N=64 the 4-cell floor does (0.750 a.u.) — twice as wide, reaching in to
  // 4.6σ of a packet at x0=2 instead of 5.1σ. The default therefore tripped its
  // OWN boundary warning at [-6,6] (3.8e-06 against a 1e-6 threshold) while the
  // identical 1D state at N=256 did not. [-8,8] puts it at 2.1e-12 and leaves
  // the amplitude-2 orbit room to run.
  2: [{ lo: -8.0, hi: 8.0, N: 64 }, { lo: -8.0, hi: 8.0, N: 64 },
      { lo: -7.0, hi: 7.0, N: 64 }, { lo: -7.0, hi: 7.0, N: 64 }],
}

/** Default potential per dimensionality (isotropic harmonic in both). */
export const DEFAULT_POTENTIAL: Record<Ndim, string> = {
  1: 'x^2/2',
  2: '(x^2 + y^2)/2',
}

/**
 * Default initial condition per dimensionality. sigma = 0.70711 (not 0.707):
 * sigma_q*sigma_k must be >= hbar/2 = 0.5 or the default state is (marginally)
 * sub-Heisenberg and the purity warning fires on first load — 0.70711^2 =
 * 0.5000045.
 *
 * The 2D default is NOT the 1D one with a second dimension bolted on at rest.
 * A packet at (x0, py0) = (2, 1) in the isotropic well traces an ELLIPSE rather
 * than sliding along x with y pinned at the origin: the (x,y) plane shows an
 * orbit, the four marginals all move, and ⟨Lz⟩ = x·py − y·px = 2 is a nonzero
 * constant — so the 2D-only plot demonstrates the conservation it exists for
 * instead of drawing a flat zero. A default is the first thing anyone sees;
 * in 2D it should show the dimension they just asked for.
 */
export const DEFAULT_IC: Record<Ndim, ICCfg> = {
  1: {
    type: 'mixture',
    components: [{ q0: [2.0], k0: [0.0], sigma_q: [0.70711],
                   sigma_k: [0.70711], weight: 1, phase: 0 }],
  },
  2: {
    type: 'mixture',
    components: [{ q0: [2.0, 0.0], k0: [0.0, 1.0], sigma_q: [0.70711, 0.70711],
                   sigma_k: [0.70711, 0.70711], weight: 1, phase: 0 }],
  },
}

/** Deep copy of one component — the form mutates these in place. */
export function cloneComponent(c: ICComponentCfg): ICComponentCfg {
  return {
    q0: [...c.q0], k0: [...c.k0], sigma_q: [...c.sigma_q],
    sigma_k: c.sigma_k ? [...c.sigma_k] : null,
    weight: c.weight, phase: c.phase,
  }
}

/** The default IC for a dimensionality, freshly copied. */
export function defaultIC(ndim: Ndim): ICCfg {
  return { type: DEFAULT_IC[ndim].type,
           components: DEFAULT_IC[ndim].components.map(cloneComponent) }
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

/** Accept either grid spelling and return the generic one. */
export function normalizeGrid(raw: unknown): GridCfg | null {
  if (!raw || typeof raw !== 'object') return null
  const g = raw as Record<string, any>
  if (Array.isArray(g.axes) && g.axes.length) {
    const axes = g.axes.map((a: Record<string, any>) => ({
      lo: num(a?.lo, 0), hi: num(a?.hi, 1), N: num(a?.N, 64),
    }))
    const ndim = (axes.length === 4 ? 2 : 1) as Ndim
    return { ndim, axes: axes.slice(0, 2 * ndim) }
  }
  if (typeof g.x1 === 'number') {
    return {
      ndim: 1,
      axes: [{ lo: num(g.x1, -6), hi: num(g.x2, 6), N: num(g.Nx, 256) },
             { lo: num(g.p1, -7), hi: num(g.p2, 7), N: num(g.Np, 256) }],
    }
  }
  return null
}

/** Accept either IC-component spelling and return the generic one. */
export function normalizeComponent(raw: unknown): ICComponentCfg {
  const c = (raw ?? {}) as Record<string, any>
  const arr = (v: unknown, fb: number[]): number[] =>
    Array.isArray(v) && v.length ? v.map((x) => num(x, 0)) : fb
  if (!Array.isArray(c.q0) && typeof c.x0 === 'number') {
    return {
      q0: [num(c.x0, 0)], k0: [num(c.p0, 0)], sigma_q: [num(c.sigma_x, 0.5)],
      sigma_k: c.sigma_p == null ? null : [num(c.sigma_p, 0.5)],
      weight: num(c.weight, 1), phase: num(c.phase, 0),
    }
  }
  return {
    q0: arr(c.q0, [0]), k0: arr(c.k0, [0]), sigma_q: arr(c.sigma_q, [0.5]),
    sigma_k: c.sigma_k == null ? null : arr(c.sigma_k, [0.5]),
    weight: num(c.weight, 1), phase: num(c.phase, 0),
  }
}

/** The form grid as a live-geometry object, for the display before the first
 *  frame lands (and for the IC preview's axes). */
export function geomOf(g: GridCfg): GeomCfg {
  return {
    ndim: g.ndim,
    lo: g.axes.map((a) => a.lo),
    hi: g.axes.map((a) => a.hi),
    N: g.axes.map((a) => a.N),
  }
}

/** Total grid cells — the number that actually bounds 2D memory. */
export function gridCells(g: GridCfg): number {
  return g.axes.reduce((n, a) => n * a.N, 1)
}

/**
 * The per-axis N values the Setup panel offers: powers of two up to the host's
 * ceiling FOR THIS NDIM, plus whatever the form currently holds.
 *
 * It lives here, not in SetupPanel, so it can be pinned by a unit test rather
 * than eyeballed through the DOM — the same reason `lib/potentialCuts.ts` was
 * extracted. Both of its bugs were invisible in the component: the list was
 * built against the RUNNING session's cap, so a form switched to 2D over a live
 * 1D session offered N up to 4096 where the API refuses anything past 128, and a
 * form switched back to 1D over a live 2D session collapsed to a single option
 * (cap 128, loop starting at 256, body never entered).
 *
 * Two floors matter:
 *  - 2D starts at 32, not 16. `boundary._band_mass` reports nothing below 32
 *    cells per axis — the edge band would otherwise cover a quarter of the axis
 *    — so a 16⁴ session has no boundary watch at all and says so nowhere.
 *    Starting at 32 means every grid on offer has a working one, and it matches
 *    DEFAULT_AXES's own "drop to 32 per axis for quick exploration". 16⁴ stays
 *    reachable through the API and through an imported config, which the
 *    `current` merge below keeps listed.
 *  - the start is clamped to `cap`, or a host that lowered WIGNERF_MAX_GRID
 *    (which CLAUDE.md recommends on VRAM-constrained hosts) would get an empty
 *    1D list the same way.
 *
 * `current` values are always listed even when over cap, so an imported
 * oversized setup renders its own value rather than a blank select; the API
 * refuses it at Restart with a message naming the ceiling.
 */
export function axisSizeOptions(ndim: number, cap: number,
                                current: number[] = []): number[] {
  const top = Math.max(cap, 16)
  const out: number[] = []
  for (let n = Math.min(ndim > 1 ? 32 : 256, top); n <= top; n *= 2) out.push(n)
  for (const n of current) if (!out.includes(n)) out.push(n)
  return out.sort((a, b) => a - b)
}

/**
 * Switch the form's dimensionality IN PLACE, rebuilding what cannot carry
 * over. Grid axes and IC components are per-dimension, so going 1D -> 2D has
 * to invent a second dimension's worth of both: the new axes copy the defaults
 * and each component's second dimension mirrors its first, centred at the
 * origin, which is the least surprising state to land in (a separable product
 * of what was there). Going back drops it.
 */
export function setNdim(c: SimConfig, ndim: Ndim) {
  if (c.grid.ndim === ndim) return
  const wasDefaultU = c.potential === DEFAULT_POTENTIAL[c.grid.ndim]
  const keep = c.grid.axes
  // The BOX carries over — a 2D run wants the same extents its 1D counterpart
  // had — but the RESOLUTION is re-chosen, because N means a different thing
  // in each: the cell count is N² against N⁴. Propagating a 1D 256 gave
  // 256⁴ = 4.3e9 cells, over the per-axis cap AND the cell ceiling, so the
  // first Restart after a switch failed on a grid nobody chose. min() rather
  // than a flat reset, so a deliberately SMALL choice survives the switch.
  const box = ndim === 2
    ? [keep[0]!, keep[0]!, keep[1]!, keep[1]!]   // y mirrors x, py mirrors px
    : [keep[0]!, keep[2]!]                       // keep x and px, drop y/py
  // ...EXCEPT when the box is still the source ndim's default, in which case
  // take the TARGET ndim's default box — the same "only if untouched" rule the
  // potential gets below, and for a sharper reason. Carrying [-6,6] into 2D
  // reproduces exactly the case DEFAULT_AXES[2] was widened to avoid: the edge
  // band is max(4, N/32) CELLS, so at N=64 the 4-cell floor makes it 0.750 a.u.
  // — reaching in to 4.60σ of the default packet at x0=2 — and the fresh 2D
  // default tripped its OWN boundary warning on the first Restart (measured band
  // mass 3.78e-06 against a 1e-6 trigger; analytic tail 2.15e-06, so real mass,
  // not detector noise). At [-8,8] the same band sits at 7.07σ: 2.12e-12.
  // A box the user CHOSE still carries over untouched — that is their domain,
  // and silently widening it would be worse than a warning.
  const boxWasDefault = keep.every((a, i) => {
    const d = DEFAULT_AXES[c.grid.ndim]![i]!
    return a.lo === d.lo && a.hi === d.hi
  })
  const src = boxWasDefault ? DEFAULT_AXES[ndim] : box
  const next = DEFAULT_AXES[ndim].map((d, i) => ({
    // extents from whichever source applies; N always the user's own choice,
    // capped at the target's default (a 1D 256 is 4.3e9 cells at ndim=2)
    lo: src[i]!.lo, hi: src[i]!.hi, N: Math.min(box[i]!.N, d.N),
  }))
  c.grid.ndim = ndim
  c.grid.axes.splice(0, c.grid.axes.length, ...next)
  for (const k of c.ic.components) {
    if (ndim === 2) {
      k.q0 = [k.q0[0]!, 0]
      k.k0 = [k.k0[0]!, 0]
      k.sigma_q = [k.sigma_q[0]!, k.sigma_q[0]!]
      k.sigma_k = k.sigma_k ? [k.sigma_k[0]!, k.sigma_k[0]!] : null
    } else {
      k.q0 = [k.q0[0]!]
      k.k0 = [k.k0[0]!]
      k.sigma_q = [k.sigma_q[0]!]
      k.sigma_k = k.sigma_k ? [k.sigma_k[0]!] : null
    }
  }
  // U(x) cannot mean U(x,y): only replace it if the user never edited it away
  // from the default, so a hand-written potential is never silently discarded.
  if (wasDefaultU) c.potential = DEFAULT_POTENTIAL[ndim]
  applyNdimInvariants(c)
}

/**
 * The two things a 2D run still cannot do, applied to the config rather than
 * argued with at Restart time — the exact counterpart of
 * applyPrecisionInvariants, and for the same reason: the backend refuses each
 * combination outright (milestones M1 and M3), so reaching one from a stale
 * localStorage entry, an imported 1D setup or a probe-adopted host default
 * would leave Restart failing with a 422 the user never chose.
 *
 * The relativistic variants were the third (M2) and landed on 2026-07-27, so
 * qr/cr are no longer filtered out here. A stored 2D config that this function
 * once stripped down to ['qn'] keeps whatever it was left with — there is
 * nothing to migrate, because the stripping was destructive at the time.
 */
export function applyNdimInvariants(c: SimConfig) {
  if (c.grid.ndim < 2) return
  c.precision = 'float64'          // M1
  c.auto_expand = false            // M3
}

/** Axis labels for this config's dimensionality ('x','p' / 'x','y','px','py'). */
export function gridLabels(c: SimConfig): readonly string[] {
  return axisLabels(c.grid.ndim)
}

export function gridAxisCount(c: SimConfig): number {
  return nAxes(c.grid.ndim)
}

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

/** Call when the user actually operates the precision control, or when an
 *  IMPORT supplies one — NOT for the programmatic adoptions (host probe,
 *  status sync), which must stay overridable by the server. */
export function markPrecisionChosen() {
  precisionWasStored = true
}

/**
 * The `precision` to put in a create payload, or null to omit the field and let
 * the host's WIGNERF_PRECISION decide.
 *
 * Omitting is the default and the point: it is the only way a float32 host can
 * win when the SPA could not read its default, and a hard-coded float64 sent as
 * though it were a decision is exactly how such a host got silently overridden.
 *
 * There are two exceptions, and both are the same rule: a form asking for a
 * float64-ONLY feature IS asking for float64, so it says so rather than
 * deferring.
 *
 *  - ndim = 2, which defers float32 entirely (the backend's MSG_F32_2D,
 *    milestone M1). Stating it also keeps the exported setup document honest
 *    about the precision the run had.
 *  - auto-expand (MSG_EXPAND_F32: in single precision a contained state's own
 *    noise passes the edge trigger and the support scan reads the whole axis).
 *    Only reachable when the /device probe failed — a probe that succeeded has
 *    already cleared auto_expand through applyPrecisionInvariants.
 *
 * The 2D branch returns a LITERAL rather than c.precision, so the answer does
 * not depend on whether applyNdimInvariants has run on this config yet.
 */
export function precisionForPayload(c: SimConfig): SimConfig['precision'] | null {
  if (c.grid.ndim > 1) return 'float64'
  if (precisionWasStored || c.auto_expand) return c.precision
  return null
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
  const g = normalizeGrid(src.grid)
  if (g) {
    target.grid.ndim = g.ndim
    target.grid.axes.splice(0, target.grid.axes.length, ...g.axes)
  }
  if (Array.isArray(src.ic?.components) && src.ic.components.length) {
    target.ic.type = src.ic.type === 'cat' ? 'cat' : 'mixture'
    target.ic.components.splice(0, target.ic.components.length,
                                ...src.ic.components.map(normalizeComponent))
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
  // ndim first: it FORCES float64, so running the precision invariants after
  // it keeps the two from disagreeing about tol
  applyNdimInvariants(target)
  applyPrecisionInvariants(target)
}

/**
 * The float32 floor on the adaptive-step tolerance. Mirrors the backend's
 * `core/protocol.py:TOL_MIN_F32`, which refuses anything below it at create AND
 * on the live path: adjust_step compares one full step against two half steps,
 * and in single precision that difference has a roundoff floor near 7e-7, so a
 * smaller tol makes the controller shrink dt every 20 steps without ever
 * converging. Move both sides together.
 */
export const TOL_MIN_F32 = 1e-5

/**
 * The two things float32 cannot do, applied to the config rather than argued
 * with at Restart time. The backend refuses both combinations outright, so
 * reaching one from a stale localStorage entry, an imported setup, or a host
 * default adopted AFTER the merge would leave Restart failing with a 422 the
 * user never chose:
 *
 *  - auto-expand, because in single precision a contained state's own spectral
 *    noise passes the 1e-6 edge trigger and the 1e-8 support scan it would size
 *    the new domain from reads the whole axis;
 *  - a tol below TOL_MIN_F32, for the reason recorded on that constant.
 *
 * It lives here rather than only in the Setup panel's watchers because it is a
 * property of the config, not of a mounted component: the panel can be hidden,
 * and the adoption path in SimulatorView.probeHost() sets precision from
 * outside it. Callers with a LIVE session must also tell it — see the
 * cfg.precision watcher in SimulatorView, which cannot be done from here.
 */
export function applyPrecisionInvariants(c: SimConfig) {
  if (c.precision !== 'float32') return
  c.auto_expand = false
  if (!(c.tol >= TOL_MIN_F32)) c.tol = TOL_MIN_F32   // NaN-safe
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
  const ng = normalizeGrid(cfg.grid)
  if (!ng) throw new Error('grid is neither {ndim, axes} nor {x1, x2, Nx, ...}')
  const names = axisLabels(ng.ndim)
  if (ng.axes.length !== 2 * ng.ndim)
    throw new Error(`a ${ng.ndim}D grid needs ${2 * ng.ndim} axes, got ${ng.axes.length}`)
  ng.axes.forEach((a, i) => {
    for (const k of ['lo', 'hi', 'N'] as const)
      if (typeof a[k] !== 'number' || !Number.isFinite(a[k]))
        throw new Error(`grid axis ${names[i]}: ${k} is missing or not a number`)
    if (!(a.hi > a.lo)) throw new Error(`grid axis ${names[i]}: need hi > lo`)
    // the API enforces this too, but a clear message here beats a 422 after
    // the user presses Restart
    if (a.N % 2 !== 0) throw new Error(`grid axis ${names[i]}: N must be even`)
  })
  if (!cfg.ic.components.length) throw new Error('the IC has no components')
  const nd = normalizeComponent(cfg.ic.components[0]).q0.length
  if (nd !== ng.ndim)
    throw new Error(`the grid is ${ng.ndim}D but the IC components are ${nd}D`)
  if (cfg.ic.type !== 'mixture' && cfg.ic.type !== 'cat')
    throw new Error(`unknown IC type "${cfg.ic.type}"`)
  if (Array.isArray(cfg.variants)
      && !cfg.variants.some((v: string) => (ALL_KEYS as readonly string[]).includes(v)))
    throw new Error('no known variants in the file')
  mergeConfig(target, cfg)
  // An imported document's precision IS a decision — it is the run someone
  // exported, and reproducing it is the whole point of the setup document (and
  // of the same JSON in an mp4's comment tag). Without this the payload omits
  // the field, the session is built at the HOST default, and the form is left
  // showing a float32 that never happened behind a "restart to apply" that no
  // restart can resolve — status.precision never CHANGES, so SimulatorView's
  // sync watcher never fires. Gated on the key being present, exactly as
  // loadConfig gates on the stored blob: an older file has no precision, and
  // marking THAT chosen would turn mergeConfig's deliberate "absent keys land
  // on float64, the safe direction" into a decision overriding a float32 host.
  if (cfg.precision != null) markPrecisionChosen()
}

export function saveConfig(c: SimConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(c))
}

/**
 * Restore the whole setup to defaults IN PLACE — the view holds the config in a
 * long-lived reactive object, so nested objects/arrays must be mutated, not
 * replaced, for existing bindings to keep working (the deep watcher then
 * persists the defaults to localStorage).
 *
 * DIMENSIONALITY SURVIVES. ndim is the choice of PROBLEM, not a setting within
 * one: a 2D user pressing "reset" wants this potential/grid/IC replaced by the
 * 2D defaults, not to be dropped back into a 1D simulation they would then have
 * to switch out of again. Everything ndim-dependent — grid axes, U, the IC —
 * comes from defaultConfig(ndim) for exactly that reason.
 */
export function resetToDefaults(c: SimConfig) {
  const d = defaultConfig(c.grid.ndim)
  c.grid.axes.splice(0, c.grid.axes.length, ...d.grid.axes)
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
  // ...but the host default may be float32, which 2D refuses (M1). Without
  // this a reset in 2D would leave the form in the one combination the API
  // rejects, and Restart would 422 on something nobody chose.
  applyNdimInvariants(c)
}

/** The whole default setup for a dimensionality. `ndim` defaults to 1 because
 *  that is what a first-ever load gets; every other caller passes the one it
 *  is resetting WITHIN. */
export function defaultConfig(ndim: Ndim = 1): SimConfig {
  return {
    grid: { ndim, axes: DEFAULT_AXES[ndim].map((a) => ({ ...a })) },
    potential: DEFAULT_POTENTIAL[ndim],
    ic: defaultIC(ndim),
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
