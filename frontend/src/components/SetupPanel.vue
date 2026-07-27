<script setup lang="ts">
/**
 * Setup controls: potential editor, physics parameters (live-applied),
 * grid geometry + grid-lines display toggle, run mode. The IC editor is a
 * separate component so the portrait layout can place it in its own column.
 */
import { computed, ref, watch } from 'vue'
import PotentialEditor from './PotentialEditor.vue'
import { labelHtml } from '../lib/axes'
import { applyPrecisionInvariants, axisSizeOptions, gridCells, gridLabels,
         markPrecisionChosen, resetToDefaults, setNdim, TOL_MIN_F32,
         type GeomCfg, type LivePhysics, type LiveRun,
         type SimConfig } from '../lib/config'

const props = defineProps<{ cfg: SimConfig; live: boolean; sign?: number
                            // live = a session exists; computing = it is
                            // producing new records right now (only then is
                            // there a run for U(x)'s "Apply live" to reach —
                            // status.running is true during playback too)
                            computing?: boolean
                            liveGrid?: GeomCfg | null; maxGrid?: number
                            // 2D memory facts (status): the total-cell ceiling
                            // and the measured device bytes per cell per worker
                            maxCells?: number | null
                            bytesPerCell?: number | null
                            livePhysics?: LivePhysics | null
                            liveRun?: LiveRun | null
                            // true once the session has COMPUTED records — before
                            // that there is no meaningful "run mode" to diverge from
                            hasRun?: boolean
                            // host facts for the Compute section: the resolved
                            // device pool of the RUNNING session, the pool this
                            // backend offers, and the history cap it granted
                            // against the ceiling it enforces
                            liveDevices?: string[] | null
                            // installed memory per device spec. Keyed, not a
                            // single number: which device binds depends on the
                            // assignment, and at one variant only ONE is used.
                            deviceTotals?: Record<string, number | null> | null
                            deviceOptions?: { spec: string; device: string }[] | null
                            // the host's POOL — what a form device of '' resolves
                            // to, hence the only way to compare that against the
                            // running session's devices
                            hostPool?: string[] | null
                            historyCapMb?: number | null
                            historyMbMax?: number | null }>()

/**
 * RUN fields are SessionCreate-only, so editing one changes NOTHING about the
 * session already running — and pressing Solve then computes under the old
 * settings while the form displays the new ones. That is invisible and it
 * misleads: on 2026-07-23 a run believed to be "batch, t₂=100" was really
 * the previous interactive session, and computed straight past t=100 with the
 * form showing 100 the whole time. `status` has carried the live mode/t2/
 * record_dt all along, so mark any field that disagrees with it.
 */
function runDiffers(field: keyof LiveRun) {
  const lr = props.liveRun
  // Before ANY computation there is no run to diverge from — the session is
  // just an implementation detail that silently tracks the form (it
  // auto-restarts on a run-field change while idle). So the run fields are
  // never "stale" until records exist.
  if (!lr || !props.hasRun) return false
  // t2 only matters in BATCH mode; interactive has no end time, so the form's
  // (hidden, leftover) t2 must NOT count as a difference against a session's
  // t2=null. A batch form vs an interactive session is caught by `mode`.
  if (field === 't2' && props.cfg.mode !== 'batch') return false
  return (lr[field] ?? null) !== (props.cfg[field] ?? null)
}
// Only the fields this SECTION owns. `precision` is a LiveRun member because
// that is where status carries it, but it is a COMPUTE control and it has its
// own amber line down there (`computeStale`) — including it here meant a
// float64 → float32 switch lit the RUN warning too, announcing "running:
// interactive (no t₂), Δt rec = 0.05" at someone who had changed neither and
// whose mode and Δt rec did not differ at all. Each section reports its own.
const runStale = computed(() =>
  (['mode', 't2', 'record_dt'] as const).some(runDiffers))

const f32 = computed(() => props.cfg.precision === 'float32')
// auto-expand is gated by float32 (single-precision noise trips its own
// detector) AND by 2D (no memory guard on a 4-axis doubling) — two different
// reasons, one disabled state, so the marker and tooltip pick the right words
const expandGated = computed(() => f32.value || props.cfg.grid.ndim > 1)
// ...and precision itself is gated by 2D, which defers float32 (M1). Same
// treatment as auto-expand rather than a control whose value is silently put
// back: applyNdimInvariants forces float64 in payload(), so picking float32
// here used to leave the select amber against a value no restart could send —
// and, on a fresh session, made syncFreshSessionToForm build TWO of them (its
// loop re-reads the form after restart(), and restart() had moved it back).
const precisionGated = computed(() => props.cfg.grid.ndim > 1)

// The two float32 refusals, worded once each. They live in TOOLTIPS and in the
// one-off amber note below, never in a standing paragraph: this panel is a
// narrow column and two permanent explanations of settings you are not allowed
// to change were eating the vertical space the actual controls need. A touch
// device has no hover, which is why the checkbox label and the tol label carry a
// compact permanent marker as well ("— float64 only", "≥1e-5") — those cost no
// line at all.
const AUTO_EXPAND_HELP = 'when W(x,p,t) approaches a domain edge, move or double'
  + ' the domain automatically at the frontier (exact — the lattice spacing is'
  + ' frozen, values are never interpolated). Applies live; detection and its'
  + ' warning run either way.'
const AUTO_EXPAND_F32_HELP = 'Unavailable in float32, and refused by the API: a'
  + " contained state's own single-precision noise passes the 1e-6 edge trigger"
  + ' within a few hundred steps, and the 1e-8 support scan that would size the'
  + ' new domain reads the whole axis, so the domain would double for no'
  + ' physical reason. Boundary detection still runs on a raised threshold and'
  + ' still warns you. Restart in float64 to auto-expand.'
const AUTO_EXPAND_2D_HELP = 'Not available for 2D runs yet, and refused by the'
  + ' API (milestone M3): in 4D every axis doubling doubles a multi-GiB working'
  + ' set, so the planner needs a memory guard it does not yet have. Boundary'
  + ' DETECTION still runs on all four axes and will warn you — size the domain'
  + ' by hand.'
const PRECISION_HELP = 'spectral working precision. float64 is the physics'
  + ' setting. float32 is a PREVIEW mode: ~3.3-3.8× faster and ~58% of the VRAM'
  + ' on CUDA (no speedup on CPU), but purity and energy drift by ~1e-4 with the'
  + ' same secular signature as boundary wrap, and ΔX·ΔP noise is ~150× the'
  + ' relativistic shear. The exponents are built in double either way.'
const PRECISION_2D_HELP = 'Not available for 2D runs yet, and refused by the'
  + ' API (milestone M1): the mixed-precision rules — float64 meshes and'
  + ' exponent construction, single-precision stepping — have not been'
  + " re-verified for the correlated 2D Bopp shift, and adjust_step's"
  + ' single-precision residual floor was measured at 256² only. It is the first'
  + ' 2D follow-up precisely because memory is the binding 2D constraint.'
const NDIM_HELP = 'spatial dimensions. 1D solves W(x,p,t); 2D solves'
  + ' W(x,y,px,py,t) and streams the six pairwise 2D projections instead of the'
  + ' 4D array. Restart-only, and it rebuilds the grid, the initial condition'
  + ' and (if untouched) U. NB 2D defers float32, auto-expand and mp4 export'
  + ' — the API refuses each with the reason.'
const TOL_HELP = 'adaptive-step relative tolerance'
// toExponential: JS prints 1e-5 as "0.00001", which is neither how the field is
// typed nor how the backend's own refusal words it
const TOL_F32_HELP = 'adaptive-step relative tolerance. In float32 the floor is '
  + TOL_MIN_F32.toExponential() + ': the controller compares one full step'
  + ' against two half steps, which agree only to ~7e-7 in single precision, so'
  + ' a smaller tol would shrink Δt every 20 steps and never converge.'

/**
 * What switching to float32 actually changed, listed once in amber until the
 * restart that applies it. Recorded at the moment of the switch rather than
 * derived, so the note states facts: a form already at tol = 0.01 had nothing
 * raised, and claiming otherwise is the kind of small lie that makes a user
 * stop reading these.
 */
const f32Applied = ref<string[]>([])

/**
 * The user operated the precision select. Apply the float32 invariants
 * SYNCHRONOUSLY, right here — this must not be a watcher.
 *
 * Watchers flush asynchronously, and SimulatorView's own cfg.precision watcher
 * is created FIRST (a child's setup runs during the parent's render, so the
 * parent's watchers hold lower ids and pre-flush jobs run in id order). It
 * calls syncFreshSessionToForm, which on a fresh session restarts inside that
 * same flush — building a create payload from a config this had not fixed up
 * yet. Result: `auto_expand: true` beside `precision: "float32"`, and a 422 for
 * a pair the user never asked for, in either run mode (2026-07-25). Doing it in
 * the handler also makes the checkbox untick the moment you pick float32 rather
 * than a tick later. The f32 watcher below stays as a backstop for any future
 * route that sets precision without coming through here; SimulatorView guards
 * the payload boundary as well, because neither belt nor braces is free.
 */
function onPrecisionChange() {
  markPrecisionChosen()
  const wasAutoExpand = props.cfg.auto_expand
  const wasTol = props.cfg.tol
  applyPrecisionInvariants(props.cfg)
  // Each entry carries its own short reason: this note is the only place a
  // touch device can read WHY, since it has no tooltip to hover.
  const done: string[] = []
  if (wasAutoExpand && !props.cfg.auto_expand)
    done.push("auto-expand off — float32's own noise trips its 1e-6 edge trigger")
  if (props.cfg.tol !== wasTol)
    done.push('tol raised from ' + wasTol + ' to ' + props.cfg.tol.toExponential()
              + ' — the step controller cannot resolve less')
  f32Applied.value = f32.value ? done : []
}
// applyPrecisionInvariants owns both float32 refusals; call it rather than
// restate one. SimulatorView watches cfg.precision and tells a LIVE session
// about the cleared auto-expand — the form clearing it alone would leave a
// running session still expanding behind an unchecked, disabled box.
watch(f32, (on) => {
  if (on) applyPrecisionInvariants(props.cfg)
  else f32Applied.value = []      // back to float64: the note no longer applies
})

/** The float32 tol floor, snapped here rather than left for the backend to
 *  refuse: below it the step controller never converges, so create 422s and a
 *  live change is popped with a bad_tol error. Same idiom as clampHistory. */
function clampTol() {
  if (f32.value && !(props.cfg.tol >= TOL_MIN_F32)) props.cfg.tol = TOL_MIN_F32
}

/** 0 means "the host's default"; anything else is a real cap the API bounds at
 *  64 MiB. Snap the dead zone between them here rather than let Restart come
 *  back with a schema error about a field the form let you type. */
function clampHistory() {
  const v = props.cfg.history_mb
  if (!Number.isFinite(v) || v <= 0) props.cfg.history_mb = 0
  else if (v < 64) props.cfg.history_mb = 64
  else if (props.historyMbMax && v > props.historyMbMax)
    props.cfg.history_mb = props.historyMbMax
}

// A device this backend does not offer (a setup file carried over from the
// workstation to the VPS, a stored 'cuda:1' after a card is pulled) must stay
// visible and named rather than silently reverting to the default — the run
// would then quietly land somewhere else.
const deviceMissing = computed(() =>
  !!props.cfg.device && !!props.deviceOptions
  && !props.deviceOptions.some((d) => d.spec === props.cfg.device))

/**
 * COMPUTE is restart-only exactly like RUN, so its fields need the same amber
 * "this is not what is running" marking — device and history_mb had none, and a
 * form reading "cuda:0" over a session on cuda:1 is the same trap as a form
 * reading float64 over a float32 run. They are not in LiveRun because neither
 * has a directly comparable value in `status`: the form holds a REQUEST ('' =
 * the host's pool, 0 = the host's ceiling) while status reports what was
 * GRANTED (the resolved device list, the clamped cap). So resolve the request
 * the way the server would, then compare.
 */
const wantDevices = computed(() => {
  const spec = props.cfg.device.trim()
  if (!spec) return props.hostPool          // null when the probe never answered
  // the API canonicalizes a bare "cuda" to cuda:0; an imported setup carrying
  // the short form must not read as a difference no restart can resolve
  return spec.split(',').map((s) => s.trim()).filter(Boolean)
             .map((s) => (s === 'cuda' ? 'cuda:0' : s))
})
const deviceDiffers = computed(() =>
  !!props.liveDevices && !!wantDevices.value
  && props.liveDevices.join(',') !== wantDevices.value.join(','))

/** The cap the form is asking for, in MiB — resolved as the server will: 0 is
 *  the host ceiling, and anything above it is CLAMPED to it, so asking for
 *  999999 on a 110000 host is not a difference. */
const wantCapMb = computed(() => {
  const max = props.historyMbMax
  if (!max) return null
  const v = props.cfg.history_mb
  return v > 0 ? Math.min(v, max) : max
})
const historyDiffers = computed(() =>
  props.historyCapMb != null && wantCapMb.value != null
  && props.historyCapMb !== wantCapMb.value)

/** The history tooltip NAMES the host ceiling. That number used to sit beside
 *  the field as "(0 = host max 110000)", which does not fit a third of this
 *  column — and the tooltip explained the 0 convention without ever saying
 *  what 0 resolves TO, which is the half you actually need. */
const historyHelp = computed(() =>
  'in-RAM frame history for this session (scrub/replay depth). 0 = the'
  + " host's WIGNERF_HISTORY_MB, which is also the ceiling — a session can ask"
  + ' for less, never more'
  + (props.historyMbMax ? ` (host max ${props.historyMbMax} MiB).` : '.'))

const computeStale = computed(() =>
  deviceDiffers.value || historyDiffers.value
  || (!!props.hasRun && runDiffers('precision')))

/**
 * Which settings reach a RUNNING session and which need a restart, on the
 * Physics heading. It ENUMERATES, so an omission reads as a promise: it once
 * listed only "grid & IC" as restart-only, leaving the RUN block unmentioned,
 * and a mode/t₂ edited in the form but never applied is indistinguishable from
 * a batch run ignoring its own t₂ (2026-07-23: a run believed to be "batch,
 * t₂=100" was really the old interactive session and computed straight past
 * t=100). Keep every group named.
 */
const WHAT_APPLIES_HELP = 'U, m, c, ℏ, tol, t dir and auto-expand apply LIVE'
  + ' at the frontier — the numeric fields on blur/Enter, U via its own'
  + ' "Apply live" button. Grid, IC, variants, RUN (mode, t₂, Δt rec) and'
  + ' COMPUTE (precision, device, history) need a session restart — the form'
  + ' marks any of those in amber while it disagrees with the running session.'

/** Physics fields apply on `@change` (blur/Enter), so a typed-but-not-yet-
 *  committed value is otherwise invisible — mark it, and say so in the note
 *  under the fields. */
function pending(field: keyof LivePhysics) {
  const lp = props.livePhysics
  return !!lp && lp[field] !== props.cfg[field]
}
const pendingPhysics = computed(() =>
  (['mass', 'c', 'hbar_eff', 'tol'] as const).some(pending))

const ndim = computed(() => props.cfg.grid.ndim)
const axisNames = computed(() => gridLabels(props.cfg))
// with real subscripts for the table's name column (see lib/axes.ts labelHtml)
const axisNamesHtml = computed(() =>
  axisNames.value.map((_, i) => labelHtml(ndim.value, i)))

// Per-axis size choices follow the HOST's ceiling for the ndim this form is
// SHOWING — WIGNERF_MAX_GRID / _MAX_GRID_2D, from /api/device, not from
// `status`, which reports the ceiling of the ndim that is RUNNING. See
// lib/config.axisSizeOptions for the floors and for what reading it off the
// session did to both directions of a dims switch.
const sizeOptions = computed(() => axisSizeOptions(
  ndim.value, props.maxGrid ?? (ndim.value > 1 ? 128 : 4096),
  props.cfg.grid.axes.map((a) => a.N)))

const cells = computed(() => gridCells(props.cfg.grid))
const overCells = computed(() =>
  props.maxCells != null && cells.value > props.maxCells)

/**
 * Estimated device footprint. Shown for 2D ONLY, and not as decoration: at N^4
 * this is the number that decides whether a session starts at all, and finding
 * out by OOM after pressing Restart is the failure this line exists to prevent.
 * The per-card figure assumes the workers spread evenly over the pool, which is
 * what session.assign_devices does.
 */
// CUDA context + cuFFT plan cache, per process per device. The frontend mirror
// of routers/sessions.CONTEXT_BYTES — move both together. No FIT_MARGIN mirror
// on purpose: that margin exists because FREE memory moves between the check
// and the first allocation, and this comparison is against TOTAL, which does
// not. Leaving it out keeps the panel strictly less trigger-happy than the
// server, which is the right direction for a pre-flight hint.
const CONTEXT_GIB = 300 / 1024

/**
 * Which devices this session's workers would land on, and how many each gets —
 * the frontend mirror of `core.session.assign_devices`. Move both together.
 *
 * The two properties that matter, and that a "spread evenly over the pool"
 * shortcut gets WRONG: only `min(pool, variants)` devices are used at all, and
 * the pool is ordered fastest-first so the earlier devices take the extra when
 * the split is uneven. A single-variant session therefore touches exactly ONE
 * device — the fastest — and says nothing whatsoever about the others.
 *
 * Getting that wrong is not academic: comparing one worker against the SMALLEST
 * pool device condemned 128×128×64×64 (13.0 GiB, one qn worker) as "will not fit
 * this host" against the 10.6 GiB 2080 Ti, while the worker was in fact bound
 * for the 23.6 GiB 3090 and the session started and computed happily. A
 * pre-flight hint that contradicts what the server then does is worse than no
 * hint, because it teaches you to ignore the line.
 */
const deviceLoad = computed(() => {
  const bpc = props.bytesPerCell
  const pool = wantDevices.value
  if (ndim.value < 2 || !bpc || !pool?.length) return null
  const per = cells.value * bpc / 1024 ** 3
  const nv = props.cfg.variants.length
  const k = Math.min(pool.length, nv)
  const base = Math.floor(nv / k)
  const extra = nv % k
  return Array.from({ length: k }, (_, j) => {
    const workers = base + (j < extra ? 1 : 0)
    return { spec: pool[j]!, workers, giB: workers * per }
  })
})

/**
 * Does any device this session would actually USE lack the memory to hold its
 * share? TOTAL, not free, deliberately: total is a static fact the form can
 * state without re-polling, so the panel can never contradict the server, whose
 * live-free refusal (`routers/sessions._fit_error`) is a strict superset.
 *
 * Unknown total ⇒ no warning, the same way _fit_error declines to refuse on
 * unknown free memory: there the cell rail is the only guard and guessing is
 * worse than staying quiet.
 */
const wontFit = computed(() => {
  const load = deviceLoad.value
  const totals = props.deviceTotals
  if (!load || !totals) return false
  return load.some((d) => {
    const t = totals[d.spec]
    return !!t && d.giB + CONTEXT_GIB > t / 1024 ** 3
  })
})

const footprint = computed(() => {
  const bpc = props.bytesPerCell
  if (ndim.value < 2 || !bpc) return ''
  const per = cells.value * bpc / 1024 ** 3
  const load = deviceLoad.value
  // The per-device figure only says something when more than one device is
  // actually USED. At one variant the split is 1×1, so "13.00 GiB per worker ·
  // 13.00 GiB/device" was the same number twice, dressed up as a distribution.
  const cardsNote = load && load.length > 1
    ? ` · ${Math.max(...load.map((d) => d.giB)).toFixed(2)} GiB/device`
    : ''
  return `≈ ${per.toFixed(2)} GiB per worker${cardsNote} · ${cells.value.toLocaleString()} cells`
})

/** Switching dimensionality rebuilds grid + IC, so it is restart-only. */
function onNdimChange(v: string) {
  setNdim(props.cfg, v === '2' ? 2 : 1)
  emit('dirty')
}

const showGrid = defineModel<boolean>('showGrid', { required: true })
const showCells = defineModel<boolean>('showCells', { required: true })

const CELL_LINES_HELP =
  'The lattice W is actually computed on, drawn faintly on the W panels and the'
  + ' IC preview, with the boundary watch\'s outer edge-band cells brighter —'
  + ' so the "in the outer N cells" in that warning points at something you can'
  + ' see. Dropped automatically when more than ~200 cell lines would land in'
  + ' the visible window; zoom in and they come back.'

const emit = defineEmits<{
  (e: 'restart'): void
  (e: 'dirty'): void
  (e: 'apply-live', params: Record<string, unknown>): void
  (e: 'potential-validity', valid: boolean): void
}>()

/** Restore the persisted setup (grid, potential, physics, run mode, IC,
 *  variants) to defaults; display prefs (layout, grid lines) are separate
 *  localStorage keys and stay untouched. */
function resetSetup() {
  if (!confirm('Reset the ENTIRE setup (grid, potential, physics, run mode, IC, variants) to defaults?')) return
  // A reset that changed NOTHING is not a change. Emitting `dirty`
  // unconditionally put "setup changed — restart to apply" over a form that
  // was already at the defaults — most visibly right after clearing local
  // data, where the very first thing the user can press announces a
  // divergence that does not exist. Same rule the backend's apply_params
  // follows: compare, and say nothing when the whole message is a no-op.
  const before = JSON.stringify(props.cfg)
  resetToDefaults(props.cfg)
  if (JSON.stringify(props.cfg) !== before) emit('dirty')
}

// Auto-expand can move the SESSION's domain away from the form's grid;
// show the live domain when they differ, with a one-click adopt so a
// restart reproduces the expanded window.
const liveDiffers = computed(() => {
  const lg = props.liveGrid
  if (!lg) return false
  const ax = props.cfg.grid.axes
  // A DIMENSIONALITY change is not what this line is for. It exists for one
  // case — auto-expand moved the session's domain away from the form, and
  // "adopt" copies the expanded window back — which only means anything within
  // one ndim. Across a switch it printed the running 1D box under a 2D form,
  // reading as if that were the 2D grid; and `adopt` would have been an active
  // trap, silently reverting grid.ndim to 1 while the IC stayed 2D, which the
  // API then refuses. The dims marker beside the heading already says it.
  if (lg.ndim !== props.cfg.grid.ndim) return false
  if (lg.N.length !== ax.length) return true
  return ax.some((a, i) =>
    a.lo !== lg.lo[i] || a.hi !== lg.hi[i] || a.N !== lg.N[i])
})
const fmt = (v: number) => String(+v.toFixed(4))
const liveText = computed(() => {
  const lg = props.liveGrid
  if (!lg) return ''
  const box = lg.lo.map((lo, i) => `[${fmt(lo)}, ${fmt(lg.hi[i]!)}]`).join(' × ')
  return `${box} ${lg.N.join('×')}`
})
function adoptLive() {
  const lg = props.liveGrid
  if (!lg) return
  props.cfg.grid.ndim = lg.ndim === 2 ? 2 : 1
  props.cfg.grid.axes.splice(0, props.cfg.grid.axes.length,
    ...lg.N.map((N, i) => ({ lo: lg.lo[i]!, hi: lg.hi[i]!, N })))
  emit('dirty')
}
</script>

<template>
  <div class="space-y-4">
    <PotentialEditor
      v-model="props.cfg.potential"
      :grid="props.cfg.grid" :hbar-eff="props.cfg.hbar_eff"
      :live="live" :computing="computing ?? false" :variants="props.cfg.variants"
      :live-expr="props.livePhysics?.potential ?? null"
      :live-ndim="props.liveGrid?.ndim ?? null"
      @apply-live="(expr) => emit('apply-live', { U: expr })"
      @validity="(v) => emit('potential-validity', v)"
      @grid-dirty="emit('dirty')"
    />

    <section class="space-y-1.5">
      <!-- the ? marks the heading as hoverable: a tooltip nobody knows is there
           is the same as no tooltip -->
      <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider
                 cursor-help w-fit" :title="WHAT_APPLIES_HELP">Physics <span
        class="text-dim normal-case">(what applies live?)</span></h3>
      <!-- Seven columns so ONE grid gives two differently-shaped rows AND lets
           them be uneven where the content is: m, c, ℏ across the first, tol +
           t dir across the second. c gets 3 of the 7 because it is the only
           field here holding a long number (137.035999 needs ~10 digits, which
           an equal third of a 320px column truncates); m and ℏ hold 1-ish, and
           tol's width goes to its LABEL, which grows by " ≥1e-5" in float32.
           The m/c/ℏ labels are one glyph, so a fixed `w-4` aligns their three
           inputs; tol's and t dir's are words and size to content. -->
      <div class="grid grid-cols-7 gap-x-2 gap-y-1 text-xs">
        <label class="col-span-2 flex items-center gap-1">
          <span class="w-4 text-muted" title="rest mass, mₑ = 1">m</span>
          <input v-model.number="props.cfg.mass" type="number" step="any" min="0"
                 class="wf-num min-w-0" :class="pending('mass') && 'wf-pending'"
                 @change="emit('apply-live', { mass: props.cfg.mass })" />
        </label>
        <label class="col-span-3 flex items-center gap-1">
          <span class="w-4 text-muted"
                title="speed of light: 137.036 physical, 1 = old toy runs">c</span>
          <input v-model.number="props.cfg.c" type="number" step="any" min="0.1"
                 class="wf-num min-w-0" :class="pending('c') && 'wf-pending'"
                 @change="emit('apply-live', { c: props.cfg.c })" />
        </label>
        <label class="col-span-2 flex items-center gap-1">
          <span class="w-4 text-muted"
                title="value of ℏ in the evolution equations (a.u.: physical value 1); dial it below 1 to watch the classical limit emerge">ℏ</span>
          <input v-model.number="props.cfg.hbar_eff" type="number" step="any" min="0.001"
                 class="wf-num min-w-0" :class="pending('hbar_eff') && 'wf-pending'"
                 @change="emit('apply-live', { hbar_eff: props.cfg.hbar_eff })" />
        </label>
        <label class="col-span-4 flex items-center gap-1"
               :title="f32 ? TOL_F32_HELP : TOL_HELP">
          <span class="shrink-0 whitespace-nowrap text-muted">tol<template
            v-if="f32"> ≥{{ TOL_MIN_F32.toExponential() }}</template></span>
          <input v-model.number="props.cfg.tol" type="number" step="any"
                 :min="f32 ? TOL_MIN_F32 : 1e-6" max="0.5"
                 class="wf-num min-w-0" :class="pending('tol') && 'wf-pending'"
                 @change="clampTol(); emit('apply-live', { tol: props.cfg.tol })" />
        </label>
        <label class="col-span-3 flex items-center gap-1">
          <span class="shrink-0 whitespace-nowrap text-muted"
                title="time direction of NEWLY computed records: flips the sign of dt in the propagator at the frontier. Already-computed history is unaffected — use the timeline to move within it. Shortcut: R">t dir</span>
          <select class="wf-num min-w-0" :value="(props.sign ?? 1) > 0 ? 1 : -1"
                  @change="emit('apply-live', { dt_sign: Number(($event.target as HTMLSelectElement).value) })">
            <option :value="1">forward</option>
            <option :value="-1">backward</option>
          </select>
        </label>
      </div>
      <!-- Only the TRANSIENT half stays inline. The live-vs-restart enumeration
           it used to share this slot with is permanent by nature, and a
           permanent three-line paragraph is exactly what the narrow first column
           cannot afford — it moved to the section heading's tooltip (WHAT_APPLIES
           _HELP), where it is still one hover away and still enumerates. -->
      <p v-if="pendingPhysics" class="text-xs text-warn">
        edited (amber): press Enter or leave the field to apply it live.
      </p>
    </section>

    <section class="space-y-1.5">
      <div class="flex items-baseline justify-between">
        <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider">Grid</h3>
        <!-- restart-only: dimensionality fixes the grid, the IC and the whole
             worker construction. Amber when it differs from the run, exactly
             like the other restart-only fields. -->
        <label class="flex items-center gap-1 text-xs"
               :class="liveGrid && liveGrid.ndim !== ndim ? 'text-warn' : 'text-muted'"
               :title="NDIM_HELP">
          <span>dims</span>
          <select :value="String(ndim)" class="wf-num !w-14"
                  @change="onNdimChange(($event.target as HTMLSelectElement).value)">
            <option value="1">1D</option>
            <option value="2">2D</option>
          </select>
        </label>
      </div>
      <!-- One row per phase-space axis: name, lo, hi, N. A table rather than
           four labelled rows because at 2D there are FOUR axes, and "x₁,x₂"
           style labels would spend the width twice over in a 320px column. -->
      <div class="grid grid-cols-[1.2rem_1fr_1fr_3.6rem] gap-x-1 gap-y-1 text-xs
                  items-center">
        <span></span>
        <span class="text-muted text-[10px]">lo</span>
        <span class="text-muted text-[10px]">hi</span>
        <span class="text-muted text-[10px]">N</span>
        <template v-for="(a, i) in props.cfg.grid.axes" :key="i">
          <span class="text-muted italic" v-html="axisNamesHtml[i]"></span>
          <input v-model.number="a.lo" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
          <input v-model.number="a.hi" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
          <select v-model.number="a.N" class="wf-num" @change="emit('dirty')">
            <option v-for="n in sizeOptions" :key="n" :value="n">{{ n }}</option>
          </select>
        </template>
      </div>
      <div class="flex items-center gap-x-3 gap-y-1 flex-wrap text-xs">
        <!-- All THREE toggles share one row: in portrait the first column is
             what the W panels are competing with, so a checkbox that needs a
             third of a line must not take a whole one. Natural widths (flex,
             not equal thirds) plus short labels are what make three fit inside
             320px — verified at the real width, per the UI-debugging note in
             CLAUDE.md. flex-wrap is only a safety valve.
             That budget is why auto-expand's gate marker is "(1D)" / "(f64)"
             rather than a spelled-out clause: it stays PERMANENT and visible,
             which is the load-bearing part (a touch device gets no hover), and
             the full reason lives in the tooltip. The float32 gate is still
             stated three ways, none of them a standing paragraph: this marker,
             the tooltip, and the one-off amber note in Compute at the switch. -->
        <label class="flex items-center gap-1 select-none"
               :class="expandGated ? 'cursor-not-allowed' : 'cursor-pointer'"
               :title="ndim > 1 ? AUTO_EXPAND_2D_HELP
                       : (f32 ? AUTO_EXPAND_F32_HELP : AUTO_EXPAND_HELP)">
          <input type="checkbox" v-model="props.cfg.auto_expand"
                 :disabled="expandGated"
                 @change="emit('apply-live', { auto_expand: props.cfg.auto_expand })" />
          <span :class="expandGated ? 'text-dim' : 'text-fg-3'">auto-expand<template
            v-if="ndim > 1"> (1D)</template><template
            v-else-if="f32"> (f64)</template></span>
        </label>
        <label class="flex items-center gap-1 cursor-pointer select-none"
               title="axis grid lines at nice value intervals — on all plots, the W panels and the IC preview">
          <input type="checkbox" v-model="showGrid" />
          <span class="text-fg-3">grid</span>
        </label>
        <label class="flex items-center gap-1 cursor-pointer select-none"
               :title="CELL_LINES_HELP">
          <input type="checkbox" v-model="showCells" />
          <span class="text-fg-3">cells</span>
        </label>
      </div>
      <!-- 2D only: what this grid will cost a card, BEFORE the restart that
           would find out by OOM. Red once past the host's cell ceiling, which
           is the refusal the create call would return. -->
      <p v-if="footprint" class="text-xs tabular-nums"
         :class="overCells || wontFit ? 'text-error' : 'text-fg-3'"
         :title="overCells
           ? `over the host's WIGNERF_MAX_CELLS_2D (${maxCells?.toLocaleString()}) — creating this session will be refused`
           : wontFit
           ? 'more memory than the smallest device in the pool physically has,'
             + ' so this grid cannot start on this host at all — the workers'
             + ' spread over the pool, so the smaller card is what binds.'
             + ' Reduce an axis or drop a variant.'
           : 'estimated device memory per variant worker at this grid. Mostly '
             + 'NOT the state: W is real (float64, 8 B/cell), and the rest is '
             + 'the step\'s machinery at full shape — two exponent slots '
             + '(2 complex meshes each), the dU/dT rate meshes, the FFT work '
             + 'arrays and adjust_step\'s two candidate states.'">
        {{ footprint }}<template v-if="overCells"> — over the host cap</template>
        <template v-else-if="wontFit"> — will not fit this host</template>
      </p>
      <p v-if="liveDiffers" class="text-xs text-warn">
        live: {{ liveText }}
        <button class="underline ml-1" title="copy the live domain into the setup so a restart reproduces it"
                @click="adoptLive">adopt</button>
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider">Run</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1"
               title="interactive: no end time — Solve keeps computing new records until you pause, streaming a live preview you can zoom. batch: Solve computes at full speed until t = t₂ with NO frame streaming — the heatmap/marginals dim and only a progress report (t, %, throughput) is sent, so heavy runs are not slowed by transferring frames; the observable curves still update. Then the button becomes Play — playback of the finished history.">
          <span class="w-14 text-muted">mode</span>
          <select v-model="props.cfg.mode" class="wf-num"
                  :class="runDiffers('mode') ? 'text-warn' : ''">
            <option value="interactive">interactive</option>
            <option value="batch">batch</option>
          </select>
        </label>
        <label class="flex items-center gap-1" v-if="props.cfg.mode === 'batch'">
          <span class="w-14 text-muted">t₂</span>
          <input v-model.number.lazy="props.cfg.t2" type="number" step="any"
                 class="wf-num"
                 :class="runDiffers('t2') ? 'text-warn' : ''" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-14 text-muted" title="physical time per record">Δt rec</span>
          <input v-model.number.lazy="props.cfg.record_dt" type="number" step="any" min="0.001"
                 class="wf-num"
                 :class="runDiffers('record_dt') ? 'text-warn' : ''" />
        </label>
        <!-- playback speed lives ONLY in the transport bar; a session
             always starts at 1.00 a.u./s -->
      </div>
      <!-- Name what is ACTUALLY running. The amber fields say "this differs";
           only this line says WHAT you are computing under, which is the fact
           you need when a run does not do what the form describes. The RUN
           fields only — the precision it used to append belongs to COMPUTE and
           is stated by COMPUTE's own line, so quoting it here just said the
           same word twice whenever both differed. -->
      <p v-if="runStale" class="text-xs text-warn">
        running: {{ liveRun!.mode === 'batch'
                    ? `batch, t₂ = ${liveRun!.t2}` : 'interactive (no t₂)' }},
        Δt rec = {{ liveRun!.record_dt }}
        — restart to apply the values above
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider">Compute</h3>
      <!-- One row, three columns — these were three full-width rows each
           carrying one short control. The label sits ABOVE its control here
           rather than beside it, unlike every other section: "precision",
           "device" and "history" are WORDS, and a third of this column is
           ~100px, which an inline label plus a select showing "float64" does
           not fit. Stacked still costs two lines instead of three. -->
      <div class="grid grid-cols-3 gap-x-2 gap-y-1 text-xs">
        <!-- The "(1D)" marker is the same permanent, line-free gate marker
             auto-expand carries, for the same reason: a touch device has no
             hover, and the full reason belongs in the tooltip. -->
        <label class="flex flex-col gap-0.5 min-w-0"
               :title="precisionGated ? PRECISION_2D_HELP : PRECISION_HELP">
          <span :class="precisionGated ? 'text-dim' : 'text-muted'">precision<template
            v-if="precisionGated"> (1D)</template></span>
          <!-- onPrecisionChange marks the choice (until the user operates THIS
               control the form only holds a placeholder and the create payload
               omits precision, so the host's WIGNERF_PRECISION decides) and
               applies the float32 invariants synchronously — see its comment
               for why a watcher is too late. -->
          <select v-model="props.cfg.precision" class="wf-num"
                  :disabled="precisionGated"
                  :class="runDiffers('precision') ? 'text-warn' : ''"
                  @change="onPrecisionChange()">
            <option value="float64">float64</option>
            <option value="float32">float32</option>
          </select>
        </label>
        <label class="flex flex-col gap-0.5 min-w-0"
               title="which device(s) this session's variant workers run on. Default spreads them over the host's whole WIGNERF_DEVICE pool, costliest variant to the fastest card; pick one device to keep a session off a card you need for something else.">
          <span class="text-muted">device</span>
          <select v-model="props.cfg.device" class="wf-num"
                  :class="deviceDiffers ? 'text-warn' : ''"
                  @change="emit('dirty')">
            <option value="">default (host pool)</option>
            <option v-for="d in props.deviceOptions ?? []" :key="d.spec" :value="d.spec">
              {{ d.device }}
            </option>
            <option v-if="deviceMissing" :value="props.cfg.device">
              {{ props.cfg.device }} — not on this server
            </option>
          </select>
        </label>
        <label class="flex flex-col gap-0.5 min-w-0" :title="historyHelp">
          <span class="text-muted">history</span>
          <!-- the unit stays to the RIGHT of the field, as it reads; the
               "0 = host max N" it used to carry alongside does not fit a third
               of this column and moved into the tooltip above, which had to
               name the number anyway -->
          <span class="flex items-center gap-1">
            <input v-model.number.lazy="props.cfg.history_mb" type="number" step="64" min="0"
                   :max="props.historyMbMax ?? undefined" class="wf-num min-w-0"
                   :class="historyDiffers ? 'text-warn' : ''"
                   @change="clampHistory(); emit('dirty')" />
            <span class="shrink-0 text-muted">MiB</span>
          </span>
        </label>
        <!-- What the switch just did to two OTHER settings — a fact about the
             switch, not a standing property of the session, so it lives only
             while the choice is PENDING and "Restart session" clears it. It
             used to open with "single precision mode", which said nothing the
             select, its tooltip and the header's float32 badge do not already
             say, and which appeared even when the switch had changed nothing
             else (a form already at tol ≥ 1e-5 with auto-expand off). What is
             left is only the part no other control reports. -->
        <p v-if="f32 && runDiffers('precision') && f32Applied.length"
           class="col-span-3 text-xs text-warn/90 -mt-0.5">
          {{ f32Applied.join('; ') }}
        </p>
        <p v-if="deviceMissing" class="col-span-3 text-xs text-warn">
          this backend has no {{ props.cfg.device }} — Restart will be refused
          until you pick another.
        </p>
      </div>
      <!-- Only when a field DISAGREES with the session. The steady-state facts
           live where they cost nothing: the devices and the history cap ride the
           timeline's own "hist 0.1 / 107 GiB · dev: …" readout, and float32 has
           the header badge. This line exists for the same reason as the RUN
           one — the amber fields say "this differs", only this says WHAT you are
           computing under, which is the fact you need when a run does not do
           what the form describes. -->
      <p v-if="computeStale" class="text-xs text-warn">
        running {{ liveRun?.precision ?? '' }} on
        {{ props.liveDevices?.join(', ') || '—' }}<template
          v-if="props.historyCapMb">, history cap {{ props.historyCapMb }} MiB</template>
        — restart to apply the values above
      </p>
      <div class="flex gap-2">
        <button class="wf-solid flex-1 py-1.5 rounded bg-sky-700 hover:bg-sky-600
                       font-medium whitespace-nowrap"
                @click="emit('restart')">Restart session</button>
        <button class="flex-1 py-1.5 rounded bg-raised hover:bg-raised-hover
                       text-fg-2 whitespace-nowrap"
                title="restore grid, U, physics, run mode, IC and variants to the defaults FOR THE CURRENT DIMENSIONALITY (dims itself is kept)"
                @click="resetSetup">Reset to defaults</button>
      </div>
    </section>
  </div>
</template>
