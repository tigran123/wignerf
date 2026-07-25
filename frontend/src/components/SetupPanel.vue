<script setup lang="ts">
/**
 * Setup controls: potential editor, physics parameters (live-applied),
 * grid geometry + grid-lines display toggle, run mode. The IC editor is a
 * separate component so the portrait layout can place it in its own column.
 */
import { computed, ref, watch } from 'vue'
import PotentialEditor from './PotentialEditor.vue'
import { applyPrecisionInvariants, markPrecisionChosen, resetToDefaults,
         TOL_MIN_F32, type GridCfg, type LivePhysics, type LiveRun,
         type SimConfig } from '../lib/config'

const props = defineProps<{ cfg: SimConfig; live: boolean; sign?: number
                            liveGrid?: GridCfg | null; maxGrid?: number
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
const runStale = computed(() =>
  (['mode', 't2', 'record_dt', 'precision'] as const).some(runDiffers))

const f32 = computed(() => props.cfg.precision === 'float32')

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
const WHAT_APPLIES_HELP = 'm, c, ℏ, tol, t dir and auto-expand apply LIVE at the'
  + ' frontier. Grid, IC, variants, RUN (mode, t₂, Δt rec) and COMPUTE'
  + ' (precision, device, history) need a session restart — the form marks any'
  + ' of those in amber while it disagrees with the running session.'

/** Physics fields apply on `@change` (blur/Enter), so a typed-but-not-yet-
 *  committed value is otherwise invisible — mark it, and say so in the note
 *  under the fields. */
function pending(field: keyof LivePhysics) {
  const lp = props.livePhysics
  return !!lp && lp[field] !== props.cfg[field]
}
const pendingPhysics = computed(() =>
  (['mass', 'c', 'hbar_eff', 'tol'] as const).some(pending))

// Nx/Np choices follow the SERVER's per-axis ceiling (WIGNERF_MAX_GRID,
// reported in status) instead of a hardcoded list; the form's current
// values stay listed even if a lower-capped backend would reject them,
// so the select never renders blank.
const sizeOptions = computed(() => {
  const out: number[] = []
  for (let n = 256; n <= Math.max(props.maxGrid ?? 4096, 256); n *= 2) out.push(n)
  for (const v of [props.cfg.grid.Nx, props.cfg.grid.Np])
    if (!out.includes(v)) out.push(v)
  return out.sort((a, b) => a - b)
})

const showGrid = defineModel<boolean>('showGrid', { required: true })

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
  resetToDefaults(props.cfg)
  emit('dirty')
}

// Auto-expand can move the SESSION's domain away from the form's grid;
// show the live domain when they differ, with a one-click adopt so a
// restart reproduces the expanded window.
const liveDiffers = computed(() => {
  const lg = props.liveGrid
  if (!lg) return false
  const g = props.cfg.grid
  return lg.x1 !== g.x1 || lg.x2 !== g.x2 || lg.Nx !== g.Nx
      || lg.p1 !== g.p1 || lg.p2 !== g.p2 || lg.Np !== g.Np
})
const fmt = (v: number) => String(+v.toFixed(4))
function adoptLive() {
  if (!props.liveGrid) return
  Object.assign(props.cfg.grid, props.liveGrid)
  emit('dirty')
}
</script>

<template>
  <div class="space-y-4">
    <PotentialEditor
      v-model="props.cfg.potential"
      :grid="props.cfg.grid" :hbar-eff="props.cfg.hbar_eff"
      :live="live" :variants="props.cfg.variants"
      :live-expr="props.livePhysics?.potential ?? null"
      @update:model-value="emit('dirty')"
      @apply-live="(expr) => emit('apply-live', { U: expr })"
      @validity="(v) => emit('potential-validity', v)"
      @grid-dirty="emit('dirty')"
    />

    <section class="space-y-1.5">
      <!-- the ? marks the heading as hoverable: a tooltip nobody knows is there
           is the same as no tooltip -->
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider
                 cursor-help w-fit" :title="WHAT_APPLIES_HELP">Physics <span
        class="text-neutral-600 normal-case">(what applies live?)</span></h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500" title="rest mass, mₑ = 1">m</span>
          <input v-model.number="props.cfg.mass" type="number" step="any" min="0"
                 class="wf-num" :class="pending('mass') && 'wf-pending'"
                 @change="emit('apply-live', { mass: props.cfg.mass })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500"
                title="speed of light: 137.036 physical, 1 = old toy runs">c</span>
          <input v-model.number="props.cfg.c" type="number" step="any" min="0.1"
                 class="wf-num" :class="pending('c') && 'wf-pending'"
                 @change="emit('apply-live', { c: props.cfg.c })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500"
                title="value of ℏ in the evolution equations (a.u.: physical value 1); dial it below 1 to watch the classical limit emerge">ℏ</span>
          <input v-model.number="props.cfg.hbar_eff" type="number" step="any" min="0.001"
                 class="wf-num" :class="pending('hbar_eff') && 'wf-pending'"
                 @change="emit('apply-live', { hbar_eff: props.cfg.hbar_eff })" />
        </label>
        <label class="flex items-center gap-1" :title="f32 ? TOL_F32_HELP : TOL_HELP">
          <span class="w-10 text-neutral-500">tol<template
            v-if="f32"> ≥{{ TOL_MIN_F32.toExponential() }}</template></span>
          <input v-model.number="props.cfg.tol" type="number" step="any"
                 :min="f32 ? TOL_MIN_F32 : 1e-6" max="0.5"
                 class="wf-num" :class="pending('tol') && 'wf-pending'"
                 @change="clampTol(); emit('apply-live', { tol: props.cfg.tol })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500"
                title="time direction of NEWLY computed records: flips the sign of dt in the propagator at the frontier. Already-computed history is unaffected — use the timeline to move within it. Shortcut: R">t dir</span>
          <select class="wf-num" :value="(props.sign ?? 1) > 0 ? 1 : -1"
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
      <p v-if="pendingPhysics" class="text-xs text-amber-400">
        edited (amber): press Enter or leave the field to apply it live.
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Grid</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">x₁,x₂</span>
          <input v-model.number="props.cfg.grid.x1" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
          <input v-model.number="props.cfg.grid.x2" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">p₁,p₂</span>
          <input v-model.number="props.cfg.grid.p1" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
          <input v-model.number="props.cfg.grid.p2" type="number" step="any"
                 class="wf-num" @change="emit('dirty')" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">Nx</span>
          <select v-model.number="props.cfg.grid.Nx" class="wf-num" @change="emit('dirty')">
            <option v-for="n in sizeOptions" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500">Np</span>
          <select v-model.number="props.cfg.grid.Np" class="wf-num" @change="emit('dirty')">
            <option v-for="n in sizeOptions" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <!-- The two toggles SHARE a row: in portrait the first column is what
             the W panels are competing with, so a checkbox that needs half a
             line must not take a whole one. Labels are short and the tooltips
             carry the full meaning. In float32 auto-expand takes the full width
             instead, because its "— float64 only" suffix wraps in half of a
             narrow column — and a wrapped cell drags its row-mate taller too,
             so pairing them there would COST a line rather than save one. The
             common case keeps the saving; the marker keeps its full wording.
             The float32 gate on auto-expand is stated three ways, none of them a
             standing paragraph: the "float64 only" suffix (permanent, and the
             only one a touch device gets), the tooltip (the full reason), and
             the one-off amber note in Compute when the switch is made. -->
        <label class="flex items-center gap-1 select-none"
               :class="f32 ? 'cursor-not-allowed col-span-2' : 'cursor-pointer'"
               :title="f32 ? AUTO_EXPAND_F32_HELP : AUTO_EXPAND_HELP">
          <input type="checkbox" v-model="props.cfg.auto_expand" :disabled="f32"
                 @change="emit('apply-live', { auto_expand: props.cfg.auto_expand })" />
          <span :class="f32 ? 'text-neutral-600' : 'text-neutral-400'">auto-expand<template
            v-if="f32"> — float64 only</template></span>
        </label>
        <label class="flex items-center gap-1 cursor-pointer select-none"
               title="axis grid lines on all plots, the W panels and the IC preview">
          <input type="checkbox" v-model="showGrid" />
          <span class="text-neutral-400">grid lines</span>
        </label>
      </div>
      <p v-if="liveDiffers" class="text-xs text-amber-400">
        live: [{{ fmt(liveGrid!.x1) }}, {{ fmt(liveGrid!.x2) }}] ×
        [{{ fmt(liveGrid!.p1) }}, {{ fmt(liveGrid!.p2) }}]
        {{ liveGrid!.Nx }}×{{ liveGrid!.Np }}
        <button class="underline ml-1" title="copy the live domain into the setup so a restart reproduces it"
                @click="adoptLive">adopt</button>
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Run</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1"
               title="interactive: no end time — Solve keeps computing new records until you pause, streaming a live preview you can zoom. batch: Solve computes at full speed until t = t₂ with NO frame streaming — the heatmap/marginals dim and only a progress report (t, %, throughput) is sent, so heavy runs are not slowed by transferring frames; the observable curves still update. Then the button becomes Play — playback of the finished history.">
          <span class="w-14 text-neutral-500">mode</span>
          <select v-model="props.cfg.mode" class="wf-num"
                  :class="runDiffers('mode') ? 'text-amber-400' : ''">
            <option value="interactive">interactive</option>
            <option value="batch">batch</option>
          </select>
        </label>
        <label class="flex items-center gap-1" v-if="props.cfg.mode === 'batch'">
          <span class="w-14 text-neutral-500">t₂</span>
          <input v-model.number.lazy="props.cfg.t2" type="number" step="any"
                 class="wf-num"
                 :class="runDiffers('t2') ? 'text-amber-400' : ''" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-14 text-neutral-500" title="physical time per record">Δt rec</span>
          <input v-model.number.lazy="props.cfg.record_dt" type="number" step="any" min="0.001"
                 class="wf-num"
                 :class="runDiffers('record_dt') ? 'text-amber-400' : ''" />
        </label>
        <!-- playback speed lives ONLY in the transport bar; a session
             always starts at 1.00 a.u./s -->
      </div>
      <!-- Name what is ACTUALLY running. The amber fields say "this differs";
           only this line says WHAT you are computing under, which is the fact
           you need when a run does not do what the form describes. -->
      <p v-if="runStale" class="text-xs text-amber-400">
        running: {{ liveRun!.mode === 'batch'
                    ? `batch, t₂ = ${liveRun!.t2}` : 'interactive (no t₂)' }},
        Δt rec = {{ liveRun!.record_dt }}, {{ liveRun!.precision }}
        — restart to apply the values above
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Compute</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1 col-span-2"
               title="spectral working precision. float64 is the physics setting. float32 is a PREVIEW mode: ~3.3-3.8× faster and ~58% of the VRAM on CUDA (no speedup on CPU), but purity and energy drift by ~1e-4 with the same secular signature as boundary wrap, and ΔX·ΔP noise is ~150× the relativistic shear. The exponents are built in double either way.">
          <span class="w-14 text-neutral-500">precision</span>
          <!-- onPrecisionChange marks the choice (until the user operates THIS
               control the form only holds a placeholder and the create payload
               omits precision, so the host's WIGNERF_PRECISION decides) and
               applies the float32 invariants synchronously — see its comment
               for why a watcher is too late. -->
          <select v-model="props.cfg.precision" class="wf-num"
                  :class="runDiffers('precision') ? 'text-amber-400' : ''"
                  @change="onPrecisionChange()">
            <option value="float64">float64</option>
            <option value="float32">float32</option>
          </select>
        </label>
        <!-- Only while the choice is PENDING: it names what the switch just did
             to two OTHER settings, which is a fact about the switch, not a
             standing property of the session. "Restart session" applies the
             precision, `runDiffers` goes false, and this clears itself — from
             then on the header badge carries the one permanent fact. The
             reasons stay in each control's tooltip, and the controls keep a
             compact marker for devices with no hover. -->
        <p v-if="f32 && runDiffers('precision')"
           class="col-span-2 text-xs text-amber-400/90 -mt-0.5">
          single precision mode<template v-if="f32Applied.length">;
          {{ f32Applied.join('; ') }}</template>
        </p>
        <label class="flex items-center gap-1 col-span-2"
               title="which device(s) this session's variant workers run on. Default spreads them over the host's whole WIGNERF_DEVICE pool, costliest variant to the fastest card; pick one device to keep a session off a card you need for something else.">
          <span class="w-14 text-neutral-500">device</span>
          <select v-model="props.cfg.device" class="wf-num flex-1"
                  :class="deviceDiffers ? 'text-amber-400' : ''"
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
        <p v-if="deviceMissing" class="col-span-2 text-xs text-amber-400">
          this backend has no {{ props.cfg.device }} — Restart will be refused
          until you pick another.
        </p>
        <label class="flex items-center gap-1 col-span-2"
               title="in-RAM frame history for this session (scrub/replay depth). 0 = the host's WIGNERF_HISTORY_MB, which is also the ceiling — a session can ask for less, never more.">
          <span class="w-14 text-neutral-500">history</span>
          <input v-model.number.lazy="props.cfg.history_mb" type="number" step="64" min="0"
                 :max="props.historyMbMax ?? undefined" class="wf-num"
                 :class="historyDiffers ? 'text-amber-400' : ''"
                 @change="clampHistory(); emit('dirty')" />
          <span class="text-neutral-500">MiB
            <template v-if="props.historyMbMax">(0 = host max {{ props.historyMbMax }})</template>
          </span>
        </label>
      </div>
      <!-- Only when a field DISAGREES with the session. The steady-state facts
           live where they cost nothing: the devices and the history cap ride the
           timeline's own "hist 0.1 / 107 GiB · dev: …" readout, and float32 has
           the header badge. This line exists for the same reason as the RUN
           one — the amber fields say "this differs", only this says WHAT you are
           computing under, which is the fact you need when a run does not do
           what the form describes. -->
      <p v-if="computeStale" class="text-xs text-amber-400">
        running {{ liveRun?.precision ?? '' }} on
        {{ props.liveDevices?.join(', ') || '—' }}<template
          v-if="props.historyCapMb">, history cap {{ props.historyCapMb }} MiB</template>
        — restart to apply the values above
      </p>
      <button class="w-full py-1.5 rounded bg-sky-800 hover:bg-sky-700 font-medium"
              @click="emit('restart')">Restart session</button>
      <button class="w-full py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-xs text-neutral-300"
              title="restore grid, potential, physics, run mode, IC and variants to their defaults"
              @click="resetSetup">Reset setup to defaults</button>
    </section>
  </div>
</template>
