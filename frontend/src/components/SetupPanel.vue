<script setup lang="ts">
/**
 * Setup controls: potential editor, physics parameters (live-applied),
 * grid geometry + grid-lines display toggle, run mode. The IC editor is a
 * separate component so the portrait layout can place it in its own column.
 */
import { computed, watch } from 'vue'
import PotentialEditor from './PotentialEditor.vue'
import { markPrecisionChosen, resetToDefaults, type GridCfg,
         type LivePhysics, type LiveRun, type SimConfig } from '../lib/config'

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

// The backend refuses float32 + auto-expand: in single precision a contained
// state's own spectral noise passes the 1e-6 edge trigger within a few hundred
// steps and the 1e-8 support scan reads the whole axis, so the domain would
// double for no physical reason. Say that where the checkbox is, and clear the
// checkbox rather than let Restart fail with a 422.
const f32 = computed(() => props.cfg.precision === 'float32')
watch(f32, (on) => { if (on && props.cfg.auto_expand) props.cfg.auto_expand = false })

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
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Physics</h3>
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
        <label class="flex items-center gap-1">
          <span class="w-10 text-neutral-500" title="adaptive-step relative tolerance">tol</span>
          <input v-model.number="props.cfg.tol" type="number" step="any" min="1e-6" max="0.5"
                 class="wf-num" :class="pending('tol') && 'wf-pending'"
                 @change="emit('apply-live', { tol: props.cfg.tol })" />
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
      <p class="text-xs" :class="pendingPhysics ? 'text-amber-400' : 'text-neutral-400'">
        <template v-if="pendingPhysics">
          edited (amber): press Enter or leave the field to apply it live.
        </template>
        <template v-else>
          <!-- This line ENUMERATES, so an omission reads as a promise. It
               used to list only "grid &amp; IC" as restart-only, leaving the RUN
               block unmentioned — and mode/t₂ changed in the form but never
               applied is indistinguishable from a batch run that ignored its
               own t₂ (2026-07-23: a run "in batch t₂=100" was really the
               old interactive session, and computed straight past t=100). -->
          m, c, ℏ, tol, t dir and auto-expand apply live at the frontier;
          grid, IC, variants, RUN (mode, t₂, Δt rec) and COMPUTE
          (precision, device, history) need a restart.
        </template>
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
        <label class="flex items-center gap-1 col-span-2 select-none"
               :class="f32 ? 'cursor-not-allowed' : 'cursor-pointer'"
               title="when W(x,p,t) approaches a domain edge, move or double the domain automatically at the frontier (exact — the lattice spacing is frozen, values are never interpolated). Applies live; detection and its warning run either way.">
          <input type="checkbox" v-model="props.cfg.auto_expand" :disabled="f32"
                 @change="emit('apply-live', { auto_expand: props.cfg.auto_expand })" />
          <span :class="f32 ? 'text-neutral-600' : 'text-neutral-400'">auto-expand domain</span>
        </label>
        <!-- never a bare disabled control: say why, here, not in a tooltip -->
        <p v-if="f32" class="col-span-2 text-xs text-neutral-500 -mt-0.5">
          unavailable in float32: a contained state's own single-precision noise
          passes the edge trigger within a few hundred steps, so the domain would
          double for no physical reason. Detection still warns you.
        </p>
        <label class="flex items-center gap-1 col-span-2 cursor-pointer select-none"
               title="axis grid lines on all plots, the W panels and the IC preview">
          <input type="checkbox" v-model="showGrid" />
          <span class="text-neutral-400">grid lines on plots</span>
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
          <span class="w-14 text-neutral-500" title="physical time per record">Δτ rec</span>
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
        Δτ rec = {{ liveRun!.record_dt }}, {{ liveRun!.precision }}
        — restart to apply the values above
      </p>
    </section>

    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Compute</h3>
      <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
        <label class="flex items-center gap-1 col-span-2"
               title="spectral working precision. float64 is the physics setting. float32 is a PREVIEW mode: ~3.3-3.8× faster and ~58% of the VRAM on CUDA (no speedup on CPU), but purity and energy drift by ~1e-4 with the same secular signature as boundary wrap, and ΔX·ΔP noise is ~150× the relativistic shear. The exponents are built in double either way.">
          <span class="w-14 text-neutral-500">precision</span>
          <!-- markPrecisionChosen: until the user operates THIS control the
               form only holds a placeholder and the create payload omits
               precision so the host's WIGNERF_PRECISION decides. Operating it
               is the decision, and from here on it is sent explicitly. -->
          <select v-model="props.cfg.precision" class="wf-num"
                  :class="runDiffers('precision') ? 'text-amber-400' : ''"
                  @change="markPrecisionChosen()">
            <option value="float64">float64</option>
            <option value="float32">float32</option>
          </select>
        </label>
        <!-- Only while the choice is PENDING. Once the session is actually
             running in float32 the header badge carries it permanently, and
             two standing warnings for one fact is one too many. -->
        <p v-if="f32 && runDiffers('precision')"
           class="col-span-2 text-xs text-amber-400/90 -mt-0.5">
          single precision mode
        </p>
        <label class="flex items-center gap-1 col-span-2"
               title="which device(s) this session's variant workers run on. Default spreads them over the host's whole WIGNERF_DEVICE pool, costliest variant to the fastest card; pick one device to keep a session off a card you need for something else.">
          <span class="w-14 text-neutral-500">device</span>
          <select v-model="props.cfg.device" class="wf-num flex-1" @change="emit('dirty')">
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
                 @change="clampHistory(); emit('dirty')" />
          <span class="text-neutral-500">MiB
            <template v-if="props.historyMbMax">(0 = host max {{ props.historyMbMax }})</template>
          </span>
        </label>
      </div>
      <p v-if="props.liveDevices?.length" class="text-xs text-neutral-500">
        running on {{ props.liveDevices.join(', ') }}<template
          v-if="props.historyCapMb">, history cap {{ props.historyCapMb }} MiB</template>
      </p>
      <button class="w-full py-1.5 rounded bg-sky-800 hover:bg-sky-700 font-medium"
              @click="emit('restart')">Restart session</button>
      <button class="w-full py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-xs text-neutral-300"
              title="restore grid, potential, physics, run mode, IC and variants to their defaults"
              @click="resetSetup">Reset setup to defaults</button>
    </section>
  </div>
</template>
