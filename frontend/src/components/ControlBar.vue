<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
import type { SessionStatus, ProgressEvent } from '../composables/useSession'
import { displayInterval } from '../lib/perf'
import { transportAction } from '../lib/transport'
import { AU_ENERGY_EV, AU_TIME_FS } from '../lib/units'

const props = defineProps<{
  status: SessionStatus | null
  lastFrame: Frame | null
  // batch compute streams no frames, so t comes from the progress report
  progress: ProgressEvent | null
  setupValid: boolean
}>()

// batch mode computing: no frames stream, the display is dimmed and t/percent
// come from the progress report instead of the (frozen/absent) lastFrame
const batchComputing = computed(() =>
  props.status?.mode === 'batch' && !!props.status?.computing)

const emit = defineEmits<{
  (e: 'command', cmd: Record<string, unknown>): void
}>()

const running = computed(() => props.status?.running ?? false)

/**
 * Delay dial: how much time is injected between played-back frames.
 * Leftmost = "0" (the default) = one record per display refresh — the
 * fastest speed at which EVERY frame is still painted. The client sends
 * the measured refresh interval for it (the server keeps honest seconds),
 * and every position is clamped to at least that interval, so delivery
 * never outpaces painting and nothing is visually skipped. To the right,
 * DIAL_STEPS log-spaced values from 20 ms up to 1.5 s per frame.
 * Computation is NEVER paced by it (workers always run flat out), and it
 * is only settable while PAUSED: pause, change, resume. The thumb
 * position is local state (`dial`) so the 1 Hz status echo cannot yank it
 * around mid-drag; the echo re-syncs it when idle.
 */
const DELAY_MIN = 0.02
const DELAY_MAX = 1.5
const DIAL_STEPS = 45         // dial positions 0 (refresh-paced) .. DIAL_STEPS
function dialToDelay(i: number): number {
  return i <= 0 ? 0
    : DELAY_MIN*Math.pow(DELAY_MAX/DELAY_MIN, (i - 1)/(DIAL_STEPS - 1))
}
function delayToDial(d: number): number {
  if (d <= displayInterval()*1.05) return 0
  const i = 1 + (DIAL_STEPS - 1)
    * Math.log(d/DELAY_MIN) / Math.log(DELAY_MAX/DELAY_MIN)
  return Math.min(Math.max(Math.round(i), 1), DIAL_STEPS)
}
const dial = ref(delayToDial(props.status?.delay ?? 0))
let dialTouched = 0
watch(() => props.status?.delay, (d) => {
  if (d != null && performance.now() - dialTouched > 750)
    dial.value = delayToDial(d)
})
const delayLabel = computed(() => {
  const d = dialToDelay(dial.value)
  return d === 0 ? '0' : d < 1 ? `${Math.round(d*1000)} ms` : `${d.toFixed(1)} s`
})
const delayTitle = computed(() => running.value
  ? 'pause first to change the frame delay — it paces playback only '
    + '(computation always runs at full speed)'
  : 'time injected between played-back frames — 0 (default) means one '
    + 'frame per display refresh, the fastest speed at which every frame '
    + 'is still painted')

/**
 * Solve / Play / Pause — the label tells you IN ADVANCE what the button
 * will do: "Solve" = pressing it computes new records (GPU/CPU work);
 * "Play" = pure playback of already-computed history; "Pause" while
 * running. Playback-only runs auto-pause at the frontier (the backend
 * flips running off and the button becomes "Solve") — computation only
 * ever starts from an explicit Solve. Batch: solving until t2 is
 * reached, pure playback afterwards. Solve is DISABLED while the setup
 * form holds invalid data (potential draft / IC preview) — computing
 * while the visible setup is broken misleads; playback stays allowed.
 */
const action = computed(() =>
  transportAction(props.status, props.lastFrame?.record ?? null))
const playLabel = computed(() =>
  action.value === 'pause' ? 'Pause' : action.value === 'play' ? 'Play' : 'Solve')
const solveBlocked = computed(() => action.value === 'solve' && !props.setupValid)

function togglePlay(ev?: Event) {
  // drop focus so a later Space is the global shortcut, never a second
  // native click on this button (double-fire made Space look erratic)
  ;(ev?.currentTarget as HTMLElement | null)?.blur()
  if (solveBlocked.value) return
  emit('command', { type: action.value === 'pause' ? 'pause' : 'play' })
}

function setDelay(ev: Event) {
  dialTouched = performance.now()
  const v = Number((ev.target as HTMLInputElement).value)
  dial.value = v
  emit('command', { type: 'delay',
                    seconds: Math.max(dialToDelay(v), displayInterval()) })
}

// The readout box has a fixed width, but its CONTENT must hold still too.
// toPrecision(4) printed a DIFFERENT number of decimals as the value grew
// (0.02419 → 0.2419 → 2.419 fs), so the text kept changing length and the
// "(… fs)" part slid about; fixed decimals plus a right-aligned
// fixed-width field pin every glyph to its column. The exported frames
// have the same rule (core/render_mpl.py).
// during batch compute there is no frame — t rides the progress report
const tNow = computed<number | null>(() =>
  batchComputing.value ? (props.progress?.t ?? null)
                       : (props.lastFrame?.t ?? null))
const tAu = computed(() => tNow.value != null ? tNow.value.toFixed(3) : '—')
const tFs = computed(() =>
  tNow.value != null ? (tNow.value*AU_TIME_FS).toFixed(3) : '—')
// percent toward t₂ while a batch run computes (a compact echo of the big
// progress bar on the dimmed heatmap)
const batchPct = computed(() =>
  batchComputing.value && props.progress
    ? `${props.progress.percent.toFixed(0)}%` : null)

/**
 * Direct t entry: whenever the session is paused and history exists, the t
 * readout becomes an editbox on click — clicking a timeline pixel cannot
 * reach a specific record among hundreds, but science can type. The entered
 * t seeks to the nearest record. During computation and playback (running)
 * the readout is display-only.
 */
const editingT = ref(false)
const tDraft = ref('')
const tInput = ref<HTMLInputElement | null>(null)
const canEditT = computed(() =>
  !running.value && (props.status?.record_extent?.[1] ?? -1) >= 0)

function startEditT() {
  if (!canEditT.value || !props.lastFrame) return
  tDraft.value = props.lastFrame.t.toFixed(3)
  editingT.value = true
  void nextTick(() => tInput.value?.select())
}

function commitT() {
  if (!editingT.value) return
  editingT.value = false
  const st = props.status
  const tv = Number(tDraft.value)
  if (!st || !Number.isFinite(tv)) return
  const [k0, k1] = st.record_extent
  const t0 = st.t_extent?.[0]
  if (t0 == null || k1 < 0) return
  // KNOWN LIMITATION: assumes one time direction across the retained
  // history. After a mid-run dt_sign flip the record times are piecewise
  // linear in k, so this lands near (not on) the requested t; the clamp
  // below keeps it inside the timeline either way.
  const step = st.record_dt * (st.sign || 1)
  const k = Math.min(Math.max(k0 + Math.round((tv - t0) / step), k0), k1)
  emit('command', { type: 'seek', record: k })
}
/**
 * Observables of the FIRST active variant, normalized across the two sources.
 * Batch compute streams no frames, so `lastFrame` is stale or absent for the
 * whole run — but the frontier record's scalars ride the progress report
 * (which is already being sent), exactly as `t` does above. Without this the
 * readouts sat at "—" while the series plots two panels away were live.
 */
/** The uncertainty PRODUCT per spatial dimension: std[i]*std[ndim+i]. One
 *  number in 1D, two in 2D — a single "ΔX·ΔP" would silently report only the
 *  x dimension of a run where y is the interesting one. */
function products(std: number[]): number[] {
  const nd = std.length / 2
  return Array.from({ length: nd }, (_, i) => std[i]! * std[nd + i]!)
}

const obs = computed<{ E: number; uncert: number[]; purity: number } | null>(() => {
  if (batchComputing.value) {
    const v = props.progress?.per_variant[0]
    if (!v || v.E == null || v.std == null || v.purity == null) return null
    return { E: v.E, uncert: products(v.std), purity: v.purity }
  }
  const v = props.lastFrame?.variants[0]
  return v ? { E: v.E, uncert: products(v.std), purity: v.purity } : null
})
const ndim = computed(() => props.lastFrame?.ndim
  ?? ((props.progress?.per_variant[0]?.std?.length ?? 2) / 2))
const uncertLabel = computed(() => ndim.value > 1
  ? 'ΔX·ΔPx, ΔY·ΔPy ='
  : 'ΔX·ΔP =')
const eHa = computed(() => obs.value ? obs.value.E.toPrecision(6) : '—')
const eEv = computed(() =>
  obs.value ? (obs.value.E*AU_ENERGY_EV).toFixed(3) : '—')
const uncert = computed(() => obs.value
  ? obs.value.uncert.map((u) => u.toFixed(4)).join(', ')
  : '—')
const purity = computed(() => obs.value ? obs.value.purity.toFixed(6) : '—')
const stepInfo = computed(() => {
  // Tag each variant with its device only when the pool actually has
  // more than one device — single-device setups keep the compact form.
  const multi = (props.status?.devices?.length ?? 1) > 1
  return (props.status?.per_variant ?? [])
    .map((v) => `${v.variant}${multi ? `[${v.device}]` : ''}: dt=${v.dt.toExponential(2)} @${v.steps_per_sec}/s`)
    .join('   ')
})
</script>

<template>
  <!-- Every readout sits in a FIXED-width box: live-updating text must
       never change the layout (page scrollbars used to flicker). -->
  <div class="flex items-center gap-4 px-3 py-2 bg-panel border-t border-line-soft text-sm text-fg whitespace-nowrap overflow-hidden">
    <button
      class="wf-solid w-20 py-1 shrink-0 rounded font-medium"
      :class="playLabel === 'Solve' ? 'bg-pink-800 hover:bg-pink-700'
                                    : 'bg-sky-700 hover:bg-sky-600'"
      :disabled="solveBlocked"
      :title="solveBlocked
        ? 'setup is invalid — fix the potential / initial condition first'
        : playLabel === 'Solve'
          ? 'will compute new records (GPU/CPU work) — shortcut: Space'
          : playLabel === 'Play'
            ? 'pure playback of computed history — shortcut: Space'
            : 'shortcut: Space'"
      @click="togglePlay($event)"
    >{{ playLabel }}</button>

    <label class="flex items-center gap-2 shrink-0"
           :class="running ? 'opacity-40' : ''" :title="delayTitle">
      <span class="text-fg-3">delay</span>
      <input type="range" :min="0" :max="DIAL_STEPS" step="1"
             :disabled="running" :value="dial" @input="setDelay"
             @change="(ev: Event) => (ev.target as HTMLInputElement).blur()"
             class="w-36" />
      <span class="tabular-nums w-24 truncate">{{ delayLabel }}</span>
    </label>

    <div class="tabular-nums w-64 truncate shrink-0">
      <span class="text-fg-3">t =</span>
      <input v-if="editingT" ref="tInput" v-model="tDraft"
             class="w-28 bg-raised border border-info rounded px-1 tabular-nums"
             @keydown.enter="commitT" @keydown.esc="editingT = false"
             @blur="commitT" />
      <span v-else
            :class="canEditT ? 'cursor-pointer underline decoration-dotted decoration-dim' : ''"
            :title="canEditT ? 'click to type t directly (seeks to the nearest record); ←/→ step ±10%, ↓/↑ one record, Home/End jump to start/end' : ''"
            @click="startEditT"><span class="wf-fixnum w-[7ch]">{{ tAu }}</span>
        a.u. (<span class="wf-fixnum w-[7ch]">{{ tFs }}</span> fs)</span>
    </div>
    <div v-if="batchPct" class="shrink-0 px-2 py-0.5 rounded bg-warn-soft text-warn-2 text-xs tabular-nums"
         title="batch compute progress toward t₂ — no frames are streamed; press Play when done to review">
      batch {{ batchPct }}</div>
    <div class="tabular-nums w-64 truncate shrink-0"><span class="text-fg-3">E =</span>
      <span class="wf-fixnum w-[9ch]">{{ eHa }}</span>
      Ha (<span class="wf-fixnum w-[8ch]">{{ eEv }}</span> eV)</div>
    <div class="tabular-nums truncate shrink-0"
         :class="ndim > 1 ? 'w-60' : 'w-36'"><span class="text-fg-3">{{ uncertLabel }}</span> {{ uncert }}</div>
    <div class="tabular-nums w-36 truncate shrink-0"
         :title="'purity of the first active variant'"><span class="text-fg-3">γ =</span> {{ purity }}</div>
    <div class="ml-auto min-w-0 truncate text-right text-xs text-muted tabular-nums"
         :title="stepInfo">{{ stepInfo }}</div>
  </div>
</template>
