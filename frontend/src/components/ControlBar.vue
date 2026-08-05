<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
import type { SessionStatus, ProgressEvent } from '../composables/useSession'
import { displayInterval } from '../lib/perf'
import { pickReadout, type ReadoutSource } from '../lib/readout'
import { transportAction } from '../lib/transport'
import { AU_ENERGY_EV, AU_TIME_FS } from '../lib/units'

const props = defineProps<{
  status: SessionStatus | null
  lastFrame: Frame | null
  // batch compute streams no frames, so t comes from the progress report
  progress: ProgressEvent | null
  // which of those two spoke last — see lib/readout
  readoutSource: ReadoutSource | null
  setupValid: boolean
}>()

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
/**
 * No session at all — `status` is null both before the first one exists and
 * after `destroy()` ran, which is where a FAILED restart leaves us: create()
 * deletes the old session before it posts, so a 422 leaves the app session-less
 * with the form still showing a valid-looking setup. Solve stayed pink and
 * enabled there and its click went nowhere, which reads as the button being
 * broken rather than as there being nothing to command.
 */
const noSession = computed(() => !props.status)
const solveBlocked = computed(() =>
  action.value === 'solve' && (!props.setupValid || noSession.value))

/**
 * Repeat a playback pass instead of stopping at the frontier. Optimistic like
 * the transport flip: the server echoes a fresh status right after, but the
 * checkbox must not lag a click by up to the status period.
 */
const loopOptimistic = ref<boolean | null>(null)
watch(() => props.status?.loop, () => { loopOptimistic.value = null })
function toggleLoop(ev: Event) {
  const on = (ev.target as HTMLInputElement).checked
  loopOptimistic.value = on
  emit('command', { type: 'loop', on })
}
const loopOn = computed(() => loopOptimistic.value ?? !!props.status?.loop)
const loopTitle = computed(() => noSession.value
  ? 'no session'
  : loopOn.value
    ? 'playback repeats from where you started it instead of pausing at the'
      + ' frontier. Turn off to let the current pass finish and stop.'
    : 'repeat playback from where you pressed Play instead of pausing at the'
      + ' end — without it the transport becomes "Solve" at the frontier and'
      + ' Space then COMPUTES rather than replaying.')

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
/**
 * Every readout below comes from ONE normalized record: the newest painted
 * frame, or — through a batch run, which streams no frames — the newest
 * progress report. `readoutSource` says which spoke last, so the values HOLD
 * across the pause that ends a batch compute instead of the whole line
 * reverting to "—" (see lib/readout for why arrival order and not the record
 * index decides). Interactive mode is unaffected: it never sends a progress
 * report, so the source is always the frame.
 */
const readout = computed(() =>
  pickReadout(props.readoutSource, props.lastFrame, props.progress))
const tAu = computed(() => readout.value.t != null ? readout.value.t.toFixed(3) : '—')
const tFs = computed(() =>
  readout.value.t != null ? (readout.value.t*AU_TIME_FS).toFixed(3) : '—')
// percent toward t₂ of a batch run (a compact echo of the big progress bar on
// the dimmed heatmap). `pct` is a progress-report-only field, so this shows
// exactly while the readouts are describing that report — through the pause
// included, which is when how far the run got is most worth knowing.
const batchPct = computed(() =>
  props.status?.mode === 'batch' && readout.value.pct != null
    ? `${readout.value.pct.toFixed(0)}%` : null)

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
  // seeded from the readout, not from lastFrame: a batch run has no frame, so
  // demanding one left the underlined-and-clickable t inert for exactly the
  // paused batch run where typing a t is most useful
  if (!canEditT.value || readout.value.t == null) return
  tDraft.value = readout.value.t.toFixed(3)
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
const uncertLabel = computed(() => readout.value.ndim > 1
  ? 'ΔX·ΔPx, ΔY·ΔPy ='
  : 'ΔX·ΔP =')
const eHa = computed(() => readout.value.E != null ? readout.value.E.toPrecision(6) : '—')
const eEv = computed(() =>
  readout.value.E != null ? (readout.value.E*AU_ENERGY_EV).toFixed(3) : '—')
const uncert = computed(() => readout.value.uncert
  ? readout.value.uncert.map((u) => u.toFixed(4)).join(', ')
  : '—')
const purity = computed(() =>
  readout.value.purity != null ? readout.value.purity.toFixed(6) : '—')
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
      :title="noSession
        ? 'no session — the last restart did not create one; see the error above'
        : solveBlocked
        ? 'setup is invalid — fix the potential / initial condition first'
        : playLabel === 'Solve'
          ? 'will compute new records (GPU/CPU work) — shortcut: Space'
          : playLabel === 'Play'
            ? 'pure playback of computed history — shortcut: Space'
            : 'shortcut: Space'"
      @click="togglePlay($event)"
    >{{ playLabel }}</button>

    <!-- Loop lives HERE, beside the transport, because it is about what
         happens when playback ENDS and that is the moment you reach for this
         row. Enabled while running, unlike the delay dial: arming it mid-pass
         is the common case — you notice the run is about to stop just before
         it does. It is why the button says "Solve" after a replay finishes:
         playback pauses at the frontier, and Space there computes instead of
         replaying. -->
    <label class="flex items-center gap-1 shrink-0 cursor-pointer select-none"
           :class="noSession ? 'opacity-40' : ''"
           :title="loopTitle">
      <input type="checkbox" :checked="loopOn" :disabled="noSession"
             @change="toggleLoop" />
      <span class="text-fg-3">loop</span>
    </label>

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
         title="how far this batch run has reached toward t₂ — no frames are streamed while it computes, so the readouts describe its newest record and hold there when it stops; press Play when done to review">
      batch {{ batchPct }}</div>
    <div class="tabular-nums w-64 truncate shrink-0"><span class="text-fg-3">E =</span>
      <span class="wf-fixnum w-[9ch]">{{ eHa }}</span>
      Ha (<span class="wf-fixnum w-[8ch]">{{ eEv }}</span> eV)</div>
    <div class="tabular-nums truncate shrink-0"
         :class="readout.ndim > 1 ? 'w-60' : 'w-36'"><span class="text-fg-3">{{ uncertLabel }}</span> {{ uncert }}</div>
    <div class="tabular-nums w-36 truncate shrink-0"
         :title="'purity of the first active variant'"><span class="text-fg-3">γ =</span> {{ purity }}</div>
    <div class="ml-auto min-w-0 truncate text-right text-xs text-muted tabular-nums"
         :title="stepInfo">{{ stepInfo }}</div>
  </div>
</template>
