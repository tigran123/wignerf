<script setup lang="ts">
/**
 * Analytic U(x) editor with instant feedback: debounced compile+sample via
 * POST /api/preview/potential, a live plot of U over the grid range, and
 * per-variant-family validity badges (quantum needs U finite on the
 * extended Bopp range; classical needs a delta-free dU/dx).
 *
 * The draft becomes the FORM's value (cfg.potential) as soon as the server
 * accepts it — there is no commit button. Every other setting in the Setup
 * panel is bound straight to cfg; U(x) was the one exception, and needing a
 * "Use at restart" press to make the form mean what it showed is what made
 * the two-button arrangement confusing. The gate is validity, not a click:
 * a half-typed expression ("x^", "x^2/") must never reach cfg, because cfg
 * is persisted to localStorage and is what a restart computes from.
 *
 * That leaves ONE button. "Apply live" pushes the draft into the RUNNING
 * session (set_params, applied at the frontier); U(x) is live-appliable, so
 * it never NEEDS a restart, and the input goes amber while the session is
 * still evolving a different expression — the same "edited but not applied"
 * signal the Physics fields use. Those commit on blur/Enter; U does not,
 * because rebuilding the Bopp-range meshes is not something to trigger by
 * tabbing out of a field.
 *
 * The draft's validity is emitted upward and the transport Solve is gated on
 * it (an invalid form must never coexist with a running computation).
 * "Apply live" is inert when the draft already IS the live U: the backend
 * drops such a no-op, and a button that looks like it did something is how a
 * run's change log filled up with U changes that never happened.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { createUplotZoom } from '../lib/uplotZoom'
import { chartPalette, theme } from '../lib/theme'
import type { VariantKey } from '../lib/variants'

const props = defineProps<{
  modelValue: string
  grid: { x1: number; x2: number; Nx: number; p1: number; p2: number; Np: number }
  hbarEff: number
  live: boolean          // a session exists and the socket is attached
  // ...and it is producing NEW records right now. NOT status.running, which
  // is also true during pure playback — nothing computes there, so there is
  // no live run for this button to reach (status.computing = running and
  // not stop_at_frontier; see core/session.py).
  computing: boolean
  variants: VariantKey[] // active selection: decides which families must be valid
  liveExpr?: string | null   // the session's CURRENT U(x), for the no-op gate
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'applyLive', expr: string): void
  (e: 'validity', valid: boolean): void
  (e: 'gridDirty'): void
}>()

interface PreviewResult {
  ok: boolean
  error?: string
  validity?: { quantum: boolean; classical: boolean }
  reasons?: string[]
  warnings?: string[]
  samples?: { x: number[]; U: (number | null)[] }
}

const draft = ref(props.modelValue)
// external config changes ("Reset setup to defaults") must reach the
// editor — a stale draft would keep showing (and validating, and offering
// to apply) the pre-reset expression
watch(() => props.modelValue, (v) => { draft.value = v })
const result = ref<PreviewResult | null>(null)
const plotEl = ref<HTMLDivElement | null>(null)
let chart: uPlot | null = null
let timer: ReturnType<typeof setTimeout> | null = null

// zoomed x window of the plot, DELIBERATELY independent of the grid's
// x1/x2 (null = follow the grid): zooming is how the "interesting domain"
// of U(x) is discovered, and the button below copies it into the grid.
// The backend samples U on this window while the validity probe keeps
// using the SIMULATION range (req.grid) — no restriction to [x1, x2].
const viewX = ref<{ x1: number; x2: number } | null>(null)
// clampX: false — zooming out past the data is the point (it re-samples
// on the wider window via the debounced compile below)
const zoom = createUplotZoom({
  clampX: false,
  onXChange: (w) => { viewX.value = w && { x1: w.min, x2: w.max } },
})

async function compile() {
  // the expression the answer will be ABOUT: the draft can move while the
  // request is in flight, and only a verdict about the current text may
  // commit it below
  const expr = draft.value
  try {
    const { data } = await api.post<PreviewResult>('/preview/potential', {
      expr,
      x1: viewX.value?.x1 ?? props.grid.x1,
      x2: viewX.value?.x2 ?? props.grid.x2,
      grid: props.grid,
      hbar_eff: props.hbarEff,
    })
    result.value = data
    if (data.ok && data.samples && chart) {
      zoom.setData(chart, [data.samples.x, data.samples.U])
    }
    // the form adopts the draft the moment the server says it is usable —
    // this is what replaces the old "Use at restart" button
    if (expr === draft.value && draftValid.value && expr !== props.modelValue)
      emit('update:modelValue', expr)
  } catch (e) {
    result.value = { ok: false, error: String(e) }
  }
}

// grid/ℏ changes move the extended Bopp range, so they can flip validity
// (a pole may enter it) — recompile on those too, not just on typing.
// viewX: wheel/drag rescale the old samples instantly client-side; the
// debounced fetch then refines/extends them to exactly the new window.
watch([draft, () => props.grid, () => props.hbarEff, viewX], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(compile, 300)
}, { deep: true })

/** Copy the zoomed-in "interesting domain" into the grid's x₁/x₂
 *  (applies at restart, like any grid edit). */
function setGridFromView() {
  const v = viewX.value
  if (!v) return
  props.grid.x1 = parseFloat(v.x1.toPrecision(6))
  props.grid.x2 = parseFloat(v.x2.toPrecision(6))
  emit('gridDirty')
}

/** Valid for the CURRENT variant selection: each active family must
 *  accept the draft (an inactive family's ✗ doesn't block). */
const draftValid = computed(() => {
  const r = result.value
  if (!r?.ok) return false
  const needQ = props.variants.some((v) => v.startsWith('q'))
  const needC = props.variants.some((v) => v.startsWith('c'))
  return (!needQ || !!r.validity?.quantum) && (!needC || !!r.validity?.classical)
})
// immediate: the gate starts pessimistic until the first compile lands
watch(draftValid, (v) => emit('validity', v), { immediate: true })

/** The running session is evolving a DIFFERENT U(x) than the draft: edited
 *  but not applied, marked in amber on the input exactly as the Physics
 *  fields mark theirs. */
const isPending = computed(() =>
  props.live && props.liveExpr != null && draft.value !== props.liveExpr)

/**
 * Something to push, somewhere to push it, and a reason to push it NOW.
 *
 * `computing` is the part that is easy to get wrong: while nothing is
 * computing there is no "live" to apply into, and Solve carries the draft
 * anyway (SimulatorView.sendCommand). An enabled button then invites a click
 * that is at best redundant and reads as the ONLY way to make the new U(x)
 * count. This button is exclusively for the case Solve cannot serve: a
 * computation already in flight, which you would otherwise have to pause.
 * Hence `computing` and not `running` — a playback-only run is `running`
 * too, and there the tooltip's "already in progress" would be a lie.
 */
const canApplyLive = computed(() =>
  props.computing && draftValid.value && isPending.value)

function makeChart(): uPlot {
  // uPlot takes axis/grid/series colours at construction only, hence the
  // destroy+rebuild on a theme change below
  const pal = chartPalette()
  return new uPlot(
    {
      width: plotEl.value!.clientWidth || 300,
      height: 110,
      title: 'U(x)',
      legend: { show: false },
      plugins: [zoom.plugin], // owns the cursor config (drag/wheel/dblclick)
      scales: { x: { time: false } },
      axes: [
        { stroke: pal.axis, grid: { stroke: pal.gridSoft } },
        { stroke: pal.axis, grid: { stroke: pal.gridSoft } },
      ],
      series: [{}, { stroke: pal.cursor, width: 1.5, points: { show: false } }],
    },
    [[], []],
    plotEl.value!,
  )
}

onMounted(() => {
  chart = makeChart()
  const ro = new ResizeObserver(() => {
    if (chart && plotEl.value)
      chart.setSize({ width: plotEl.value.clientWidth, height: 110 })
  })
  ro.observe(plotEl.value!)
  // theme change: rebuild in place, keeping the data — exactly as the other
  // two charts do. The last response's samples ARE that data, so re-fetching
  // them would blank the trace for a whole round-trip (and spend a
  // /preview/potential call) to redraw a curve we already have.
  watch(theme, () => {
    chart?.destroy()
    chart = makeChart()
    const s = result.value?.samples
    if (s) zoom.setData(chart, [s.x, s.U])
  })
  onBeforeUnmount(() => ro.disconnect())
  void compile()
})

onBeforeUnmount(() => chart?.destroy())
</script>

<template>
  <section class="space-y-1.5">
    <!-- the header's `uppercase` styling must not mangle math notation:
         U(x) keeps its case -->
    <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider">Potential <span class="normal-case">U(x)</span></h3>
    <input
      v-model="draft"
      spellcheck="false"
      class="w-full bg-input border border-line rounded px-2 py-1 font-mono text-sm"
      :class="isPending && 'wf-pending'"
      :title="isPending
        ? 'the session is still evolving a different U(x) — Apply live pushes this one at the frontier'
        : ''"
      placeholder="e.g. x^2/2, -1/sqrt(x^2+2), x^4/4 - x^2/2"
    />
    <div class="flex items-center gap-2 text-xs">
      <template v-if="result">
        <span v-if="!result.ok" class="text-error">{{ result.error }}</span>
        <template v-else>
          <span :class="result.validity?.quantum ? 'text-ok' : 'text-error'"
                :title="(result.reasons ?? []).join('; ')">
            quantum {{ result.validity?.quantum ? '✓' : '✗' }}
          </span>
          <span :class="result.validity?.classical ? 'text-ok' : 'text-error'"
                :title="(result.reasons ?? []).join('; ')">
            classical {{ result.validity?.classical ? '✓' : '✗' }}
          </span>
        </template>
      </template>
    </div>
    <div v-if="result?.warnings?.length" class="text-[11px] text-warn space-y-0.5">
      <div v-for="(w, i) in result.warnings" :key="i">⚠ {{ w }}</div>
    </div>
    <div v-if="result?.reasons?.length && result.ok" class="text-[11px] text-error space-y-0.5">
      <div v-for="(r, i) in result.reasons" :key="i">{{ r }}</div>
    </div>
    <div ref="plotEl" class="wf-plot"></div>
    <div v-if="viewX" class="flex justify-end">
      <button
        class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
        title="copy the current zoom window into x₁,x₂ (applies at restart); double-click the plot to reset the zoom"
        @click="setGridFromView"
      >x₁,x₂ ← view</button>
    </div>
    <!-- ONE button: the form already holds the draft (see the header
         comment), so the only thing left to ask for is the live push. The
         gate is explained by the title and by the amber input, NOT by a
         standing paragraph — the old one said "already the live U(x)" on
         every page load, which is the steady state, so it was permanent
         furniture explaining a button nobody had tried to press yet. -->
    <button
      class="wf-solid w-full py-1 rounded bg-emerald-700 hover:bg-emerald-600 text-xs"
      :disabled="!canApplyLive"
      :title="!live ? 'no session — Solve computes from this U(x)'
        : !draftValid ? 'invalid for the active variants — see the reasons above'
        : !isPending ? 'this is already the U(x) the session is evolving'
        : !computing ? 'nothing is computing, so there is no live run to apply'
          + ' into — press Solve and it will compute with this U(x)'
        : 'push this U(x) into the computation already in progress, applied'
          + ' at the frontier (records before it keep the old U)'"
      @click="emit('applyLive', draft)"
    >Apply live</button>
  </section>
</template>
