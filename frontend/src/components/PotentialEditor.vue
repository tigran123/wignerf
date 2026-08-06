<script setup lang="ts">
/**
 * Analytic U(x) / U(x,y) editor with instant feedback: debounced compile+sample via
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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { createUplotZoom } from '../lib/uplotZoom'
import { chartPalette, theme } from '../lib/theme'
import type { VariantKey } from '../lib/variants'
import { labels as axisLabels } from '../lib/axes'
import { cutAlongX, cutAlongY, cutAt, cutLabel, isLattice } from '../lib/potentialCuts'
import type { GridCfg } from '../lib/config'

const props = defineProps<{
  modelValue: string
  grid: GridCfg
  hbarEff: number
  live: boolean          // a session exists and the socket is attached
  // ...and it is producing NEW records right now. NOT status.running, which
  // is also true during pure playback — nothing computes there, so there is
  // no live run for this button to reach (status.computing = running and
  // not stop_at_frontier; see core/session.py).
  computing: boolean
  variants: VariantKey[] // active selection: decides which families must be valid
  liveExpr?: string | null   // the session's CURRENT U, for the no-op gate
  /** ndim of the RUNNING session, or null when none — see ndimDiffers */
  liveNdim?: number | null
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
  ndim?: number
  validity?: { quantum: boolean; classical: boolean }
  reasons?: string[]
  warnings?: string[]
  /** 1D: U along x. 2D: an n x n lattice, from which the axis cuts are read. */
  samples?: { x: number[]; y?: number[]; U: (number | null)[] | (number | null)[][] }
}

const ndim = computed(() => props.grid.ndim)
const axisNames = computed(() => axisLabels(ndim.value))
/**
 * In 2D the editor plots the two AXIS CUTS, U(x, 0) and U(0, y), rather than a
 * heatmap: the cuts are what you read numbers off — the depth of a well, where
 * a barrier sits — and they carry the pole and asymmetry information that
 * matters for validity. The server sends the whole n x n lattice, so both cuts
 * come out of one response with no extra request.
 *
 * TWO charts, not two traces on one. uPlot's AlignedData has a single shared
 * abscissa, and these two cuts do not share one: each axis carries its OWN
 * zoom window, and the two move independently. Overlaying them drew U(0, y) —
 * indexed by y — at the x sample positions, which is right only when the two
 * windows happen to coincide (they do for the isotropic default box, which is
 * exactly why it looked correct) and silently rescales the y cut by
 * (x2-x1)/(y2-y1) otherwise. 1D is unchanged: one chart, one trace.
 *
 * BOTH cuts zoom, with the app's whole gesture set (drag / wheel / Shift-wheel /
 * dblclick-reset) and one createUplotZoom instance each, and each window is sent
 * back as the range to SAMPLE. A y zoom therefore also moves where the X cut is
 * taken — it is read at the y sample nearest the origin, which a zoomed-away
 * window no longer contains — and titleX says so ("U(x, 3.2)"), the mirror of
 * what an x zoom already does to the y cut's title.
 */

/** "U(x)" / "U(x,y)" — the function's own name, which is a fact about the
 *  grid's dimensionality and must not be hard-coded anywhere it is shown. */
const uName = computed(() => `U(${axisNames.value.slice(0, ndim.value).join(',')})`)

const PLACEHOLDERS: Record<number, string> = {
  1: 'e.g. x^2/2, -1/sqrt(x^2+2), x^4/4 - x^2/2',
  2: 'e.g. (x^2+y^2)/2, x^2*y - y^3/3, -1/sqrt(x^2+y^2+1)',
}

const draft = ref(props.modelValue)
// external config changes ("Reset setup to defaults") must reach the
// editor — a stale draft would keep showing (and validating, and offering
// to apply) the pre-reset expression
watch(() => props.modelValue, (v) => { draft.value = v })
const result = ref<PreviewResult | null>(null)
const plotEl = ref<HTMLDivElement | null>(null)
// the y-cut's own chart, mounted only at ndim > 1 (see the note above)
const plotElY = ref<HTMLDivElement | null>(null)
let chart: uPlot | null = null
let chartY: uPlot | null = null
let timer: ReturnType<typeof setTimeout> | null = null

// zoomed windows of the two plots, DELIBERATELY independent of the grid's
// own extents (null = follow the grid): zooming is how the "interesting domain"
// of U is discovered, and the buttons below copy either one into the grid.
// The backend samples U on these windows while the validity probes keep
// using the SIMULATION range (req.grid) — no restriction to [x1, x2].
const viewX = ref<{ x1: number; x2: number } | null>(null)
// viewY exists only at ndim > 1 and is dropped when ndim falls back to 1 (see
// rebuild): the chart it belongs to is v-if'd away there, so a surviving window
// would be pinned with nothing left to double-click and would reappear,
// unasked, on the next switch back to 2D.
const viewY = ref<{ y1: number; y2: number } | null>(null)
// clampX: false — zooming out past the data is the point (it re-samples
// on the wider window via the debounced compile below).
// Each instance reports its own ABSCISSA window, which on the second chart is
// y; Shift-wheel moves the U scale only and is deliberately not reported,
// since U's range is autoscaled and never re-sampled.
const zoom = createUplotZoom({
  clampX: false,
  onXChange: (w) => { viewX.value = w && { x1: w.min, x2: w.max } },
})
const zoomY = createUplotZoom({
  clampX: false,
  onXChange: (w) => { viewY.value = w && { y1: w.min, y2: w.max } },
})

async function compile() {
  // the expression the answer will be ABOUT: the draft can move while the
  // request is in flight, and only a verdict about the current text may
  // commit it below
  const expr = draft.value
  try {
    const ax = props.grid.axes
    const { data } = await api.post<PreviewResult>('/preview/potential', {
      expr,
      x1: viewX.value?.x1 ?? ax[0]!.lo,
      x2: viewX.value?.x2 ?? ax[0]!.hi,
      // undefined is dropped from the body and the server fills in the grid's
      // own y extent (routers/preview.preview_potential). NOT `?? ax[1]!.lo`
      // like x above: at ndim=1 axes[1] is the MOMENTUM axis, not y.
      y1: viewY.value?.y1,
      y2: viewY.value?.y2,
      grid: props.grid,
      hbar_eff: props.hbarEff,
    })
    result.value = data
    if (data.ok && data.samples && matchesNdim(data.samples)) draw(data.samples)
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
// viewX/viewY: wheel/drag rescale the old samples instantly client-side; the
// debounced fetch then refines/extends them to exactly the new window. Both
// windows are in here — a y zoom that only stretched the lattice's existing
// n2 samples would show no more detail than the full extent already did.
watch([draft, () => props.grid, () => props.hbarEff, viewX, viewY], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(compile, 300)
}, { deep: true })

/**
 * Does this response's shape match the layout that currently exists? A 1D
 * response carries a flat U trace and a 2D one an n x n lattice, and a 2D
 * layout has a second chart to fill from that lattice's rows. Feed one to the
 * other and uPlot reads a data array that is not there — it throws, and the
 * half-built chart collapses to min-content width, which on screen is the
 * title typeset one word per line. Shape-based rather than ndim-based so any
 * stale response is inert, not just an ndim switch.
 */
function matchesNdim(sm: NonNullable<PreviewResult['samples']>): boolean {
  return isLattice(sm) === (ndim.value > 1)
}

/** Which coordinate each cut was actually TAKEN at (null in 1D), so the titles
 *  name it instead of claiming 0 — see lib/potentialCuts.ts. */
const cutFrom = ref<{ x: number; y: number } | null>(null)

// Both go through the zoom's setData, never uPlot's own: that autoscales, so
// every 300 ms refetch would throw away the window just set.
function draw(sm: NonNullable<PreviewResult['samples']>) {
  if (chart) zoom.setData(chart, cutAlongX(sm) as unknown as uPlot.AlignedData)
  if (chartY && isLattice(sm))
    zoomY.setData(chartY, cutAlongY(sm) as unknown as uPlot.AlignedData)
  cutFrom.value = cutAt(sm)
}

const titleX = computed(() => {
  if (ndim.value < 2) return 'U(x)'
  return `U(x, ${cutLabel(cutFrom.value?.y, result.value?.samples?.y)})`
})
const titleY = computed(() =>
  `U(${cutLabel(cutFrom.value?.x, result.value?.samples?.x)}, y)`)

/** Copy the zoomed-in "interesting domain" of spatial axis `i` into the grid
 *  (applies at restart, like any grid edit). Array order is (q..., k...), so
 *  axes[0] is x and axes[1] is y; i=1 comes only from the y button, which
 *  renders only while viewY is set, which happens only at ndim=2. */
function setGridFromView(i: number) {
  const v = i === 0
    ? viewX.value && [viewX.value.x1, viewX.value.x2]
    : viewY.value && [viewY.value.y1, viewY.value.y2]
  if (!v) return
  props.grid.axes[i]!.lo = parseFloat(v[0]!.toPrecision(6))
  props.grid.axes[i]!.hi = parseFloat(v[1]!.toPrecision(6))
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
/**
 * The form's dimensionality differs from the RUNNING session's. Then U is
 * restart-only like the grid and the IC: a 2D expression cannot be pushed into
 * a 1D session at all (the backend would reject `y` as a free symbol), so
 * marking it "edited, Apply live pushes it" would promise something nothing
 * can keep. The dims marker in the Grid section already says "restart" — one
 * amber per fact.
 */
const ndimDiffers = computed(() =>
  props.liveNdim != null && props.liveNdim !== ndim.value)

const isPending = computed(() =>
  props.live && !ndimDiffers.value
  && props.liveExpr != null && draft.value !== props.liveExpr)

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
      title: titleX.value,
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

/** The y cut, on its own abscissa and with its own zoom — the same gesture set
 *  as the x chart, reported as the request's y1/y2. Shorter than the x chart —
 *  this is the secondary reading, in the column the W panels are competing
 *  with. No `cursor` of its own: zoomY.plugin's opts hook assigns o.cursor
 *  wholesale (drag/wheel/dblclick), so anything set here would be silently
 *  replaced — the x chart carries none for the same reason. */
function makeChartY(): uPlot {
  const pal = chartPalette()
  return new uPlot(
    {
      width: plotElY.value!.clientWidth || 300,
      height: 90,
      title: titleY.value,
      legend: { show: false },
      plugins: [zoomY.plugin],
      scales: { x: { time: false } },
      axes: [
        { stroke: pal.axis, grid: { stroke: pal.gridSoft } },
        { stroke: pal.axis, grid: { stroke: pal.gridSoft } },
      ],
      series: [{},
        { stroke: pal.text, width: 1.5, dash: [5, 4], points: { show: false } }],
    },
    [[], []],
    plotElY.value!,
  )
}

/** Rebuild whichever charts should exist right now, keeping the data. uPlot
 *  takes titles, series and axis colours at construction, so a theme change, a
 *  dimensionality change and a moved cut coordinate all land here. */
function rebuild() {
  chart?.destroy()
  chart = makeChart()
  chartY?.destroy()
  if (ndim.value > 1 && plotElY.value) {
    chartY = makeChartY()
  } else {
    // The y axis has stopped existing, so display state that ADDRESSES it must
    // go with it (the ICEditor.cutAxis rule): the chart is v-if'd away, leaving
    // a pinned window with nothing left to double-click, which would then come
    // back unasked on the next switch to 2D.
    chartY = null
    zoomY.reset()
    viewY.value = null
  }
  const sm = result.value?.samples
  if (sm && matchesNdim(sm)) {
    draw(sm)
  } else {
    // The samples belong to the OTHER dimensionality: not merely mis-shaped
    // but the wrong function. Drop them and let the recompile the grid
    // watcher already scheduled refill — which also re-runs validity, so
    // Solve stays gated until U has been checked against the new grid.
    result.value = null
  }
}

onMounted(() => {
  chart = makeChart()
  // mounting straight into 2D (a stored 2D setup, a reload): plotElY is
  // rendered by the initial pass, so its chart is built here rather than
  // waiting for the first theme/ndim change
  if (ndim.value > 1 && plotElY.value) chartY = makeChartY()
  const ro = new ResizeObserver(() => {
    if (chart && plotEl.value)
      chart.setSize({ width: plotEl.value.clientWidth, height: 110 })
    if (chartY && plotElY.value)
      chartY.setSize({ width: plotElY.value.clientWidth, height: 90 })
  })
  ro.observe(plotEl.value!)
  // theme change: rebuild in place, keeping the data — exactly as the other
  // two charts do. The last response's samples ARE that data, so re-fetching
  // them would blank the trace for a whole round-trip (and spend a
  // /preview/potential call) to redraw a curve we already have.
  // ndim rides along because it decides whether the y chart exists at all, and
  // the titles because a zoomed-away window changes which cut was taken. The
  // ndim branch waits a tick: plotElY is v-if'd on it and is not in the DOM
  // until Vue has flushed the render.
  watch([theme, ndim, titleX, titleY], async () => {
    await nextTick()
    if (plotEl.value) rebuild()
  })
  onBeforeUnmount(() => ro.disconnect())
  void compile()
})

onBeforeUnmount(() => { chart?.destroy(); chartY?.destroy() })
</script>

<template>
  <section class="space-y-1.5">
    <!-- the header's `uppercase` styling must not mangle math notation:
         U(x) keeps its case -->
    <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider">Potential <span class="normal-case">{{ uName }}</span></h3>
    <input
      v-model="draft"
      spellcheck="false"
      class="w-full bg-input border border-line rounded px-2 py-1 font-mono text-sm"
      :class="isPending && 'wf-pending'"
      :title="isPending
        ? `the session is still evolving a different ${uName} — Apply live pushes this one at the frontier`
        : ''"
      :placeholder="PLACEHOLDERS[ndim] ?? PLACEHOLDERS[1]"
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
    <!-- 2D: the y cut on its OWN abscissa. It cannot share the x chart's —
         each axis zooms independently — see the note at the top. -->
    <div v-if="ndim > 1" ref="plotElY" class="wf-plot"></div>
    <div v-if="viewX || viewY" class="flex justify-end gap-1">
      <button
        v-if="viewX"
        class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
        title="copy the current zoom window into x₁,x₂ (applies at restart); double-click the plot to reset the zoom"
        @click="setGridFromView(0)"
      >x₁,x₂ ← view</button>
      <button
        v-if="viewY"
        class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
        title="copy the current zoom window into y₁,y₂ (applies at restart); double-click the plot to reset the zoom"
        @click="setGridFromView(1)"
      >y₁,y₂ ← view</button>
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
        : ndimDiffers ? `the session is ${liveNdim}D and this ${uName} is ${ndim}D — press Restart session`
        : !isPending ? `this is already the ${uName} the session is evolving`
        : !computing ? 'nothing is computing, so there is no live run to apply'
          + ' into — press Solve and it will compute with this U(x)'
        : 'push this U(x) into the computation already in progress, applied'
          + ' at the frontier (records before it keep the old U)'"
      @click="emit('applyLive', draft)"
    >Apply live</button>
  </section>
</template>
