<script setup lang="ts">
/**
 * Overlaid marginal distributions of all active variants for ONE phase-space
 * axis, updated per received frame. uPlot instance lives outside Vue
 * reactivity; data arrives via the frame-source callback.
 *
 * One plot per AXIS, so a 2D run gets four rather than two overlaying both
 * dimensions: colour already means variant and dash already means variant, so
 * an overlay would have to encode the dimension in line width or opacity, and
 * "which curve is rho(y)" is not a thing to squint at. The column scrolls; the
 * per-variant checkboxes thin it out.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { perfStage } from '../lib/perf'
import { loadHidden, saveHidden } from '../lib/plotPrefs'
import type { Frame } from '../lib/protocol'
import { marginalTitle } from '../lib/axes'
import { createUplotZoom, setPlotTitleHtml } from '../lib/uplotZoom'
import { chartPalette, theme } from '../lib/theme'
import { VARIANT_META, variantColor, type VariantKey } from '../lib/variants'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  variants: VariantKey[]
  /** which phase-space axis this plot marginalizes onto (see lib/axes.ts) */
  axis: number
  ndim: number
  // natural-order axis seed, until the first frame lands: a1 + i*(a2-a1)/N
  a1: number
  a2: number
  n: number
  showGrid?: boolean
}>()

const el = ref<HTMLDivElement | null>(null)
const prefId = computed(() => `marg${props.axis}` as const)
// per-plot display-only visibility (persisted)
const hidden = ref(loadHidden(`marg${props.axis}`))
// created in setup so the zoom window survives the grid-lines rebuild
const zoom = createUplotZoom()
let chart: uPlot | null = null
let unsub: (() => void) | null = null
let lastData: uPlot.AlignedData | null = null

function toggle(v: VariantKey) {
  const h = hidden.value
  if (h.has(v)) h.delete(v)
  else h.add(v)
  saveHidden(prefId.value, h)
  chart?.setSeries(props.variants.indexOf(v) + 1, { show: !h.has(v) })
  // setSeries auto-rescales y, dropping a pinned y-zoom — re-assert it
  if (chart) zoom.reapply(chart)
}

// The axis is a per-FRAME fact: auto-expand regrids move/double the domain
// mid-run, and scrubbing across the regrid boundary flips it back and
// forth. Props seed the pre-first-frame axis; the frame handler rebuilds
// in place when the painted frame's geometry differs (no remount — that
// would drop the zoom and blank the plot until the next frame).
function buildAxis(a1: number, a2: number, n: number) {
  return Array.from({ length: n }, (_, i) => a1 + (i * (a2 - a1)) / n)
}
let axis = buildAxis(props.a1, props.a2, props.n)
let axisKey = `${props.a1}|${props.a2}|${props.n}`

function makeChart(width: number) {
  const series: uPlot.Series[] = [
    {},
    // distinct dashes: coincident curves (harmonic Q==C, c=137 rel~nonrel)
    // stay individually visible through the gaps of the ones on top
    ...props.variants.map((v) => ({
      label: v,
      stroke: variantColor(v),
      dash: VARIANT_META[v].dash,
      width: 1.5,
      points: { show: false },
      // in the config (not just setSeries) so visibility survives the
      // grid-lines destroy+rebuild
      show: !hidden.value.has(v),
    })),
  ]
  // construction-time in uPlot, hence the destroy+rebuild on a theme change
  const pal = chartPalette()
  const grid = { show: props.showGrid ?? true, stroke: pal.grid, width: 1 }
  return new uPlot(
    {
      width,
      height: 130,
      // plain here, then rewritten as HTML below: uPlot's title is textContent
      title: marginalTitle(props.ndim, props.axis),
      legend: { show: false },
      plugins: [zoom.plugin], // owns the cursor config (drag/wheel/dblclick)
      scales: { x: { time: false } },
      axes: [
        { stroke: pal.axis, grid, ticks: { stroke: pal.tick } },
        { stroke: pal.axis, grid, ticks: { stroke: pal.tick } },
      ],
      series,
    },
    [axis, ...props.variants.map(() => new Array(axis.length).fill(null))],
    el.value!,
  )
}

onMounted(() => {
  chart = makeChart(el.value!.clientWidth || 360)
  setPlotTitleHtml(chart, marginalTitle(props.ndim, props.axis, true))
  const ro = new ResizeObserver(() => {
    if (chart && el.value) chart.setSize({ width: el.value.clientWidth, height: 130 })
  })
  ro.observe(el.value!)
  // the session fan-out is already rAF-timed; the Float32Array views go
  // into uPlot as-is (it only indexes them) — no Array.from boxing copies
  unsub = props.frameSource((f: Frame) => {
    if (!chart) return
    const a = props.axis
    const a1 = f.lo[a]!
    const a2 = f.hi[a]!
    const n = f.N[a]!
    const key = `${a1}|${a2}|${n}`
    if (key !== axisKey) {
      axisKey = key
      axis = buildAxis(a1, a2, n)
    }
    lastData = [
      axis,
      ...f.variants.map((v) => v.marg[props.axis]),
    ] as unknown as uPlot.AlignedData
    const t0 = performance.now()
    // zoom-preserving per painted frame: same cost class as a plain
    // autoscaling setData (no new per-frame allocations)
    zoom.setData(chart, lastData)
    perfStage('plots', performance.now() - t0)
  })
  // grid-lines toggle and theme change: rebuild the chart in place, keeping
  // the data — a remount would blank the plot until the next frame arrives
  watch([() => props.showGrid, theme], () => {
    chart?.destroy()
    chart = makeChart(el.value?.clientWidth || 360)
    setPlotTitleHtml(chart, marginalTitle(props.ndim, props.axis, true))
    if (lastData) zoom.setData(chart, lastData)
  })
  onBeforeUnmount(() => ro.disconnect())
})

onBeforeUnmount(() => {
  unsub?.()
  chart?.destroy()
})
</script>

<template>
  <div class="flex items-center gap-1">
    <div v-if="variants.length > 1" class="flex flex-col gap-0.5 shrink-0">
      <label v-for="v in variants" :key="v"
             class="flex items-center gap-0.5 cursor-pointer select-none"
             :title="`show/hide ${VARIANT_META[v].label} in this plot only`">
        <input type="checkbox" class="scale-75" :checked="!hidden.has(v)"
               @change="toggle(v)" />
        <span class="text-[9px] leading-none"
              :style="{ color: variantColor(v) }">{{ v.toUpperCase() }}</span>
      </label>
    </div>
    <div ref="el" class="wf-plot flex-1 min-w-0"></div>
  </div>
</template>

<!-- NB the `.wf-plot .u-*` rules that used to live here were global and
     styled SeriesPlot and PotentialEditor too; they are now in style.css,
     themed off --wf-chart-*. -->

