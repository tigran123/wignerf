<script setup lang="ts">
/**
 * One small uPlot line chart, driven entirely by props.
 *
 * Extracted from the pattern PotentialEditor.vue open-codes (construct with a
 * palette read at build time, destroy+rebuild on a theme change, ResizeObserver
 * -> setSize, optional zoom plugin, HTML title). PotentialEditor is deliberately
 * NOT migrated onto this yet: its x chart is entangled with a zoom -> re-sample
 * -> "x1,x2 <- view" loop and with a stale-response drop keyed on the response
 * SHAPE, neither of which belongs in a generic chart, and neither of which is
 * covered by an automated test. This ships with one verified user first.
 *
 * SERIES CARRY A PALETTE ROLE, NOT A COLOUR. Passing colour strings would make
 * the parent recompute them on every theme flip AND make the rebuild watch
 * notice — two coupled obligations. A role keeps CSS the single source of truth
 * (style.css defines every --wf-chart-* once per theme) and costs nothing.
 */
import { onBeforeUnmount, onMounted, ref, watch, nextTick } from 'vue'
import uPlot from 'uplot'
// Its own import, as MarginalsPlot and SeriesPlot each have. Vite dedupes it,
// so this costs nothing in the bundle — and without it a generic chart renders
// correctly only where one of those two happens to be mounted, which is true
// today (PlotsColumn always is) and is not a property this component controls.
import 'uplot/dist/uPlot.min.css'

import { chartPalette, theme, type ChartPalette } from '../lib/theme'
import { setPlotTitleHtml, type UplotZoom } from '../lib/uplotZoom'

export interface LineSeries {
  /** a --wf-chart-* role, resolved at construction from chartPalette() */
  role: keyof ChartPalette
  dash?: number[]
  width?: number
}

const props = withDefaults(defineProps<{
  /** null renders an EMPTY tuple of the right arity — see below */
  data: uPlot.AlignedData | null
  series: LineSeries[]
  height?: number
  titleHtml?: string
  showGrid?: boolean
  cursor?: boolean
  zoom?: UplotZoom | null
}>(), { height: 110, showGrid: true, cursor: false, zoom: null })

const el = ref<HTMLDivElement | null>(null)
let chart: uPlot | null = null

/**
 * uPlot throws when handed fewer data arrays than it has series, and the
 * half-built chart then collapses to min-content width — which shows up as the
 * title typeset one word per line rather than as an error. So an empty chart
 * gets an empty array PER SERIES, never a fixed [[], []].
 */
function empty(): uPlot.AlignedData {
  return Array.from({ length: props.series.length + 1 },
                    () => [] as number[]) as unknown as uPlot.AlignedData
}

function make(): uPlot {
  // uPlot takes axis/grid/series colours at construction only, hence the
  // destroy+rebuild on a theme change below.
  const pal = chartPalette()
  const grid = props.showGrid ? { stroke: pal.gridSoft } : { show: false }
  return new uPlot(
    {
      width: el.value!.clientWidth || 300,
      height: props.height,
      // uPlot creates the .u-title element ONLY when a title is given at
      // construction, so setPlotTitleHtml below has nothing to fill without
      // this — the chart renders, untitled, and looks like a missing feature.
      // The text is immediately overwritten; it exists to make the element.
      title: props.titleHtml ? ' ' : undefined,
      legend: { show: false },
      // The zoom plugin OWNS the cursor config (drag-select, wheel, dblclick),
      // so a chart that has one must not also be handed `cursor: {show:false}`
      // — that silently disables the interaction it was given.
      cursor: (props.zoom || props.cursor) ? undefined : { show: false },
      plugins: props.zoom ? [props.zoom.plugin] : [],
      scales: { x: { time: false } },
      axes: [{ stroke: pal.axis, grid }, { stroke: pal.axis, grid }],
      series: [{}, ...props.series.map((s) => ({
        stroke: pal[s.role],
        width: s.width ?? 1.5,
        dash: s.dash,
        points: { show: false },
      }))],
    },
    props.data ?? empty(),
    el.value!,
  )
}

function apply() {
  if (!chart) return
  const d = props.data ?? empty()
  if (props.zoom) props.zoom.setData(chart, d)
  else chart.setData(d)
}

function rebuild() {
  chart?.destroy()
  chart = make()
  // the title element is new on every rebuild, so it is re-applied here rather
  // than only when the text changes
  if (props.titleHtml) setPlotTitleHtml(chart, props.titleHtml)
  apply()
}

onMounted(() => {
  chart = make()
  if (props.titleHtml) setPlotTitleHtml(chart, props.titleHtml)
  apply()
  const ro = new ResizeObserver(() => {
    if (chart && el.value)
      chart.setSize({ width: el.value.clientWidth, height: props.height })
  })
  ro.observe(el.value!)
  watch(() => props.data, apply)
  watch([theme, () => props.titleHtml, () => props.series.length,
         () => props.height, () => props.showGrid], async () => {
    await nextTick()
    if (el.value) rebuild()
  })
  onBeforeUnmount(() => ro.disconnect())
})

onBeforeUnmount(() => { chart?.destroy(); chart = null })
</script>

<template>
  <div ref="el" class="wf-plot w-full"></div>
</template>
