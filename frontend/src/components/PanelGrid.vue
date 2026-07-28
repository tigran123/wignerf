<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
import type { GeomCfg } from '../lib/config'
import type { ProgressEvent } from '../composables/useSession'
import { VARIANT_META, variantColor, type VariantKey } from '../lib/variants'
import { createViewWindow, remapView, type Domain } from '../lib/viewWindow'
import { planeLabel, planeLabelHtml, planes as planesOf } from '../lib/axes'
import { panelMode, panelPlaneIdx, panelVariantIdx } from '../lib/panelView'
import { AU_TIME_FS } from '../lib/units'
import WignerPanel from './WignerPanel.vue'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  variants: VariantKey[]   // in bundle order (= session config order)
  /** live geometry of the PAINTED frame — per-record, so a regrid follows */
  geom: GeomCfg
  showGrid: boolean
  showCells?: boolean
  // batch mode streams no frames: 'computing' dims the heatmaps and shows a
  // live progress card; 'review' shows a "press Play" hint over the (blank)
  // panels of a finished run; null = normal live/playback display.
  batchOverlay?: 'computing' | 'review' | null
  progress?: ProgressEvent | null
}>()

const throughput = computed(() =>
  (props.progress?.per_variant ?? [])
    .map((v) => `${v.variant} ${v.steps_per_sec}/s`).join('   '))

/**
 * WHAT THE PANELS SHOW — the two readings and their state live in
 * lib/panelView.ts, because the Export panel seeds its own plane/variant
 * choice from them (see that module's docstring). Both come out of one cell
 * list here, so PanelGrid keeps a single code path and WignerPanel never
 * learns which mode it is in.
 */
const mode = panelMode
const planeIdx = panelPlaneIdx
const variantIdx = panelVariantIdx

const ndim = computed(() => props.geom.ndim)
const allPlanes = computed(() => planesOf(ndim.value))
// a stored index can outlive a switch back to 1D, where only plane 0 exists
const plane = computed(() =>
  allPlanes.value[Math.min(planeIdx.value, allPlanes.value.length - 1)]!)

interface Cell {
  key: string
  variant: VariantKey
  variantIndex: number
  plane: readonly [number, number]
  label: string
}

const cells = computed<Cell[]>(() => {
  if (ndim.value === 1 || mode.value === 'variants') {
    const pl = plane.value
    return props.variants.map((v, i) => ({
      key: `${v}:${pl[0]},${pl[1]}`,
      variant: v, variantIndex: i, plane: pl,
      label: VARIANT_META[v].label,
    }))
  }
  const i = Math.min(variantIdx.value, props.variants.length - 1)
  const v = props.variants[i]!
  return allPlanes.value.map((pl) => ({
    key: `${v}:${pl[0]},${pl[1]}`,
    variant: v, variantIndex: i, plane: pl,
    // the PLANE only: every panel here is the same variant, which the selector
    // names once. Repeating "Quantum, non-relativistic" six times is noise, and
    // the long form ran under the control cluster in the top-right panel.
    label: planeLabelHtml(ndim.value, pl),
  }))
})

// Zoom/pan coupling: decoupled by default (each panel its own window);
// the "link zoom" toggle drives all panels from one shared window. One window
// per CELL, keyed by cell identity so a mode switch does not shuffle them.
const LINK_KEY = 'wignerf.linkZoom'
const linked = ref(localStorage.getItem(LINK_KEY) === '1')
watch(linked, (v) => localStorage.setItem(LINK_KEY, v ? '1' : '0'))

/**
 * Linking only MEANS anything when every panel shows the same axis pair. In the
 * phase portrait they show six DIFFERENT pairs — (x,y) against (px,py) against
 * (x,py) — spanning different quantities in different units, so one shared
 * fractional window would map to six unrelated physical regions. The toggle is
 * hidden there and the panels use their own windows; the stored preference is
 * untouched, so it comes back on the way to 'variants'.
 */
const canLink = computed(() => ndim.value === 1 || mode.value === 'variants')
const linkedNow = computed(() => linked.value && canLink.value)

const views = new Map<string, ReturnType<typeof createViewWindow>>()
const shared = createViewWindow()
// coupling adopts the window of the panel the user last zoomed/panned
let lastTouched = ''
function viewFor(key: string) {
  let v = views.get(key)
  if (!v) {
    v = createViewWindow()
    views.set(key, v)
    watch(v, () => { lastTouched = key })
  }
  return v
}

/** This cell's plane extents, for the regrid remap. */
function domainOf(g: GeomCfg, pl: readonly [number, number]): Domain {
  return { a1: g.lo[pl[0]]!, a2: g.hi[pl[0]]!, b1: g.lo[pl[1]]!, b2: g.hi[pl[1]]! }
}

// auto-expand regrid (the geom object is replaced per painted frame when its
// geometry changed): keep every zoom window on the same PHYSICAL region. Each
// window is remapped through ITS OWN axis pair.
watch(() => props.geom, (ng, og) => {
  if (!og || !ng || og === ng || og.ndim !== ng.ndim) return
  for (const c of cells.value) {
    const v = views.get(c.key)
    if (v) remapView(v, domainOf(og, c.plane), domainOf(ng, c.plane))
  }
  remapView(shared, domainOf(og, plane.value), domainOf(ng, plane.value))
})

function toggleLink() {
  if (!linked.value) {
    Object.assign(shared, viewFor(lastTouched || cells.value[0]!.key))
  } else {
    // decouple in place: every panel keeps the current shared window
    for (const c of cells.value) Object.assign(viewFor(c.key), shared)
  }
  linked.value = !linked.value
}

const gridClass = computed(() => {
  const n = cells.value.length
  if (n <= 1) return 'grid-cols-1 grid-rows-1'
  if (n === 2) return 'grid-cols-2 grid-rows-1'
  if (n <= 4) return 'grid-cols-2 grid-rows-2'
  return 'grid-cols-3 grid-rows-2'
})
</script>

<template>
  <div class="relative w-full h-full min-h-0">
    <div class="grid gap-1 w-full h-full min-h-0" :class="[gridClass,
           batchOverlay === 'computing' ? 'opacity-20 grayscale pointer-events-none' : '']">
      <div v-for="c in cells" :key="c.key"
           class="min-h-0 border rounded overflow-hidden"
           :style="{ borderColor: variantColor(c.variant) + '66' }">
        <WignerPanel :frame-source="frameSource" :variant-index="c.variantIndex"
                     :plane="c.plane" :label="c.label" :show-grid="showGrid"
                     :show-cells="showCells"
                     :view="linkedNow ? shared : viewFor(c.key)" />
      </div>
    </div>
    <div class="absolute top-1 right-2 z-10 flex items-center gap-2 text-xs
                text-white bg-black/75 px-1.5 py-0.5 rounded select-none">
      <!-- 2D only: 24 variant x plane combinations do not fit on screen, so
           pick a reading. Hidden at ndim=1, where there is nothing to pick. -->
      <template v-if="ndim > 1">
        <select v-model="mode" class="bg-transparent outline-none cursor-pointer"
                title="compare variants on one plane, or see every plane of one variant">
          <option value="variants" class="text-black">compare variants</option>
          <option value="phase" class="text-black">phase portrait</option>
        </select>
        <select v-if="mode === 'variants'" v-model.number="planeIdx"
                class="bg-transparent outline-none cursor-pointer"
                title="which 2D reduction every panel shows">
          <option v-for="(pl, i) in allPlanes" :key="i" :value="i" class="text-black">
            {{ planeLabel(ndim, pl) }}
          </option>
        </select>
        <select v-else v-model.number="variantIdx"
                class="bg-transparent outline-none cursor-pointer"
                title="whose phase-space portrait to show">
          <option v-for="(v, i) in variants" :key="v" :value="i" class="text-black">
            {{ VARIANT_META[v].label }}
          </option>
        </select>
      </template>
      <label v-if="canLink && cells.length > 1"
             class="flex items-center gap-1 cursor-pointer"
             title="couple zoom/pan across all panels (coupling adopts the last-zoomed panel's view)">
        <input type="checkbox" :checked="linked" @change="toggleLink" />
        <span>link zoom</span>
      </label>
    </div>

    <!-- batch mode: no frames are streamed while computing, so stand a
         progress card in for the dimmed heatmaps (the observable series keep
         updating live). A finished run shows a "review" hint over its blank
         panels. -->
    <div v-if="batchOverlay"
         class="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
      <div class="min-w-[22rem] max-w-[80%] rounded-lg bg-scrim border border-line px-5 py-4 text-center text-fg shadow-xl">
        <template v-if="batchOverlay === 'computing'">
          <div class="text-sm font-medium text-warn-2">Batch computing — no live preview</div>
          <div class="mt-3 h-2 w-full rounded bg-raised overflow-hidden">
            <div class="h-full bg-warn transition-[width] duration-200"
                 :style="{ width: `${progress?.percent ?? 0}%` }"></div>
          </div>
          <div class="mt-2 tabular-nums text-sm">
            <span class="wf-fixnum">{{ (progress?.percent ?? 0).toFixed(1) }}</span>%
            <span class="text-muted mx-2">·</span>
            t = <span class="wf-fixnum">{{ (progress?.t ?? 0).toFixed(3) }}</span> a.u.
            (<span class="wf-fixnum">{{ ((progress?.t ?? 0)*AU_TIME_FS).toFixed(3) }}</span> fs)
            <span class="text-muted">/ t₂ =
              <span class="wf-fixnum">{{ (progress?.t2 ?? 0).toFixed(3) }}</span> a.u.
              (<span class="wf-fixnum">{{ ((progress?.t2 ?? 0)*AU_TIME_FS).toFixed(3) }}</span> fs)</span>
          </div>
          <div v-if="throughput" class="mt-1 text-xs text-fg-3 tabular-nums">{{ throughput }}</div>
          <div class="mt-2 text-xs text-muted">
            the observable plots keep updating; press Play when done to review the run
          </div>
        </template>
        <template v-else>
          <div class="text-sm font-medium text-fg-2">Batch run finished</div>
          <div class="mt-1 text-xs text-muted">
            no frames were streamed while computing — press Play to review the run
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
