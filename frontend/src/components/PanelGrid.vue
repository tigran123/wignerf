<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import type { ProgressEvent } from '../composables/useSession'
import { VARIANT_META, variantColor, type VariantKey } from '../lib/variants'
import { createViewWindow, remapView } from '../lib/viewWindow'
import { AU_TIME_FS } from '../lib/units'
import WignerPanel from './WignerPanel.vue'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  variants: VariantKey[]   // in bundle order (= session config order)
  domain: GridCfg
  showGrid: boolean
  // batch mode streams no frames: 'computing' dims the heatmaps and shows a
  // live progress card; 'review' shows a "press Play" hint over the (blank)
  // panels of a finished run; null = normal live/playback display.
  batchOverlay?: 'computing' | 'review' | null
  progress?: ProgressEvent | null
}>()

const throughput = computed(() =>
  (props.progress?.per_variant ?? [])
    .map((v) => `${v.variant} ${v.steps_per_sec}/s`).join('   '))

// Zoom/pan coupling: decoupled by default (each panel its own window);
// the "link zoom" toggle drives all panels from one shared window.
const LINK_KEY = 'wignerf.linkZoom'
const linked = ref(localStorage.getItem(LINK_KEY) === '1')
watch(linked, (v) => localStorage.setItem(LINK_KEY, v ? '1' : '0'))

const views = [createViewWindow(), createViewWindow(),
               createViewWindow(), createViewWindow()]
const shared = createViewWindow()
// coupling adopts the window of the panel the user last zoomed/panned
let lastTouched = 0
views.forEach((v, i) => watch(v, () => { lastTouched = i }))

// auto-expand regrid (the domain object is replaced per painted frame when
// its geometry changed): keep every zoom window on the same PHYSICAL region
watch(() => props.domain, (nd, od) => {
  if (!od || !nd || od === nd) return
  for (const v of views) remapView(v, od, nd)
  remapView(shared, od, nd)
})

function toggleLink() {
  if (!linked.value) {
    Object.assign(shared, views[lastTouched])
  } else {
    // decouple in place: every panel keeps the current shared window
    for (const v of views) Object.assign(v, shared)
  }
  linked.value = !linked.value
}

const gridClass = computed(() => {
  const n = props.variants.length
  if (n <= 1) return 'grid-cols-1 grid-rows-1'
  if (n === 2) return 'grid-cols-2 grid-rows-1'
  return 'grid-cols-2 grid-rows-2'
})
</script>

<template>
  <div class="relative w-full h-full min-h-0">
    <div class="grid gap-1 w-full h-full min-h-0" :class="[gridClass,
           batchOverlay === 'computing' ? 'opacity-20 grayscale pointer-events-none' : '']">
      <div v-for="(v, i) in variants" :key="v"
           class="min-h-0 border rounded overflow-hidden"
           :style="{ borderColor: variantColor(v) + '66' }">
        <WignerPanel :frame-source="frameSource" :variant-index="i"
                     :label="VARIANT_META[v].label"
                     :domain="domain" :show-grid="showGrid"
                     :view="linked ? shared : views[i]" />
      </div>
    </div>
    <label v-if="variants.length > 1"
           class="absolute top-1 right-2 z-10 flex items-center gap-1 text-xs
                  text-white bg-black/75 px-1.5 py-0.5 rounded cursor-pointer select-none"
           title="couple zoom/pan across all panels (coupling adopts the last-zoomed panel's view)">
      <input type="checkbox" :checked="linked" @change="toggleLink" />
      <span>link zoom</span>
    </label>

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
