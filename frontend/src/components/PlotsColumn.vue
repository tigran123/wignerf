<script setup lang="ts">
/**
 * The diagnostics column: one marginal per phase-space axis, then E(t), one
 * ΔQ·ΔK(t) per spatial dimension, purity γ(t), and — in 2D only — ⟨Lz⟩(t).
 *
 * So 1D shows five plots as it always did, and 2D shows nine. The column
 * scrolls rather than overlaying pairs: colour and dash already mean "variant",
 * and an overlay would have to encode the axis in line width.
 */
import { computed } from 'vue'
import MarginalsPlot from './MarginalsPlot.vue'
import SeriesPlot from './SeriesPlot.vue'
import type { Frame } from '../lib/protocol'
import type { GeomCfg } from '../lib/config'
import type { VariantKey } from '../lib/variants'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  sessionId: string | null
  variants: VariantKey[]
  /** live geometry (form values until the first frame lands) */
  geom: GeomCfg
  lastFrame: Frame | null
  showGrid: boolean
  plotsKey: string
  // batch compute streams no frames — the marginals (frame-fed) are dimmed;
  // the series plots below stay live (they poll REST, not frames)
  batchComputing?: boolean
}>()

const ndim = computed(() => props.geom.ndim)
const axisIdx = computed(() => props.geom.N.map((_, i) => i))
const dims = computed(() => Array.from({ length: ndim.value }, (_, i) => i))
</script>

<template>
  <div class="flex flex-col gap-2">
    <!-- frame-fed views: dimmed during batch compute (no frames stream) -->
    <div class="flex flex-col gap-2 transition-opacity"
         :class="batchComputing ? 'opacity-20 grayscale pointer-events-none' : ''"
         :title="batchComputing ? 'batch mode streams no frames while computing — press Play when done to review' : undefined">
      <!-- The colour scale used to head this column. It has moved INTO each W
           plot: it is a per-panel fact (every panel autoscales to its own
           range) and one shared bar mislabelled the rest, and this is the
           tallest of the three portrait columns, so a row spent here is a row
           the panels start later by. See Colorbar.vue. -->
      <MarginalsPlot v-for="a in axisIdx" :key="'m' + a + plotsKey"
                     :frame-source="frameSource" :variants="variants"
                     :axis="a" :ndim="ndim" :show-grid="showGrid"
                     :a1="geom.lo[a]!" :a2="geom.hi[a]!" :n="geom.N[a]!" />
    </div>
    <!-- cursor-t: the painted frame's time, so the series carry the same
         moving marker the exported video does -->
    <SeriesPlot :key="'e' + plotsKey" :session-id="sessionId"
                :variants="variants" which="E" :ndim="ndim"
                :show-grid="showGrid" :cursor-t="lastFrame?.t ?? null" />
    <SeriesPlot v-for="d in dims" :key="'u' + d + plotsKey"
                :session-id="sessionId" :variants="variants"
                which="uncertainty" :dim="d" :ndim="ndim"
                :show-grid="showGrid" :cursor-t="lastFrame?.t ?? null" />
    <SeriesPlot :key="'g' + plotsKey" :session-id="sessionId"
                :variants="variants" which="purity" :ndim="ndim"
                :show-grid="showGrid" :cursor-t="lastFrame?.t ?? null" />
    <!-- angular momentum exists only in 2D, and is the one invariant there
         that no 1D run has: conserved by a central U, not by an anisotropic
         one, which makes it the fastest visual check that a 2D potential is
         what you think it is -->
    <SeriesPlot v-if="ndim > 1" :key="'lz' + plotsKey" :session-id="sessionId"
                :variants="variants" which="lz" :ndim="ndim"
                :show-grid="showGrid" :cursor-t="lastFrame?.t ?? null" />
  </div>
</template>
