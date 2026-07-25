<script setup lang="ts">
/** The diagnostics column: marginals, E(t), ΔX·ΔP(t), purity γ(t). */
import MarginalsPlot from './MarginalsPlot.vue'
import SeriesPlot from './SeriesPlot.vue'
import type { Frame } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import type { VariantKey } from '../lib/variants'

defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  sessionId: string | null
  variants: VariantKey[]
  grid: GridCfg
  lastFrame: Frame | null
  showGrid: boolean
  plotsKey: string
  // batch compute streams no frames — the colorbar and marginals (frame-fed)
  // are dimmed; the series plots below stay live (they poll REST, not frames)
  batchComputing?: boolean
}>()
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
      <MarginalsPlot :key="'r' + plotsKey" :frame-source="frameSource"
                     :variants="variants" which="rho" :show-grid="showGrid"
                     :a1="grid.x1" :a2="grid.x2" :n="grid.Nx" />
      <MarginalsPlot :key="'p' + plotsKey" :frame-source="frameSource"
                     :variants="variants" which="phi" :show-grid="showGrid"
                     :a1="grid.p1" :a2="grid.p2" :n="grid.Np" />
    </div>
    <!-- cursor-t: the painted frame's time, so the series carry the same
         moving marker the exported video does -->
    <SeriesPlot :key="'e' + plotsKey" :session-id="sessionId"
                :variants="variants" which="E" :show-grid="showGrid"
                :cursor-t="lastFrame?.t ?? null" />
    <SeriesPlot :key="'u' + plotsKey" :session-id="sessionId"
                :variants="variants" which="uncertainty" :show-grid="showGrid"
                :cursor-t="lastFrame?.t ?? null" />
    <SeriesPlot :key="'g' + plotsKey" :session-id="sessionId"
                :variants="variants" which="purity" :show-grid="showGrid"
                :cursor-t="lastFrame?.t ?? null" />
  </div>
</template>
