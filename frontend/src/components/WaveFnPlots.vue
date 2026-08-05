<script setup lang="ts">
/**
 * ψ and its Fourier image φ, as Re / Im / |·|².
 *
 * TWO CHARTS, not two traces on one — ψ is indexed by a spatial coordinate and
 * φ by a momentum one, and uPlot's AlignedData carries a single shared abscissa.
 * That is the same bug lib/potentialCuts.ts records for the 2D U(x,y) cuts, and
 * it is invisible on an isotropic default box because the two windows coincide.
 *
 * RE AND IM CAN COINCIDE EXACTLY, and the styling has to survive that. It is
 * not a corner case: ψ = e^-(x-2)²/2 + i·e^-(x+2)²/2 gives φ ∝
 * e^-p²/2[(cos2p − sin2p) + i(cos2p − sin2p)], i.e. Re ≡ Im at EVERY p. Any two
 * equal-weight packets a relative factor of i apart do it.
 *
 * Two cues were tried and both failed there. A shared colour plus a dash — the
 * repo's usual anti-occlusion device — draws as one solid line, because the
 * dash's gaps reveal the identical curve underneath. Distinct colours plus a
 * dash draws as one line of alternating segments: both hues are present (134 px
 * teal against 190 px purple, measured) but at 1.5 px they read as a single
 * muddy curve.
 *
 * What works is DIFFERENT WIDTHS. Re is a wide base, Im a thin dashed line on
 * top of it, so coincidence renders as a thin dark core inside a broad band —
 * two curves, unmistakably. Where they differ, they simply look like two curves.
 * The dash and the hues stay as the second and third cues.
 */
import { computed, onBeforeUnmount } from 'vue'

import LineChart, { type LineSeries } from './LineChart.vue'
import { createUplotZoom } from '../lib/uplotZoom'
import { conjugate } from '../lib/axes'
import { waveSeries, waveTitleHtml, type WaveCut } from '../lib/wavefn'

const props = defineProps<{
  psi: WaveCut | null
  phi: WaveCut | null
  ndim: number
  /** which spatial axis the cuts run along — needed for the TITLES, which must
   *  stand even when there is no data to title. */
  cutAxis: number
  /** why there is no data, when there is none */
  error?: string
  showGrid?: boolean
}>()

// Order IS z-order — uPlot draws series in array order, so Im lands on top of
// Re. That is deliberate and it is why Re is the WIDE one: see the note above.
const SERIES: LineSeries[] = [
  { role: 'waveRe', width: 3.5 },            // Re — the wide base
  { role: 'waveIm', dash: [5, 4], width: 1.5 },  // Im — thin, on top
  { role: 'waveAbs', width: 1.5 },           // |·|² — its own role, see theme.ts
]

/**
 * Drag-select / wheel / Shift-wheel / dblclick-reset, the same gesture set every
 * other chart in the app has.
 *
 * ONE INSTANCE PER CHART, created here in setup rather than inside LineChart, so
 * the window survives that component's destroy+rebuild on a theme change — the
 * reason PotentialEditor and the series plots do it this way too.
 *
 * `clampX: false` is deliberate: these traces are decimated to at most `n`
 * samples, so zooming in reveals the lattice rather than more detail. Unlike the
 * potential editor there is no re-sample loop behind it — the window is a pure
 * client-side view and never becomes a new request.
 */
const psiZoom = createUplotZoom({ clampX: false })
const phiZoom = createUplotZoom({ clampX: false })
onBeforeUnmount(() => { psiZoom.reset(); phiZoom.reset() })

const psiData = computed(() => (props.psi ? waveSeries(props.psi) : null))
const phiData = computed(() => (props.phi ? waveSeries(props.phi) : null))
// The titles come from the CUT, not from the response, so an empty chart is
// still labelled `ψ(x)` rather than a bare `ψ`. A title that degrades whenever
// the data does turns "this request failed" into "this component is broken".
const psiTitle = computed(() => waveTitleHtml(
  'ψ', props.ndim, props.psi ?? { axis: props.cutAxis, at: null, v: [], re: [], im: [] }))
const phiTitle = computed(() => waveTitleHtml(
  'φ', props.ndim, props.phi
    ?? { axis: conjugate(props.ndim, props.cutAxis), at: null, v: [], re: [], im: [] }))
</script>

<template>
  <div class="space-y-1">
    <!-- ONE shared legend row for both charts: uPlot's own legends would cost
         ~40px for the same information twice. -->
    <div class="flex items-center gap-3 text-[10px] text-fg-3">
      <span class="flex items-center gap-1">
        <span class="inline-block w-4 border-t-2" style="border-color: var(--wf-wave-re)"></span>Re
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block w-4 border-t-2 border-dashed"
              style="border-color: var(--wf-wave-im)"></span>Im
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block w-4 border-t-2" style="border-color: var(--wf-wave-abs)"></span>|·|²
      </span>
    </div>
    <div v-if="error" class="text-[11px] text-error">ψ could not be sampled: {{ error }}</div>
    <LineChart :data="psiData" :series="SERIES" :title-html="psiTitle"
               :height="110" :show-grid="showGrid" :zoom="psiZoom" />
    <LineChart :data="phiData" :series="SERIES" :title-html="phiTitle"
               :height="110" :show-grid="showGrid" :zoom="phiZoom" />
  </div>
</template>
