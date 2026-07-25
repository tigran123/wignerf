<script setup lang="ts">
/**
 * Colour scale of ONE W plot, overlaid in a corner of that plot.
 *
 * It matches the shader's SYMMETRIC diverging scaling exactly: white pinned at
 * W = 0, colour intensity proportional to |W| with the shared scale
 * max(Wmax, -Wmin) (render/WignerRenderer.ts). For a frame with tiny Wmin the
 * left end is therefore near-white, not saturated blue — same as the panel.
 *
 * PER PLOT, and overlaid, for two reasons. Every panel autoscales to its OWN
 * range (WignerPanel uploads `f.variants[variantIndex]`, and the renderer sets
 * q = [v.wmin, v.wmax] from it), so one shared bar beside a 2x2 grid labelled
 * three of the four panels wrongly — measured on a cat state in x²/2 + 0.3x⁴,
 * QN vs CN at t = 15: wmin -1.87e-1 vs -2.70e-1 (44% apart) and wmax +3.18e-1
 * vs +3.51e-1. At t = 0 every variant agrees (record 0 IS the IC), which is
 * exactly why a single bar looked right. And overlaid because it then costs no
 * layout height at all: it used to head the diagnostics column, the tallest of
 * the three in portrait, delaying where the panels begin.
 */
import { computed } from 'vue'

const props = defineProps<{
  /** null while no frame has been painted yet — the bar then hides itself. */
  min: number | null
  max: number | null
  /** where to sit. Default tucks it under a W panel's name label, which keeps
   *  it clear of GridOverlay's axis tick row along the bottom AND groups the
   *  panel's identity with its scale. The IC preview has no name label, so it
   *  passes the top slot instead. */
  place?: string
}>()

function bwr(u: number): string {
  u = Math.min(1, Math.max(0, u))
  let r: number, g: number, b: number
  if (u < 0.5) {
    const s = u * 2
    r = s; g = s; b = 1
  } else {
    const s = (u - 0.5) * 2
    r = 1; g = 1 - s; b = 1 - s
  }
  return `rgb(${Math.round(255 * r)},${Math.round(255 * g)},${Math.round(255 * b)})`
}

const ok = computed(() => props.min != null && props.max != null)
const gradient = computed(() => {
  if (!ok.value) return 'linear-gradient(to right, #00f, #fff, #f00)'
  const min = props.min as number, max = props.max as number
  const scale = Math.max(max, -min, 1e-300)
  const f0 = Math.min(100, Math.max(0, (100 * (0 - min)) / (max - min || 1)))
  return `linear-gradient(to right, ${bwr(0.5 + 0.5 * min / scale)} 0%, ` +
    `rgb(255,255,255) ${f0}%, ${bwr(0.5 + 0.5 * max / scale)} 100%)`
})
</script>

<template>
  <!-- absolute: zero layout cost, so a plot loses no drawing area to it -->
  <div v-if="ok" class="absolute z-10 w-28 text-[9px] leading-none text-white
                        tabular-nums bg-black/70 rounded px-1 py-0.5
                        pointer-events-none"
       :class="place ?? 'top-7 left-2'"
       :title="`colour scale of THIS plot: white at W = 0, intensity ∝ |W| on the
                symmetric scale max(Wmax, −Wmin). Every panel autoscales to its
                own range, so these numbers are this panel's alone.`">
    <div class="h-1.5 rounded-sm border border-white/25" :style="{ background: gradient }"></div>
    <div class="flex justify-between mt-0.5">
      <span>{{ (min as number).toExponential(1) }}</span>
      <span class="text-white/60">0</span>
      <span>{{ (max as number).toExponential(1) }}</span>
    </div>
  </div>
</template>
