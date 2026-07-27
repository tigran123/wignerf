<script setup lang="ts">
/**
 * Axis grid overlay for the phase-space canvases (W panels, IC preview):
 * an SVG of tick lines at "nice" data intervals with edge labels, drawn
 * above the WebGL canvas. pointer-events: none — dragging/panning below
 * is unaffected.
 *
 * The axes are named by POSITION (a horizontal, b vertical) and their glyphs
 * come in as props: at ndim=2 a panel may show any of six pairs, so "x" and
 * "p" cannot be baked in here.
 *
 * Two independent layers, two toggles. The TICK lines sit at nice data
 * intervals and are about reading values off the plot. The CELL lines are the
 * actual lattice W is computed on, with the outer edge band the boundary watch
 * sums drawn brighter — so "1.3e-4 of its integral is in the outer 4 cells"
 * points at something you can see. Both stay theme-INDEPENDENT grey: they are
 * drawn ON the bwr heatmap, not on the page.
 */
import { computed } from 'vue'
import { cellLines, type AxisLattice } from '../lib/cells'

const props = defineProps<{
  /** horizontal axis extents */
  a1: number
  a2: number
  /** vertical axis extents (drawn upward) */
  b1: number
  b2: number
  /** axis glyphs, e.g. 'x' and 'p', or 'y' and 'px' */
  aLabel?: string
  bLabel?: string
  /** full-domain lattice of each axis (NOT the view window) — needed to place
   *  cell boundaries, which are a property of the grid, not of the zoom */
  aAxis?: AxisLattice
  bAxis?: AxisLattice
  /** draw the computed cell lattice + the edge band */
  showCells?: boolean
  /** draw the nice-interval tick lines, their value labels and the axis names.
   *  Independent of showCells: either layer can be on alone. */
  showTicks?: boolean
}>()

// SVG has no <sub>, so a two-letter name is drawn as head + a smaller,
// baseline-shifted tspan — the same subscript the HTML labels get.
const head = (n?: string) => (n && n.length === 2 ? n[0] : n) ?? ''
const tail = (n?: string) => (n && n.length === 2 ? n[1] : '') ?? ''

function niceTicks(a: number, b: number, target = 8): number[] {
  const span = b - a
  if (!(span > 0)) return []
  const raw = span / target
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= target)!
  const ticks: number[] = []
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-12 * span; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  }
  return ticks
}

const at = computed(() => niceTicks(props.a1, props.a2))
const bt = computed(() => niceTicks(props.b1, props.b2))

const ac = computed(() => (props.showCells && props.aAxis
  ? cellLines(props.aAxis, props.a1, props.a2)
  : { lattice: [], band: [] }))
const bc = computed(() => (props.showCells && props.bAxis
  ? cellLines(props.bAxis, props.b1, props.b2)
  : { lattice: [], band: [] }))

const fa = (v: number) => (100 * (v - props.a1)) / (props.a2 - props.a1)
const fb = (v: number) => 100 - (100 * (v - props.b1)) / (props.b2 - props.b1)

function fmt(v: number): string {
  return Math.abs(v) < 1e-12 ? '0' : String(parseFloat(v.toPrecision(6)))
}
</script>

<template>
  <svg class="absolute inset-0 w-full h-full pointer-events-none select-none"
       preserveAspectRatio="none">
    <!-- cell lattice, faint; then the edge-band cells, brighter. Drawn first
         so the tick lines and labels stay the legible layer on top. -->
    <line v-for="(v, i) in ac.lattice" :key="'ca' + i"
          :x1="fa(v) + '%'" :x2="fa(v) + '%'" y1="0" y2="100%"
          stroke="rgba(120,120,120,0.13)" stroke-width="1" />
    <line v-for="(v, i) in bc.lattice" :key="'cb' + i"
          :y1="fb(v) + '%'" :y2="fb(v) + '%'" x1="0" x2="100%"
          stroke="rgba(120,120,120,0.13)" stroke-width="1" />
    <line v-for="(v, i) in ac.band" :key="'ba' + i"
          :x1="fa(v) + '%'" :x2="fa(v) + '%'" y1="0" y2="100%"
          stroke="rgba(120,120,120,0.42)" stroke-width="1" />
    <line v-for="(v, i) in bc.band" :key="'bb' + i"
          :y1="fb(v) + '%'" :y2="fb(v) + '%'" x1="0" x2="100%"
          stroke="rgba(120,120,120,0.42)" stroke-width="1" />
    <line v-for="v in (showTicks ? at : [])" :key="'a' + v"
          :x1="fa(v) + '%'" :x2="fa(v) + '%'" y1="0" y2="100%"
          :stroke="v === 0 ? 'rgba(120,120,120,0.55)' : 'rgba(120,120,120,0.28)'"
          stroke-width="1" />
    <line v-for="v in (showTicks ? bt : [])" :key="'b' + v"
          :y1="fb(v) + '%'" :y2="fb(v) + '%'" x1="0" x2="100%"
          :stroke="v === 0 ? 'rgba(120,120,120,0.55)' : 'rgba(120,120,120,0.28)'"
          stroke-width="1" />
    <text v-for="v in (showTicks ? at : [])" :key="'al' + v"
          :x="fa(v) + '%'" y="99%" dx="2"
          fill="#737373" font-size="9">{{ fmt(v) }}</text>
    <text v-for="v in (showTicks ? bt : [])" :key="'bl' + v"
          x="0" :y="fb(v) + '%'" dx="2" dy="-2"
          fill="#737373" font-size="9">{{ fmt(v) }}</text>
    <!-- axis names beside the "0" tick labels: the horizontal one right of the
         bottom-edge 0, the vertical one just above the left-edge 0 -->
    <text v-if="showTicks && aLabel && a1 < 0 && a2 > 0" :x="fa(0) + '%'" y="99%" dx="14"
          fill="#404040" font-size="14" font-style="italic"
          font-weight="bold">{{ head(aLabel)
      }}<tspan v-if="tail(aLabel)" font-size="10" dy="3">{{ tail(aLabel) }}</tspan></text>
    <text v-if="showTicks && bLabel && b1 < 0 && b2 > 0" x="0" :y="fb(0) + '%'" dx="2" dy="-14"
          fill="#404040" font-size="14" font-style="italic"
          font-weight="bold">{{ head(bLabel)
      }}<tspan v-if="tail(bLabel)" font-size="10" dy="3">{{ tail(bLabel) }}</tspan></text>
  </svg>
</template>
