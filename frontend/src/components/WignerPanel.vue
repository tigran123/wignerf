<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import Colorbar from './Colorbar.vue'
import GridOverlay from './GridOverlay.vue'
import type { Frame } from '../lib/protocol'
import { variantName } from '../lib/protocol'
import { label as axisLabel, planeIndex, planeTitle } from '../lib/axes'
import { isZoomed, panBy, resetView, zoomAt, type ViewWindow } from '../lib/viewWindow'
import type { PlaneViewReq } from '../lib/planeView'
import type { AxisLattice } from '../lib/cells'
import { WignerRenderer } from '../render/WignerRenderer'

const emit = defineEmits<{ view: [PlaneViewReq] }>()

const props = defineProps<{
  /** Register a frame handler with the session; returns unsubscribe. */
  frameSource: (h: (f: Frame) => void) => () => void
  /** Which variant of the bundle this panel shows. */
  variantIndex: number
  /**
   * Which 2D plane of that variant, as its axis pair (see lib/axes.ts). A 1D
   * record has exactly one, [0, 1], and that plane IS W.
   */
  plane?: readonly [number, number]
  label?: string
  showGrid?: boolean
  /** draw the computed cell lattice + the boundary watch's edge band */
  showCells?: boolean
  /** Zoom/pan window driving this panel. PanelGrid swaps it between the
   *  panel's own and the shared one when "link zoom" toggles. */
  view?: ViewWindow
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
let ro: ResizeObserver | null = null
/** Decimation of the plane last painted; 1 = every computed cell. */
const reduction = ref(1)
// The variant name from the last painted frame, used only when the parent gives
// no label. A plain ref seeded from props.label was a reactivity bug: PanelGrid
// REUSES a panel instance across a view-mode switch (the cell key is the same,
// which is what preserves its zoom window), so a panel that started life
// labelled "Quantum, non-relativistic" kept that text after the label prop
// became "x,y".
const vidName = ref('')
/** This panel's variant id, from the painted frame — the server keys views by
 *  it, and the panel's own `variantIndex` is a position in the bundle. */
const vidOf = ref<number | null>(null)
const title = computed(() => props.label ?? vidName.value)
const subtitle = ref('')
// This panel's OWN colour range, for its overlaid colorbar. Per panel because
// the renderer autoscales per frame from exactly these two numbers (q = [wmin,
// wmax]); the variants drift apart as they evolve (see Colorbar.vue), and at
// ndim=2 the six reductions of ONE state differ by orders of magnitude.
const wmin = ref<number | null>(null)
const wmax = ref<number | null>(null)
const glError = ref('')
const panning = ref(false)
const renderer = new WignerRenderer()
let unsub: (() => void) | null = null
let lastX = 0
let lastY = 0

const pair = computed<readonly [number, number]>(() => props.plane ?? [0, 1])

/**
 * The FULL extents of this panel's own two axes, taken from the painted frame
 * — geometry is a per-record fact, so a scrub across a regrid boundary just
 * works. Kept as a ref (not a prop) so a 2D panel needs no ndim plumbing.
 */
const full = ref<{ a1: number; a2: number; b1: number; b2: number;
                  na: number; nb: number } | null>(null)

/** This plane's two lattices, for the cell overlay. Taken from the painted
 *  frame like the extents are, so a scrub across a regrid draws the cells that
 *  record really had. */
const lattice = computed<[AxisLattice, AxisLattice] | null>(() => {
  const d = full.value
  if (!d) return null
  return [{ lo: d.a1, hi: d.a2, n: d.na }, { lo: d.b1, hi: d.b2, n: d.nb }]
})

/** Extents of the current view WINDOW, for the axis overlay. */
const viewDomain = computed(() => {
  const d = full.value
  if (!d) return null
  const v = props.view ?? { x0: 0, x1: 1, y0: 0, y1: 1 }
  return {
    a1: d.a1 + v.x0 * (d.a2 - d.a1),
    a2: d.a1 + v.x1 * (d.a2 - d.a1),
    b1: d.b1 + v.y0 * (d.b2 - d.b1),
    b2: d.b1 + v.y1 * (d.b2 - d.b1),
  }
})

const labels = ref<[string, string]>(['x', 'p'])

/**
 * Tell the parent what this panel is showing, so the server can send that and
 * nothing else. PHYSICAL extents, because they survive an auto-expand regrid
 * with no bookkeeping — the same region keeps meaning the same thing when the
 * domain doubles under it, where a fraction quietly comes to mean somewhere
 * else.
 *
 * Emitted rather than sent: the request has to be BATCHED with every other
 * panel's (see PanelGrid), or four panels would each overwrite the others'
 * entries in a map the server keys by panel.
 */
function requestView() {
  const d = viewDomain.value
  const vid = vidOf.value
  if (!d || vid == null) return
  const [pw, ph] = renderer.pixelSize()
  if (!pw || !ph) return
  const [a, b] = pair.value
  emit('view', { vid, a, b, a1: d.a1, a2: d.a2, b1: d.b1, b2: d.b2,
                 na: pw, nb: ph })
}

function onWheel(e: WheelEvent) {
  if (!props.view) return
  const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
  zoomAt(props.view,
    (e.clientX - r.left) / r.width,
    1 - (e.clientY - r.top) / r.height,
    e.deltaY < 0 ? 0.85 : 1 / 0.85)
}

function onDown(e: PointerEvent) {
  if (!props.view) return
  panning.value = true
  lastX = e.clientX
  lastY = e.clientY
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}

function onMove(e: PointerEvent) {
  if (!panning.value || !props.view) return
  const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
  // screen y is down, the vertical-axis fraction is up
  panBy(props.view, (e.clientX - lastX) / r.width, -(e.clientY - lastY) / r.height)
  lastX = e.clientX
  lastY = e.clientY
}

function onUp() { panning.value = false }

function onDblClick() {
  if (props.view) resetView(props.view)
}

onMounted(() => {
  try {
    renderer.init(canvas.value!)
  } catch (e) {
    glError.value = String(e)
    return
  }
  // the session fan-out is already rAF-timed (one frame per animation
  // frame), so upload + draw run directly, once per painted frame
  unsub = props.frameSource((f: Frame) => {
    const v = f.variants[props.variantIndex]
    if (!v) return
    const [a, b] = pair.value
    // BY PAIR, not by index: the server omits planes this client is not
    // showing, and a positional lookup into a list with holes silently draws
    // the wrong plane. (It is still the canonical order — but nothing here
    // needs to depend on that.)
    const p = v.planes.find((q) => q.a === a && q.b === b)
    if (!p) return
    vidName.value = variantName(v.vid)
    vidOf.value = v.vid
    subtitle.value = planeTitle(f.ndim, [a, b], true)
    labels.value = [axisLabel(f.ndim, a), axisLabel(f.ndim, b)]
    full.value = { a1: f.lo[a]!, a2: f.hi[a]!, b1: f.lo[b]!, b2: f.hi[b]!,
                   na: f.N[a]!, nb: f.N[b]! }
    // na === 0 is "not sent" — keep the last texture rather than blanking the
    // panel, which is what a frame arriving between a request and its answer
    // would otherwise do.
    if (p.na === 0) return
    wmin.value = p.wmin
    wmax.value = p.wmax
    reduction.value = Math.max(p.step[0], p.step[1])
    renderer.upload(p, f.N[a]!, f.N[b]!)
    renderer.render()
    requestView()
  })
  // zoom/pan repaint: this is what redraws while PAUSED; the getter tracks
  // both the window's fields AND its identity (own <-> shared on "link
  // zoom" toggles); render() no-ops until the first frame arrives
  watch(
    () => (props.view ? [props.view.x0, props.view.x1, props.view.y0, props.view.y1] : null),
    (v) => {
      if (!v) return
      renderer.setView(v[0]!, v[1]!, v[2]!, v[3]!)
      renderer.render()
      requestView()
    },
  )
  // A panel's pixel size is a request input, and nothing else notices it
  // changing: the canvas is CSS-sized, so a window resize or a layout toggle
  // moves it with no Vue reactivity at all.
  ro = new ResizeObserver(() => { renderer.render(); requestView() })
  ro.observe(canvas.value!)
  requestView()
})

onBeforeUnmount(() => {
  unsub?.()
  ro?.disconnect()
  renderer.dispose()
})
</script>

<template>
  <div class="relative w-full h-full min-h-0 touch-none"
       :class="view && isZoomed(view) ? (panning ? 'cursor-grabbing' : 'cursor-grab') : ''"
       @wheel.prevent="onWheel"
       @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp"
       @pointercancel="onUp" @dblclick="onDblClick">
    <!-- the shader covers every pixel, so this only shows before the first
         draw — a black flash on a light page, hence a themed surface -->
    <canvas ref="canvas" class="w-full h-full block bg-panel"></canvas>
    <GridOverlay v-if="viewDomain && (showGrid || showCells)"
                 :a1="viewDomain.a1" :a2="viewDomain.a2"
                 :b1="viewDomain.b1" :b2="viewDomain.b2"
                 :a-label="labels[0]" :b-label="labels[1]"
                 :show-ticks="showGrid"
                 :a-axis="lattice?.[0]" :b-axis="lattice?.[1]"
                 :show-cells="showCells" />
    <div class="absolute top-1 left-2 text-xs text-white bg-black/75 px-1.5 py-0.5 rounded">
      <span v-html="title"></span>
      <!-- at ndim=2 the panel must say WHICH reduction it is; at 1D subtitle
           is empty, because the single plane is the state itself -->
      <span v-if="subtitle" class="opacity-75">· <span v-html="subtitle"></span></span>
      <!-- Reduced panels SAY SO. Area-averaging is honest but invisible: the
           image looks like a smooth W rather than an under-resolved one, so
           fringes finer than a pixel are gone with nothing on screen to
           suggest it. Costs no line — it rides the label chip already there. -->
      <span v-if="reduction > 1" class="opacity-75"
            :title="`This panel is drawn from every ${reduction}th computed cell along each axis — `
              + `each pixel is the average of ${reduction * reduction} of them, which is what W on a `
              + `coarser grid looks like. Interference finer than one pixel is averaged away; zoom in `
              + `to get the computed resolution back. The server sends only this, which is what keeps `
              + `a large grid interactive.`">
        · ↓{{ reduction }}×</span>
    </div>
    <!-- this panel's own colour scale; overlaid, so it costs no drawing area -->
    <Colorbar :min="wmin" :max="wmax" />
    <div v-if="glError" class="absolute inset-0 grid place-items-center text-error text-sm p-4">
      {{ glError }}
    </div>
  </div>
</template>
