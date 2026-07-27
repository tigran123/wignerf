<script setup lang="ts">
/**
 * Initial-condition editor: a list of Gaussian components (mixture or cat)
 * plus a live phase-space preview (POST /api/preview/wigner returns the
 * same binary bundle as the stream, rendered by the same WebGL renderer).
 * Components are draggable directly on the preview canvas — pointer down
 * near a peak marker grabs it; the two coordinates of the DISPLAYED PLANE
 * follow the pointer.
 *
 * At ndim=2 the preview shows one selectable plane, and dragging edits exactly
 * that plane's pair: drag in (x,y) to move q0, in (x,px) to move x0 and px0.
 * That is the whole 2D interaction — no new gesture, just a plane to pick.
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import Colorbar from './Colorbar.vue'
import GridOverlay from './GridOverlay.vue'
import { decodeFrame } from '../lib/protocol'
import { apiErrorText } from '../lib/apierror'
import { label as axisLabel, labels as axisLabels, labelHtml, planeLabel,
         planes as planesOf } from '../lib/axes'
import { defaultIC, type GridCfg, type ICCfg,
         type ICComponentCfg } from '../lib/config'
import { createViewWindow, panBy, resetView, zoomAt } from '../lib/viewWindow'
import { edgeBand, type AxisLattice } from '../lib/cells'
import { WignerRenderer } from '../render/WignerRenderer'

const props = defineProps<{
  ic: ICCfg
  grid: GridCfg
  hbarEff: number
  showGrid?: boolean
  showCells?: boolean
  /** axes the live session's boundary watch already reports — see edgeNotice */
  liveEdgeAxes?: string[] | null
  /**
   * The header is already showing an error. The preview endpoint refuses the
   * same things session creation does — a grid over the host's cell ceiling,
   * above all — so both calls fail with the identical sentence and the user
   * read it twice, once at the top and once under this plot. This line is for
   * telling you YOUR IC is wrong; when something larger is already being
   * reported, it stays quiet. previewOk still goes false, so Solve remains
   * gated — the header says why.
   */
  quiet?: boolean
}>()

const emit = defineEmits<{
  (e: 'changed'): void
  (e: 'validity', valid: boolean): void
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
const overlay = ref<HTMLDivElement | null>(null)
const selected = ref(0)
const deficit = ref('')
const warnings = ref<string[]>([])
/** Axes whose edge band the SAMPLED total W trips, from X-Wignerf-Edge. */
const edgeFound = ref<{ axis: string; mass: number }[]>([])

/**
 * The edge warning, for the axes the RUNNING session is not already warning
 * about. The session's boundary watch reports the same finding from record 0 on
 * — record 0 IS this IC — so showing both put one fact on screen twice, once as
 * a header line and once as a paragraph here. Differencing the axis sets rather
 * than hiding this outright keeps the case the header cannot cover: an IC you
 * have EDITED but not yet restarted into, whose new axes appear here alone.
 *
 * NB the caller must feed this something that survives a DISMISSAL. It used to
 * get `session.boundary` alone — the transient event — and the header's × sets
 * that to null, so dismissing the header made this line appear instead: the same
 * fact stated twice over, just sequentially rather than at once. SimulatorView
 * therefore falls back to `status.boundary.axes`, which is the session's standing
 * boundary state and is not what the × clears.
 */
const edgeNotice = computed(() => {
  const live = new Set(props.liveEdgeAxes ?? [])
  const fresh = edgeFound.value.filter((e) => !live.has(e.axis))
  if (!fresh.length) return ''
  const worst = fresh.reduce((a, b) => (b.mass > a.mass ? b : a))
  const ax = props.grid.axes
  const i = axisLabels(props.grid.ndim).indexOf(worst.axis)
  const cells = i >= 0 && ax[i] ? edgeBand(ax[i]!.N) : 0
  return `this IC reaches the ${fresh.map((e) => e.axis).join(', ')} edge — ` +
         `${worst.mass.toExponential(1)} of its integral is in the outer ` +
         `${cells} cells.`
})
const renderer = new WignerRenderer()
let timer: ReturnType<typeof setTimeout> | null = null
let dragging = -1
const panning = ref(false)
let lastX = 0
let lastY = 0

// zoom/pan window of the preview, independent of the main panels' one
const view = createViewWindow()

const ndim = computed(() => props.grid.ndim)
const allPlanes = computed(() => planesOf(ndim.value))
const planeIdx = ref(0)
const plane = computed<readonly [number, number]>(() =>
  allPlanes.value[Math.min(planeIdx.value, allPlanes.value.length - 1)]!)
const planeLabels = computed<[string, string]>(() =>
  [axisLabel(ndim.value, plane.value[0]), axisLabel(ndim.value, plane.value[1])])

/** Full extents of the displayed plane's two axes. */
const planeExtents = computed(() => {
  const [a, b] = plane.value
  const ax = props.grid.axes
  return { a1: ax[a]!.lo, a2: ax[a]!.hi, b1: ax[b]!.lo, b2: ax[b]!.hi }
})

/** The displayed plane's two lattices, for the cell overlay. Straight off the
 *  FORM's grid — the preview is built at exactly that grid. */
const planeLattice = computed<[AxisLattice, AxisLattice]>(() => {
  const [a, b] = plane.value
  const ax = props.grid.axes
  return [{ lo: ax[a]!.lo, hi: ax[a]!.hi, n: ax[a]!.N },
          { lo: ax[b]!.lo, hi: ax[b]!.hi, n: ax[b]!.N }]
})

/** Extents of the current view WINDOW: the pointer↔phase-space mapping and
 *  the axis overlay both go through it. */
const viewExtents = computed(() => {
  const g = planeExtents.value
  return {
    a1: g.a1 + view.x0 * (g.a2 - g.a1),
    a2: g.a1 + view.x1 * (g.a2 - g.a1),
    b1: g.b1 + view.y0 * (g.b2 - g.b1),
    b2: g.b1 + view.y1 * (g.b2 - g.b1),
  }
})

/**
 * Read/write one component coordinate by PHASE-SPACE AXIS index: axis a < ndim
 * is q0[a], otherwise k0[a-ndim]. Everything below (markers, dragging, the
 * parameter form) goes through this, so none of it has to know which half of
 * the component it is touching.
 */
function coord(c: ICComponentCfg, a: number): number {
  const nd = ndim.value
  return a < nd ? (c.q0[a] ?? 0) : (c.k0[a - nd] ?? 0)
}

function setCoord(c: ICComponentCfg, a: number, v: number) {
  const nd = ndim.value
  if (a < nd) c.q0[a] = v
  else c.k0[a - nd] = v
}

const state = reactive({ previewOk: false, error: '' })
// kept so switching planes repaints without a round-trip: the reductions are
// all in the bundle already
let lastFrame: ReturnType<typeof decodeFrame> | null = null
const wmin = ref<number | null>(null)
const wmax = ref<number | null>(null)

// gate for the transport Solve: an IC (or grid) the preview endpoint
// rejects must not coexist with a running computation; starts pessimistic
// until the first preview lands
watch(() => state.previewOk, (v) => emit('validity', v), { immediate: true })

function derivedSigmaP(sx: number): number {
  return props.hbarEff / (2 * sx)
}

/** Repaint from the bundle we already hold when only the plane changed. */
function repaintPlane() {
  const f = lastFrame
  if (!f) return
  const [a, b] = plane.value
  const pl = f.variants[0]?.planes.find((q) => q.a === a && q.b === b)
  if (!pl) return
  wmin.value = pl.wmin
  wmax.value = pl.wmax
  renderer.upload(pl)
  renderer.render()
}

watch(plane, () => { resetView(view); repaintPlane() })

async function refresh() {
  try {
    const { data, headers } = await api.post('/preview/wigner', {
      type: props.ic.type,
      components: props.ic.components,
      grid: props.grid,
      hbar_eff: props.hbarEff,
    }, { responseType: 'arraybuffer' })
    const f = decodeFrame(data as ArrayBuffer)
    const v = f.variants[0]!
    const [a, b] = plane.value
    const pl = v.planes.find((q) => q.a === a && q.b === b) ?? v.planes[0]!
    // the preview's OWN colour range, for its overlaid bar: it is a W plot like
    // the panels and autoscales the same way, so it carries its own scale
    wmin.value = pl.wmin
    wmax.value = pl.wmax
    lastFrame = f
    renderer.upload(pl)
    renderer.render()
    deficit.value = String(headers['x-wignerf-norm-deficit'] ?? '')
    // percent-encoded server-side: HTTP headers are latin-1, the messages
    // carry Unicode (sigma, hbar, rho...)
    const w = decodeURIComponent(String(headers['x-wignerf-warnings'] ?? ''))
    warnings.value = w ? w.split(' | ') : []
    const e = String(headers['x-wignerf-edge'] ?? '')
    edgeFound.value = e ? e.split(',').map((s) => {
      const [a, m] = s.split(':')
      return { axis: a!, mass: Number(m) }
    }) : []
    state.previewOk = true
    state.error = ''
  } catch (e: unknown) {
    state.previewOk = false
    // apiErrorText, like every other failed call in the app — it unwraps the
    // arraybuffer body this request type produces (see lib/apierror.ts)
    state.error = apiErrorText(e)
  }
}

function scheduleRefresh(notify = true) {
  if (notify) emit('changed')
  if (timer) clearTimeout(timer)
  timer = setTimeout(refresh, 150)
}

watch(() => [props.ic, props.grid, props.hbarEff], () => scheduleRefresh(false),
  { deep: true })

// -- drag-to-place / pan / zoom ---------------------------------------------

/** Pointer -> the displayed plane's two coordinates (a horizontal, b up). */
function toData(ev: { clientX: number; clientY: number }): { a: number; b: number } {
  const r = overlay.value!.getBoundingClientRect()
  const fx = (ev.clientX - r.left) / r.width
  const fy = (ev.clientY - r.top) / r.height
  const v = viewExtents.value
  return {
    a: v.a1 + fx * (v.a2 - v.a1),
    b: v.b2 - fy * (v.b2 - v.b1),
  }
}

function markerStyle(c: ICComponentCfg) {
  const v = viewExtents.value
  const [pa, pb] = plane.value
  const fx = (coord(c, pa) - v.a1) / (v.a2 - v.a1)
  const fy = 1 - (coord(c, pb) - v.b1) / (v.b2 - v.b1)
  return { left: `${100 * fx}%`, top: `${100 * fy}%` }
}

/** Nearest component within grab range of the pointer, or -1. The radius
 *  is view-relative, so grabbing works the same at any zoom. */
function nearestComponent(ev: { clientX: number; clientY: number }): number {
  const d = toData(ev)
  const v = viewExtents.value
  const [pa, pb] = plane.value
  const sa = (v.a2 - v.a1) / 15
  const sb = (v.b2 - v.b1) / 15
  let best = -1
  let bestDist = 1
  props.ic.components.forEach((c, i) => {
    const dist = Math.hypot((coord(c, pa) - d.a) / sa, (coord(c, pb) - d.b) / sb)
    if (dist < bestDist) { best = i; bestDist = dist }
  })
  return best
}

function onDown(ev: PointerEvent) {
  const best = nearestComponent(ev)
  if (best >= 0) {
    // markers win; empty space pans
    dragging = best
    selected.value = best
  } else {
    panning.value = true
    lastX = ev.clientX
    lastY = ev.clientY
  }
  ;(ev.target as HTMLElement).setPointerCapture(ev.pointerId)
}

function onMove(ev: PointerEvent) {
  if (dragging >= 0) {
    const d = toData(ev)
    const c = props.ic.components[dragging]!
    const [pa, pb] = plane.value
    setCoord(c, pa, Math.round(d.a * 1000) / 1000)
    setCoord(c, pb, Math.round(d.b * 1000) / 1000)
    scheduleRefresh()
    return
  }
  if (!panning.value) return
  const r = overlay.value!.getBoundingClientRect()
  panBy(view, (ev.clientX - lastX) / r.width, -(ev.clientY - lastY) / r.height)
  lastX = ev.clientX
  lastY = ev.clientY
}

function onUp() {
  dragging = -1
  panning.value = false
}

function onWheel(ev: WheelEvent) {
  const r = overlay.value!.getBoundingClientRect()
  zoomAt(view,
    (ev.clientX - r.left) / r.width,
    1 - (ev.clientY - r.top) / r.height,
    ev.deltaY < 0 ? 0.85 : 1 / 0.85)
}

function onDblClick(ev: MouseEvent) {
  // double-clicking a marker must never surprise-reset the view
  if (nearestComponent(ev) >= 0) return
  resetView(view)
}

// -- component list ----------------------------------------------------------

function addComponent() {
  // new mixture components start as minimal packets (sigma_p = hbar/2sigma_x)
  // so adding one never makes W sub-Heisenberg by default
  const nd = ndim.value
  const zeros = () => Array.from({ length: nd }, () => 0)
  const halves = () => Array.from({ length: nd }, () => 0.5)
  props.ic.components.push({
    q0: zeros(), k0: zeros(), sigma_q: halves(),
    sigma_k: props.ic.type === 'cat' ? null
      : Array.from({ length: nd }, () => derivedSigmaP(0.5)),
    weight: 1, phase: 0,
  })
  selected.value = props.ic.components.length - 1
  scheduleRefresh()
}

function removeComponent(i: number) {
  if (props.ic.components.length <= 1) return
  props.ic.components.splice(i, 1)
  selected.value = Math.min(selected.value, props.ic.components.length - 1)
  scheduleRefresh()
}

function resetIC() {
  if (!confirm('Reset the initial condition to the default single Gaussian?')) return
  // the default FOR THIS DIMENSIONALITY — a 2D editor must not be handed a
  // one-coordinate component the grid would then reject
  const d = defaultIC(ndim.value)
  const before = JSON.stringify(props.ic)
  props.ic.type = d.type
  props.ic.components.splice(0, props.ic.components.length, ...d.components)
  selected.value = 0
  // repaint either way, but only a real change marks the session restart-dirty
  // (see resetSetup: an already-default IC has not "changed")
  scheduleRefresh(JSON.stringify(props.ic) !== before)
}

function setType(t: 'mixture' | 'cat') {
  props.ic.type = t
  for (const c of props.ic.components) {
    if (t === 'cat') c.sigma_k = null
    else if (c.sigma_k == null) c.sigma_k = c.sigma_q.map(derivedSigmaP)
  }
  scheduleRefresh()
}

const sel = computed(() => props.ic.components[selected.value])

/** Number input -> a component field. Written once rather than per input
 *  because v-model cannot bind an array element through a helper. */
function onNum(ev: Event, set: (v: number) => void) {
  const v = Number((ev.target as HTMLInputElement).value)
  if (!Number.isFinite(v)) return
  set(v)
  scheduleRefresh()
}

/** Editing sigma_q must carry a MIXTURE's derived sigma_k along only when it
 *  was itself derived; a hand-set sigma_k is the user's number and stays. */
function setSigmaQ(c: ICComponentCfg, d: number, v: number) {
  const was = c.sigma_q[d] ?? 0.5
  c.sigma_q[d] = v
  if (c.sigma_k && Math.abs((c.sigma_k[d] ?? 0) - derivedSigmaP(was)) < 1e-9)
    c.sigma_k[d] = derivedSigmaP(v)
}

onMounted(() => {
  renderer.init(canvas.value!)
  // zoom/pan repaint: the uploaded texture persists between previews
  watch(view, (v) => {
    renderer.setView(v.x0, v.x1, v.y0, v.y1)
    renderer.render()
  })
  void refresh()
})

onBeforeUnmount(() => {
  if (timer) clearTimeout(timer)
  renderer.dispose()
})
</script>

<template>
  <section class="space-y-1.5">
    <h3 class="text-xs font-semibold text-fg-3 uppercase tracking-wider">
      Initial condition
    </h3>

    <div class="flex gap-1 text-xs">
      <button
        v-for="t in (['mixture', 'cat'] as const)" :key="t"
        class="flex-1 py-1 rounded border"
        :class="ic.type === t
          ? 'bg-info-soft border-info text-info-fg'
          : 'bg-panel border-line text-fg-3 hover:bg-raised'"
        :title="t === 'mixture'
          ? 'statistical mixture: W >= 0, independent σₓ, σₚ'
          : 'coherent superposition (cat): interference fringes, σₚ derived'"
        @click="setType(t)"
      >{{ t }}</button>
    </div>

    <!-- 2D only: which reduction the preview shows. Dragging edits exactly
         this plane's two coordinates, so the selector IS the 2D interaction. -->
    <div v-if="ndim > 1" class="flex items-center gap-1 text-xs">
      <span class="text-muted">plane</span>
      <select v-model.number="planeIdx" class="wf-num flex-1"
              title="which 2D reduction to preview — dragging a marker edits this plane's two coordinates">
        <option v-for="(pl, i) in allPlanes" :key="i" :value="i">
          {{ planeLabel(ndim, pl) }}
        </option>
      </select>
    </div>

    <!-- phase-space preview with draggable peaks -->
    <div class="relative aspect-square w-full border border-line rounded overflow-hidden">
      <canvas ref="canvas" class="w-full h-full block bg-panel"></canvas>
      <GridOverlay v-if="(showGrid ?? true) || showCells"
                   :a1="viewExtents.a1" :a2="viewExtents.a2"
                   :b1="viewExtents.b1" :b2="viewExtents.b2"
                   :a-label="planeLabels[0]" :b-label="planeLabels[1]"
                   :show-ticks="showGrid ?? true"
                   :a-axis="planeLattice[0]" :b-axis="planeLattice[1]"
                   :show-cells="showCells" />
      <div ref="overlay" class="absolute inset-0 touch-none"
           :class="panning ? 'cursor-grabbing' : 'cursor-crosshair'"
           @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp"
           @pointercancel="onUp" @wheel.prevent="onWheel" @dblclick="onDblClick">
        <!-- These rings sit on the bwr HEATMAP, not on the page, and the
             heatmap is the same in both themes — so they are deliberately
             not tokenised (same reasoning as GridOverlay's greys). -->
        <div v-for="(c, i) in ic.components" :key="i"
             class="absolute w-3 h-3 -ml-1.5 -mt-1.5 rounded-full border-2 pointer-events-none"
             :class="i === selected ? 'border-yellow-300' : 'border-neutral-400/70'"
             :style="markerStyle(c)"></div>
      </div>
      <!-- the preview's own colour scale, overlaid so it costs no height in a
           column the W panels are competing with. No name label here, so it
           takes the top slot the panels reserve for theirs. -->
      <Colorbar :min="wmin" :max="wmax" place="top-1 left-2" />
    </div>
    <div v-if="state.error && !quiet" class="text-[11px] text-error">{{ state.error }}</div>
    <div v-if="deficit" class="text-xs text-fg-3">
      norm deficit on grid: {{ deficit }}
    </div>
    <div v-for="(w, i) in warnings" :key="i" class="text-[11px] text-warn">⚠ {{ w }}</div>
    <div v-if="edgeNotice" class="text-[11px] text-warn">⚠ {{ edgeNotice }}</div>

    <!-- component list -->
    <div class="flex items-center gap-1 text-xs">
      <button v-for="(c, i) in ic.components" :key="i"
              class="px-2 py-0.5 rounded border"
              :class="i === selected
                ? 'border-select text-select'
                : 'border-line text-fg-3'"
              @click="selected = i">{{ i + 1 }}</button>
      <button class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover"
              @click="addComponent">+</button>
      <button class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover disabled:opacity-40"
              :disabled="ic.components.length <= 1"
              @click="removeComponent(selected)">−</button>
      <button class="ml-auto px-2 py-0.5 rounded bg-raised hover:bg-raised-hover"
              title="reset the IC to the default single Gaussian"
              @click="resetIC">↺ defaults</button>
    </div>

    <!-- step="any": with a discrete step= the browser rejects perfectly
         good values (0.60, drag-placed coordinates, even 1.0) -->
    <!-- One row per SPATIAL dimension: centre and width for that dimension's
         coordinate and its conjugate momentum. 1D renders the single x/p row
         it always did; 2D adds y/py below it. -->
    <div v-if="sel" class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
      <template v-for="d in ndim" :key="'d' + d">
        <label class="flex items-center gap-1">
          <span class="w-8 text-muted"><span v-html="labelHtml(ndim, d - 1)"></span>₀</span>
          <input :value="sel.q0[d - 1]" type="number" step="any" class="wf-num"
                 @change="onNum($event, (v) => { sel!.q0[d - 1] = v })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-8 text-muted"><span v-html="labelHtml(ndim, ndim + d - 1)"></span>₀</span>
          <input :value="sel.k0[d - 1]" type="number" step="any" class="wf-num"
                 @change="onNum($event, (v) => { sel!.k0[d - 1] = v })" />
        </label>
        <label class="flex items-center gap-1">
          <span class="w-8 text-muted">σ<span v-html="labelHtml(ndim, d - 1)"></span></span>
          <input :value="sel.sigma_q[d - 1]" type="number" step="any" min="0.01"
                 class="wf-num"
                 @change="onNum($event, (v) => { setSigmaQ(sel!, d - 1, v) })" />
        </label>
        <label class="flex items-center gap-1" :title="ic.type === 'cat'
                 ? 'derived: ℏ/(2σ) — a Gaussian wavefunction is a minimal packet' : ''">
          <span class="w-8 text-muted">σ<span v-html="labelHtml(ndim, ndim + d - 1)"></span></span>
          <input v-if="ic.type === 'mixture' && sel.sigma_k"
                 :value="sel.sigma_k[d - 1]" type="number" step="any" min="0.01"
                 class="wf-num"
                 @change="onNum($event, (v) => { sel!.sigma_k![d - 1] = v })" />
          <input v-else :value="derivedSigmaP(sel.sigma_q[d - 1] ?? 0.5).toFixed(4)"
                 disabled class="wf-num opacity-50" />
        </label>
      </template>
      <label class="flex items-center gap-1"
             :title="ic.type === 'mixture'
               ? 'relative weight: ensemble probability wⱼ/Σw in ρ = Σ wⱼ ρⱼ'
               : 'relative weight: |cⱼ|² in ψ ∝ Σ cⱼψⱼ, cⱼ = √wⱼ·exp(iφⱼ)'">
        <span class="w-8 text-muted">w</span>
        <input v-model.number="sel.weight" type="number" step="any" min="0.01"
               class="wf-num" @change="scheduleRefresh()" />
      </label>
      <label class="flex items-center gap-1"
             :title="'phase of cⱼ = √wⱼ·exp(iφⱼ) — sets the interference fringe phase (e.g. even/odd cat); meaningless for a mixture (no coherence)'">
        <span class="w-8 text-muted">φ</span>
        <input v-model.number="sel.phase" type="number" step="any"
               :disabled="ic.type === 'mixture'"
               class="wf-num" :class="{ 'opacity-50': ic.type === 'mixture' }"
               @change="scheduleRefresh()" />
      </label>
    </div>
  </section>
</template>

<!-- NB `.wf-num` used to be defined here in a non-scoped <style>, which made
     it look local to a component that is only one of its three users (the
     Setup and Export panels style every input with it too). It now lives in
     style.css, themed off --wf-input/--wf-line. -->

