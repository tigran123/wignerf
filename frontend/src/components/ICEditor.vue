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
import { defaultIC, icPayload, type GridCfg, type ICCfg,
         type ICComponentCfg } from '../lib/config'
import { IC_KINDS, icTabLabel, icTabTitle, isExprKind,
         type ICExprKind, type ICKind } from '../lib/icKinds'
import { conjugate } from '../lib/axes'
import type { WaveCut } from '../lib/wavefn'
import ICExprInput from './ICExprInput.vue'
import WaveFnPlots from './WaveFnPlots.vue'
import { createViewWindow, panBy, resetView, zoomAt } from '../lib/viewWindow'
import { type AxisLattice } from '../lib/cells'
import { WignerRenderer } from '../render/WignerRenderer'

const props = defineProps<{
  ic: ICCfg
  grid: GridCfg
  hbarEff: number
  /** The form's spectral precision — passed to the preview endpoint only so its
   *  over-the-cap refusal quotes the SESSION footprint the Setup panel quotes.
   *  The preview itself always builds in float64; see the call site. */
  precision: 'float64' | 'float32'
  showGrid?: boolean
  showCells?: boolean
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
  /** axes of THIS form's IC that sit in the edge band. Reported upward rather
   *  than rendered here: the same fact about the same detector belongs in one
   *  sentence in one place — see SimulatorView's wOf. */
  (e: 'edge', found: { axis: string; mass: number }[]): void
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
const overlay = ref<HTMLDivElement | null>(null)
const selected = ref(0)
const deficit = ref('')
const warnings = ref<string[]>([])
/** Axes whose edge band the SAMPLED total W trips, from X-Wignerf-Edge. */
const edgeFound = ref<{ axis: string; mass: number }[]>([])

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

// -- the expression draft -----------------------------------------------------
//
// THE DRAFT LIVES HERE, NOT IN THE INPUT, and it travels in the preview REQUEST.
// PotentialEditor can keep its draft local because /preview/potential takes
// `expr` as a field of its own; this component posts `props.ic`, the COMMITTED
// config. Commit only on the server's verdict while previewing only the
// committed text and the server is never asked about the draft, so nothing ever
// commits — a deadlock, not a race.
//
// Sending the draft has a second payoff: an invalid one makes the preview fail,
// which drops `previewOk` and gates Solve. So the gate describes what payload()
// would actually send, which is the real argument for auto-committing at all —
// stronger than the localStorage one, because an invalid IC is otherwise
// discovered only at Restart, i.e. exactly when it destroys the session.
const kind = computed(() => props.ic.type)
const exprKind = computed<ICExprKind | null>(() =>
  (isExprKind(kind.value) ? kind.value : null))
const draft = ref('')
watch(() => (exprKind.value ? props.ic.expr[exprKind.value] : ''),
      (v) => { draft.value = v }, { immediate: true })
/** the draft differs from what the server last accepted */
const exprPending = computed(() =>
  !!exprKind.value && draft.value !== props.ic.expr[exprKind.value])

// Which spatial axis the ψ/φ cuts are taken along at ndim=2. DISPLAY state like
// planeIdx — deliberately not in cfg, since it changes nothing that is computed.
const cutAxis = ref(0)
// ...and it must come back to 0 when there is no longer a second axis to point
// at. The select is inside v-if="ndim > 1", so at ndim=1 a stale 1 is
// unreachable AND unrecoverable — and it is not inert, because the chart titles
// fall back to it whenever the response is null, which is exactly when a 2D ψ
// stops compiling at ndim=1. That fallback then reads "ψ(p)" and "φ(x)":
// wavefunction labels on a momentum axis and vice versa, the one confusion this
// component's whole axis vocabulary exists to prevent. The backend clamps its
// own copy, so this only ever showed on the failure path.
watch(ndim, (n) => { if (cutAxis.value >= n) cutAxis.value = 0 })
const psiCut = ref<WaveCut | null>(null)
const phiCut = ref<WaveCut | null>(null)
/** why the two charts are empty, when they are */
const waveError = ref('')

/** X-Wignerf-IC-Norm: "method:raw:momentum_mass". */
const icNorm = ref('')
const normScale = computed(() => {
  const [method, raw] = icNorm.value.split(':')
  if (method !== 'grid-sum' || !raw) return ''
  return Number(raw).toPrecision(4)
})
/** dμ, the phase-space measure — spelled out in the tooltip rather than in the
 *  line, because "∫W dx dy dpx dpy" is most of a 320px column at ndim=2 and the
 *  two lines that use it are already the panel's least urgent. */
const measure = computed(() => 'd' + axisLabels(ndim.value).join(' d'))

/**
 * What the deficit is, in words — the number itself is ambiguous three ways
 * (absolute vs relative, signed vs not, against what) and the formula beside it
 * settles only the first two.
 */
const deficitTitle = computed(() => `dμ = ${measure.value}. ` + (kind.value === 'psi'
  ? 'how much of ψ\'s probability lies OUTSIDE the spatial box: the state is '
    + 'normalised over the extended box, so what the grid holds is 1 minus this. '
    + 'It says nothing about the momentum box — see the warning that does.'
  : 'how far the SAMPLED W is from unit norm. The normalisation is analytic and '
    + 'exact, so this is grid truncation: mass in the tails the box does not '
    + 'reach. An absolute difference, not a relative one.'))

const state = reactive({ previewOk: false, error: '' })

/**
 * The ⚠ lines: the preview's own quality warnings. These are STEADY — they
 * change only when the form does — so they need no dwell. The edge finding is
 * deliberately not among them; it goes up to the header, where the same
 * sentence covers the live state too.
 */
const noticeLines = computed(() => warnings.value)
const noticeText = computed(() => noticeLines.value.join('\n'))
/** What was dismissed, as its own text — so a DIFFERENT warning still shows. */
const noticeDismissed = ref('')
const noticeKey = computed(() =>
  `${state.error && !props.quiet ? state.error : ''}\n${noticeText.value}`)
const noticeShown = computed(() =>
  noticeKey.value.trim() !== '' && noticeKey.value !== noticeDismissed.value)
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
  renderer.upload(pl, f.N[a]!, f.N[b]!)
  renderer.render()
}

watch(plane, () => { resetView(view); repaintPlane() })

/** Monotone request id: a newer request is always the authoritative one. */
let seq = 0
/** The wavefunction request's OWN counter. The two calls are independent — one
 *  builds the state, the other samples psi — so a refresh that concerns only the
 *  W preview must not be able to invalidate a wave response in flight. */
let waveSeq = 0
/** The last body actually sent, so a commit does not re-fire the preview. */
let lastBody = ''
/** The wavefunction call's OWN de-dup key, for the same reason it has its own
 *  counter: it depends on `cutAxis`, which the W preview does not, so sharing
 *  one key made the cut selector INERT. Changing the cut left the W body
 *  byte-identical, refresh() returned at the de-dup, and refreshWave — which
 *  sits below it — was never reached. Both charts kept the old cut and both
 *  titles kept naming the old axis, permanently and with no error anywhere. */
let lastWaveBody = ''

async function refresh() {
  const sent = draft.value
  // The wire shape, with the DRAFT substituted for the committed expression —
  // icPayload sends what cfg holds, and cfg is exactly what has not been
  // updated yet.
  const body = {
    ...icPayload(props.ic),
    ...(exprKind.value ? { expr: sent } : {}),
    grid: props.grid,
    hbar_eff: props.hbarEff,
    precision: props.precision,
  }
  // BEFORE the de-dup below, never after: this call has its own inputs and its
  // own key, so whether the W preview needs repeating says nothing about
  // whether this one does.
  void refreshWave(sent)
  // De-dup, because committing below mutates props.ic and so re-fires the deep
  // watch — a second identical request, and at 8192² that is half a second of
  // GPU build for nothing. The body IS the whole editor state, so comparing it
  // is correct by construction.
  const key = JSON.stringify(body)
  if (key === lastBody) return
  lastBody = key
  // THE SEQUENCE IS BUMPED ONLY WHEN A REQUEST IS ACTUALLY SENT, and it must
  // stay that way. It used to be the first line of this function, so the
  // de-dup above returned AFTER incrementing it — and every successful commit
  // re-fires the deep watch, lands here, de-dups out, and silently invalidated
  // the responses still in flight from the request it just declined to repeat.
  // The wavefunction traces were the visible casualty: typing an expression
  // left the two charts permanently empty, titled with their null fallbacks,
  // and no later edit could refill them because the body never changed again.
  const mine = ++seq
  try {
    // `precision` rides in `body` above. It is NOT what the preview computes in
    // (that is always float64 on its own backend) — it is what the SESSION
    // would use, and the endpoint needs it so its over-the-cap refusal quotes
    // the same footprint the Setup panel's footprint line does. It is in the
    // body unconditionally rather than only when chosen: this is the form's
    // displayed value, and the two lines sit on screen together, so they must
    // agree even while the form is still deferring.
    const { data, headers } = await api.post('/preview/wigner', body,
                                             { responseType: 'arraybuffer' })
    if (mine !== seq) return          // a newer request is authoritative
    const f = decodeFrame(data as ArrayBuffer)
    const v = f.variants[0]!
    const [a, b] = plane.value
    const pl = v.planes.find((q) => q.a === a && q.b === b) ?? v.planes[0]!
    // the preview's OWN colour range, for its overlaid bar: it is a W plot like
    // the panels and autoscales the same way, so it carries its own scale
    wmin.value = pl.wmin
    wmax.value = pl.wmax
    lastFrame = f
    renderer.upload(pl, f.N[pl.a]!, f.N[pl.b]!)
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
    emit('edge', edgeFound.value)
    icNorm.value = String(headers['x-wignerf-ic-norm'] ?? '')
    state.previewOk = true
    state.error = ''
    commit(sent)
  } catch (e: unknown) {
    if (mine !== seq) return
    state.previewOk = false
    // apiErrorText, like every other failed call in the app — it unwraps the
    // arraybuffer body this request type produces (see lib/apierror.ts)
    state.error = apiErrorText(e)
    // NOTHING FROM THE PREVIOUS IC MAY OUTLIVE ITS OWN STATE. These four all
    // describe a W that no longer exists, and leaving them left the panel
    // asserting facts about it beside the error saying it could not be built:
    // the compile error stacked UNDER a stale ⚠ quality warning, a stale
    // "normalised: ∫W dμ was 3.142" readout below the plot, and — because the
    // same edit sets restartNeeded — the HEADER stating an edge finding about
    // an initial condition that does not compile.
    warnings.value = []
    deficit.value = ''
    icNorm.value = ''
    edgeFound.value = []
    emit('edge', [])
    // The body was recorded before the request, so without this a transient
    // failure (a backend restart, a 503, a dropped connection) is permanent:
    // the identical body de-dups out, ↺ defaults on an already-default IC
    // changes nothing, and Solve stays gated behind the stale error with no way
    // back except editing a field to a different value and editing it back.
    lastBody = ''
  }
}

/**
 * Promote a draft the server accepted into `cfg`.
 *
 * Gated on the response still describing the CURRENT text, exactly as
 * PotentialEditor's is, and for a sharper reason: `cfg` is persisted to
 * localStorage AND is what POST /sessions sends, and unlike U(x) the IC has no
 * live path — an invalid one is discovered at Restart, which is precisely when
 * it destroys the session (useSession.create deletes the old one before it
 * posts). So a half-typed `exp(-x^2/` must never reach it.
 */
function commit(sent: string) {
  const k = exprKind.value
  if (!k || sent !== draft.value || sent === props.ic.expr[k]) return
  props.ic.expr[k] = sent
  // The assignment fires the deep watch with notify=false, so the restart-dirty
  // flag has to be raised explicitly here. Typing a change and typing it back
  // raises nothing, because of the equality check above.
  emit('changed')
}

/** The two line charts. Separate from the W preview on purpose: this one is
 *  cheap and JSON, that one is a binary frame that can cost a multi-GiB
 *  transient, and moving the cut selector must not pay for one. */
async function refreshWave(sent: string) {
  if (kind.value !== 'psi') {
    // The counter moves here too. Clearing is a STATE CHANGE like any other, so
    // a response already in flight must not be allowed to land after it —
    // flicking mixture → psi inside the debounce would otherwise let the old
    // request repopulate charts this call had just emptied.
    waveSeq++
    lastWaveBody = ''
    psiCut.value = phiCut.value = null
    waveError.value = ''
    return
  }
  const body = {
    expr: sent, grid: props.grid, hbar_eff: props.hbarEff,
    cut_axis: cutAxis.value,
  }
  const key = JSON.stringify(body)
  if (key === lastWaveBody) return
  lastWaveBody = key
  // Bumped only now that a request is certain to go out — the rule the W
  // preview above states at length, and it applies to every counter here.
  const mine = ++waveSeq
  try {
    const { data } = await api.post('/preview/wavefunction', body)
    if (mine !== waveSeq) return
    // Its failure gates nothing — the W response owns the Solve gate — but it
    // must SAY something. Blanking two charts with no explanation is how this
    // reads as "the plots are broken" rather than "that expression did not
    // compile". The two calls can diverge: they validate ψ on different boxes
    // (this one on the extended spatial box alone) and they are independent
    // requests, so either can fail transports the other survives.
    psiCut.value = data.ok ? data.psi : null
    phiCut.value = data.ok ? data.phi : null
    waveError.value = data.ok ? '' : String(data.error ?? 'could not sample ψ')
    if (!data.ok) lastWaveBody = ''
  } catch (e: unknown) {
    if (mine !== waveSeq) return
    lastWaveBody = ''      // a transient failure must stay retryable
    psiCut.value = phiCut.value = null
    waveError.value = apiErrorText(e)
  }
}

/** `delay` is a parameter because typing is continuous where dragging is not:
 *  the expression box uses PotentialEditor's 300 ms, everything else the 150 ms
 *  a drag has always had. */
function scheduleRefresh(notify = true, delay = 150) {
  if (notify) emit('changed')
  if (timer) clearTimeout(timer)
  timer = setTimeout(refresh, delay)
}

watch([draft, cutAxis], () => scheduleRefresh(false, 300))

// precision is in the key even though it changes NOTHING the preview computes:
// it changes what the endpoint's over-the-cap refusal quotes, so without it a
// float64 → float32 switch left the old figure standing under the plot —
// contradicting the Setup panel, which updates immediately, by a factor of 1.9.
watch(() => [props.ic, props.grid, props.hbarEff, props.precision],
  () => scheduleRefresh(false), { deep: true })

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
  // NEVER on an expression tab. The components are deliberately kept alive
  // there (so switching back is non-destructive), but their markers are not
  // drawn — so without this a click anywhere near a hidden component's
  // coordinates would move it, mutate cfg and mark the session restart-dirty
  // with no visible effect whatsoever. That is the direct cost of keeping them.
  if (isExprKind(kind.value)) return -1
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
  if (!confirm('Reset the initial condition — type, Gaussian components and '
               + 'both expressions — to defaults?')) return
  // the default FOR THIS DIMENSIONALITY — a 2D editor must not be handed a
  // one-coordinate component the grid would then reject
  const d = defaultIC(ndim.value)
  const before = JSON.stringify(props.ic)
  props.ic.type = d.type
  props.ic.components.splice(0, props.ic.components.length, ...d.components)
  props.ic.expr.wexpr = d.expr.wexpr
  props.ic.expr.psi = d.expr.psi
  selected.value = 0
  // repaint either way, but only a real change marks the session restart-dirty
  // (see resetSetup: an already-default IC has not "changed")
  scheduleRefresh(JSON.stringify(props.ic) !== before)
}

function setType(t: ICKind) {
  props.ic.type = t
  // The σ_k fix-up is a MIXTURE/CAT concern; running it for an expression
  // target would make mixture → W(x,p) → mixture stop being a round trip.
  if (!isExprKind(t)) {
    for (const c of props.ic.components) {
      if (t === 'cat') c.sigma_k = null
      else if (c.sigma_k == null) c.sigma_k = c.sigma_q.map(derivedSigmaP)
    }
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
        v-for="t in IC_KINDS" :key="t"
        class="flex-1 py-1 rounded border"
        :class="ic.type === t
          ? 'bg-info-soft border-info text-info-fg'
          : 'bg-panel border-line text-fg-3 hover:bg-raised'"
        :title="icTabTitle(t, ndim)"
        @click="setType(t)"
      >{{ icTabLabel(t, ndim) }}</button>
    </div>

    <!-- the expression tabs' one control -->
    <ICExprInput v-if="exprKind" v-model="draft" :kind="exprKind" :ndim="ndim"
                 :pending="exprPending" />

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
      <!-- Which spatial axis the ψ/φ cuts run along. The MOMENTUM axis is not
           a second choice: it follows from axes.conjugate, so the pair can
           never be mismatched. -->
      <template v-if="ic.type === 'psi'">
        <span class="text-muted">cut</span>
        <select v-model.number="cutAxis" class="wf-num flex-1"
                title="which axis the ψ and φ cuts are taken along; the momentum axis is its conjugate">
          <option v-for="d in ndim" :key="d" :value="d - 1">
            {{ axisLabel(ndim, d - 1) }},{{ axisLabel(ndim, conjugate(ndim, d - 1)) }}
          </option>
        </select>
      </template>
    </div>

    <!-- phase-space preview with draggable peaks.
         TWO nested boxes on purpose: the OUTER one is the positioning anchor for
         the notice strip below, the INNER one clips the canvas. They cannot be
         one box — `overflow-hidden` is what keeps the heatmap inside its rounded
         border, and it also clips any absolutely-positioned child that hangs
         BELOW the box, which is exactly where a `top-full` strip lives. Merging
         them silently deletes the warnings from the screen while leaving them in
         the DOM, so nothing but a screenshot can see it. -->
    <div class="relative w-full">
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
        <div v-for="(c, i) in (exprKind ? [] : ic.components)" :key="i"
             class="absolute w-3 h-3 -ml-1.5 -mt-1.5 rounded-full border-2 pointer-events-none"
             :class="i === selected ? 'border-yellow-300' : 'border-neutral-400/70'"
             :style="markerStyle(c)"></div>
      </div>
      <!-- the preview's own colour scale, overlaid so it costs no height in a
           column the W panels are competing with. No name label here, so it
           takes the top slot the panels reserve for theirs. -->
      <Colorbar :min="wmin" :max="wmax" place="top-1 left-2" />
    </div>

      <!--
        TRANSIENT NOTICES ARE AN OVERLAY, NEVER FLOW CONTENT — the same rule the
        header strip follows, and for the same measured reason. The edge finding
        that used to live here was differenced against the RUNNING session's
        tripped axes, so a packet sloshing through the edge band made it appear
        and vanish every record or two; in flow that moved everything below it by
        38 px, on and off, for the whole computation (measured: ten controls
        taking two positions each, 1184/1222 …). The panel is a thing you WATCH,
        so a message arriving must not relayout it.

        Anchored to the preview box's bottom edge, so it overlays the top of
        whatever follows — chrome, never data — exactly the trade the header
        strip makes.

        NO DWELL HERE, and that is not an omission. What this strip carries now
        is the compile error and the preview's quality warnings, and both are
        STEADY — they change only when the form does, which is the user's own
        doing. `sticky` is for a notice driven by the RUNNING session, which is
        the header's job; see noticeLines.
      -->
      <div v-if="noticeShown"
           class="absolute left-0 right-0 top-full z-20 flex gap-1 px-1 py-0.5
                  bg-panel/95 border-b border-line-soft shadow-sm">
        <div class="flex-1 space-y-0.5">
          <div v-if="state.error && !quiet" class="text-[11px] text-error">
            {{ state.error }}
          </div>
          <div v-for="(w, i) in noticeLines" :key="i" class="text-[11px] text-warn">
            ⚠ {{ w }}
          </div>
        </div>
        <!-- Dismissible, like the header strip: this hangs OVER the panel below
             it, so a standing warning you have already read must be closable
             rather than permanently obscuring the ψ/φ legend. Keyed on the TEXT,
             so dismissing one warning never hides the next, different one. -->
        <button class="self-start text-fg-3 hover:text-fg-1 px-1 leading-none"
                title="dismiss (returns if the warning changes)"
                @click="noticeDismissed = noticeKey">×</button>
      </div>
    </div>
    <!-- ψ and its Fourier image, on the ψ tab only -->
    <WaveFnPlots v-if="ic.type === 'psi'" :psi="psiCut" :phi="phiCut"
                 :ndim="ndim" :cut-axis="cutAxis" :error="waveError"
                 :show-grid="showGrid ?? true" />

    <!-- The formula IS the label, so the number cannot be read as a relative
         error or as a signed one — both of which it was mistaken for. Steady
         readouts of the CURRENT preview, so they belong in flow: they change
         only when the form does, which is the user's own doing. -->
    <div v-if="deficit" class="text-xs text-fg-3" :title="deficitTitle">
      |∫W dμ − 1| = {{ deficit }}
    </div>
    <!-- What replaces the deficit line for an auto-normalised W: the integral
         it was divided BY. Without it, "why does multiplying my expression by 5
         change nothing?" is a question the panel invites and never answers. -->
    <div v-if="normScale" class="text-xs text-fg-3">
      normalised: ∫W dμ was {{ normScale }}
    </div>

    <!-- component list — Gaussian kinds only; the components stay in cfg on the
         expression tabs (so switching back is non-destructive) but editing them
         there would be editing something nothing draws -->
    <div class="flex items-center gap-1 text-xs">
      <template v-if="!exprKind">
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
      </template>
      <button class="ml-auto px-2 py-0.5 rounded bg-raised hover:bg-raised-hover"
              title="reset the initial condition — kind, Gaussian components and both expression drafts — to its defaults"
              @click="resetIC">↺ defaults</button>
    </div>

    <!-- step="any": with a discrete step= the browser rejects perfectly
         good values (0.60, drag-placed coordinates, even 1.0) -->
    <!-- One row per SPATIAL dimension: centre and width for that dimension's
         coordinate and its conjugate momentum. 1D renders the single x/p row
         it always did; 2D adds y/py below it. -->
    <div v-if="sel && !exprKind" class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
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

