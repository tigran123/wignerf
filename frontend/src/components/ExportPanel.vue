<script setup lang="ts">
/**
 * Export: an mp4 of an already-computed record range, and the run's setup
 * (its initial conditions) as a document that can be imported back.
 *
 * VIDEO — the backend renders the frames off its own history
 * (routers/export.py); the browser only configures the job, watches its
 * progress and downloads the file. Export is PAUSED-only: a running session
 * evicts old records once the history cap is reached, and the whole point of
 * the feature is to film a range you have already played back and judged
 * interesting. Progress arrives as 'export' events on the session WebSocket;
 * a 1 s REST poll runs alongside so a missed event can never leave the bar
 * stuck.
 *
 * SETUP — GET /sessions/{id}/setup serves what the run STARTED from (the
 * server rewinds its own live changes; see core/describe.setup_document).
 * Import fills the setup form and marks the session restart-dirty — never
 * restarts by itself — and reads either that .json or an exported .mp4,
 * which carries the same document in its metadata (lib/mp4meta.ts).
 */
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { api } from '../api'
import type { ExportEvent, SessionStatus } from '../composables/useSession'
import { apiErrorText } from '../lib/apierror'
import { planeLabel, planes as planesOf, planeTitle } from '../lib/axes'
import { importConfig, type SimConfig } from '../lib/config'
import { defaultDiagnostics, diagnosticLabel, diagnosticTitle,
         diagnosticsAvailable, panelGridOf, panelPixels, PANEL_PX_MIN,
         type DiagId } from '../lib/exportLayout'
import { extractWignerfDoc } from '../lib/mp4meta'
import { panelMode, panelPlaneIdx, panelVariantIdx } from '../lib/panelView'
import { fmtTime } from '../lib/units'
import { theme, type ThemeName } from '../lib/theme'
import { VARIANT_META, variantColor, type VariantKey } from '../lib/variants'

const props = defineProps<{
  status: SessionStatus | null
  sessionId: string | null
  event: ExportEvent | null
  variants: VariantKey[]
  currentRecord: number
  /** the SPA's "grid lines on plots" setting — the video follows it, on
   *  the charts AND on the W heatmaps */
  showGrid: boolean
  /** the setup form — an import writes into it (same in-place mutation the
   *  SetupPanel's own fields do) */
  cfg: SimConfig
}>()

const emit = defineEmits<{
  (e: 'command', cmd: Record<string, unknown>): void
  (e: 'dirty'): void
}>()

const RESOLUTIONS = [
  { label: '1920 × 1080 (FHD)', width: 1920, height: 1080 },
  { label: '2560 × 1440 (QHD)', width: 2560, height: 1440 },
  { label: '3840 × 2160 (4K UHD)', width: 3840, height: 2160 },
]

interface ExportForm {
  k0: number
  k1: number
  stride: number
  fps: number
  // stored as the pixel width, NOT as an index into RESOLUTIONS: the list
  // changes over time and a stale index would silently pick another size
  width: number
  variants: VariantKey[]
  // WHAT THE FRAME SHOWS. `planes` are indices into axes.planes(ndim) — the
  // same index the panel grid stores — and the panels are the cartesian
  // product of these and `variants`, so the SPA's two readings are the two
  // edges of one control rather than modes of their own. `diagnostics` are
  // plot ids shared with lib/plotPrefs.ts; an EMPTY list is legal and gives a
  // panels-only video.
  planes: number[]
  diagnostics: DiagId[]
  // '' = the host's WIGNERF_EXPORT_ENCODER. Per JOB, not per session: the
  // right choice depends on what else is running, since nvenc's real benefit
  // is freeing CPU cores for the parallel frame RENDER (the bottleneck).
  encoder: '' | 'auto' | 'cpu' | 'nvenc'
}

const ENCODERS = [
  { value: '', label: 'default (host)' },
  { value: 'nvenc', label: 'GPU (h264_nvenc)' },
  { value: 'cpu', label: 'CPU (libx264)' },
] as const

// The frame's light/dark theme. Deliberately NOT persisted alongside the
// encoding prefs: it FOLLOWS the app theme (re-seeded whenever the panel
// opens, below), and a remembered value would silently stop tracking it.
// The override is per job because filming a dark video out of a light
// session is a legitimate thing to want.
const jobTheme = ref<ThemeName>(theme.value)

const STORAGE_KEY = 'wignerf.export'

/**
 * The diagnostics choice is remembered PER NDIM, not as one list. The two are
 * genuinely different sets — 1D has five plots, 2D has nine, and only three
 * ids appear in both — so carrying one over gives the worst of each: a 1D
 * choice reused at 2D silently drops ⟨Lz⟩ and ΔY·ΔPy, the two plots a 2D user
 * is most likely to be here for.
 */
const diagByNdim: Record<string, DiagId[]> = {}

// Declared HERE and not beside is2d below, because the persistence watcher a
// few lines down reads it and Vue evaluates a watch source IMMEDIATELY to seed
// its old value — a later `const` is still in its temporal dead zone at that
// moment, which threw "Cannot access 'ndim' before initialization" on every
// page load and left the panel's watcher dead.
const ndim = computed(() => props.status?.ndim ?? 1)

function loadForm(): ExportForm {
  const d: ExportForm = { k0: 0, k1: 0, stride: 1, fps: 30, width: 1920,
                          variants: [...props.variants], planes: [0],
                          diagnostics: defaultDiagnostics(1), encoder: '' }
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (s && typeof s === 'object') {
      for (const k of ['stride', 'fps'] as const)
        if (typeof s[k] === 'number') d[k] = s[k]
      if (RESOLUTIONS.some((r) => r.width === s.width)) d.width = s.width
      if (ENCODERS.some((e) => e.value === s.encoder)) d.encoder = s.encoder
      if (s.diagnostics && typeof s.diagnostics === 'object')
        for (const [nd, ids] of Object.entries(s.diagnostics))
          if (Array.isArray(ids)) diagByNdim[nd] = ids as DiagId[]
    }
  } catch { /* corrupted storage -> defaults */ }
  return d
}

const form = reactive(loadForm())
const resolution = computed(() =>
  RESOLUTIONS.find((r) => r.width === form.width) ?? RESOLUTIONS[0])
// the RANGE is never persisted (it belongs to this session's history), and
// neither is the PLANE choice — that is seeded from the panel grid every time
// this opens (below), and a remembered value would fight the seeding exactly
// as a remembered theme would stop tracking the app's
function saveForm() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(
    { stride: form.stride, fps: form.fps, width: form.width,
      encoder: form.encoder, diagnostics: diagByNdim }))
}

watch(() => [form.stride, form.fps, form.width, form.encoder], saveForm)

// The diagnostics set is recorded by the TOGGLE, never by a watcher on the
// form. A watcher cannot tell a choice from a re-seed: `status.ndim` arrives
// after the first render, so the form still holds the 1D default at the moment
// it flips to 2, and a watcher keyed on (ndim, diagnostics) faithfully stored
// that 1D set as the user's 2D preference — which then won over
// defaultDiagnostics(2) the first time the panel was opened, so a fresh 2D
// export offered rho(x)/rho(y) and no <Lz>. Measured in a browser before this
// was written this way.
function rememberDiagnostics() {
  diagByNdim[String(ndim.value)] = [...form.diagnostics]
  saveForm()
}

const open = ref(false)
const busy = ref(false)
const error = ref('')
const job = ref<ExportEvent | null>(null)
let poll = 0

/**
 * The retained extent, from the SERVER on demand. The streamed status is
 * pushed at ~1 Hz and lags a burst of frames, so right after a pause it can
 * still report a frontier hundreds of records old — seeding the range from
 * it would silently export less than the user computed. `GET /sessions/{id}`
 * reads history.extent() directly; the status echo only ever widens it.
 */
const liveExtent = ref<[number, number] | null>(null)
const extent = computed(() =>
  liveExtent.value ?? props.status?.record_extent ?? [-1, -1])

async function refreshExtent() {
  if (!props.sessionId) return
  try {
    const { data } = await api.get<SessionStatus>(`/sessions/${props.sessionId}`)
    liveExtent.value = data.record_extent
  } catch { /* session gone — the status fallback still applies */ }
}

watch(() => props.status?.record_extent, (e) => {
  if (e && liveExtent.value && e[1] > liveExtent.value[1]) liveExtent.value = e
})
watch(() => props.sessionId, () => {
  liveExtent.value = null
  // a restart closes the old session, and closing it deletes its exports:
  // a "ready" badge pointing at a deleted file would be a lie
  job.value = null
  clearInterval(poll)
})
const hasHistory = computed(() => extent.value[1] >= 0)
const running = computed(() => props.status?.running ?? false)
// The button stays live while the session computes: rendering itself needs
// a paused session (the backend 409s, and a running session evicts old
// records), but a DISABLED button whose only explanation is a tooltip is
// how this feature first looked broken. Opening the panel explains the
// gate, and Render pauses the run first.
const is2d = computed(() => ndim.value > 1)     // ndim is declared above
const disabled = computed(() => !hasHistory.value || !props.sessionId)
const disabledWhy = computed(() =>
  !hasHistory.value ? 'nothing computed yet'
    : 'render the computed range to an mp4 video, or save/load this '
      + "simulation's setup")

// -- what the frame shows ---------------------------------------------------

/**
 * A 2D record carries six planes per variant (up to 24 panels) and nine
 * diagnostics against 1D's five, which is more than a frame can hold — the SPA
 * answers that by scrolling its column and offering two panel readings,
 * neither of which a video frame can do. So the job selects.
 *
 * Panels are the CARTESIAN PRODUCT of the chosen planes and variants, which
 * makes PanelGrid's two readings the two edges of one control: the presets
 * below just set the checkboxes.
 */
const allPlanes = computed(() => planesOf(ndim.value))
const allDiagnostics = computed(() => diagnosticsAvailable(ndim.value))

function togglePlane(i: number) {
  const at = form.planes.indexOf(i)
  if (at >= 0) {
    if (form.planes.length === 1) return       // the API needs >= 1
    form.planes.splice(at, 1)
  } else {
    form.planes.push(i)
    form.planes.sort((a, b) => a - b)
  }
}

function toggleDiagnostic(id: DiagId) {
  const at = form.diagnostics.indexOf(id)
  // an empty set is legal — a panels-only video is a real thing to want
  if (at >= 0) form.diagnostics.splice(at, 1)
  else {
    form.diagnostics.push(id)
    const order = allDiagnostics.value
    form.diagnostics.sort((a, b) => order.indexOf(a) - order.indexOf(b))
  }
  rememberDiagnostics()
}

/** The two readings PanelGrid offers, as one-click presets. */
function presetCompareVariants() {
  form.planes = [Math.min(panelPlaneIdx.value, allPlanes.value.length - 1)]
  form.variants = [...props.variants]
}

function presetPhasePortrait() {
  form.planes = allPlanes.value.map((_p, i) => i)
  const v = props.variants[Math.min(panelVariantIdx.value,
                                    props.variants.length - 1)]
  if (v) form.variants = [v]
}

const grid = computed(() =>
  panelGridOf(form.planes.length, form.variants.length))
const panelPx = computed(() =>
  panelPixels(grid.value.rows, grid.value.cols, form.diagnostics.length,
              resolution.value.width, resolution.value.height))
/** Tiny panels are the one way this dialog can produce a useless video, so it
 *  says so BEFORE the render rather than after — the same job the Setup
 *  panel's 2D footprint line does. */
const panelsCramped = computed(() =>
  Math.min(panelPx.value.w, panelPx.value.h) < PANEL_PX_MIN)

/** t of a record index, from the timeline's own linear mapping. */
function tOf(k: number): string {
  const st = props.status
  const t0 = st?.t_extent?.[0]
  if (!st || t0 == null) return ''
  return fmtTime(t0 + (k - extent.value[0])*st.record_dt*(st.sign || 1))
}

const frames = computed(() => {
  const n = Math.floor((form.k1 - form.k0)/Math.max(1, form.stride)) + 1
  return Math.max(0, n)
})
const duration = computed(() => frames.value/Math.max(1, form.fps))

// A range the user has not touched follows the history: pausing on their
// behalf (Render while computing) lands more records, and the export must
// cover them instead of stopping where the panel happened to open.
const rangeTouched = ref(false)

function useAll() {
  form.k0 = extent.value[0]
  form.k1 = extent.value[1]
  rangeTouched.value = false
}

function useFromHere() {
  form.k0 = Math.min(Math.max(props.currentRecord, extent.value[0]),
                     extent.value[1])
  form.k1 = extent.value[1]
  rangeTouched.value = true
}

// opening the panel seeds the range from the live extent (the previous
// session's indices mean nothing here) and drops a finished job's state
watch(open, async (v) => {
  if (!v) return
  await refreshExtent()
  useAll()
  jobTheme.value = theme.value   // follow the app unless overridden below
  form.variants = form.variants.filter((k) => props.variants.includes(k))
  if (!form.variants.length) form.variants = [...props.variants]
  // ...and seed the CONTENT from the panel grid, so Render films what is on
  // screen. Same path the theme takes, and re-seeded on every open for the
  // same reason: a remembered value would silently stop tracking.
  if (panelMode.value === 'phase' && is2d.value) presetPhasePortrait()
  else presetCompareVariants()
  // remembered per ndim; an empty list is a legal CHOICE (a panels-only
  // video), so only an absent entry falls back to the default
  const stored = diagByNdim[String(ndim.value)]
  form.diagnostics = stored
    ? stored.filter((d) => allDiagnostics.value.includes(d))
    : defaultDiagnostics(ndim.value)
  // a FINISHED job is kept: rendering continues while the popover is
  // closed, and reopening it is how you collect the file (clearing here
  // threw away the only link to a file that still exists on the server)
  error.value = ''
})

// live progress from the session socket (same job only)
watch(() => props.event, (e) => {
  if (e && (!job.value || e.job_id === job.value.job_id)) job.value = e
})

function toggleVariant(v: VariantKey) {
  const i = form.variants.indexOf(v)
  if (i >= 0) {
    if (form.variants.length === 1) return    // the API needs >= 1
    form.variants.splice(i, 1)
  } else {
    form.variants.push(v)
    form.variants.sort((a, b) => props.variants.indexOf(a) - props.variants.indexOf(b))
  }
}

/** Pause the run and wait for the server to confirm it (in-flight records
 *  land after the command, so the frontier keeps moving for a moment). */
async function pauseAndSettle(): Promise<boolean> {
  emit('command', { type: 'pause' })
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 250))
    try {
      const { data } = await api.get<SessionStatus>(`/sessions/${props.sessionId}`)
      if (!data.running) {
        liveExtent.value = data.record_extent
        return true
      }
    } catch { return false }
  }
  return false
}

async function start() {
  if (!props.sessionId || busy.value) return
  busy.value = true
  error.value = ''
  job.value = null            // a new render supersedes the previous file
  const r = resolution.value
  try {
    if (running.value) {
      if (!(await pauseAndSettle())) {
        error.value = 'could not pause the session — try the transport button'
        return
      }
      if (!rangeTouched.value) useAll()   // cover what the pause just added
    }
    const { data } = await api.post<ExportEvent>(
      `/sessions/${props.sessionId}/export`,
      { k0: form.k0, k1: form.k1, stride: form.stride, fps: form.fps,
        width: r.width, height: r.height, variants: form.variants,
        planes: form.planes, diagnostics: form.diagnostics,
        show_grid: props.showGrid, theme: jobTheme.value,
        ...(form.encoder ? { encoder: form.encoder } : {}) })
    job.value = data
    startPoll(data.job_id)
  } catch (e: unknown) {
    error.value = apiErrorText(e)
  } finally {
    busy.value = false
  }
}

function startPoll(jid: string) {
  clearInterval(poll)
  poll = window.setInterval(async () => {
    try {
      const { data } = await api.get<ExportEvent>(`/exports/${jid}`)
      job.value = data
      if (data.state !== 'running' && data.state !== 'queued') clearInterval(poll)
    } catch {
      clearInterval(poll)       // job dropped (session closed / TTL)
    }
  }, 1000)
}

async function cancel() {
  const j = job.value
  if (!j) return
  clearInterval(poll)
  try { await api.delete(`/exports/${j.job_id}`) } catch { /* already gone */ }
  job.value = null
}

/** The file lives on the server until its TTL; hand the browser a plain
 *  download link built on the API base (so an nginx prefix is honoured). */
const downloadUrl = computed(() =>
  job.value && job.value.state === 'done'
    ? `${api.defaults.baseURL}/exports/${job.value.job_id}/file`
    : '')

const pct = computed(() => {
  const j = job.value
  return j && j.total ? Math.round((100*j.done)/j.total) : 0
})

/**
 * How fast the SERVER is rendering frames, for the progress line.
 *
 * Two decimals under 10 fps because that is where this feature lives: a 2D
 * 24-panel matrix renders at ~6 fps and a large 1D frame slower still, so
 * "6.4" and "0.72" are the numbers people read, and rounding those to integers
 * would print "6" and "1" for a 40% difference. Blank until the server has a
 * sample rather than showing a confident 0.
 */
const renderRate = computed(() => {
  const r = job.value?.render_fps
  if (!r) return ''
  return `${r >= 10 ? r.toFixed(0) : r.toFixed(2)} fps render`
})

/**
 * The header button IS the notification: rendering continues while the
 * popover is closed (the poll and the WS 'export' events keep running), so
 * the button carries the progress and then the finished state — reopening
 * the panel is how you collect the file.
 */
const buttonLabel = computed(() => {
  const j = job.value
  if (!j || open.value) return '⤓ export'
  if (j.state === 'running' || j.state === 'queued')
    return `⤓ export ${pct.value}%`
  if (j.state === 'done') return '⤓ export ready'
  if (j.state === 'error') return '⤓ export failed'
  return '⤓ export'
})
const buttonClass = computed(() => {
  const s = open.value ? null : job.value?.state
  if (s === 'done') return 'bg-emerald-700 hover:bg-emerald-600 text-white'
  if (s === 'error') return 'bg-red-800 hover:bg-red-700 text-white'
  if (s === 'running' || s === 'queued')
    return 'bg-sky-800 hover:bg-sky-700 text-white'
  return 'bg-raised hover:bg-raised-hover'
})
const buttonTitle = computed(() => {
  const j = job.value
  if (!open.value && j?.state === 'done')
    return `${j.filename} is ready (${(j.bytes/2**20).toFixed(1)} MiB) — `
      + 'click to download; the server keeps it for 30 minutes'
  if (!open.value && j?.state === 'error')
    return `export failed: ${j.error ?? 'unknown error'}`
  if (!open.value && (j?.state === 'running' || j?.state === 'queued'))
    return `rendering ${j.done}/${j.total} frames — click to watch or cancel`
  return disabledWhy.value
})

// -- setup (initial conditions) ---------------------------------------------

/** The server serves the ORIGINAL config of this session, not the form:
 *  live changes and auto-expand move the form/session apart, and what you
 *  want to keep is the state the run started from. */
const setupUrl = computed(() =>
  props.sessionId ? `${api.defaults.baseURL}/sessions/${props.sessionId}/setup` : '')

const fileInput = ref<HTMLInputElement | null>(null)
const importMsg = ref('')
const importOk = ref(false)
// The message tells you to restart, and a new session id IS that restart —
// so it must stop being on screen the moment the advice has been taken. It
// used to clear only at the top of the NEXT import, which left "press
// Restart session to run it" standing over a session already running the
// imported setup. Deliberately NOT cleared when the panel closes/reopens:
// an import you have not acted on yet is still actionable.
watch(() => props.sessionId, () => { importMsg.value = '' })

async function onImportFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''                 // re-importing the same file must re-fire
  if (!file) return
  importMsg.value = ''
  try {
    const mp4 = /\.mp4$/i.test(file.name) || file.type === 'video/mp4'
    let doc: unknown
    if (mp4) {
      doc = await extractWignerfDoc(file)
      if (doc == null)
        throw new Error('this mp4 carries no wignerf metadata (not exported '
                        + 'from here, or re-encoded since)')
    } else {
      const text = await file.text()
      try {
        doc = JSON.parse(text)
      } catch {
        // the raw SyntaxError ("Unexpected token '#'…") explains nothing
        // about what this dialog wanted
        throw new Error('not a wignerf setup file — expected the .json this '
                        + 'panel downloads, or an mp4 exported here')
      }
    }
    importConfig(props.cfg, doc)
    emit('dirty')
    importOk.value = true
    // "Restart session", NOT "or Solve": an import moves grid/IC/variants,
    // which are SessionCreate-only, so Solve would compute the old ones
    // behind the header's amber "setup changed" (the auto-restart in
    // SimulatorView.syncFreshSessionToForm only fires when the document
    // also moves mode/t₂/Δt rec/precision on a fresh idle session).
    importMsg.value = `imported ${file.name} — press "Restart session" to run it`
  } catch (e: unknown) {
    importOk.value = false
    importMsg.value = e instanceof Error ? e.message : String(e)
  }
}

onBeforeUnmount(() => clearInterval(poll))
</script>

<template>
  <div class="relative">
    <button class="px-2 py-0.5 rounded text-xs tabular-nums
                   disabled:opacity-40 disabled:cursor-not-allowed"
            :class="buttonClass" :disabled="disabled" :title="buttonTitle"
            @click="open = !open;
                    (($event as MouseEvent).currentTarget as HTMLElement)?.blur()">
      {{ buttonLabel }}
    </button>
    <div v-if="open" class="fixed inset-0 z-40" @click="open = false"></div>
    <div v-if="open"
         class="absolute left-0 top-full mt-2 z-50 w-[26rem] rounded border border-line
                bg-panel shadow-xl p-3 text-xs space-y-2">
      <h4 class="font-semibold text-fg-2">Export</h4>

      <h5 class="text-muted uppercase tracking-wider text-[11px] pt-1">
        Video (mp4) — the computed range
      </h5>

      <div class="grid grid-cols-[3.5rem_1fr] gap-x-2 gap-y-1.5 items-center">
        <span class="text-muted">records</span>
        <div class="flex items-center gap-1">
          <input v-model.number="form.k0" type="number" class="wf-num w-20"
                 :min="extent[0]" :max="extent[1]"
                 @input="rangeTouched = true" />
          <span class="text-dim">…</span>
          <input v-model.number="form.k1" type="number" class="wf-num w-20"
                 :min="extent[0]" :max="extent[1]"
                 @input="rangeTouched = true" />
          <button class="px-1.5 py-0.5 rounded bg-raised hover:bg-raised-hover whitespace-nowrap"
                  title="the whole retained history"
                  @click="refreshExtent().then(useAll)">all</button>
          <button class="px-1.5 py-0.5 rounded bg-raised hover:bg-raised-hover whitespace-nowrap"
                  title="from the current time position to the frontier"
                  @click="refreshExtent().then(useFromHere)">here →</button>
        </div>

        <span class="text-muted">t</span>
        <div class="text-fg-3 tabular-nums truncate"
             :title="'physical time of the first and last exported record'">
          {{ tOf(form.k0) }} → {{ tOf(form.k1) }}
        </div>

        <span class="text-muted">every</span>
        <div class="flex items-center gap-1">
          <input v-model.number="form.stride" type="number" min="1" max="1000"
                 class="wf-num w-16" />
          <span class="text-fg-3">record(s)</span>
          <span class="ml-2 text-muted">fps</span>
          <input v-model.number="form.fps" type="number" min="1" max="120"
                 class="wf-num w-16" />
        </div>

        <span class="text-muted">size</span>
        <select v-model.number="form.width" class="wf-num w-40"
                title="4K renders ~4× more pixels per frame — expect the
                       export to take about four times as long as FHD">
          <option v-for="r in RESOLUTIONS" :key="r.label" :value="r.width">
            {{ r.label }}
          </option>
        </select>

        <span class="text-muted">encoder</span>
        <select v-model="form.encoder" class="wf-num w-40"
                title="the bottleneck is rendering the frames, not encoding them,
                       so this mostly decides how many CPU cores are left for the
                       render pool. Default probes for a working h264_nvenc and
                       falls back to libx264.">
          <option v-for="e in ENCODERS" :key="e.value" :value="e.value">
            {{ e.label }}
          </option>
        </select>

        <span class="text-muted">theme</span>
        <select v-model="jobTheme" class="wf-num w-40"
                title="the frame's colour scheme. Defaults to the app's own
                       theme every time this panel opens, so the video reads
                       like the screen; override it to film a dark video from
                       a light session or vice versa. The W heatmap (bwr,
                       white at W = 0) is the same either way.">
          <option value="light">light</option>
          <option value="dark">dark</option>
        </select>

        <span class="text-muted">variants</span>
        <div class="flex items-center gap-2">
          <label v-for="v in props.variants" :key="v"
                 class="flex items-center gap-1 cursor-pointer select-none"
                 :title="VARIANT_META[v].label">
            <input type="checkbox" :checked="form.variants.includes(v)"
                   @change="toggleVariant(v)" />
            <span :style="{ color: variantColor(v) }">{{ v.toUpperCase() }}</span>
          </label>
        </div>

        <!-- 2D only: one plane per panel, and there are six of them. At
             ndim=1 there is nothing to choose. -->
        <template v-if="is2d">
          <span class="text-muted">planes</span>
          <div class="flex flex-wrap items-center gap-x-2 gap-y-1">
            <label v-for="(pl, i) in allPlanes" :key="i"
                   class="flex items-center gap-1 cursor-pointer select-none"
                   :title="planeTitle(ndim, pl)">
              <input type="checkbox" :checked="form.planes.includes(i)"
                     @change="togglePlane(i)" />
              <span>{{ planeLabel(ndim, pl) }}</span>
            </label>
          </div>

          <span></span>
          <div class="flex items-center gap-1">
            <button class="px-1.5 py-0.5 rounded bg-raised hover:bg-raised-hover
                           whitespace-nowrap"
                    title="one plane across every variant — the reading the
                           panel grid calls 'compare variants'"
                    @click="presetCompareVariants">compare variants</button>
            <button class="px-1.5 py-0.5 rounded bg-raised hover:bg-raised-hover
                           whitespace-nowrap"
                    title="every plane of one variant — the reading the panel
                           grid calls 'phase portrait'"
                    @click="presetPhasePortrait">phase portrait</button>
          </div>
        </template>

        <span class="text-muted">plots</span>
        <div class="flex flex-wrap items-center gap-x-2 gap-y-1">
          <label v-for="d in allDiagnostics" :key="d"
                 class="flex items-center gap-1 cursor-pointer select-none"
                 :title="diagnosticTitle(ndim, d)">
            <input type="checkbox" :checked="form.diagnostics.includes(d)"
                   @change="toggleDiagnostic(d)" />
            <span>{{ diagnosticLabel(ndim, d) }}</span>
          </label>
        </div>
      </div>

      <p class="text-fg-3">
        {{ frames }} frames → {{ duration.toFixed(1) }} s of video,
        {{ grid.cells }} panel<template v-if="grid.cells !== 1">s</template>
        (<span class="tabular-nums">{{ grid.cols }}×{{ grid.rows }}</span>,
        ~<span class="tabular-nums">{{ panelPx.w }}×{{ panelPx.h }}</span> px
        each), {{ jobTheme }} theme, grid lines
        {{ showGrid ? 'on' : 'off' }} (follows the setup panel). Each frame
        also carries the parameters + IC needed to reproduce the run.
      </p>

      <p v-if="panelsCramped" class="text-warn">
        Those panels are thumbnails at this size. Render at a larger size, or
        drop planes or variants — {{ grid.cells }} panels share the frame.
      </p>

      <p v-if="running" class="text-warn">
        The session is still computing. Rendering needs a paused session —
        Render will pause it first (computation resumes with Solve).
      </p>

      <p v-if="error" class="text-error">{{ error }}</p>

      <div v-if="job" class="space-y-1">
        <div class="h-2 rounded bg-raised overflow-hidden">
          <div class="h-full bg-sky-600 transition-[width] duration-200"
               :style="{ width: pct + '%' }"></div>
        </div>
        <p class="text-fg-3 tabular-nums">
          {{ job.state }} — {{ job.done }}/{{ job.total }} frames
          <!-- The RENDER rate, which is not the video's fps two fields up: a
               1000-frame export is minutes of waiting, and this is the only
               thing on screen that says whether it is progressing at 8 fps or
               at 0.5. Averaged over the whole run once it finishes, so it
               stays readable instead of freezing on its last second. -->
          <span v-if="renderRate" :title="job.state === 'done'
                  ? 'frames rendered per second, averaged over the whole render'
                  : 'frames rendered per second right now. This is the RENDER '
                    + 'rate on the server, not the video\'s frame rate — the '
                    + 'finished mp4 plays at ' + job.fps + ' fps whatever this says.'">
            · {{ renderRate }}
          </span>
          <span v-if="job.state === 'done'">
            ({{ (job.bytes/2**20).toFixed(1) }} MiB)
          </span>
          <span v-if="job.error" class="text-error">{{ job.error }}</span>
        </p>
      </div>

      <div class="flex items-center gap-2 pt-1">
        <button class="wf-solid px-2 py-1 rounded bg-sky-700 hover:bg-sky-600 font-medium"
                :disabled="busy || frames < 1
                           || job?.state === 'running' || job?.state === 'queued'"
                @click="start">{{ running ? 'Pause &amp; render' : 'Render' }}</button>
        <a v-if="downloadUrl" :href="downloadUrl" :download="job!.filename"
           class="px-2 py-1 rounded bg-emerald-700 hover:bg-emerald-600 font-medium text-white">
          ⤓ download
        </a>
        <button v-if="job?.state === 'running' || job?.state === 'queued'"
                class="px-2 py-1 rounded bg-raised hover:bg-raised-hover"
                @click="cancel">Cancel</button>
      </div>

      <h5 class="text-muted uppercase tracking-wider text-[11px] pt-2
                 border-t border-line-soft">
        Setup — the initial conditions
      </h5>
      <p class="text-fg-3">
        The state this run STARTED from: grid, U(x), the IC, variants,
        m/c/ℏ/tol and the run mode. Live parameter changes and auto-expand
        are not part of it — they belong to the run, and the video's
        metadata block is where they are recorded.
      </p>
      <div class="flex items-center gap-2">
        <a v-if="setupUrl" :href="setupUrl"
           class="px-2 py-1 rounded bg-raised hover:bg-raised-hover"
           title="download this simulation's initial conditions as a .json file">
          ⤓ download setup
        </a>
        <button class="px-2 py-1 rounded bg-raised hover:bg-raised-hover"
                title="load a setup from a .json file, or straight from an mp4
                       exported here (it carries the same document)"
                @click="fileInput?.click()">⤒ import setup…</button>
        <input ref="fileInput" type="file" class="hidden"
               accept=".json,.mp4,application/json,video/mp4"
               @change="onImportFile" />
      </div>
      <p v-if="importMsg" :class="importOk ? 'text-ok' : 'text-error'">
        {{ importMsg }}
      </p>

      <div class="flex pt-1">
        <button class="ml-auto px-2 py-1 rounded bg-raised hover:bg-raised-hover"
                @click="open = false">Close</button>
      </div>
    </div>
  </div>
</template>
