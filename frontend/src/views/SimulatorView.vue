<script setup lang="ts">
/**
 * Simulator view. Two layouts (persisted):
 * - landscape: [setup+IC column] [W panel grid] [plots column]
 * - portrait (tall screens): three columns on top — setup | IC | plots —
 *   with the W panel grid full-width underneath.
 *
 * Live-applicable changes (U, c, mass, ℏ, tol, dt sign) go through
 * set_params into the running session; grid/IC/variant changes require a
 * restart (fresh session from the same-config Cauchy data).
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import ControlBar from '../components/ControlBar.vue'
import ExportPanel from '../components/ExportPanel.vue'
import ICEditor from '../components/ICEditor.vue'
import PanelGrid from '../components/PanelGrid.vue'
import PlotsColumn from '../components/PlotsColumn.vue'
import SetupPanel from '../components/SetupPanel.vue'
import Timeline from '../components/Timeline.vue'
import { useSession } from '../composables/useSession'
import { applyPrecisionInvariants, loadConfig, precisionForPayload,
         precisionIsUserChosen, saveConfig, setHostPrecision } from '../lib/config'
import { apiErrorText } from '../lib/apierror'
import { displayInterval } from '../lib/perf'
import { transportAction } from '../lib/transport'
import { theme, toggleTheme } from '../lib/theme'
import { ALL_VARIANTS, VARIANT_META, variantColor, type VariantKey } from '../lib/variants'

const session = useSession()
const currentRecord = ref(0)
const createError = ref('')
const restartCount = ref(0)
const showSetup = ref(true)
const restartNeeded = ref(false)
const reconnecting = ref(false)

// Solve gate: the transport must not start a computation while the setup
// form holds invalid data (family-invalid potential draft, IC the preview
// endpoint rejects). The editors emit their verdicts after every compile/
// preview round-trip; hiding the setup discards the (unapplied) drafts, so
// the gate reopens.
const potentialValid = ref(false)
const icValid = ref(false)
const setupValid = computed(() =>
  !showSetup.value || (potentialValid.value && icValid.value))

const layout = ref<'landscape' | 'portrait'>(
  (localStorage.getItem('wignerf.layout') as 'landscape' | 'portrait') ?? 'landscape')
const showHelp = ref(false)
watch(layout, (v) => localStorage.setItem('wignerf.layout', v))

const showGrid = ref(localStorage.getItem('wignerf.grid') !== '0')
watch(showGrid, (v) => localStorage.setItem('wignerf.grid', v ? '1' : '0'))

// setup persists across reloads: a hard refresh must not silently reset
// mode/t2/grid/IC to defaults
const cfg = reactive(loadConfig())
watch(cfg, () => saveConfig(cfg), { deep: true })
const activeVariants = ref<VariantKey[]>([...cfg.variants])
const activeGrid = ref({ ...cfg.grid })
const sessionId = computed(() => session.info.value?.session_id ?? null)
// batch mode while it computes new records: no frames stream, so the heatmap
// and marginals are dimmed and a progress report stands in for the display
// (the observable series keep updating from their own cheap REST poll)
const batchComputing = computed(() =>
  session.status.value?.mode === 'batch' && !!session.status.value?.computing)
// what the heatmap should overlay in batch mode: the live progress while
// computing, or a "press Play to review" hint once a finished run sits at the
// frontier with nothing painted (batch streamed no frames, so it is blank)
const batchOverlay = computed<'computing' | 'review' | null>(() => {
  const st = session.status.value
  if (st?.mode !== 'batch') return null
  if (st.computing) return 'computing'
  if (!session.lastFrame.value
      && transportAction(st, null) === 'play') return 'review'
  return null
})
// NOTE: showGrid is deliberately NOT part of this key — toggling grid
// lines must never remount/blank the panels (charts rebuild internally)
const plotsKey = computed(() => String(restartCount.value))

let unsub: (() => void) | null = null
let userPaused = false     // survives a reconnect; see recover()

function payload() {
  // The form must be self-consistent BEFORE it is serialized, and this is the
  // last place that can be guaranteed. The invariants are normally applied at
  // the point of change, but a restart can happen in the SAME flush as the
  // change that triggered it (the watcher below calls syncFreshSessionToForm,
  // and it runs before the Setup panel's own watchers — a child's setup runs
  // during the parent's render, so the parent's watchers flush first). A
  // payload built from a half-fixed config asked for float32 + auto-expand and
  // came back a 422 naming a pair the user never chose. Idempotent, so this
  // costs nothing when the fix-up has already run.
  applyPrecisionInvariants(cfg)
  const p: Record<string, unknown> = JSON.parse(JSON.stringify(cfg))
  if (cfg.mode === 'interactive') delete p.t2
  // '' / 0 are the form's way of saying "whatever the host is configured for";
  // the API wants the field absent, not an empty string or a zero it would
  // reject against its own ge=64 bound.
  if (!cfg.device) delete p.device
  if (!cfg.history_mb) delete p.history_mb
  // Same idea, and load-bearing: until the user picks a precision, DON'T send
  // one — let WIGNERF_PRECISION decide (the one exception being auto-expand,
  // which only float64 can provide; see precisionForPayload). The session's
  // real precision comes back in `status` and the watcher below syncs the form.
  const prec = precisionForPayload(cfg)
  if (prec === null) delete p.precision
  else p.precision = prec
  // delay 0 is the dial's "one frame per display refresh" position: the
  // server needs honest seconds, and a fresh session must not flood
  p.delay = Math.max(cfg.delay, displayInterval())
  return p
}

async function restart() {
  createError.value = ''
  try {
    unsub?.()
    await session.create(payload())
    activeVariants.value = [...cfg.variants]
    activeGrid.value = { ...cfg.grid }
    currentRecord.value = 0
    restartNeeded.value = false
    restartCount.value++
    unsub = session.onFrame((f) => {
      currentRecord.value = f.record
      // geometry is a per-record fact (auto-expand regrids): follow the
      // PAINTED frame so panels, axis overlays and marginal axes stay in
      // sync while scrubbing across a regrid boundary in either direction
      const g = activeGrid.value
      if (f.Nx !== g.Nx || f.Np !== g.Np || f.x1 !== g.x1 || f.x2 !== g.x2
          || f.p1 !== g.p1 || f.p2 !== g.p2)
        activeGrid.value = { x1: f.x1, x2: f.x2, Nx: f.Nx,
                             p1: f.p1, p2: f.p2, Np: f.Np }
    })
  } catch (e: unknown) {
    createError.value = apiErrorText(e)
  }
}

/**
 * An mp4 render dies if the session it reads from moves on: restarting
 * deletes the session (and with it the job and its partial file), and
 * computing new records evicts the old ones out from under the renderer.
 * Both used to happen silently — so ask first, and on "yes" cancel the job
 * outright instead of leaving it to fail somewhere mid-file.
 */
const exportRunning = computed(() => {
  const e = session.exportEvent.value
  return e && (e.state === 'running' || e.state === 'queued') ? e : null
})

async function mayDiscardExport(what: string): Promise<boolean> {
  const e = exportRunning.value
  if (!e) return true
  const pct = e.total ? Math.round((100*e.done)/e.total) : 0
  if (!confirm(`An mp4 export is rendering (${pct}% — ${e.done}/${e.total} `
               + `frames).\n\n${what} cancels it and discards the partial `
               + 'file.\n\nContinue?'))
    return false
  try { await api.delete(`/exports/${e.job_id}`) } catch { /* already gone */ }
  return true
}

/** The user-initiated restart (the automatic ones — first mount, backend
 *  recovery — must never prompt). */
async function requestRestart() {
  if (await mayDiscardExport('Restarting the session')) await restart()
}

/** Transport commands, gated the same way: only a command that will COMPUTE
 *  threatens the render (playback adds no records, so it cannot evict). */
async function sendCommand(cmd: Record<string, unknown>) {
  const willCompute = cmd.type === 'play'
    && transportAction(session.status.value,
                       session.lastFrame.value?.record ?? null) === 'solve'
  if (willCompute && !(await mayDiscardExport('Computing new records'))) return
  // THE FORM IS AUTHORITATIVE AT THE MOMENT COMPUTATION STARTS. U(x) is the
  // only live-appliable field with no implicit commit — the numeric Physics
  // fields apply on blur/Enter, which pressing Solve performs — so without
  // this a validated edit could sit in the form while the session went on
  // computing the OLD potential. Measured before the fix: 369 records of
  // x^2/2 behind a form reading x^2/2 + 5*x^4, the only signal being the
  // amber input, in a panel that can be hidden. Sent BEFORE the play command
  // (same WS, handled in order) so the frontier is already under the new U;
  // the header's "✓ applied U(x)" flash is the confirmation. Playback is
  // deliberately excluded — it computes nothing, so U is irrelevant to it.
  const st = session.status.value
  if (willCompute && potentialValid.value && st
      && cfg.potential !== st.potential)
    applyLive({ U: cfg.potential })
  // Remember the user's transport intent across a reconnect (see recover()).
  if (cmd.type === 'pause') userPaused = true
  if (cmd.type === 'play') userPaused = false
  session.send(cmd)
}

function toggleVariant(v: VariantKey) {
  const i = cfg.variants.indexOf(v)
  if (i >= 0) {
    if (cfg.variants.length === 1) return   // >=1 constraint
    cfg.variants.splice(i, 1)
  } else {
    cfg.variants.push(v)
    cfg.variants.sort((a, b) => ALL_VARIANTS.indexOf(a) - ALL_VARIANTS.indexOf(b))
  }
  // variant changes discard the computed history (a new variant has no
  // state at the current t), so they only take effect on YOUR restart —
  // never behind your back
  restartNeeded.value = true
}

function applyLive(params: Record<string, unknown>) {
  session.send({ type: 'set_params', params })
}

// What the SESSION is actually evolving (status echoes every live change):
// the setup form marks fields that differ from it as edited-but-unapplied,
// and greys out an "Apply live" that would be a no-op.
const livePhysics = computed(() => {
  const st = session.status.value
  if (!st || !session.connected.value) return null
  return { potential: st.potential, mass: st.mass, c: st.c,
           hbar_eff: st.hbar_eff, tol: st.tol }
})

// The RUN settings of the session actually running. Restart-only, so unlike
// livePhysics a difference is not "pending" but INERT — the Setup panel says
// so rather than leaving the form looking like it took effect.
const liveRun = computed(() => {
  const st = session.status.value
  if (!st || !session.connected.value) return null
  return { mode: st.mode, t2: st.t2, record_dt: st.record_dt,
           precision: st.precision }
})
// Whether the session has COMPUTED anything beyond the initial IC record.
// Until it has, there is no meaningful "run mode" — the form is the sole
// source of truth and the (idle) session silently tracks it (see the watch
// below). So the run-field staleness warning only applies once a run exists.
const hasRun = computed(() =>
  (session.status.value?.record_extent?.[1] ?? -1) > 0)

// Serialized identity of the run-defining fields (t₂ only matters in batch),
// used to tell whether a FRESH session still matches the form.
function runKeyOf(mode: string, t2: number | null, record_dt: number,
                  precision: string) {
  return JSON.stringify([mode, mode === 'batch' ? t2 : null, record_dt,
                         precision])
}
const formRunKey = () =>
  runKeyOf(cfg.mode, cfg.t2, cfg.record_dt, cfg.precision)

// A run-defining field (mode / t₂ / Δt rec) is SessionCreate-only. On a FRESH
// session — connected, idle, nothing computed beyond the initial IC record —
// silently adopt the change by re-creating the session, so switching e.g. to
// interactive before you have computed anything just takes effect (no stale
// "running: batch — restart to apply", and Solve then runs the mode you SEE,
// not the one the session happened to be created with). Once a run has
// actually produced records the change stays INERT and the amber warning
// protects that run — the user restarts explicitly.
//
// Restarts are SERIALIZED and COALESCED. create() clears status/connected
// while it runs, so a second edit arriving mid-restart would otherwise be
// dropped (the watch guard bails on the null status) and leave the new session
// on the FIRST edit's values while the form shows the later one — with no
// warning, because a fresh session shows none. Instead, once we start managing
// an idle session we loop until what we created matches the current form.
let autoRestarting = false
async function syncFreshSessionToForm() {
  if (autoRestarting) return              // a loop is already running; it re-reads
                                          // the form each pass and will catch this edit
  const st = session.status.value
  if (!session.connected.value || !st || st.running
      || (st.record_extent?.[1] ?? -1) > 0) return   // a real run exists: don't touch
  autoRestarting = true
  try {
    let sent = runKeyOf(st.mode, st.t2, st.record_dt,  // the live session's config
                        st.precision)
    while (sent !== formRunKey()) {
      sent = formRunKey()
      await restart()                     // restart() reads cfg as it stands NOW
    }
  } finally {
    autoRestarting = false
  }
}
watch(() => [cfg.mode, cfg.t2, cfg.record_dt, cfg.precision],
      () => { void syncFreshSessionToForm() })

// Boundary watch surfacing: a dismissible amber warning while W sits in
// the edge band (the server posts an all-clear that removes it), and a
// transient notice for each auto-expand regrid.
const boundaryText = computed(() => {
  const b = session.boundary.value
  if (!b) return ''
  const axes = b.axes.join(', ')
  if (b.action === 'capped')
    return `W(x,p,t) reached the ${axes} edge but the domain is at the ` +
           `${b.max_grid ?? ''}-cell cap (WIGNERF_MAX_GRID) — mass is wrapping`
  if (b.action === 'invalid_potential')
    return b.message ?? 'cannot expand: U(x) is invalid on the larger domain'
  return `W(x,p,t) is approaching the ${axes} edge — mass will wrap around` +
         (cfg.auto_expand ? '' : '; enable auto-expand or restart with a larger domain')
})
// the server's status is the authority on the live toggle (delay-dial
// pattern): a reattach to a surviving session must not show a stale box
watch(() => session.status.value?.auto_expand, (v) => {
  if (typeof v === 'boolean' && v !== cfg.auto_expand) cfg.auto_expand = v
})
// Same pattern for precision, and it is what makes a failed /device probe
// harmless: an unchosen form omits `precision` from the create payload, so the
// SERVER picks, and its answer comes straight back in status. Syncing to it
// keeps the form honest and stops `runDiffers('precision')` from showing a
// phantom "restart to apply" against a difference the user never made and
// could not resolve. A user who HAS chosen is never overwritten — their
// pending choice is exactly what the amber marker is for.
watch(() => session.status.value?.precision, (v) => {
  if (v && !precisionIsUserChosen() && v !== cfg.precision) {
    cfg.precision = v
    applyPrecisionInvariants(cfg)
  }
})
// float32 forbids auto-expand, and applyPrecisionInvariants has already cleared
// the FORM. Make the LIVE session agree, because nothing else can: the watcher
// above cannot (status.auto_expand does not CHANGE, so it never fires), the
// checkbox is disabled in float32, and auto_expand is not in LivePhysics so
// there is no amber marker either — a session left quietly expanding behind an
// unchecked, unreachable box. Keyed on PRECISION, not on cfg.auto_expand, so it
// never duplicates the checkbox's own apply-live; that path leaves precision
// alone. Covers the select, an import and probeHost's adoption in one place.
// send() no-ops on a closed socket, so this is safe before a session exists.
watch(() => cfg.precision, () => {
  if (cfg.precision === 'float32' && session.status.value?.auto_expand
      && session.connected.value)
    applyLive({ auto_expand: false })
})
// Live parameter changes: the Physics fields apply on change (blur/Enter)
// with no other confirmation anywhere, which reads as "nothing happened" —
// so echo what the server actually took. Labels mirror describe.FIELD_LABEL.
const PARAM_LABEL: Record<string, string> = {
  U: 'U(x)', mass: 'm', c: 'c', hbar_eff: 'ℏ', tol: 'tol',
  dt_sign: 't dir', auto_expand: 'auto-expand',
}
const paramFlash = ref('')
let paramTimer = 0
watch(session.paramsApplied, (p) => {
  if (!p) return
  const fmt = (k: string, v: string | number | boolean) =>
    k === 'auto_expand' ? (v ? 'on' : 'off')
    : k === 'dt_sign' ? (Number(v) < 0 ? 'backward' : 'forward')
    : String(v)
  const parts = Object.entries(p.applied).map(([k, v]) => {
    const label = PARAM_LABEL[k] ?? k
    return k in p.before ? `${label} ${fmt(k, p.before[k])} → ${fmt(k, v)}`
                         : `${label} = ${fmt(k, v)}`
  })
  paramFlash.value = `applied ${parts.join(', ')} at record ${p.at_record}`
  clearTimeout(paramTimer)
  paramTimer = window.setTimeout(() => { paramFlash.value = '' }, 6000)
})

const regridFlash = ref('')
let regridTimer = 0
watch(session.regrid, (r) => {
  if (!r) return
  const g = r.grid
  const verb = Object.values(r.kind).includes('double') ? 'expanded' : 'moved'
  regridFlash.value = `domain ${verb} to [${+g.x1.toFixed(4)}, ${+g.x2.toFixed(4)}] × ` +
    `[${+g.p1.toFixed(4)}, ${+g.p2.toFixed(4)}], ${g.Nx}×${g.Np}, at record ${r.at_record}`
  clearTimeout(regridTimer)
  regridTimer = window.setTimeout(() => { regridFlash.value = '' }, 6000)
})

/**
 * Self-heal after an unexpected WebSocket close (e.g. a backend restart,
 * common with `uvicorn --reload`): keep probing until the backend answers;
 * if our session survived, reattach the socket, otherwise recreate the
 * session from the current config.
 */
let recovering = false
session.onClose(() => { void recover() })

async function recover() {
  if (recovering) return
  recovering = true
  reconnecting.value = true
  const sid = session.info.value?.session_id
  // was playback in progress when the socket dropped? (the browser drops it
  // under a large-grid frame flood.) If so, resume after reattaching — the
  // server seeds the replay from the cursor, so this CONTINUES to the end
  // rather than leaving the user stranded mid-playback.
  const wasPlaying = session.status.value?.running === true
  // ...but a Pause pressed DURING the outage must win. Each drop re-issued
  // play unconditionally, so while the socket was dropping repeatedly (open
  // DevTools captures every 32 MiB frame, which is enough to cause that) the
  // transport flipped to Play and instantly back to Pause and playback could
  // not be stopped at all. userPaused is set by sendCommand and cleared by an
  // explicit play, so the resume fires only if the user still wants one.
  userPaused = false
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500))
      try {
        if (!sid) break
        await api.get(`/sessions/${sid}`)
        session.reconnect()          // session survived: reattach
        if (wasPlaying && !userPaused)
          setTimeout(() => { if (!userPaused) sendCommand({ type: 'play' }) }, 400)
        return
      } catch (e: unknown) {
        const st = (e as { response?: { status?: number } }).response?.status
        if (st === 404) break        // backend is up, session is gone
        // no response at all: backend still down — keep waiting
      }
    }
    await restart()                  // recreate from the current config
  } finally {
    recovering = false
    reconnecting.value = false
  }
}

// Keyboard shortcuts: Space = play/pause (documented in the transport
// button's tooltip), R = reverse time direction, and — paused only —
// ←/→ = seek ±10% of the computed history, Home/End = first record /
// frontier. BUTTON is excluded so a focused button keeps its native
// Space=click and never double-fires with this handler (transport
// controls also blur themselves after use).
function onKey(ev: KeyboardEvent) {
  // before the focus guard: Esc dismisses the help popover from anywhere
  if (ev.code === 'Escape' && showHelp.value) {
    showHelp.value = false
    return
  }
  const tag = (ev.target as HTMLElement).tagName
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA'
      || tag === 'BUTTON') return
  if (ev.code === 'Space') {
    ev.preventDefault()
    // same gate as the transport button: never start a computation
    // (action 'solve') while the setup form is invalid
    const act = transportAction(session.status.value,
                                session.lastFrame.value?.record ?? null)
    if (act === 'solve' && !setupValid.value) return
    // sendCommand: same export-in-flight guard as the transport button
    void sendCommand({ type: act === 'pause' ? 'pause' : 'play' })
  } else if (ev.code === 'KeyR') {
    const sign = session.status.value?.sign ?? 1
    session.send({ type: 'set_params', params: { dt_sign: sign > 0 ? -1 : 1 } })
  } else if (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight'
             || ev.code === 'ArrowUp' || ev.code === 'ArrowDown'
             || ev.code === 'Home' || ev.code === 'End') {
    const st = session.status.value
    const [k0, k1] = st?.record_extent ?? [0, -1]
    if (!st || st.running || k1 < 0) return   // paused-only, needs history
    ev.preventDefault()
    // The backend clamps every seek into the TRUE extent, and our cached
    // record_extent can be stale (pausing races the final record's
    // completion, and no status is pushed while paused) — so send the
    // unclamped target and let the backend be the authority. End uses a
    // sentinel index for "the frontier, wherever it really is".
    // No optimistic cursor state on purpose: a held key recomputes from
    // the last painted frame, so stepping is paced by frame delivery
    // instead of flying through the whole timeline between paints.
    // ←/→ jump 10% of the computed history; ↑/↓ are the neighbouring keys
    // for the fine version — one record per press. Both increase to the
    // right/up, so →/↑ move forward in time.
    const step = Math.max(1, Math.round((k1 - k0) / 10))
    const cur = Math.round(session.lastFrame.value?.record ?? st.cursor)
    const k = ev.code === 'Home' ? 0
            : ev.code === 'End'  ? Number.MAX_SAFE_INTEGER
            : ev.code === 'ArrowLeft' ? Math.max(0, cur - step)
            : ev.code === 'ArrowRight' ? cur + step
            : ev.code === 'ArrowDown' ? Math.max(0, cur - 1)
            :                           cur + 1
    if (k !== cur) session.send({ type: 'seek', record: k })
  }
}

// A genuine page departure (tab close, reload, navigation) does NOT run Vue's
// onBeforeUnmount reliably, and its awaited DELETE would not complete during
// unload — so fire a keepalive DELETE here to free the session's RAM+VRAM at
// once instead of leaving it to the idle TTL. Skip a bfcache suspend
// (e.persisted): that page may be restored, and an open WebSocket usually
// disqualifies bfcache anyway. This is distinct from recover() (a live-tab
// socket drop), which is never a pagehide.
const onPageHide = (e: PageTransitionEvent) => { if (!e.persisted) session.beaconDestroy() }

// The host's device pool, for the Setup panel's per-session device select.
// Fetched once (it cannot change without a backend restart) and separately
// from `status`, because the form needs the CHOICES even before a session
// exists. A stored device this backend does not have stays in the list as a
// flagged entry rather than silently disappearing — see SetupPanel.
const deviceOptions = ref<{ spec: string; device: string }[] | null>(null)
// The POOL, as distinct from the choices: what a session gets when the form's
// device is '' (= "the host's default"). The Setup panel needs it to tell
// whether a form device of '' still matches the RUNNING session's devices — a
// restart-only field with no live counterpart to compare against otherwise.
const hostPool = ref<string[] | null>(null)
const historyCapMb = computed(() => {
  const b = session.status.value?.history_cap_bytes
  return b == null ? null : Math.round(b/(1024*1024))
})

/**
 * Host facts the form needs before it can create anything: the device
 * `choices` (the pool PLUS cpu, which is always a legal target but never
 * appears in an "auto" pool on a CUDA host) and WIGNERF_PRECISION.
 *
 * Awaited before the FIRST restart so the form shows the right default at once
 * and no session is created only to be recreated. The cost is a cached
 * round-trip: measured 758 ms cold (two CUDA contexts) and ~1 ms warm,
 * `lru_cache`d server-side, so only the first page load after a backend start
 * pays — and it is bounded by an explicit timeout, because awaiting an
 * endpoint with axios's default (none) would let a hung backend stall the
 * whole app at startup.
 *
 * It is NOT load-bearing for correctness. Precision is omitted from the create
 * payload until the user chooses one, so a probe that times out, 404s or
 * returns junk costs a device list and a briefly-stale form value — never the
 * wrong precision. That is deliberate: this used to send a hard-coded float64
 * as though it were a decision, silently overriding a float32 host whenever
 * this call failed.
 */
async function probeHost() {
  try {
    const { data } = await api.get<{ choices: { spec: string; device: string }[]
                                     devices?: { spec: string }[]
                                     precision?: string }>(
      '/device', { timeout: 5000 })
    deviceOptions.value = data.choices ?? []
    hostPool.value = data.devices?.map((d) => d.spec) ?? null
    // Install the host's default so "Reset setup to defaults" agrees with the
    // server, and adopt it if this browser has never chosen a precision.
    setHostPrecision(data.precision)
    if (!precisionIsUserChosen() && data.precision
        && data.precision !== cfg.precision) {
      cfg.precision = data.precision as typeof cfg.precision
      // mergeConfig already enforced this at load, but the host default lands
      // AFTER that — a stored auto_expand would otherwise ride into a float32
      // session and 422 on a combination the user never picked.
      applyPrecisionInvariants(cfg)
    }
  } catch {
    deviceOptions.value = []
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('pagehide', onPageHide)
  void probeHost().then(restart)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('pagehide', onPageHide)
  unsub?.()
  void session.destroy()
})
</script>

<template>
  <div class="h-screen flex flex-col overflow-hidden bg-app text-fg">
    <header class="px-3 py-2 text-sm border-b border-line-soft flex items-center gap-4 flex-wrap">
      <button class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
              @click="showSetup = !showSetup">
        {{ showSetup ? '◀ hide setup' : '▶ setup' }}
      </button>
      <span class="font-semibold tracking-wide">wignerf</span>

      <div class="flex items-center gap-3">
        <label v-for="v in ALL_VARIANTS" :key="v"
               class="flex items-center gap-1 cursor-pointer select-none"
               :title="VARIANT_META[v].label">
          <input type="checkbox" :checked="cfg.variants.includes(v)"
                 @change="toggleVariant(v)" />
          <span :style="{ color: variantColor(v) }">{{ v.toUpperCase() }}</span>
        </label>
      </div>

      <button class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
              :title="'toggle landscape/portrait layout'"
              @click="layout = layout === 'landscape' ? 'portrait' : 'landscape'">
        {{ layout === 'landscape' ? '⬒ portrait' : '⬓ landscape' }}
      </button>

      <button class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
              :title="`switch to the ${theme === 'dark' ? 'light' : 'dark'} theme`"
              @click="toggleTheme">
        {{ theme === 'dark' ? '☀ light' : '☾ dark' }}
      </button>

      <ExportPanel :status="session.status.value" :session-id="sessionId"
                   :event="session.exportEvent.value" :variants="activeVariants"
                   :current-record="currentRecord" :show-grid="showGrid"
                   :cfg="cfg"
                   @command="session.send" @dirty="restartNeeded = true" />

      <div class="relative">
        <button class="px-2 py-0.5 rounded bg-raised hover:bg-raised-hover text-xs"
                title="keyboard shortcuts and mouse controls"
                @click="showHelp = !showHelp;
                        (($event as MouseEvent).currentTarget as HTMLElement)?.blur()">? help</button>
        <div v-if="showHelp" class="fixed inset-0 z-40" @click="showHelp = false"></div>
        <div v-if="showHelp"
             class="absolute left-0 top-full mt-2 z-50 w-96 rounded border border-line bg-panel shadow-xl p-3 text-xs space-y-2">
          <h4 class="font-semibold text-fg-2">Keyboard</h4>
          <div class="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-fg-2">
            <span><kbd class="wf-kbd">Space</kbd></span>
            <span>Solve / Play / Pause (the transport button)</span>
            <span><kbd class="wf-kbd">R</kbd></span>
            <span>reverse time direction (applies at the frontier)</span>
            <span><kbd class="wf-kbd">←</kbd> <kbd class="wf-kbd">→</kbd></span>
            <span>while paused: step the time position by 10% of the computed history</span>
            <span><kbd class="wf-kbd">↓</kbd> <kbd class="wf-kbd">↑</kbd></span>
            <span>while paused: step ONE record back / forward</span>
            <span><kbd class="wf-kbd">Home</kbd> <kbd class="wf-kbd">End</kbd></span>
            <span>while paused: jump to the first record / the frontier</span>
            <span>click <span class="text-fg-3">t =</span></span>
            <span>while paused: type an exact time — <kbd class="wf-kbd">Enter</kbd> seeks to the
              nearest record, <kbd class="wf-kbd">Esc</kbd> cancels</span>
          </div>
          <h4 class="font-semibold text-fg-2 pt-1">Mouse (plots &amp; W panels)</h4>
          <div class="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-fg-2">
            <span>drag</span>
            <span>charts: zoom to selection (x, y or box) · W panels &amp; IC preview: pan</span>
            <span>wheel</span>
            <span>zoom at the cursor (charts: x-axis, <kbd class="wf-kbd">Shift</kbd> = y-axis)</span>
            <span>double-click</span>
            <span>reset the view</span>
          </div>
        </div>
      </div>

      <!-- Persistent, not a flash: every number on screen (E, ΔX·ΔP, purity)
           is preview-grade for as long as this session lives, and the one
           thing that must never happen is a physics claim made from a run
           whose reduced precision had scrolled off the header. -->
      <span v-if="session.status.value?.precision === 'float32'"
            class="text-warn text-xs border border-warn/60 rounded px-1.5 py-0.5"
            title="This session's spectral working precision is float32 — a fast PREVIEW mode. Purity and energy drift by ~1e-4 with the same secular signature as boundary wrap, and ΔX·ΔP noise is ~150× the relativistic shear. Restart in float64 before reading any of these curves as physics.">
        float32 · preview — single precision mode
      </span>
      <span v-if="restartNeeded" class="text-warn text-xs">
        setup changed —
        <button class="underline" @click="requestRestart">restart</button> to apply
      </span>
      <span v-if="boundaryText" class="text-warn text-xs">
        ⚠ {{ boundaryText }}
        <button class="underline" title="dismiss (reappears if the boundary state changes)"
                @click="session.boundary.value = null">×</button>
      </span>
      <span v-if="paramFlash" class="text-ok text-xs">✓ {{ paramFlash }}</span>
      <span v-if="regridFlash" class="text-info text-xs">⤢ {{ regridFlash }}</span>
      <span v-if="reconnecting" class="ml-auto text-warn">
        backend disconnected — reconnecting…
      </span>
      <span v-else-if="!session.connected.value" class="ml-auto text-warn">connecting…</span>
    </header>

    <div v-if="createError" class="px-3 py-2 text-error text-sm">{{ createError }}</div>

    <!-- ================= landscape ================= -->
    <main v-if="layout === 'landscape'" class="flex-1 min-h-0 flex gap-2 p-2">
      <aside v-if="showSetup" class="w-80 shrink-0 overflow-y-auto space-y-4 pr-1 text-sm">
        <SetupPanel :cfg="cfg" :live="session.connected.value"
                    :computing="session.status.value?.computing ?? false"
                    :sign="session.status.value?.sign ?? 1"
                    :live-grid="session.status.value?.grid ?? null"
                    :live-physics="livePhysics"
                    :live-run="liveRun" :has-run="hasRun"
                    :max-grid="session.status.value?.max_grid ?? 4096"
                    :live-devices="session.status.value?.devices ?? null"
                    :device-options="deviceOptions" :host-pool="hostPool"
                    :history-cap-mb="historyCapMb" :history-mb-max="session.status.value?.history_mb_max ?? null"
                    v-model:show-grid="showGrid"
                    @dirty="restartNeeded = true" @restart="requestRestart"
                    @apply-live="applyLive"
                    @potential-validity="(v: boolean) => potentialValid = v" />
        <ICEditor :ic="cfg.ic" :grid="cfg.grid" :hbar-eff="cfg.hbar_eff"
                  :show-grid="showGrid" @changed="restartNeeded = true"
                  @validity="(v: boolean) => icValid = v" />
      </aside>

      <div class="flex-1 min-w-0 min-h-0">
        <PanelGrid :key="plotsKey" :frame-source="session.onFrame"
                   :variants="activeVariants" :domain="activeGrid"
                   :batch-overlay="batchOverlay" :progress="session.progress.value"
                   :show-grid="showGrid" />
      </div>

      <aside class="w-[380px] shrink-0 overflow-y-auto">
        <PlotsColumn :frame-source="session.onFrame" :session-id="sessionId"
                     :variants="activeVariants" :grid="activeGrid"
                     :last-frame="session.lastFrame.value"
                     :batch-computing="batchComputing"
                     :show-grid="showGrid" :plots-key="plotsKey" />
      </aside>
    </main>

    <!-- ================= portrait ================== -->
    <main v-else class="flex-1 min-h-0 flex flex-col gap-2 p-2">
      <div v-if="showSetup" class="grid grid-cols-3 gap-3 max-h-[46%] min-h-0 text-sm">
        <div class="overflow-y-auto min-h-0 pr-1">
          <SetupPanel :cfg="cfg" :live="session.connected.value"
                    :computing="session.status.value?.computing ?? false"
                    :sign="session.status.value?.sign ?? 1"
                    :live-grid="session.status.value?.grid ?? null"
                    :live-physics="livePhysics"
                    :live-run="liveRun" :has-run="hasRun"
                    :max-grid="session.status.value?.max_grid ?? 4096"
                    :live-devices="session.status.value?.devices ?? null"
                    :device-options="deviceOptions" :host-pool="hostPool"
                    :history-cap-mb="historyCapMb" :history-mb-max="session.status.value?.history_mb_max ?? null"
                    v-model:show-grid="showGrid"
                      @dirty="restartNeeded = true" @restart="requestRestart"
                      @apply-live="applyLive"
                      @potential-validity="(v: boolean) => potentialValid = v" />
        </div>
        <div class="overflow-y-auto min-h-0 pr-1">
          <ICEditor :ic="cfg.ic" :grid="cfg.grid" :hbar-eff="cfg.hbar_eff"
                    :show-grid="showGrid" @changed="restartNeeded = true"
                    @validity="(v: boolean) => icValid = v" />
        </div>
        <div class="overflow-y-auto min-h-0 pr-1">
          <PlotsColumn :frame-source="session.onFrame" :session-id="sessionId"
                       :variants="activeVariants" :grid="activeGrid"
                       :last-frame="session.lastFrame.value"
                       :batch-computing="batchComputing"
                       :show-grid="showGrid" :plots-key="plotsKey" />
        </div>
      </div>

      <div class="flex-1 min-w-0 min-h-0">
        <PanelGrid :key="plotsKey" :frame-source="session.onFrame"
                   :variants="activeVariants" :domain="activeGrid"
                   :batch-overlay="batchOverlay" :progress="session.progress.value"
                   :show-grid="showGrid" />
      </div>
    </main>

    <div v-for="(e, i) in session.errors.value" :key="i"
         class="px-3 py-1 text-xs text-error">
      {{ e }}
      <button class="underline ml-2 text-error" title="dismiss this error"
              @click="session.errors.value.splice(i, 1)">×</button>
    </div>

    <Timeline
      :status="session.status.value"
      :current-record="currentRecord"
      @seek="(r) => session.send({ type: 'seek', record: r })"
    />
    <ControlBar
      :status="session.status.value"
      :last-frame="session.lastFrame.value"
      :progress="session.progress.value"
      :setup-valid="setupValid"
      @command="sendCommand"
    />
  </div>
</template>
