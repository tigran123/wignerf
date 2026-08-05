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
import { applyPrecisionInvariants, geomOf,
         loadConfig, precisionForPayload, precisionIsUserChosen, saveConfig,
         setHostPrecision, icPayload, type GeomCfg } from '../lib/config'
import { labels as axisLabels } from '../lib/axes'
import { apiErrorText } from '../lib/apierror'
import { displayInterval } from '../lib/perf'
import { transportAction } from '../lib/transport'
import { sticky } from '../lib/sticky'
import { edgeBand } from '../lib/cells'
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
// The computed lattice + the boundary watch's edge band, on the phase-space
// canvases only (a cell line means nothing on a uPlot series). Off by default:
// it is a diagnostic you reach for, and at 1D's larger N it is a lot of ink.
// Its own key, a sibling of wignerf.grid, so "Reset setup to defaults" leaves
// it alone like the other display prefs.
const showCells = ref(localStorage.getItem('wignerf.cells') === '1')
watch(showCells, (v) => localStorage.setItem('wignerf.cells', v ? '1' : '0'))

// setup persists across reloads: a hard refresh must not silently reset
// mode/t2/grid/IC to defaults
const cfg = reactive(loadConfig())
watch(cfg, () => saveConfig(cfg), { deep: true })
const activeVariants = ref<VariantKey[]>([...cfg.variants])
// The LIVE geometry, per painted frame (auto-expand moves it). Distinct from
// cfg.grid, which is the form: seeded from it so the panels have axes before
// the first frame lands.
const activeGeom = ref<GeomCfg>(geomOf(cfg.grid))
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
  // change that triggered it (the watcher below calls restartLoop,
  // and it runs before the Setup panel's own watchers — a child's setup runs
  // during the parent's render, so the parent's watchers flush first). A
  // payload built from a half-fixed config asked for float32 + auto-expand and
  // came back a 422 naming a pair the user never chose. Idempotent, so this
  // costs nothing when the fix-up has already run.
  applyPrecisionInvariants(cfg)
  const p: Record<string, unknown> = JSON.parse(JSON.stringify(cfg))
  // The form keeps BOTH expression drafts and the Gaussian components at once;
  // a session spec describes ONE run and so carries one kind's shape (see
  // icPayload). Deep-copied like everything else above it: icPayload returns
  // `cfg.ic.components` itself, which is the live reactive Proxy, and putting
  // that back into `p` would undo the snapshot the line above exists to take —
  // axios serializes in a microtask, so an edit landing in between would be
  // picked up by a request that had already been described as sent.
  p.ic = JSON.parse(JSON.stringify(icPayload(cfg.ic)))
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

// Restarts STARTED, bumped before the POST. `requestRestart` samples it to tell
// whether the form it is about has already been sent (see there).
let restartsStarted = 0

async function restart() {
  createError.value = ''
  restartsStarted++
  try {
    unsub?.()
    await session.create(payload())
    activeVariants.value = [...cfg.variants]
    activeGeom.value = geomOf(cfg.grid)
    currentRecord.value = 0
    restartNeeded.value = false
    restartCount.value++
    unsub = session.onFrame((f) => {
      currentRecord.value = f.record
      // geometry is a per-record fact (auto-expand regrids): follow the
      // PAINTED frame so panels, axis overlays and marginal axes stay in
      // sync while scrubbing across a regrid boundary in either direction
      const g = activeGeom.value
      if (f.ndim !== g.ndim || f.N.some((n, i) => n !== g.N[i])
          || f.lo.some((v, i) => v !== g.lo[i])
          || f.hi.some((v, i) => v !== g.hi[i]))
        activeGeom.value = { ndim: f.ndim, lo: [...f.lo], hi: [...f.hi], N: [...f.N] }
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

/**
 * The user-initiated restart (the automatic ones — first mount, backend
 * recovery — must never prompt). Goes through `restartLoop` rather than calling
 * restart() itself; see the flag it sets.
 *
 * `at` is sampled SYNCHRONOUSLY, at the click, and is what keeps "Reset setup to
 * defaults" to ONE session: the reset mutates the form and then asks for this
 * restart, so the run-key watcher's flush — queued by the mutation and
 * therefore running before this function resumes from its first await — may
 * already have created a session from the very form this request is about.
 * Requesting another would compute the same thing twice, at whatever the grid
 * costs. A create that started BEFORE the click is not counted: it cannot have
 * carried a change made after it began.
 */
async function requestRestart() {
  const at = restartsStarted
  if (!(await mayDiscardExport('Restarting the session'))) return
  if (restartsStarted > at) return
  restartRequested = true
  await restartLoop()
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
// An EXPLICIT restart waiting on the loop — the header/panel buttons and "Reset
// setup to defaults". It is served BY the loop instead of calling restart()
// itself because both requests can arrive in ONE flush: a reset moves the grid
// and the run fields together, so the watcher below fires alongside the panel's
// own request, and two overlapping create()s each destroy the other's session —
// leaving an orphan holding its workers' VRAM until the idle sweeper reaps it,
// and whichever POST happened to finish last as the survivor.
let restartRequested = false
/** The one path a restart may be started from, except the automatic ones
 *  (first mount, backend recovery) which have no form change to coalesce. */
async function restartLoop() {
  if (autoRestarting) return              // a loop is already running; it re-reads
                                          // the form each pass and will catch this edit
  const st = session.status.value
  // A real run exists: don't touch it — unless the restart was ASKED for, which
  // is the user accepting exactly that.
  if (!restartRequested && (!session.connected.value || !st || st.running
      || (st.record_extent?.[1] ?? -1) > 0)) return
  autoRestarting = true
  try {
    let sent = st ? runKeyOf(st.mode, st.t2, st.record_dt,  // the live session's
                             st.precision) : ''            // config
    while (restartRequested || sent !== formRunKey()) {
      restartRequested = false
      sent = formRunKey()
      await restart()                     // restart() reads cfg as it stands NOW
    }
  } finally {
    autoRestarting = false
  }
}
watch(() => [cfg.mode, cfg.t2, cfg.record_dt, cfg.precision],
      () => { void restartLoop() })

// Boundary watch surfacing: a dismissible amber warning while W sits in
// the edge band (the server posts an all-clear that removes it), and a
// transient notice for each auto-expand regrid.

/**
 * The dimensionality of what is RUNNING, not of the form. `dims` is
 * restart-only, so it can sit at 2D over a live 1D session — and everything
 * below describes that session: the boundary warning would otherwise name
 * W(x,y,px,py,t) while reporting a 1D session's x edge, and offer the 2D
 * remedy for a domain that can auto-expand. Falls back to the form only before
 * the first status arrives.
 */
const sessionNdim = computed(() =>
  session.status.value?.ndim ?? cfg.grid.ndim)
/**
 * `W(x,p,t)` — or `W(x,p,0)` while the only state that exists IS the initial
 * condition.
 *
 * THE TIME ARGUMENT IS WHAT DISTINGUISHES THE TWO FACTS, and it is the whole
 * reason this warning is now said once, in one place. The edge detector runs on
 * the IC preview and on the live record; at record 0 those are the SAME state,
 * so the fact was genuinely duplicated — and it used to be reported twice, in
 * two different sentences ("W(x,p,t) has reached…" in the header, "this IC
 * reaches…" under the preview), in two places, appearing at different times,
 * carrying the identical number. Differencing the axis sets hid the duplicate
 * but made the surviving copy flicker in and out during a run, which is worse:
 * the reader cannot tell whether a fact that comes and goes is one fact or two.
 *
 * So: one sentence, one place, and the argument says which state it is about.
 */
const wOf = computed(() =>
  `W(${axisLabels(sessionNdim.value).join(',')},${atRecordZero.value ? '0' : 't'})`)

/** Nothing has been computed beyond the Cauchy data, so the state being
 *  reported on IS the initial condition. */
const atRecordZero = computed(() =>
  (session.status.value?.record_extent?.[1] ?? -1) <= 0)

/**
 * The largest edge-band reading among the tripped axes. Naming the NUMBER
 * matters: "approaching the edge" alone says nothing about how close, and the
 * quantity is a probability — the fraction of ∫W lying in the outer cells of
 * that axis's marginal — not anything to do with the particle mass `m` in the
 * Physics panel. See core/boundary.py.
 */
const boundaryWorst = computed(() => {
  const b = session.boundary.value
  if (!b?.mass) return null
  const vals = b.axes.map((a) => b.mass[a] ?? 0)
  return vals.length ? Math.max(...vals) : null
})

/**
 * Axes the FORM's initial condition trips, straight from the IC preview.
 *
 * It reaches the header rather than being drawn under the preview because it is
 * the SAME fact the boundary watch reports, from the same detector — see wOf.
 * It is used only while the form differs from the running session: when they
 * agree, the session has already computed record 0 from this very IC and its own
 * reading is the authoritative one. That condition is `restartNeeded`, which
 * does not flicker, so neither does the choice between the two sources.
 */
const icEdge = ref<{ axis: string; mass: number }[]>([])

const icEdgeText = computed(() => {
  if (!restartNeeded.value || !icEdge.value.length) return ''
  const worst = icEdge.value.reduce((a, b) => (b.mass > a.mass ? b : a))
  const i = axisLabels(cfg.grid.ndim).indexOf(worst.axis)
  const cells = i >= 0 && cfg.grid.axes[i] ? edgeBand(cfg.grid.axes[i]!.N) : 0
  return `W(${axisLabels(cfg.grid.ndim).join(',')},0) reaches the `
    + `${icEdge.value.map((e) => e.axis).join(', ')} edge — `
    + `${worst.mass.toExponential(1)} of its integral is in the outer `
    + `${cells} cells. This is the IC in the form, not the running session — `
    + `restart to compute it.`
})

const boundaryRaw = computed(() => {
  const b = session.boundary.value
  if (!b) return ''
  const axes = b.axes.join(', ')
  const w = boundaryWorst.value
  const amount = w == null ? '' : `${w.toExponential(1)} of its integral `
  if (b.action === 'capped')
    return `${wOf.value} reached the ${axes} edge but the domain is at the ` +
           `${b.max_grid ?? ''}-cell cap (WIGNERF_MAX_GRID) — ${amount}` +
           're-entering from the opposite side'
  if (b.action === 'invalid_potential')
    return b.message ?? 'cannot expand: U is invalid on the larger domain'
  // Distinct from 'capped' on purpose: that one means plan_axis stopped at the
  // per-axis cell ceiling, this one means the guard made us give a doubling up —
  // and `limit` says which ceiling did it, because the remedies are unrelated
  // (a bigger card vs. a host setting). The server's message carries the numbers
  // either way; the fallback has to follow `limit` or it names the wrong one.
  // TWO wordings, because a denied doubling can still leave a window SHIFT to
  // commit: `applied` non-empty means a regrid went out with this, and the
  // ⤢ flash beside this line says the domain moved. One sentence covering both
  // read as a self-contradiction on screen.
  if (b.action === 'no_room') {
    const why = b.message ?? (b.limit === 'cells'
      ? 'over the host\'s total-cell rail'
      : 'not enough device memory')
    return Object.keys(b.applied ?? {}).length
      ? `${axes} could not be doubled — ${why}; the window was moved instead.`
      : `${wOf.value} reached the ${axes} edge and the domain could not be ` +
        `expanded — ${why}.`
  }
  // ONE LINE, and short enough to finish reading. This warning can clear again
  // within a couple of records as a state drifts back out of the band, so a
  // sentence that explains periodicity and offers a remedy was regularly gone
  // before it had been read. What survives is the two facts that cannot be got
  // anywhere else — WHICH axis, and HOW MUCH is in HOW MANY cells; the
  // reasoning and the remedy move to the title, which a hover can hold still.
  return `${wOf.value} has reached the ${axes} edge — ${amount}is in the ` +
         `outer ${boundaryBand.value} cells.`
})

/** Band width in cells per side, from the server (never re-derived here). */
const boundaryBand = computed(() => {
  const b = session.boundary.value
  const w = b?.band
  if (!w) return '?'
  const each = [...new Set(b.axes.map((a) => w[a]).filter((n) => n != null))]
  // one number when the tripped axes agree (they do unless their N differ)
  return each.length === 1 ? String(each[0]) : each.sort((x, y) => x - y).join('/')
})

/** The explanation the message no longer carries, on hover. */
// Held for MIN_DWELL_MS after the condition clears. The boundary watch flips
// its axis set every record or two when a packet sloshes through the edge band,
// so the raw computed appeared and vanished faster than its sentence could be
// read — see lib/sticky. Display only: boundaryRaw stays instantaneous.
// ONE TIMER PER SOURCE, combined afterwards — never sticky() over the join.
//
// These two are independent: `boundaryRaw` comes from the live boundary watch
// and flips every record or two while a packet crosses the edge band, and
// `icEdgeText` is a steady property of the form. Wrapping `a || b` in one
// sticky makes the dwell conditional on the OTHER line being empty: with an
// edited IC that also trips an edge, every clear of `boundaryRaw` leaves a
// non-empty value, which sticky correctly reads as a REPLACEMENT and applies at
// once — so two different sentences alternated at record rate with no dwell at
// all, the precise flicker sticky was added to stop. It also made the × look
// inert, since nulling session.boundary substituted the other sentence in the
// same flush. A dwell that only works when the message is alone is not a dwell.
const boundaryHeld = sticky(() => boundaryRaw.value)
const icEdgeHeld = sticky(() => icEdgeText.value)
const boundaryText = computed(() => boundaryHeld.value || icEdgeHeld.value)

/**
 * Dismissal is BY TEXT, and both halves of that matter.
 *
 * Clearing `session.boundary` alone — which is all the × used to do — cannot
 * dismiss this any more, twice over: the sentence may come from `icEdgeText` or
 * from the session's standing `status.boundary`, neither of which that touches,
 * and even for the transient event `sticky` goes on displaying the last value
 * for its dwell. The button looked inert because it was.
 *
 * Keyed on the text rather than a boolean so that dismissing one warning never
 * hides the NEXT, different one — the same rule the IC editor's strip follows.
 * The transient event is still cleared, because leaving it set would keep the
 * identical sentence alive and make the dismissal look like it did nothing the
 * moment anything else re-rendered.
 */
const boundaryDismissed = ref('')
const boundaryShown = computed(() =>
  !!boundaryText.value && boundaryText.value !== boundaryDismissed.value)

function dismissBoundary() {
  boundaryDismissed.value = boundaryText.value
  session.boundary.value = null
}

/**
 * "Your session is gone and this is a new one." Set by recover() on the 404
 * path, which used to replace a run holding computed records with an empty
 * session in silence — on screen that was only "0 / [0, 0]", with nothing to
 * say where the work went.
 *
 * Its own `sticky`, never one over a join with the other notices (see above),
 * and dismissed BY TEXT like the boundary warning so a later, different loss
 * still shows. No timer clears it: unlike paramFlash this is not a
 * confirmation of something the user just did, it is a report of work
 * destroyed, and it should still be there when they look back at the screen.
 */
const lostSession = ref('')
const lostSessionSeen = ref('')
const lostSessionHeld = sticky(() => lostSession.value)
const lostSessionShown = computed(() =>
  lostSessionHeld.value && lostSessionHeld.value !== lostSessionSeen.value
    ? lostSessionHeld.value : '')

const boundaryTitle = computed(() => {
  const b = session.boundary.value
  if (!b) return ''
  // 'no_room' gets the remedy its message deliberately leaves out — and WHICH
  // remedy depends on `limit`, because the two ceilings have nothing in common.
  // The device one is hardware and wants more of it; the cell rail is a host
  // setting, where freeing VRAM and switching to float32 both do exactly
  // nothing (a cell count does not depend on precision), so saying so is the
  // point rather than a caveat.
  if (b.action === 'no_room' && b.limit === 'cells')
    return 'The domain could not be doubled because the result is over this'
         + ' host\'s WIGNERF_MAX_CELLS_2D rail — a configured total-cell'
         + ' ceiling, not a hardware limit. Boundary detection keeps running,'
         + ' and a whole-cell window shift is still applied whenever one is'
         + ' available. Freeing the card will NOT help, and neither will'
         + ' float32: the rail counts cells, which do not depend on precision.'
         + ' Raise WIGNERF_MAX_CELLS_2D on the host, or restart on a grid with'
         + ' room to double.'
  if (b.action === 'no_room')
    return 'The domain could not be doubled because the device this session\'s'
         + ' workers are on cannot hold the result. Boundary detection keeps'
         + ' running, and a whole-cell window shift is still applied whenever'
         + ' one is available. To get the expansion: free the card (another'
         + ' session, another process), restart with fewer variants so each one'
         + ' gets more of it, restart on a roomier device, or restart in float32'
         + ' — which is half the footprint, but note it refuses auto-expand.'
  if (b.action !== 'warn') return ''
  // ONE piece of advice at either dimensionality since M3 (2026-08-01). This
  // used to read "Restart with a larger domain." in 2D, because auto-expand was
  // gated there and offering it would have been a dead end. It is not gated any
  // more, so pointing a 2D user at a restart would send them the long way round.
  // The SESSION's toggle, not the form's: this describes the run that raised the
  // warning, and the two can differ for a moment before a status echo.
  const advice = (session.status.value?.auto_expand ?? cfg.auto_expand)
    ? 'Auto-expand is on, so the domain will be regridded.'
    : 'Enable auto-expand, or restart with a larger domain.'
  // In a float32 2D session the SOLVER puts mass in the band by itself, so the
  // reading is true but this advice is not the right one. Measured 2026-07-27 on
  // the shipping 2D default, a fully contained state: 1.0e-3 of the integral in
  // the band at 32^4 after 12000 steps against 3.1e-5 for the identical state in
  // float64, latching the warning within ~7 s of wall clock, with purity down 2%
  // — a degraded run, not a domain that is too small. 48^4 stays clear and 64^4
  // flickers, so it is the COARSE grid that cannot carry single precision. Say so
  // rather than sending someone to widen a box that was never the problem.
  const f32 = session.status.value?.precision === 'float32'
              && sessionNdim.value > 1
  const cause = f32
    ? ' In single precision at a coarse 2D grid the solver itself migrates mass ' +
      'outward — measured 1e-3 of the integral at 32⁴ against 3e-5 for the same ' +
      'state in float64 — so if purity has drifted too, suspect the precision ' +
      'rather than the domain: the remedy is float64, or a finer grid.'
    : ''
  return 'The spectral domain is periodic, so whatever reaches an edge ' +
         're-enters from the opposite side and the run then evolves the wrong ' +
         '(torus) problem — the tells are a secular energy drift and a slow ' +
         `purity decay. ${advice}${cause}`
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
// float32 forbids auto-expand (single-precision noise passes its own detector)
// and applyPrecisionInvariants has already cleared the FORM. Make the LIVE
// session agree, because nothing else can: the watcher above cannot
// (status.auto_expand does not CHANGE, so it never fires), the checkbox is
// disabled in that state, and auto_expand is not in LivePhysics so there is no
// amber marker either — a session left quietly expanding behind an unchecked,
// unreachable box.
// ndim was in this key until M3 (2026-08-01) because 2D refused auto-expand too,
// and it mattered MORE there than for precision: dims is restart-only, so a
// switch could sit here for a long time with the old 1D session still running
// and still regridding. That gate is gone — a 2D session auto-expands like any
// other — so precision is the whole condition again.
// Keyed on precision, not on cfg.auto_expand, so it never duplicates the
// checkbox's own apply-live; that path leaves this alone. Covers the select, an
// import and probeHost's adoption in one place. send() no-ops on a closed
// socket, so this is safe before a session exists.
watch(() => cfg.precision, () => {
  if (cfg.precision === 'float32'
      && session.status.value?.auto_expand && session.connected.value)
    applyLive({ auto_expand: false })
})
// Live parameter changes: the Physics fields apply on change (blur/Enter)
// with no other confirmation anywhere, which reads as "nothing happened" —
// so echo what the server actually took. Labels mirror describe.FIELD_LABEL,
// U included: describe.param_lines writes U(x,y) at ndim=2 and this flash has
// to say the same words about the same change.
const PARAM_LABEL: Record<string, string> = {
  mass: 'm', c: 'c', hbar_eff: 'ℏ', tol: 'tol',
  dt_sign: 't dir', auto_expand: 'auto-expand',
}
const uLabel = computed(() => {
  const nd = sessionNdim.value
  return `U(${axisLabels(nd).slice(0, nd).join(',')})`
})
const paramFlash = ref('')
let paramTimer = 0
watch(session.paramsApplied, (p) => {
  if (!p) return
  const fmt = (k: string, v: string | number | boolean) =>
    k === 'auto_expand' ? (v ? 'on' : 'off')
    : k === 'dt_sign' ? (Number(v) < 0 ? 'backward' : 'forward')
    : String(v)
  const parts = Object.entries(p.applied).map(([k, v]) => {
    const label = k === 'U' ? uLabel.value : (PARAM_LABEL[k] ?? k)
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
  const box = g.lo
    .map((lo, i) => `[${+lo.toFixed(4)}, ${+g.hi[i]!.toFixed(4)}]`)
    .join(' × ')
  regridFlash.value = `domain ${verb} to ${box}, ${g.N.join('×')}, ` +
    `at record ${r.at_record}`
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
  // How much history is at stake, sampled BEFORE we probe: if the session
  // turns out to be gone, this is the only place that still knows.
  const hadRecords = (session.status.value?.record_extent?.[1] ?? -1) + 1
  try {
    // Probe IMMEDIATELY, then back off. Every millisecond here counts against
    // the server's detach grace, and the delay used to be paid up front for
    // nothing: the socket is already closed by the time onClose fires.
    for (let attempt = 0; ; attempt++) {
      if (attempt) await new Promise((r) => setTimeout(r, 1500))
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
    // The session is gone and we are about to build a new one. SAY SO: this
    // path silently replaced a run holding computed records with an empty
    // session, and on screen that was just "0 / [0, 0]" with nothing in the UI
    // and nothing in the log to explain where the work went.
    lostSession.value = hadRecords > 1
      ? `the previous session expired while the connection was down — ${hadRecords} computed records were discarded`
      : 'the previous session expired while the connection was down'
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
/** Installed memory per device spec, for the Setup panel's 2D footprint line.
 *  Keyed rather than reduced to a minimum: which device binds depends on how
 *  assign_devices distributes the workers — see SetupPanel.deviceLoad. */
const hostDeviceTotals = ref<Record<string, number | null> | null>(null)
/**
 * The host's ceilings PER NDIM, from /api/device.
 *
 * Deliberately not from `status`: those are resolved once, for the ndim of the
 * session that is RUNNING, while the form has to describe the ndim it is
 * SHOWING — and `dims` is restart-only, so the two disagree for exactly as long
 * as a switch waits for its restart. Taking them off `status` meant a form
 * switched to 2D over a live 1D session offered N up to 4096 against an API
 * ceiling of 128 AND hid the 2D footprint estimate entirely — the one number
 * that says whether a session can start, and it was missing precisely before
 * the first 2D restart. In the other direction a form back at 1D over a live 2D
 * session collapsed its N select to one option.
 *
 * The literals mirror config.py's defaults and are FALLBACKS only, in the spirit
 * of the precision handling below: a probe that fails costs a ceiling the API
 * still enforces with a message, never a wrong result.
 */
const hostLimits = ref({
  maxGrid: { 1: 4096, 2: 128 } as Record<number, number>,
  maxCells: { 1: null, 2: 2 ** 27 } as Record<number, number | null>,
  // per NDIM and per PRECISION, for the same reason: both are restart-only, so
  // a figure flat on either axis would make the footprint line disagree with
  // the server for as long as a switch waits for its restart — and precision
  // disagrees by 1.8x, which at 64^4 is 2.75 GiB/worker against 1.50.
  bytesPerCell: {
    1: { float64: 192, float32: 104 },
    2: { float64: 176, float32: 96 },
  } as Record<number, Record<string, number>>,
})
const historyCapMb = computed(() => {
  const b = session.status.value?.history_cap_bytes
  return b == null ? null : Math.round(b/(1024*1024))
})
/**
 * The per-cell footprint the Setup panel's line estimates from, at EITHER ndim.
 *
 * It was 2D-only, because the backend had no 1D figure to give: the device fit
 * check skipped ndim=1 on the grounds that WIGNERF_MAX_GRID already bounded a
 * 2D array. It bounds it to ~3 GiB/worker at that var's default and to ~12 GiB
 * at 8192, and an unaffordable 1D session used to start and then kill a worker
 * with a cupy OOM. Now the server refuses it, so the panel must be able to say
 * so BEFORE the restart rather than leave the user to read a 422.
 *
 * Keyed off the FORM's precision and the FORM's ndim, never the session's: both
 * are restart-only, so the panel's job is to describe what a Restart WOULD
 * create. Reading the running session's precision here would put a stale figure
 * under a form the user had just switched to float32 to make a grid fit.
 */
const footprintBytesPerCell = computed(() =>
  hostLimits.value.bytesPerCell[cfg.grid.ndim]?.[cfg.precision] ?? null)

/**
 * Host facts the form needs before it can create anything: the device
 * `choices` (the pool PLUS cpu, which is always a legal target but never
 * appears in an "auto" pool on a CUDA host), WIGNERF_PRECISION, and the per-ndim
 * grid ceilings (see `hostLimits` for why those cannot come off `status`).
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
                                     devices?: { spec: string
                                                 total_bytes?: number | null }[]
                                     precision?: string
                                     max_grid?: Record<string, number>
                                     max_cells?: Record<string, number | null>
                                     bytes_per_cell_2d?: Record<string, number>
                                     bytes_per_cell?: Record<string,
                                                             Record<string, number>>
                                   }>(
      '/device', { timeout: 5000 })
    deviceOptions.value = data.choices ?? []
    hostPool.value = data.devices?.map((d) => d.spec) ?? null
    // TOTAL, not free: a static fact the panel can compare against without
    // re-polling a number that moves under it. Keyed by spec — reducing it to
    // the pool minimum was wrong, because a session with fewer variants than
    // devices never touches the smaller cards at all.
    const totals: Record<string, number | null> = {}
    for (const d of data.devices ?? []) totals[d.spec] = d.total_bytes ?? null
    hostDeviceTotals.value = Object.keys(totals).length ? totals : null
    // Normalized key by key rather than assigned wholesale: the JSON keys are
    // strings, and an older backend (or the probe's own error path) carries
    // neither field — a missing one must keep its literal fallback, not become
    // undefined and index to NaN in the Setup panel.
    for (const nd of [1, 2]) {
      const g = data.max_grid?.[String(nd)]
      if (typeof g === 'number') hostLimits.value.maxGrid[nd] = g
      const c = data.max_cells?.[String(nd)]
      if (c === null || typeof c === 'number') hostLimits.value.maxCells[nd] = c
    }
    // Same key-by-key treatment, and for the same reason: a backend from before
    // M1 sends a bare number here, so anything that is not a per-precision
    // figure has to leave the literal fallbacks standing rather than index to
    // undefined and render the footprint line as NaN. The per-ndim key is newer
    // still (it only exists since the fit check stopped skipping 1D), so fall
    // back to the 2D-only spelling before falling back to the literals.
    for (const nd of [1, 2]) {
      const row = data.bytes_per_cell?.[String(nd)]
        ?? (nd === 2 ? data.bytes_per_cell_2d : undefined)
      for (const p of ['float64', 'float32']) {
        const v = row?.[p]
        if (typeof v === 'number') hostLimits.value.bytesPerCell[nd]![p] = v
      }
    }
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
    <!-- `relative` anchors the transient-notices overlay at the bottom of this
         bar; see the comment on it for why it must not sit in the flow. -->
    <header class="relative px-3 py-2 text-sm border-b border-line-soft
                   flex items-center gap-4 flex-wrap">
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
      <span v-if="reconnecting" class="ml-auto text-warn">
        backend disconnected — reconnecting…
      </span>
      <!-- NOT while createError is up: a failed create leaves no socket to
           connect, so "connecting…" is a false progress report on a state that
           will never change. The error line below says what actually happened. -->
      <span v-else-if="!session.connected.value && !createError"
            class="ml-auto text-warn">connecting…</span>

      <!-- A HARD failure, and the one notice that IS flow content — it must be
           readable, and it does not come and go while you watch the heatmaps.
           `basis-full` gives it its own line inside this flex-wrap header,
           which also pushes the absolute transient strip below it: the strip is
           anchored `top-full`, so with the error outside the header the two
           landed on the SAME y and the amber "setup changed — restart to
           apply" was painted straight over the error text. That pairing is not
           exotic — a restart that 422s sets BOTH, every time. -->
      <div v-if="createError" class="basis-full text-error text-sm pt-1">
        {{ createError }}
      </div>

      <!-- TRANSIENT NOTICES OVERLAY, and why it is absolutely positioned.
           These four come and go WHILE you are watching the heatmaps, and the
           header is `flex-wrap` above a `flex-1` main: inline, each arrival
           wrapped the header to a second line and moved the W panels down 32 px
           at 1280 px wide (measured), then back up when it cleared. The
           boundary warning made that pathological — it can legitimately toggle
           as a state drifts across the edge — but a single jump of the thing you
           are watching is wrong on its own. `top-full` takes them out of the
           vertical flow, so the panels NEVER move; they overlay the top ~24 px
           of the columns instead, which is where they always appeared anyway.
           Full width, so no message has to be truncated to fit a header row.
           The float32 badge stays inline on purpose: it is a permanent property
           of the session, set before there is anything to watch. -->
      <div v-if="restartNeeded || boundaryShown || paramFlash || regridFlash
                 || lostSessionShown"
           class="absolute left-0 right-0 top-full z-30 flex items-center
                  gap-4 flex-wrap px-3 py-1 text-xs
                  bg-panel/95 border-b border-line-soft shadow-sm">
        <span v-if="restartNeeded" class="text-warn">
          setup changed —
          <button class="underline" @click="requestRestart">restart</button> to apply
        </span>
        <span v-if="boundaryShown" class="text-warn" :title="boundaryTitle">
          ⚠ {{ boundaryText }}
          <button class="underline" title="dismiss (reappears if the warning changes)"
                  @click="dismissBoundary()">×</button>
        </span>
        <span v-if="paramFlash" class="text-ok">✓ {{ paramFlash }}</span>
        <span v-if="regridFlash" class="text-info">⤢ {{ regridFlash }}</span>
        <span v-if="lostSessionShown" class="text-warn"
              title="A session is held only while a client is attached, plus a
                     grace period after the socket drops. The connection was
                     down for longer than that, so the server freed its history
                     and its GPU memory, and a fresh session was created from
                     the same setup. Nothing is wrong with the setup — press
                     Solve to compute again.">
          ⚠ {{ lostSessionShown }}
          <button class="underline" title="dismiss"
                  @click="lostSessionSeen = lostSessionShown">×</button>
        </span>
      </div>
    </header>

    <!-- ================= landscape ================= -->
    <main v-if="layout === 'landscape'" class="flex-1 min-h-0 flex gap-2 p-2">
      <aside v-if="showSetup" class="w-80 shrink-0 overflow-y-auto space-y-4 pr-1 text-sm">
        <SetupPanel :cfg="cfg" :live="session.connected.value"
                    :computing="session.status.value?.computing ?? false"
                    :sign="session.status.value?.sign ?? 1"
                    :live-grid="session.status.value?.grid ?? null"
                    :live-physics="livePhysics"
                    :live-run="liveRun" :has-run="hasRun"
                    :max-grid="hostLimits.maxGrid"
                    :max-cells="hostLimits.maxCells[cfg.grid.ndim] ?? null"
                    :bytes-per-cell="footprintBytesPerCell"
                    :live-devices="session.status.value?.devices ?? null"
                    :device-options="deviceOptions" :host-pool="hostPool"
                    :device-totals="hostDeviceTotals"
                    :history-cap-mb="historyCapMb" :history-mb-max="session.status.value?.history_mb_max ?? null"
                    v-model:show-grid="showGrid"
                    v-model:show-cells="showCells"
                    @dirty="restartNeeded = true" @restart="requestRestart"
                    @apply-live="applyLive"
                    @potential-validity="(v: boolean) => potentialValid = v" />
        <ICEditor :ic="cfg.ic" :grid="cfg.grid" :hbar-eff="cfg.hbar_eff"
                  :precision="cfg.precision" :quiet="!!createError"
                  :show-grid="showGrid" :show-cells="showCells"
                  @edge="(f) => { icEdge = f }"
                  @changed="restartNeeded = true"
                  @validity="(v: boolean) => icValid = v" />
      </aside>

      <div class="flex-1 min-w-0 min-h-0">
        <PanelGrid :key="plotsKey" :frame-source="session.onFrame"
                   :variants="activeVariants" :geom="activeGeom"
                   :batch-overlay="batchOverlay" :progress="session.progress.value"
                   :show-grid="showGrid" :show-cells="showCells" />
      </div>

      <aside class="w-[380px] shrink-0 overflow-y-auto">
        <PlotsColumn :frame-source="session.onFrame" :session-id="sessionId"
                     :variants="activeVariants" :geom="activeGeom"
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
                    :max-grid="hostLimits.maxGrid"
                    :max-cells="hostLimits.maxCells[cfg.grid.ndim] ?? null"
                    :bytes-per-cell="footprintBytesPerCell"
                    :live-devices="session.status.value?.devices ?? null"
                    :device-options="deviceOptions" :host-pool="hostPool"
                    :device-totals="hostDeviceTotals"
                    :history-cap-mb="historyCapMb" :history-mb-max="session.status.value?.history_mb_max ?? null"
                    v-model:show-grid="showGrid"
                    v-model:show-cells="showCells"
                      @dirty="restartNeeded = true" @restart="requestRestart"
                      @apply-live="applyLive"
                      @potential-validity="(v: boolean) => potentialValid = v" />
        </div>
        <div class="overflow-y-auto min-h-0 pr-1">
          <ICEditor :ic="cfg.ic" :grid="cfg.grid" :hbar-eff="cfg.hbar_eff"
                    :precision="cfg.precision" :quiet="!!createError"
                    :show-grid="showGrid" :show-cells="showCells"
                  @edge="(f) => { icEdge = f }"
                  @changed="restartNeeded = true"
                    @validity="(v: boolean) => icValid = v" />
        </div>
        <div class="overflow-y-auto min-h-0 pr-1">
          <PlotsColumn :frame-source="session.onFrame" :session-id="sessionId"
                       :variants="activeVariants" :geom="activeGeom"
                       :last-frame="session.lastFrame.value"
                       :batch-computing="batchComputing"
                       :show-grid="showGrid" :plots-key="plotsKey" />
        </div>
      </div>

      <div class="flex-1 min-w-0 min-h-0">
        <PanelGrid :key="plotsKey" :frame-source="session.onFrame"
                   :variants="activeVariants" :geom="activeGeom"
                   :batch-overlay="batchOverlay" :progress="session.progress.value"
                   :show-grid="showGrid" :show-cells="showCells" />
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
      :readout-source="session.readoutSource.value"
      :setup-valid="setupValid"
      @command="sendCommand"
    />
  </div>
</template>
