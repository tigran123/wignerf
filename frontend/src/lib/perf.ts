/**
 * Lightweight runtime counters for the frame pipeline, exposed as
 * window.__wfPerf — the same debug-surface pattern as window.__wfSeries.
 * Every hook is cheap enough to stay compiled in permanently:
 *
 *   __wfPerf.reset()    — zero the counters (bracket a measurement)
 *   __wfPerf.snapshot() — rates, per-stage avg/total ms, environment info
 *
 * This module also owns the measured display refresh interval: the "0"
 * position of the delay dial maps to it, so playback at "0" delivers one
 * record per display refresh — the fastest speed at which every frame is
 * still painted.
 */

interface Stage { ms: number; n: number }

let stages: Record<string, Stage> = {}
let msgs = 0
let bytes = 0
let frames = 0
let drops = 0
let t0 = performance.now()

export function perfMsg(nbytes: number) { msgs++; bytes += nbytes; wMsgs++ }
export function perfFrame() { frames++; wFrames++ }
export function perfDrop(n: number) { drops += n; wDrops += n }

/**
 * Sliding-window rates for the ON-SCREEN readout. The counters above are
 * cumulative-since-reset (for bracketing a __wfPerf measurement); a readout
 * needs an instantaneous rate instead.
 *
 * Both rates are reported because they answer different questions, and the
 * gap between them is the whole story at large grids: `received` is how fast
 * the server delivers records, `painted` is how fast they reach the screen.
 * When painted < received the client is DROPPING (useSession's queue collapses
 * to newest), which looks like fast playback while it is really skipping —
 * the readout must not let that pass for speed.
 */
let wT0 = performance.now()
let wFrames = 0
let wMsgs = 0
let wDrops = 0
let rPainted = 0
let rReceived = 0
let rDropped = 0

export function perfRates(): { painted: number; received: number; dropped: number } {
  const now = performance.now()
  const dt = (now - wT0) / 1000
  // ~3 updates/s: long enough to be steady, short enough to track a stall
  if (dt >= 0.35) {
    rPainted = wFrames / dt
    rReceived = wMsgs / dt
    rDropped = wDrops / dt
    wT0 = now
    wFrames = wMsgs = wDrops = 0
  }
  return { painted: rPainted, received: rReceived, dropped: rDropped }
}

export function perfStage(name: string, ms: number) {
  const s = (stages[name] ??= { ms: 0, n: 0 })
  s.ms += ms
  s.n++
}

/** One-time environment facts (GL renderer string, refresh interval). */
export const perfInfo: Record<string, string | number> = {}

function reset() {
  stages = {}
  msgs = bytes = frames = drops = 0
  t0 = performance.now()
}

function snapshot() {
  const dt = Math.max((performance.now() - t0) / 1000, 1e-9)
  const avg: Record<string, number> = {}
  const total: Record<string, number> = {}
  for (const [k, s] of Object.entries(stages)) {
    avg[k] = +(s.ms / Math.max(s.n, 1)).toFixed(3)
    total[k] = +s.ms.toFixed(1)
  }
  return {
    seconds: +dt.toFixed(2),
    received_per_s: +(msgs / dt).toFixed(1),
    mib_per_s: +(bytes / dt / 1048576).toFixed(2),
    painted_per_s: +(frames / dt).toFixed(1),
    queue_drops: drops,
    stage_avg_ms: avg,
    stage_total_ms: total,
    info: { ...perfInfo },
  }
}

;(window as unknown as Record<string, unknown>).__wfPerf = { reset, snapshot }

/**
 * Display refresh interval in seconds. Measured once at app start (median
 * of 20 rAF deltas, clamped to [1/240, 1/24] against hidden-tab stalls);
 * the 60 Hz default applies until the measurement lands (~1/3 s).
 */
let refresh = 1 / 60
export function displayInterval(): number { return refresh }
export function measureDisplayInterval() {
  const ts: number[] = []
  const tick = (t: number) => {
    ts.push(t)
    if (ts.length < 21) { requestAnimationFrame(tick); return }
    const d = ts.slice(1).map((v, i) => v - ts[i]).sort((a, b) => a - b)
    refresh = Math.min(Math.max(d[Math.floor(d.length / 2)] / 1000, 1 / 240), 1 / 24)
    perfInfo.refresh_interval_ms = +(refresh * 1000).toFixed(2)
  }
  requestAnimationFrame(tick)
}
