/**
 * A message that, once shown, stays shown long enough to be read.
 *
 * Several notices in this app are derived from live state and therefore come
 * and go at the rate that state changes rather than at the rate a person reads.
 * The boundary watch is the clearest case: a packet sloshing in and out of the
 * edge band flips its axis set every record or two, and the warning it drives
 * appeared and vanished faster than a sentence can be finished — measured
 * flipping on and off repeatedly inside a 10 s window on a coarse grid.
 *
 * This holds the last non-empty value for at least `ms` after it appears. A
 * REPLACEMENT is not delayed — a newer message is more useful than an older one
 * and simply restarts the clock — so the only thing deferred is the clearing.
 *
 * Note this is display-only. The underlying state stays instantaneous, which is
 * what the transport, the Solve gate and every test read; nothing here can make
 * a stale fact act on anything.
 */
import { onScopeDispose, ref, watch, type Ref } from 'vue'

/** How long any transient notice stays up. Long enough to read one sentence. */
export const MIN_DWELL_MS = 5000

export function sticky(source: () => string, ms = MIN_DWELL_MS): Ref<string> {
  const out = ref(source())
  let shownAt = out.value ? Date.now() : 0
  let timer: ReturnType<typeof setTimeout> | null = null

  const clearTimer = () => {
    if (timer !== null) { clearTimeout(timer); timer = null }
  }

  watch(source, (v) => {
    clearTimer()
    if (v) {
      // a new message supersedes the old one immediately and restarts the clock
      out.value = v
      shownAt = Date.now()
      return
    }
    const left = shownAt + ms - Date.now()
    if (left <= 0) { out.value = ''; return }
    timer = setTimeout(() => { out.value = ''; timer = null }, left)
  })

  onScopeDispose(clearTimer)
  return out
}
