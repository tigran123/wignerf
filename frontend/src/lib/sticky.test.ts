import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { effectScope, ref } from 'vue'

import { MIN_DWELL_MS, sticky } from './sticky'

/** sticky() uses a watcher, so it needs a scope to be created in and flushed. */
async function withScope(fn: () => void | Promise<void>) {
  const scope = effectScope()
  await scope.run(fn as () => Promise<void>)
  scope.stop()
}

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { vi.useRealTimers() })

describe('a transient notice stays long enough to be read', () => {
  it('holds a message for the dwell after the condition clears', async () => {
    await withScope(async () => {
      const src = ref('')
      const out = sticky(() => src.value)
      src.value = '⚠ reached the x edge'
      await vi.advanceTimersByTimeAsync(0)
      expect(out.value).toBe('⚠ reached the x edge')

      // the underlying state flickers off almost at once, as the boundary
      // watch really does when a packet sloshes through the edge band
      src.value = ''
      await vi.advanceTimersByTimeAsync(100)
      expect(out.value).toBe('⚠ reached the x edge')

      await vi.advanceTimersByTimeAsync(MIN_DWELL_MS)
      expect(out.value).toBe('')
    })
  })

  it('shows a NEW message at once rather than queueing it', async () => {
    // A newer message is more useful than an older one; only the CLEARING is
    // deferred. Queueing would put the display further and further behind.
    await withScope(async () => {
      const src = ref('first')
      const out = sticky(() => src.value)
      await vi.advanceTimersByTimeAsync(0)
      src.value = 'second'
      await vi.advanceTimersByTimeAsync(0)
      expect(out.value).toBe('second')
    })
  })

  it('restarts the clock on a replacement', async () => {
    await withScope(async () => {
      const src = ref('first')
      const out = sticky(() => src.value)
      await vi.advanceTimersByTimeAsync(MIN_DWELL_MS - 200)
      src.value = 'second'
      await vi.advanceTimersByTimeAsync(0)
      src.value = ''
      // the first message's clock has nearly run out, but the second's has not
      await vi.advanceTimersByTimeAsync(MIN_DWELL_MS - 200)
      expect(out.value).toBe('second')
      await vi.advanceTimersByTimeAsync(300)
      expect(out.value).toBe('')
    })
  })

  it('clears immediately when the message has already been up long enough', async () => {
    await withScope(async () => {
      const src = ref('shown')
      const out = sticky(() => src.value)
      await vi.advanceTimersByTimeAsync(MIN_DWELL_MS + 1000)
      src.value = ''
      await vi.advanceTimersByTimeAsync(0)
      expect(out.value).toBe('')
    })
  })

  it('flicker cannot outrun it — a message flipping every 100ms stays up', async () => {
    await withScope(async () => {
      const src = ref('')
      const out = sticky(() => src.value)
      for (let i = 0; i < 20; i++) {
        src.value = i % 2 ? '' : '⚠ edge'
        await vi.advanceTimersByTimeAsync(100)
        expect(out.value).toBe('⚠ edge')
      }
    })
  })
})

describe('the dwell must survive the way it is USED', () => {
  it('a scope that stops with a pending clear leaves no timer behind', async () => {
    const src = ref('⚠ edge')
    const scope = effectScope()
    let out!: ReturnType<typeof sticky>
    await scope.run(async () => {
      out = sticky(() => src.value)
      src.value = ''
      await vi.advanceTimersByTimeAsync(0)
    })
    // the clear is scheduled, not applied
    expect(out.value).toBe('⚠ edge')
    expect(vi.getTimerCount()).toBe(1)
    scope.stop()
    // onScopeDispose(clearTimer) — a component unmounted mid-dwell must not
    // leave a callback pointing at a ref nobody is watching any more
    expect(vi.getTimerCount()).toBe(0)
    await vi.advanceTimersByTimeAsync(MIN_DWELL_MS + 1000)
    expect(out.value).toBe('⚠ edge')
  })

  it('ONE sticky over two joined sources has no dwell at all', async () => {
    // The rule this pins: when several lines share a strip, the FLICKERING one
    // gets its own timer. Joining them into a single sticky string means that
    // losing one line still leaves a non-empty value, which sticky correctly
    // reads as a REPLACEMENT and applies at once — so the dwell silently stops
    // working the moment the other line is present. This shipped once, in
    // SimulatorView's header: sticky(() => boundaryRaw || icEdgeText).
    await withScope(async () => {
      const flickering = ref('⚠ live: reached the x edge')
      const steady = ref('⚠ this IC reaches the p edge')
      const joined = sticky(() => flickering.value || steady.value)
      await vi.advanceTimersByTimeAsync(0)
      expect(joined.value).toBe('⚠ live: reached the x edge')

      flickering.value = ''                     // one record later
      await vi.advanceTimersByTimeAsync(0)
      // ...and the sentence changed INSTANTLY, well inside the dwell
      expect(joined.value).toBe('⚠ this IC reaches the p edge')
    })
  })

  it('one sticky per source keeps the dwell for each', async () => {
    await withScope(async () => {
      const flickering = ref('⚠ live: reached the x edge')
      const steady = ref('⚠ this IC reaches the p edge')
      const a = sticky(() => flickering.value)
      const b = sticky(() => steady.value)
      const shown = () => a.value || b.value
      await vi.advanceTimersByTimeAsync(0)
      expect(shown()).toBe('⚠ live: reached the x edge')

      flickering.value = ''
      await vi.advanceTimersByTimeAsync(100)
      expect(shown()).toBe('⚠ live: reached the x edge')   // held
      await vi.advanceTimersByTimeAsync(MIN_DWELL_MS)
      expect(shown()).toBe('⚠ this IC reaches the p edge')  // then hands over
    })
  })
})
