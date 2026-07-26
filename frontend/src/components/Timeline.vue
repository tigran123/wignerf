<script setup lang="ts">
import { computed, onUnmounted, ref } from 'vue'
import type { SessionStatus } from '../composables/useSession'
import { perfRates } from '../lib/perf'

const props = defineProps<{
  status: SessionStatus | null
  currentRecord: number
}>()

const emit = defineEmits<{
  (e: 'seek', record: number): void
}>()

const extent = computed(() => props.status?.record_extent ?? [-1, -1])

// in-RAM history usage vs the WIGNERF_HISTORY_MB cap — the retained
// (scrubbable) window IS this buffer, so the timeline is its natural home.
// The DEVICES ride along: both are steady-state facts about the running
// session, and this readout already costs a line that is drawn anyway. The
// Setup panel used to repeat them in a standing "running on …, history cap …"
// paragraph, which spent vertical space in a narrow column to say what is
// visible here for free.
const hist = computed(() => {
  const st = props.status
  if (!st || st.history_cap_bytes == null) return ''
  const gib = (b: number) => {
    const g = b / 2 ** 30
    return g >= 10 ? g.toFixed(0) : g >= 0.1 ? g.toFixed(1) : g.toFixed(2)
  }
  const dev = st.devices?.length ? `  ·  dev: ${st.devices.join(', ')}` : ''
  return `hist ${gib(st.history_bytes)} / ${gib(st.history_cap_bytes)} GiB${dev}`
})
// Frame rate, polled off the perf counters. TWO numbers, because at large
// grids they diverge and the difference is the point: painted/s is what the
// eye gets, received/s is what the server delivered. painted < received means
// useSession's queue is collapsing to newest — i.e. records are being SKIPPED,
// which reads on screen as fast playback when it is really loss. One number
// alone would keep that indistinguishable from genuine speed.
const rates = ref({ painted: 0, received: 0, dropped: 0 })
const timer = window.setInterval(() => { rates.value = perfRates() }, 300)
onUnmounted(() => window.clearInterval(timer))
const fps = computed(() => {
  const r = rates.value
  if (r.painted < 0.05 && r.received < 0.05) return ''   // idle: stay quiet
  return `${r.painted.toFixed(1)}/${r.received.toFixed(1)} fps`
})
const dropping = computed(() => rates.value.dropped > 0.5)

const span = computed(() => Math.max(1, extent.value[1] - extent.value[0]))
const cursorPct = computed(() => {
  if (extent.value[1] < 0) return 0
  return (100 * (props.currentRecord - extent.value[0])) / span.value
})

function click(ev: MouseEvent) {
  if (extent.value[1] < 0) return
  const el = ev.currentTarget as HTMLElement
  const frac = (ev.clientX - el.getBoundingClientRect().left) / el.clientWidth
  const rec = Math.round(extent.value[0] + frac * span.value)
  emit('seek', rec)
}
</script>

<template>
  <div
    class="relative h-4 mx-3 my-1 bg-raised rounded cursor-pointer select-none"
    title="click to seek"
    @click="click"
  >
    <!-- computed (retained) extent fill -->
    <div class="absolute inset-y-0 left-0 bg-track rounded"
         :style="{ width: '100%' }" v-if="extent[1] >= 0"></div>
    <!-- cursor -->
    <div class="absolute inset-y-0 w-0.5 bg-info"
         :style="{ left: cursorPct + '%' }" v-if="extent[1] >= 0"></div>
    <!-- Both readouts sit ON the fill bar, so they need weight and a halo to
         stay legible (at 10px/text-fg-3 they were unreadable). The halo is
         --wf-label-shadow: it inverts with the theme, or it would smear the
         text instead of separating it from the bar. -->
    <div class="absolute right-1 top-0 flex items-center gap-2 text-[11px] leading-4
                font-medium text-fg tabular-nums
                [text-shadow:var(--wf-label-shadow)]"
         v-if="extent[1] >= 0">
      <!-- fixed width: the rate changes 3x/s and must not shove the record
           counter sideways as digits come and go -->
      <span v-if="fps" class="inline-block w-[6.5rem] text-right"
            :class="dropping ? 'text-warn-2' : 'text-fg-2'"
            :title="dropping
              ? 'painted/s / received/s — the client cannot paint as fast as the server delivers, so it is DROPPING records (queue collapses to newest). The animation runs ahead by skipping.'
              : 'painted/s / received/s — equal means every delivered record reached the screen'">
        {{ fps }}
      </span>
      <span>{{ currentRecord }} / [{{ extent[0] }}, {{ extent[1] }}]</span>
    </div>
    <div class="absolute left-1 top-0 text-[11px] leading-4 font-medium text-fg-2
                tabular-nums [text-shadow:var(--wf-label-shadow)]"
         v-if="hist"
         title="in-RAM frame history used / cap (this session's history_mb, bounded by WIGNERF_HISTORY_MB) — the oldest records evict when full, shrinking the scrubbable window. dev: the device(s) this session's variant workers run on.">
      {{ hist }}
    </div>
  </div>
</template>
