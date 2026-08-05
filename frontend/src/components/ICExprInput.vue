<script setup lang="ts">
/**
 * The expression box for the W(x,p) and ψ(x) initial-condition tabs.
 *
 * A CONTROLLED input: it owns no draft of its own. The draft lives in ICEditor
 * because that is what has to put it in the preview REQUEST — see the note
 * there. Everything this component knows is what to render.
 */
import { computed } from 'vue'

import { exprPlaceholder, exprVariables, type ICExprKind } from '../lib/icKinds'

const props = defineProps<{
  kind: ICExprKind
  ndim: number
  /** the draft differs from what the server last accepted */
  pending: boolean
}>()

const draft = defineModel<string>({ required: true })

const name = computed(() =>
  `${props.kind === 'wexpr' ? 'W' : 'ψ'}(${exprVariables(props.kind, props.ndim).join(', ')})`)

/**
 * The pending marker means something narrower here than on the potential's
 * input: the session status carries no IC at all, so there is no "differs from
 * the running session" amber to be had, and no Apply-live button to enable.
 * It says only that what you are LOOKING at is not what you typed.
 */
const pendingTitle = 'not yet accepted — the preview below is still the last '
  + 'expression the server could build'
</script>

<template>
  <div class="space-y-1">
    <label class="flex items-center gap-1 text-xs">
      <span class="text-muted whitespace-nowrap">{{ name }} =</span>
    </label>
    <input v-model="draft" type="text" spellcheck="false"
           class="wf-num w-full font-mono text-sm"
           :class="pending && 'wf-pending'"
           :title="pending ? pendingTitle : ''"
           :placeholder="exprPlaceholder(kind, ndim)" />
  </div>
</template>
