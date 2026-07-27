/**
 * Per-plot variant visibility preferences (display-only — never touches the
 * computed variant set or restart logic). One localStorage object keyed by
 * plot id; stale plot/variant entries are kept so preferences survive
 * variant-set changes and come back when a variant is re-enabled.
 */
import type { VariantKey } from './variants'

/**
 * Plot ids. The marginals are keyed by AXIS index and the uncertainty products
 * by spatial DIMENSION, because a 2D run has four of the former and two of the
 * latter. The old 'rho'/'phi' keys simply stop matching, which costs a user
 * their hidden-series choices once and nothing else.
 */
export type PlotId =
  | `marg${number}`
  | 'E'
  | `uncertainty${number}`
  | 'purity'
  | 'lz'

const KEY = 'wignerf.hiddenSeries'

type Stored = Partial<Record<PlotId, VariantKey[]>>

function readAll(): Stored {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? '{}') as unknown
    return typeof v === 'object' && v !== null ? (v as Stored) : {}
  } catch {
    return {}
  }
}

export function loadHidden(id: PlotId): Set<VariantKey> {
  const v = readAll()[id]
  return new Set(Array.isArray(v) ? v : [])
}

export function saveHidden(id: PlotId, hidden: Set<VariantKey>): void {
  const all = readAll()
  all[id] = [...hidden]
  localStorage.setItem(KEY, JSON.stringify(all))
}
