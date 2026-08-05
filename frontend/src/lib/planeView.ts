/**
 * What a panel is showing, as the server needs to hear it — the request half of
 * backend display downsampling (backend/core/planeview.py picks the pyramid
 * level and window from this).
 *
 * PHYSICAL extents, not fractions of the domain, because that is what survives
 * an auto-expand regrid: "this region of phase space" keeps its meaning when
 * the domain doubles under it, where a fraction silently comes to mean
 * somewhere else. It is also why a scrub across a regrid boundary needs no
 * bookkeeping — the server answers each record against its own geometry.
 */

export interface PlaneViewReq {
  vid: number
  a: number
  b: number
  a1: number
  a2: number
  b1: number
  b2: number
  /** device pixels along each axis; the server snaps them to a power of two */
  na: number
  nb: number
}

/**
 * Stable key for one panel's request. (variant, plane) and not the panel's grid
 * position: in `variants` mode four panels show ONE plane of four variants, and
 * in the phase portrait six panels show six planes of one — only the pair
 * identifies a request in both.
 */
export function viewKey(v: PlaneViewReq): string {
  return planeKey(v.vid, v.a, v.b)
}

/** The same key from the parts, for a panel that has nothing to request yet. */
export function planeKey(vid: number, a: number, b: number): string {
  return `${vid}:${a},${b}`
}

/**
 * Whether two requests differ enough to be worth re-sending.
 *
 * Every request costs a re-send of the current record at the new resolution, so
 * the pointer-move stream behind a pan must not become a request stream. The
 * comparison is deliberately in terms of what the SERVER would do with them:
 * the pixel counts get snapped to a power of two there, so a resize of a few
 * pixels changes nothing, and a window that moved by less than a per-cent of
 * its own width lands on the same base cell.
 */
export function viewChanged(a: PlaneViewReq | undefined, b: PlaneViewReq): boolean {
  if (!a) return true
  if (pow2(a.na) !== pow2(b.na) || pow2(a.nb) !== pow2(b.nb)) return true
  const wa = Math.abs(b.a2 - b.a1)
  const wb = Math.abs(b.b2 - b.b1)
  if (Math.abs(wa - Math.abs(a.a2 - a.a1)) > 0.01 * wa) return true
  if (Math.abs(wb - Math.abs(a.b2 - a.b1)) > 0.01 * wb) return true
  return Math.abs(a.a1 - b.a1) > 0.01 * wa || Math.abs(a.b1 - b.b1) > 0.01 * wb
}

function pow2(v: number): number {
  return 1 << Math.max(0, 31 - Math.clz32(Math.max(1, Math.round(v))))
}

/**
 * The next viewport to send: what was sent last, updated by this burst, minus
 * the panels that have gone away.
 *
 * A MERGE and not a replacement, because the server reads one message as the
 * COMPLETE picture — a panel missing from it is one this client is not showing,
 * and stops being sent at all. Bursts, though, are routinely partial: a frame
 * paint re-registers every panel at once, but an unlinked zoom or a single
 * panel's resize speaks for one. Replacing the map with that burst retracted
 * the other three panels' planes, which then arrived header-only.
 *
 * `live` is keyed by viewKey, so a panel that has just unmounted is dropped
 * rather than kept alive by a stale entry — otherwise the server keeps cropping
 * and sending a plane nothing is drawing.
 */
export function mergeViews(sent: Map<string, PlaneViewReq>,
                           pending: Map<string, PlaneViewReq>,
                           live: Set<string>): Map<string, PlaneViewReq> {
  const next = new Map<string, PlaneViewReq>()
  for (const k of live) {
    const v = pending.get(k) ?? sent.get(k)
    if (v) next.set(k, v)
  }
  return next
}

/** Whether `next` is worth sending in place of `sent` (see viewChanged). */
export function viewsChanged(sent: Map<string, PlaneViewReq>,
                             next: Map<string, PlaneViewReq>): boolean {
  if (sent.size !== next.size) return true
  for (const [k, v] of next) {
    if (viewChanged(sent.get(k), v)) return true
  }
  return false
}
