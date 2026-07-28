/**
 * WHAT THE W PANELS SHOW — shared state, not PanelGrid's private business.
 *
 * At ndim=1 there is one plane and the grid is the familiar one panel per
 * variant. At ndim=2 there are six planes per variant, i.e. up to 24
 * combinations, so the user picks one of two readings:
 *
 *  'variants' — one selected plane across every variant. The quantum-vs-
 *               classical comparison this whole UI was built for.
 *  'phase'    — every plane of one selected variant: that state's full phase
 *               portrait. Called "phase portrait" in the UI, never just
 *               "portrait", which is already the name of a LAYOUT orientation
 *               two controls away in the same header.
 *
 * This lives in a module singleton for the reason lib/theme.ts does: more than
 * one component needs it. PanelGrid owns the controls, and the Export panel
 * SEEDS its own plane/variant choice from here when it opens — so "Render"
 * films what you are looking at, exactly as ExportSpec.theme already follows
 * the app's own toggle. Like the theme, the export may then override it per
 * job without writing back.
 */
import { ref, watch } from 'vue'

export type ViewMode = 'variants' | 'phase'

const MODE_KEY = 'wignerf.panelMode'
const PLANE_KEY = 'wignerf.panelPlane'

// 'portrait' was the stored value before the rename; migrate rather than
// silently resetting someone's choice
const storedMode = localStorage.getItem(MODE_KEY)
export const panelMode = ref<ViewMode>(
  storedMode === 'phase' || storedMode === 'portrait' ? 'phase' : 'variants')

/** Index into axes.planes(ndim). A stored index can outlive a switch back to
 *  1D, where only plane 0 exists — every reader clamps. */
export const panelPlaneIdx = ref(
  Math.max(0, Number(localStorage.getItem(PLANE_KEY)) || 0))

/** Which variant the phase portrait shows. Deliberately NOT persisted: it is
 *  an index into a variant set that a restart can change out from under it. */
export const panelVariantIdx = ref(0)

watch(panelMode, (v) => localStorage.setItem(MODE_KEY, v))
watch(panelPlaneIdx, (v) => localStorage.setItem(PLANE_KEY, String(v)))
