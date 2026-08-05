/**
 * Binary frame-bundle decoder — the TypeScript mirror of
 * backend/core/protocol.py (see its docstring for the layout). Verified
 * against python-generated golden fixtures in protocol.test.ts, one 1D and
 * one 2D.
 *
 * A record carries a LIST OF 2D PLANES, not the state: W(x,y,px,py) can be
 * neither drawn nor sent, so a 2D session streams the six pairwise
 * projections. At ndim=1 the list holds exactly one plane whose complement is
 * empty — W itself — so 1D is the general case here, not a special one.
 *
 * Since v5 a plane is a WINDOW: the client tells the server which physical
 * region each panel shows and at what pixel size, and gets back a crop
 * decimated to match. So the plane's dimensions are on the wire and must NOT
 * be re-derived from the header's N (which stays the computed grid, and is
 * still what the marginals and the axis overlays are indexed by).
 *
 * Every section length is a multiple of 4, so every TypedArray below is a
 * zero-copy view into the received ArrayBuffer.
 */

export const MAGIC = 0x57
// v5: per-plane window (size/offset/step) + natural order, so a plane can be a
// decimated crop of the zoom window rather than the whole axis pair; v4 added
// per-record ndim and the plane list (1D is one plane = W)
export const VERSION = 5
export const MSG_FRAME = 1

export const FLAG_LIVE_PREVIEW = 1 << 0
export const FLAG_REPLAY = 1 << 1

/** Reduction kind of a plane. Cuts are milestone M4/M5 — see CLAUDE.md. */
export const MODE_PROJECTION = 0

export interface PlaneFrame {
  /** phase-space axis indices of this plane, a < b (see lib/axes.ts) */
  a: number
  b: number
  mode: number // MODE_PROJECTION
  wmin: number
  wmax: number
  na: number // rows SENT along axis a (not N[a] — this may be a crop)
  nb: number // columns sent along axis b
  /**
   * The window this plane covers, in BASE cells of the record's own axes:
   * sample (i, j) is the mean of base cells [off[0] + i*step[0], +step[0]) x
   * [off[1] + j*step[1], +step[1]). A whole plane is off 0, step 1.
   */
  off: [number, number]
  step: [number, number]
  /** (na, nb) row-major, NATURAL order. Empty when na === 0 — that plane was
   *  not requested and carries no payload. */
  data: Uint16Array
}

export interface VariantFrame {
  vid: number // bit0 quantum, bit1 relativistic
  dt: number
  E: number
  purity: number // gamma = (2*pi*hbar)^ndim * int W^2 = Tr rho^2
  lz: number // <x*py - y*px>; 0 at ndim=1, where it is undefined
  mean: number[] // one per phase-space axis
  std: number[]
  planes: PlaneFrame[] // in lib/axes.ts PLANES order
  marg: Float32Array[] // one per axis, natural order
}

export interface Frame {
  record: number
  t: number
  // grid geometry is a per-record fact: auto-expand may move/double the
  // domain mid-run, and a replayed record keeps the grid it was computed on
  ndim: number
  N: number[] // 2*ndim axis counts
  lo: number[]
  hi: number[]
  flags: number
  variants: VariantFrame[]
}

export function decodeFrame(buf: ArrayBuffer): Frame {
  const dv = new DataView(buf)
  if (dv.getUint8(0) !== MAGIC) throw new Error('bad magic')
  const version = dv.getUint8(1)
  if (version !== VERSION) throw new Error(`protocol version ${version} != ${VERSION}`)
  if (dv.getUint8(2) !== MSG_FRAME) throw new Error('unexpected msg_type')
  const nVariants = dv.getUint8(3)
  const record = dv.getUint32(4, true)
  const t = dv.getFloat64(8, true)
  const flags = dv.getUint32(16, true)
  const ndim = dv.getUint8(20)
  const nMarg = dv.getUint8(22)
  const nax = 2 * ndim
  if (nMarg !== nax) throw new Error(`ndim ${ndim} wants ${nax} marginals, got ${nMarg}`)

  let off = 24
  const N: number[] = []
  for (let a = 0; a < nax; a++) N.push(dv.getUint32(off + 4 * a, true))
  off += 4 * nax
  const lo: number[] = []
  const hi: number[] = []
  for (let a = 0; a < nax; a++) {
    lo.push(dv.getFloat64(off + 16 * a, true))
    hi.push(dv.getFloat64(off + 16 * a + 8, true))
  }
  off += 16 * nax

  const variants: VariantFrame[] = []
  for (let i = 0; i < nVariants; i++) {
    const vid = dv.getUint8(off)
    const nPlanes = dv.getUint8(off + 1)
    const dt = dv.getFloat32(off + 4, true)
    const E = dv.getFloat32(off + 8, true)
    const purity = dv.getFloat32(off + 12, true)
    const lz = dv.getFloat32(off + 16, true)
    off += 20
    const mean: number[] = []
    const std: number[] = []
    for (let a = 0; a < nax; a++) {
      mean.push(dv.getFloat32(off + 8 * a, true))
      std.push(dv.getFloat32(off + 8 * a + 4, true))
    }
    off += 8 * nax

    const planes: PlaneFrame[] = []
    for (let j = 0; j < nPlanes; j++) {
      const a = dv.getUint8(off)
      const b = dv.getUint8(off + 1)
      const mode = dv.getUint8(off + 2)
      const wmin = dv.getFloat32(off + 4, true)
      const wmax = dv.getFloat32(off + 8, true)
      // The sizes come off the WIRE, not from the header's N: a plane may be a
      // decimated crop of the zoom window, and na === 0 means it was not
      // requested at all (header present, no payload, so the plane list keeps
      // its canonical order).
      const na = dv.getUint32(off + 12, true)
      const nb = dv.getUint32(off + 16, true)
      const off0 = dv.getUint32(off + 20, true)
      const off1 = dv.getUint32(off + 24, true)
      const step0 = dv.getUint32(off + 28, true)
      const step1 = dv.getUint32(off + 32, true)
      off += 36
      const data = new Uint16Array(buf, off, na * nb)
      off += 2 * na * nb
      planes.push({ a, b, mode, wmin, wmax, na, nb,
                    off: [off0, off1], step: [step0, step1], data })
    }
    const marg: Float32Array[] = []
    for (let a = 0; a < nax; a++) {
      marg.push(new Float32Array(buf, off, N[a]!))
      off += 4 * N[a]!
    }
    variants.push({ vid, dt, E, purity, lz, mean, std, planes, marg })
  }
  if (off !== buf.byteLength) throw new Error(`trailing bytes: ${off} != ${buf.byteLength}`)
  return { record, t, ndim, N, lo, hi, flags, variants }
}

export function variantName(vid: number): string {
  return (vid & 1 ? 'quantum' : 'classical') + ' ' +
    (vid & 2 ? 'relativistic' : 'non-relativistic')
}
