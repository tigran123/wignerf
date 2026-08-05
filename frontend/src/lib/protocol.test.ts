import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { FLAG_REPLAY, MODE_PROJECTION, decodeFrame } from './protocol'
import { planes as planesOf } from './axes'

const here = dirname(fileURLToPath(import.meta.url))

function load(name: string) {
  const bin = readFileSync(join(here, '__fixtures__', `${name}.bin`))
  const meta = JSON.parse(
    readFileSync(join(here, '__fixtures__', `${name}.json`), 'utf8'),
  )
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)
  return { buf: buf as ArrayBuffer, meta }
}

// Both goldens: a 1D record (one plane = W, two marginals) and a 2D one (six
// planes, four marginals). A 1D-only fixture would let a 2D decode bug ship,
// and both use anisotropic axis counts so a transposed index cannot pass.
describe.each([['frame', 1], ['frame2d', 2]])(
  'decodeFrame (%s)',
  (name, ndim) => {
    const { buf, meta } = load(name)

    it('matches the python-generated golden bundle', () => {
      const f = decodeFrame(buf)
      expect(f.record).toBe(meta.record)
      expect(f.t).toBeCloseTo(meta.t, 12)
      expect(f.ndim).toBe(ndim)
      expect(f.N).toEqual(meta.N)
      expect(f.lo).toEqual(meta.lo)
      expect(f.hi).toEqual(meta.hi)
      expect(f.flags).toBe(FLAG_REPLAY)
      expect(f.variants.length).toBe(meta.variants.length)

      for (let i = 0; i < f.variants.length; i++) {
        const v = f.variants[i]!
        const m = meta.variants[i]
        expect(v.vid).toBe(m.vid)
        for (const k of ['E', 'purity', 'dt', 'lz'] as const) {
          expect(v[k]).toBeCloseTo(m[k], 6)
        }
        expect(v.mean.length).toBe(2 * ndim)
        expect(v.std.length).toBe(2 * ndim)
        for (let a = 0; a < 2 * ndim; a++) {
          expect(v.mean[a]).toBeCloseTo(m.mean[a], 6)
          expect(v.std[a]).toBeCloseTo(m.std[a], 6)
        }

        // planes arrive in the canonical order lib/axes.ts declares
        expect(v.planes.map((p) => [p.a, p.b])).toEqual(
          planesOf(ndim).map((p) => [p[0], p[1]]),
        )
        expect(v.planes.length).toBe(m.planes.length)
        for (let j = 0; j < v.planes.length; j++) {
          const p = v.planes[j]!
          const pm = m.planes[j]
          expect([p.a, p.b, p.mode]).toEqual([pm.a, pm.b, MODE_PROJECTION])
          expect(p.wmin).toBeCloseTo(pm.wmin, 6)
          expect(p.wmax).toBeCloseTo(pm.wmax, 6)
          // Sizes come off the WIRE, never from the header's N — a plane may
          // be a decimated crop (the 1D fixture is one, with a different
          // off/step on each axis so a swapped pair cannot pass).
          expect([p.na, p.nb]).toEqual([pm.na, pm.nb])
          expect(p.off).toEqual(pm.off)
          expect(p.step).toEqual(pm.step)
          expect(p.data.length).toBe(pm.na * pm.nb)
          expect(Array.from(p.data)).toEqual(pm.wq)
          // the window has to stay inside the axis it is a window of
          expect(p.off[0] + p.na * p.step[0]).toBeLessThanOrEqual(meta.N[pm.a])
          expect(p.off[1] + p.nb * p.step[1]).toBeLessThanOrEqual(meta.N[pm.b])
        }

        expect(v.marg.length).toBe(2 * ndim)
        for (let a = 0; a < 2 * ndim; a++) {
          expect(v.marg[a]!.length).toBe(meta.N[a])
          for (let j = 0; j < v.marg[a]!.length; j++) {
            expect(v.marg[a]![j]).toBeCloseTo(m.marg[a][j], 6)
          }
        }
      }
    })

    it('rejects a wrong protocol version', () => {
      const bad = new Uint8Array(buf.slice(0) as ArrayBuffer)
      bad[1] = 99
      expect(() => decodeFrame(bad.buffer as ArrayBuffer)).toThrow(/version/)
    })
  },
)
