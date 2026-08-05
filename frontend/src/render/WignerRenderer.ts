/**
 * WebGL2 heatmap renderer for W(x,p) — a plain class, deliberately OUTSIDE
 * Vue reactivity. Design (see plan):
 *
 * - R16UI texture + usampler2D + texelFetch: the received Uint16Array view
 *   uploads untouched (zero copies, exact 16-bit fidelity). Integer
 *   textures cannot LINEAR-filter, so bilinear interpolation is done
 *   manually in the shader.
 * - The payload is in NATURAL order and may be a CROP of the plane, decimated
 *   to about the panel's pixel size (backend display downsampling). The
 *   texture therefore covers uTex0..uTex1 of the domain rather than all of it,
 *   and the shader maps the requested view window into that. Both the old
 *   half-period unshift offset and the old toroidal wrap are gone with the
 *   full-period assumption they rested on — see the fragment shader.
 * - Diverging colormap centered at W=0 with SYMMETRIC two-sided scaling:
 *   u = 0.5 + 0.5*W/max(Wmax, -Wmin), so color intensity is proportional
 *   to |W| with one shared scale. (The asymmetric MidpointNormalize of
 *   dynamics/midnorm.py would hand the whole blue half of the map to
 *   Wmin however tiny — rendering ~1e-6 numerical/quantization noise as
 *   saturated blue. Genuine negativity, e.g. cat-state fringes, is
 *   comparable to the peak and stays vividly blue under either scheme.)
 *   uQ = the frame's own (wmin, wmax), always used for dequantization;
 *   uC = the color scale — equal to uQ when autoscaling, or a locked pair.
 */

import type { PlaneFrame } from '../lib/protocol'
import { bwrLUT } from '../lib/colormaps'
import { perfInfo, perfStage } from '../lib/perf'

const VS = `#version 300 es
in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

const FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uW;    // width = n along axis b, height = n along axis a
uniform sampler2D uLUT;
uniform vec2 uQ;          // (wmin, wmax) of THIS frame - dequantization
uniform vec2 uC;          // color scale (min, max); == uQ when autoscaling
uniform vec2 uSize;       // texture (width, height) = (n along b, n along a)
uniform vec2 uView0;      // view window corners in domain fractions:
uniform vec2 uView1;      // (a horizontal, b vertical-up); (0,0)/(1,1) = full
uniform vec2 uTex0;       // the domain fractions the TEXTURE actually covers,
uniform vec2 uTex1;       // same (a, b) order. Equals uView0/1 for a full plane.
in vec2 vUV;
out vec4 fragColor;

float fetchW(ivec2 ij) {
  uint q = texelFetch(uW, ij, 0).r;
  return uQ.x + (uQ.y - uQ.x) * (float(q) / 65535.0);
}

float sampleW(vec2 st) {  // st in [0,1] over the SERVED window, clamped
  vec2 xy = st * uSize - 0.5;
  vec2 f = fract(xy);
  ivec2 sz = ivec2(uSize);
  ivec2 hi = sz - 1;
  // CLAMP, not the modulo wrap this had while every texture was a full
  // period. A served window is a CROP: wrapping it would fold the far edge of
  // a zoomed tile onto its near edge, which is not the torus and not anything.
  // The cost at the full view is half a texel of interpolation at the domain
  // seam, and viewWindow clamps inside [0,1] so the wrapped continuation is
  // never on screen anyway.
  ivec2 a = clamp(ivec2(floor(xy)), ivec2(0), hi);
  ivec2 b = min(a + 1, hi);
  float w00 = fetchW(a);
  float w10 = fetchW(ivec2(b.x, a.y));
  float w01 = fetchW(ivec2(a.x, b.y));
  float w11 = fetchW(b);
  return mix(mix(w00, w10, f.x), mix(w01, w11, f.x), f.y);
}

void main() {
  // screen: axis a horizontal (texture HEIGHT t), axis b vertical up (texture
  // WIDTH s) — hence the .yx swap. Records arrive in NATURAL order now
  // (frame.build unshifts on the device), so the half-period +0.5 offset this
  // used to carry is gone with the shift it undid.
  vec2 dom = mix(uView0, uView1, vUV);
  vec2 rel = (dom - uTex0) / max(uTex1 - uTex0, vec2(1e-9));
  vec2 st = clamp(vec2(rel.y, rel.x), 0.0, 1.0);
  float w = sampleW(st);
  // symmetric diverging scale: W=0 -> LUT center (white), intensity
  // proportional to |W| on both sides
  float scale = max(max(uC.y, -uC.x), 1e-30);
  float u = 0.5 + 0.5 * w / scale;
  fragColor = texture(uLUT, vec2(clamp(u, 0.0, 1.0), 0.5));
}`

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!
  gl.shaderSource(sh, src)
  gl.compileShader(sh)
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh))
  }
  return sh
}

export class WignerRenderer {
  private gl: WebGL2RenderingContext | null = null
  private prog: WebGLProgram | null = null
  private texW: WebGLTexture | null = null
  private texLUT: WebGLTexture | null = null
  private uQ: WebGLUniformLocation | null = null
  private uC: WebGLUniformLocation | null = null
  private uSize: WebGLUniformLocation | null = null
  private uView0: WebGLUniformLocation | null = null
  private uView1: WebGLUniformLocation | null = null
  private uTex0: WebGLUniformLocation | null = null
  private uTex1: WebGLUniformLocation | null = null
  private nx = 0
  private np = 0
  // view window in domain fractions (a horizontal, b vertical-up); full domain
  private vx0 = 0
  private vx1 = 1
  private vy0 = 0
  private vy1 = 1
  // the domain fractions the uploaded texture covers. Equal to the full domain
  // for a whole plane; a sub-window when the server served a crop.
  private tx0 = 0
  private tx1 = 1
  private ty0 = 0
  private ty1 = 1
  private q: [number, number] = [0, 1]
  /** When set, the color scale is locked to this (min, max); otherwise
   *  every frame autoscales to its own range. */
  colorLock: [number, number] | null = null

  init(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false })
    if (!gl) throw new Error('WebGL2 is not available')
    this.gl = gl
    // expose the real renderer once — a "SwiftShader" here means software
    // rendering, which caps large-grid playback at a few fps
    if (!perfInfo.gl_renderer) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info')
      const name = String(gl.getParameter(
        dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER))
      perfInfo.gl_renderer = name
      console.info('wignerf WebGL renderer:', name)
    }
    const prog = gl.createProgram()!
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS))
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(prog))
    }
    this.prog = prog
    gl.useProgram(prog)

    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, 'aPos')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)

    this.uQ = gl.getUniformLocation(prog, 'uQ')
    this.uC = gl.getUniformLocation(prog, 'uC')
    this.uSize = gl.getUniformLocation(prog, 'uSize')
    this.uView0 = gl.getUniformLocation(prog, 'uView0')
    this.uView1 = gl.getUniformLocation(prog, 'uView1')
    this.uTex0 = gl.getUniformLocation(prog, 'uTex0')
    this.uTex1 = gl.getUniformLocation(prog, 'uTex1')

    // LUT: 256x1 RGBA8, LINEAR (it is a float texture, filtering is fine)
    this.texLUT = gl.createTexture()
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.texLUT)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA,
      gl.UNSIGNED_BYTE, bwrLUT())
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.uniform1i(gl.getUniformLocation(prog, 'uLUT'), 1)
    gl.uniform1i(gl.getUniformLocation(prog, 'uW'), 0)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 2)
  }

  private ensureTexture(Nx: number, Np: number) {
    const gl = this.gl!
    if (this.texW && this.nx === Nx && this.np === Np) return
    if (this.texW) gl.deleteTexture(this.texW)
    this.texW = gl.createTexture()
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.texW)
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16UI, Np, Nx)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    this.nx = Nx
    this.np = Np
  }

  /** Upload one quantized 2D plane. Its data is (na, nb) row-major, i.e. the
   *  texture is nb wide (the plane's second axis) and na tall (its first).
   *
   *  `fullA`/`fullB` are the RECORD's axis counts for that pair — the plane may
   *  be a decimated crop, and they are what its off/step are measured against.
   *
   *  This is why a 4D phase space needs no renderer change at all: a plane
   *  reduction is exactly the 2D array a 1D W was, so the manual bilinear and
   *  the window mapping apply unaltered. */
  upload(p: PlaneFrame, fullA: number, fullB: number) {
    const gl = this.gl
    if (!gl) return
    const t0 = performance.now()
    this.ensureTexture(p.na, p.nb)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.texW)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, p.nb, p.na,
      gl.RED_INTEGER, gl.UNSIGNED_SHORT, p.data)
    // Which slice of the domain this texture covers. The crop is expressed in
    // BASE cells (off + n*step against the record's own N), so this is exact
    // integer arithmetic — a full plane gives 0..1 and the shader degenerates
    // to what it did before.
    this.tx0 = p.off[0] / fullA
    this.tx1 = (p.off[0] + p.na * p.step[0]) / fullA
    this.ty0 = p.off[1] / fullB
    this.ty1 = (p.off[1] + p.nb * p.step[1]) / fullB
    this.q = [p.wmin, p.wmax]
    perfStage('upload', performance.now() - t0)
  }

  /** Drawing-buffer size in device pixels — what the panel asks the server to
   *  match. Reads the canvas rather than any cached value, so it is honest
   *  immediately after a resize and before the next render(). */
  pixelSize(): [number, number] {
    const c = this.gl?.canvas as HTMLCanvasElement | undefined
    if (!c) return [0, 0]
    const dpr = window.devicePixelRatio || 1
    return [Math.max(1, Math.round(c.clientWidth * dpr)),
            Math.max(1, Math.round(c.clientHeight * dpr))]
  }

  /** Set the zoom/pan view window in domain fractions (a horizontal, b up).
   *  Takes effect on the next render(). */
  setView(x0: number, x1: number, y0: number, y1: number) {
    this.vx0 = x0
    this.vx1 = x1
    this.vy0 = y0
    this.vy1 = y1
  }

  render() {
    const gl = this.gl
    if (!gl || !this.texW) return
    const t0 = performance.now()
    const canvas = gl.canvas as HTMLCanvasElement
    const dpr = window.devicePixelRatio || 1
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr))
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr))
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    gl.viewport(0, 0, w, h)
    gl.useProgram(this.prog)
    const c = this.colorLock ?? this.q
    gl.uniform2f(this.uQ, this.q[0], this.q[1])
    gl.uniform2f(this.uC, c[0], c[1])
    gl.uniform2f(this.uSize, this.np, this.nx)
    gl.uniform2f(this.uView0, this.vx0, this.vy0)
    gl.uniform2f(this.uView1, this.vx1, this.vy1)
    gl.uniform2f(this.uTex0, this.tx0, this.ty0)
    gl.uniform2f(this.uTex1, this.tx1, this.ty1)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    perfStage('draw', performance.now() - t0)
  }

  /**
   * Release everything, INCLUDING the context itself.
   *
   * `loseContext()` is the load-bearing line, and deleting the objects above it
   * is not a substitute: a WebGL context lives as long as its canvas, and a
   * detached canvas is reclaimed only when the GC gets to it — which is
   * non-deterministic, exactly as it is for the backend's closed sessions.
   * Meanwhile the context still counts against the browser's live-context cap
   * (16 in Chrome), and past the cap the browser does not fail the new
   * context: it silently KILLS THE OLDEST one.
   *
   * That is what made the IC preview go blank on Restart in 2D. Restart bumps
   * plotsKey, PanelGrid remounts, and every panel builds a new context while
   * the old one is still alive — at ndim=2 that is SIX per Restart against 1D's
   * one, so the cap arrives in two or three Restarts. The oldest context in the
   * page is the IC editor's, created at first mount, so the IC editor is what
   * the browser takes. Nothing reports it: `webglcontextlost` fires on a
   * component that never listened, and every subsequent GL call succeeds as a
   * no-op, so the canvas is simply blank for the rest of the page's life. A
   * reload fixes it, which is exactly what makes it read as a mystery.
   *
   * A/B measured with a headless context counter that holds each context
   * object from creation (re-querying getContext misreports, which cost an
   * afternoon): across six Restarts in a TWO-panel layout, detached-but-alive
   * contexts go 2, 4, 6, 8, 10, 12 without this line and stay at 0 with it.
   * Two panels per Restart is the cheap case — the six-panel 2D layout reaches
   * the cap three times faster.
   */
  dispose() {
    const gl = this.gl
    if (!gl) return
    if (this.texW) gl.deleteTexture(this.texW)
    if (this.texLUT) gl.deleteTexture(this.texLUT)
    if (this.prog) gl.deleteProgram(this.prog)
    this.texW = this.texLUT = this.prog = null
    gl.getExtension('WEBGL_lose_context')?.loseContext()
    this.gl = null
  }
}
