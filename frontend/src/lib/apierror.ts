/**
 * The human-readable text of a failed API call.
 *
 * FastAPI's `detail` is a STRING for the HTTPExceptions we raise ourselves, but
 * an ARRAY of pydantic error objects for any body-validation failure — so a
 * schema refusal used to reach the UI as the entire raw JSON blob, `type`/`loc`/
 * `ctx` and a verbatim copy of the whole request `input` included, with the one
 * sentence that mattered buried somewhere in the middle. Those refusal messages
 * are written to be read (see protocol.MSG_EXPAND_F32); this is what lets them
 * be.
 */
export function apiErrorText(e: unknown): string {
  const err = e as { response?: { data?: unknown } }
  // A responseType:'arraybuffer' request (the IC preview fetches a binary frame
  // bundle) hands back its ERROR body as bytes too, so `data` is an ArrayBuffer
  // holding the JSON envelope. Decoding it here rather than at the call site is
  // what keeps the rule in CLAUDE.md true — a failed API call goes through this
  // function, never through data.detail — and is why the IC editor used to
  // print a raw {"detail":"..."} blob under its plot.
  if (err?.response?.data instanceof ArrayBuffer) {
    try {
      const txt = new TextDecoder().decode(err.response.data)
      return apiErrorText({ response: { data: JSON.parse(txt) } })
    } catch {
      return new TextDecoder().decode(err.response.data) || String(e)
    }
  }
  const detail = (err as { response?: { data?: { detail?: unknown } } })
    ?.response?.data?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail) && detail.length) {
    const seen = new Set<string>()
    for (const d of detail) {
      const o = (d ?? {}) as { msg?: unknown; loc?: unknown[] }
      // pydantic prefixes every custom validator message with "Value error, "
      const msg = typeof o.msg === 'string'
        ? o.msg.replace(/^Value error,\s*/, '') : JSON.stringify(d)
      // "body" is noise — every one of these is a body error
      const field = Array.isArray(o.loc)
        ? o.loc.filter((k) => k !== 'body').join('.') : ''
      seen.add(field ? `${field}: ${msg}` : msg)
    }
    return [...seen].join(' · ')
  }
  return String(e)
}
