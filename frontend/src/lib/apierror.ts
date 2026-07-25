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
  const detail = (e as { response?: { data?: { detail?: unknown } } })
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
