import { describe, expect, it } from 'vitest'
import { apiErrorText } from './apierror'

/**
 * FastAPI answers a schema refusal with an ARRAY of pydantic error objects, and
 * both call sites used to read `detail` as a string — so the UI printed the raw
 * blob (type/loc/ctx and a verbatim copy of the whole request `input`) with the
 * one readable sentence buried in it. These messages are written to be read.
 */
describe('apiErrorText', () => {
  it('passes our own HTTPException string through', () => {
    expect(apiErrorText({ response: { data: { detail: 'nothing to export' } } }))
      .toBe('nothing to export')
  })

  it('extracts the message from a pydantic validation array', () => {
    const detail = [{
      type: 'value_error',
      loc: ['body'],
      msg: 'Value error, auto-expand is not available in float32: it fires on '
        + 'edge-band mass at 1e-6.',
      input: { grid: { Nx: 8192 }, potential: 'x^2/2' },
      ctx: { error: {} },
    }]
    const text = apiErrorText({ response: { data: { detail } } })
    expect(text).toBe('auto-expand is not available in float32: it fires on '
                      + 'edge-band mass at 1e-6.')
    // none of the machine-readable noise reaches the user
    expect(text).not.toContain('value_error')
    expect(text).not.toContain('8192')
    expect(text).not.toContain('Value error,')
  })

  it('names the field for a per-field error, and joins several', () => {
    const detail = [
      { loc: ['body', 'tol'], msg: 'Input should be less than 1' },
      { loc: ['body', 'grid', 'Nx'], msg: 'Input should be a valid integer' },
    ]
    expect(apiErrorText({ response: { data: { detail } } }))
      .toBe('tol: Input should be less than 1 · '
            + 'grid.Nx: Input should be a valid integer')
  })

  it('collapses duplicates rather than repeating one message', () => {
    const d = { loc: ['body'], msg: 'Value error, mass = 0 requires rel' }
    expect(apiErrorText({ response: { data: { detail: [d, d] } } }))
      .toBe('mass = 0 requires rel')
  })

  it('falls back to the error itself when there is no usable detail', () => {
    expect(apiErrorText(new Error('Network Error'))).toContain('Network Error')
    expect(apiErrorText({ response: { data: {} } })).toBe('[object Object]')
    expect(apiErrorText({ response: { data: { detail: [] } } }))
      .toBe('[object Object]')
  })

  it('survives a malformed entry instead of throwing', () => {
    expect(apiErrorText({ response: { data: { detail: [{ loc: ['body'] }] } } }))
      .toContain('loc')
  })
})
