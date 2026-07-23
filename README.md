# wignerf — Interactive Wigner Function Simulator

Live client-server simulator of the Wigner function *W(x, p, t)* in 1D phase
space. The state is evolved by the spectral split-operator method of Cabrera,
Bondar, Jacobs and Rabitz, *Efficient method to generate time evolution of the
Wigner function for open quantum systems*
([arXiv:1212.3406](https://arxiv.org/abs/1212.3406)) — an O(N log N)
FFT scheme that propagates *W* directly, without ever forming the density
matrix.

Up to four propagator variants run side by side in lockstep — **quantum**
(Moyal) and **classical** (Liouville), each **non-relativistic**
(T = p²/2m) and **relativistic** (T = √(p²c² + m²c⁴) − mc²) — so the
quantum-classical and relativistic corrections are visible as differences
between panels at the same physical time, not as separate runs you have to
line up afterwards.

A solver backend streams quantized frames over a binary WebSocket to a Vue SPA
that paints them on the GPU; the same history can be scrubbed, replayed, and
exported to mp4.

## Features

- **Four variants in lockstep.** Each variant integrates with its own adaptive
  timestep but lands exactly on a shared record grid τₖ = t₁ + k·Δt_rec, so the
  same record index is the same physical time in every panel.
- **Analytic potentials.** Type *U(x)* as an expression (sympy-parsed, screened,
  differentiated and lambdified) with an instant plot preview and per-variant
  validity reporting — a Heaviside step is quantum-valid but classically
  undefined, and the UI says so instead of producing nonsense.
- **Initial conditions.** Sums of Gaussians with independent σₓ, σ_p, and cat
  states with the analytic pairwise cross-Wigner term.
- **Live parameters.** *U(x)*, c, mass, ħ_eff, tolerance and time direction
  apply at the computation frontier without a restart; every change is logged
  and reproduced in the export's metadata block.
- **Auto-expanding domain.** The spectral domain is a torus, so a spreading
  state eventually wraps and the run silently solves the wrong problem. Edge
  mass is watched every record; with auto-expand on, the grid regrids onto an
  exact fixed lattice (whole-cell shift or power-of-two doubling, never
  interpolated) before any mass wraps.
- **Diagnostics.** Energy, ΔX·ΔP and purity γ = 2πħ∬W²dxdp per record, with
  the marginals ρ(x) and φ(p) alongside the phase-space panels.
- **mp4 export.** Any already-computed record range is rendered on the backend
  (matplotlib → ffmpeg/libx264) into a video that reads like the screen, with a
  metadata block documenting the run — and the same document embedded in the
  file's `comment` tag, so a kept video restores its own setup on import.
- **Multi-GPU.** Variant workers spread across a pool of CUDA devices, the
  costliest variants to the fastest card (measured +41% for a 4-variant run at
  1024² across an RTX 3090 + 2080 Ti pair). Runs CPU-only without CUDA.

Units are Hartree atomic units throughout (ħ = mₑ = e = 1, c ≈ 137.036);
SI (fs / Å / eV) appears in display labels only.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ (Vite 8)
- Optional, for GPU: an NVIDIA driver — **no system CUDA Toolkit**. CuPy
  JIT-compiles its kernels through NVRTC and gets the headers from PyPI via the
  `[ctk]` extra.
- Optional, for mp4 export: a system `ffmpeg` with libx264. Without it the
  export endpoint returns 503; everything else works.

## Install (after `git clone`)

```sh
git clone git@github.com:tigran123/wignerf.git
cd wignerf
```

`start.sh` only **runs** the server — it never installs or builds. Create the
venv and build the SPA once after cloning, and again after pulling. This keeps
the systemd service sandboxed with a read-only home; installing or building
inside the service would need write access to `~/.cache/uv`, `node_modules`,
`frontend/dist` and more.

**Backend** — create the venv, install the pinned dependencies, and precompile
bytecode. The service runs with a **read-only home**, so Python can never write
`__pycache__/*.pyc` at runtime; precompiling here, while the tree is still
writable, keeps server startup fast. `--compile-bytecode` handles the venv,
`compileall` handles our own source:

```sh
cd backend
uv venv
uv pip sync --compile-bytecode requirements.txt requirements-dev.txt   # add requirements-gpu.txt on a CUDA host
.venv/bin/python -m compileall main.py config.py core routers
```

**Frontend** — install the node dependencies and build the SPA into
`frontend/dist`. For a deployment behind an nginx prefix, export
`APP_ROOT_PATH` **first**, so the build bakes in the right base and API path —
the runtime service's `EnvironmentFile` does not reach the build. On the dev
machine (prefix `/`) it can be omitted:

```sh
cd ../frontend
# export APP_ROOT_PATH=/wignerf     # only for a prefixed prod build
npm ci
npm run build
```

**GPU note.** The pinned `cupy-cuda13x[ctk]` needs a CUDA 13 capable driver,
which dropped Maxwell, Pascal and Volta support. On an older card (e.g. a
GTX 1060) use `cupy-cuda12x[ctk]` in `requirements-gpu.in` instead.

## Upgrade (after `git pull`)

Re-sync the backend dependencies in case the pins changed, and rebuild the SPA
from scratch — the old `frontend/dist` must be removed so a stale build is
never served:

```sh
cd backend
uv pip sync --compile-bytecode requirements.txt requirements-dev.txt   # add requirements-gpu.txt on a CUDA host
.venv/bin/python -m compileall main.py config.py core routers

cd ../frontend
# export APP_ROOT_PATH=/wignerf     # only for a prefixed prod build
rm -rf dist
npm ci
npm run build
```

Then restart: `./start.sh`, or `sudo systemctl restart wignerf`.

To change dependencies, edit the `.in` file and recompile — never hand-edit a
`.txt`:

```sh
cd backend
uv pip compile requirements.in -o requirements.txt
uv pip sync --compile-bytecode requirements.txt requirements-dev.txt
```

## Run

Production-style — one process serving the API and the built SPA:

```sh
./start.sh                  # http://localhost:8010
```

It errors out if `backend/.venv` or `frontend/dist` is missing.

Development, with SPA hot reload — two terminals, since Vite proxies `/api`
(WebSocket included) to the backend:

```sh
# terminal 1
cd backend && .venv/bin/uvicorn main:app --port 8010 --ws-per-message-deflate false

# terminal 2
cd frontend && npm run dev          # http://localhost:5173
```

> **Always pass `--ws-per-message-deflate false`.** uvicorn's default WebSocket
> compression zlib-squeezes every multi-MiB frame bundle on the event loop and
> caps the stream at ~10–25 records/s — measured 12× slower than uncompressed
> on localhost. Browsers negotiate the extension silently, so it presents as a
> rendering problem. `start.sh` already passes it.

## Configuration

All configuration is environment-driven (`backend/config.py`); `wignerf.env`
holds the per-machine values for the systemd unit.

| Variable | Default | Meaning |
|---|---|---|
| `WIGNERF_DEVICE` | `auto` | Device pool: `auto` \| `cpu` \| `cuda:N` \| comma list (`cuda:1,cuda:0`). `auto` = every CUDA device, fastest first; an explicit list is trusted as written, its order being the speed ranking. Indices follow PCI order, matching `nvidia-smi`. |
| `WIGNERF_PORT` | `8010` | Listen port. |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session — the scrub/replay window. 32 GiB ≈ 4000 four-variant records at 1024². Lower it on small hosts. |
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling, for session creation and auto-expand alike. A 4096² working set is ~1.3 GiB per variant worker. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto. Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download; a file is deleted once downloaded, when its session closes, at shutdown, or 30 minutes after finishing. Under systemd's `PrivateTmp=yes` the default is a RAM tmpfs — point it at a disk path for long 4K renders. |
| `APP_ENV` | — | `development` enables uvicorn `--reload`. |
| `APP_ROOT_PATH` | `/` | URL prefix when the SPA is mounted behind a path (e.g. `/wignerf`). Drives both uvicorn `--root-path` and the Vite `base`, so it must be exported at **build** time too. |

## Tests

```sh
cd backend && .venv/bin/pytest          # GPU and ffmpeg tests skip when unavailable
cd frontend && npm run test && npm run build
```

`backend/tests/test_propagator.py` holds the physics invariants and is the
correctness anchor — in particular, quantum ≡ classical for a harmonic
potential (the Moyal corrections vanish for a quadratic Hamiltonian). Run it
after touching the propagator, the grid, or the fftshift bookkeeping.

Other useful scripts:

```sh
.venv/bin/python scripts/ws_smoke.py    # streaming smoke test against a live server
.venv/bin/python scripts/bench.py cpu cuda:1
```

## Deployment

`wignerf.service` is a systemd unit for the reference deployment (uvicorn on
127.0.0.1 behind nginx). Install it with:

```sh
sudo cp wignerf.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wignerf
```

The unit's `WorkingDirectory`, `EnvironmentFile` and `ExecStart` are absolute
paths — adjust them if the checkout is not at `/home/tigran/python/wignerf`.

Two constraints are load-bearing and commented in the file: `CacheDirectory`
gives CuPy a persistent kernel cache — and matplotlib a persistent font cache
via `MPLCONFIGDIR` — because the read-only home plus `PrivateTmp` otherwise
leaves both rebuilding from scratch on every restart (a minute of CuPy
recompilation blocking the first frame; ~0.7 s of matplotlib font scanning);
and `PrivateDevices` **must** stay `no` — `yes` hides `/dev/nvidia*`, and the
solver falls back to CPU without saying anything.

## docs/ (not tracked)

`docs/` is deliberately git-ignored and holds reference material rather than
part of the program:

- `Efficient-Method-2015.pdf` — the method paper; fetch your own copy from
  [arXiv:1212.3406](https://arxiv.org/abs/1212.3406).
- `solve4D.py` — an older batch 4D solver, kept as a historical reference for
  the eventual multi-D work.

## Status

A personal research tool, under active development. Next up: destructive
forking (resume computation from any record, which needs periodic float64
checkpoints alongside the quantized display history), save/load of a whole
simulation, then Lindblad dissipation and multi-D.
