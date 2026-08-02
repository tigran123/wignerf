# wignerf — Interactive Wigner Function Simulator

Live client-server simulator of the Wigner function — *W(x, p, t)* over 1D
space, or *W(x, y, pₓ, p_y, t)* over 2D space — evolved by the spectral
split-operator method of Cabrera, Bondar, Jacobs and Rabitz, *Efficient method
to generate time evolution of the Wigner function for open quantum systems*
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

- **1D or 2D space, one solver.** A session picks `ndim`: 1D evolves
  *W(x, p, t)* on (Nx, Np), 2D evolves *W(x, y, pₓ, p_y, t)* on
  (Nx, Ny, Npx, Npy). There is no second solver — 1D is `ndim=1` of the same
  generic code, deliberately, because this project navigates by diagnostics and
  two propagators that could quietly disagree would poison exactly that. A 4D
  array can be neither drawn nor sent, so each worker reduces it **on the
  device** to the six pairwise 2D projections and one marginal per axis, and
  the state itself never leaves the GPU: at 64⁴ that is 50 KiB per variant per
  record against 33 MiB for the raw state, so scrubbing stays instant and the
  history cap stops being a 2D constraint. At `ndim=1` the single plane's
  complement is empty — that plane *is* W — so 1D is the general case rather
  than a special one.
- **Four variants in lockstep.** Each variant integrates with its own adaptive
  timestep but lands exactly on a shared record grid τₖ = t₁ + k·Δt_rec, so the
  same record index is the same physical time in every panel.
- **Analytic potentials.** Type *U(x)* — or *U(x, y)* in 2D — as an expression
  (sympy-parsed, screened, differentiated and lambdified) with an instant plot
  preview and per-variant validity reporting: a Heaviside step is quantum-valid
  but classically undefined, and the UI says so instead of producing nonsense.
  In 2D the editor draws the two axis cuts as separate charts, each labelled
  with the coordinate its cut was actually taken at.
- **Initial conditions.** Sums of Gaussians with independent σ per axis, and cat
  states with the analytic pairwise cross-Wigner term. Both families factorise
  over dimensions, so 2D needed no new closed form — a 2D pair's cross-Wigner is
  the outer product of the two 1D cores, and the 1D expressions are reused
  verbatim per dimension.
- **Live parameters.** *U(x)*, c, mass, ħ_eff, tolerance and time direction
  apply at the computation frontier without a restart; every change is logged
  and reproduced in the export's metadata block.
- **Auto-expanding domain.** The spectral domain is a torus, so a spreading
  state eventually wraps and the run silently solves the wrong problem. Edge
  mass is watched every record — at either dimensionality — and with
  auto-expand on the grid regrids onto an exact fixed lattice (whole-cell shift
  or power-of-two doubling, never interpolated) before any mass wraps. In 2D a
  doubling doubles a multi-GiB working set, so the planner asks the driver
  whether the devices can afford it first, and says which one cannot when they
  cannot — a shift costs nothing and is never subject to that, so a drifting
  state keeps getting relief even on a full card.
- **Diagnostics.** Energy, one ΔQ·ΔK per dimension and purity
  γ = (2πħ)^ndim ∫W² per record, with a marginal per axis alongside the
  phase-space panels — plus ⟨L_z⟩(t) in 2D. Everything but the purity is
  computed from the same reductions the panels are drawn from, leaving ∫W² as
  the only pass over the full array.
- **mp4 export.** Any already-computed record range is rendered on
  the backend (matplotlib → ffmpeg/libx264) into a video that reads like the
  screen, with a metadata block documenting the run — and the same document
  embedded in the file's `comment` tag, so a kept video restores its own setup
  on import.
- **Multi-GPU.** Variant workers spread across a pool of CUDA devices, the
  costliest variants to the fastest card (measured +41% for a 4-variant run at
  1024² across an RTX 3090 + 2080 Ti pair). Runs CPU-only without CUDA.

Units are Hartree atomic units throughout (ħ = mₑ = e = 1, c ≈ 137.036);
SI (fs / Å / eV) appears in display labels only.

### What 2D does not do yet

Nothing. Every feature is available at either dimensionality: the relativistic
variants and float32 (2026-07-27), mp4 export of a 2D run (2026-07-28) and
auto-expand (2026-08-01) were each gated behind an explicit refusal while the
work they named was outstanding, and all four gates are gone.

Two of them carry a caveat worth knowing. The **relativistic** variants
restore the anharmonic-shear diagnostic in 2D; the one to know is that a
**massless** (m = 0) relativistic run loses purity at a rate set by how well the
momentum grid resolves the kink in T = c|p| — ~7e-6 per 100 steps at 32⁴, falling
~10× at 48⁴, independent of the timestep, so the remedy is a finer momentum axis
and not a smaller dt.

**float32** is the other, and in 2D it is a memory setting rather than a speed
one: it costs 96 bytes per cell against float64's 176, but it is only 1.5–2.6×
faster here against 3.3–3.8× in 1D, because the 4D step transforms two axes at a
time and single precision gains far less on that. Two things to expect from it.
It is still a preview mode, so purity and energy drift with the same secular
signature as boundary wrap. And at a **coarse** 2D grid it moves enough mass
outward by itself to raise the boundary warning on a perfectly contained state —
measured 1e-3 of the integral at 32⁴ against 3e-5 for the same state in float64,
within seconds. 48⁴ and above are clear. If that warning appears and purity has
drifted too, the remedy is float64 or a finer grid, not a wider box.

Sizing a 2D run: a worker costs ~176 B per cell in float64 and ~96 in float32,
both flat across sizes, of which the state itself is only 5% (the rest is the
step's machinery at full shape). That is 0.17 GiB per worker at 32⁴ and 2.75 GiB
at 64⁴ in float64, or 0.09 and 1.50 in float32 — and 80⁴ (6.71 against 3.66) is
reachable only in single precision on these cards. Measured on an RTX 3090:
610 steps/s at 32⁴, 130 at 48⁴, 35.1 at 64⁴, 13.8 at 80⁴ — so **32⁴ is for
exploration and 64⁴ is a serious run**. Reproduce any of it with
`scripts/bench.py --ndim 2 --precision both [--footprint]`. Whether a session starts is not
decided by a cell count but by asking the driver how much memory the cards its
workers actually land on have free; see `WIGNERF_MAX_CELLS_2D` below.

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
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling for **1D** sessions, at creation and for auto-expand alike. A 4096² working set is ~3.0 GiB per variant worker. |
| `WIGNERF_MAX_GRID_2D` | `128` | Per-axis ceiling for **2D** sessions. A sanity rail only: a 4D array grows as N⁴, so a per-axis cap is no real guard — 128⁴ is ~44 GiB per worker with every axis inside the rail. |
| `WIGNERF_MAX_CELLS_2D` | `134217728` (2²⁷) | Total-cell rail for 2D — a cheap deterministic stop for absurd values (a dimensionality switch that carried N over would mean 1024⁴ from the 1D default, which is why the form caps it at the target's own), and the only guard on a host whose free memory cannot be read. Deliberately not the operative limit: that is a per-device fit check which runs the worker→device assignment, then compares the estimate against the driver's free memory, so the **smaller card binds** and unknown free memory never refuses. |
| `WIGNERF_PRECISION` | `float64` | Default spectral working precision (`float64` \| `float32`); a session may override it. float32 is a **preview mode** — ~3.3–3.8× faster in 1D but only 1.5–2.6× in 2D, and ~54–55% of the working set on CUDA, nothing at all on CPU — and it costs the diagnostics this project navigates by, so do not make it the host default anywhere a result might be read off. It refuses auto-expand and `tol` below 1e-5. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto. Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download; a file is deleted once downloaded, when its session closes, at shutdown, or 30 minutes after finishing. Under systemd's `PrivateTmp=yes` the default is a RAM tmpfs — point it at a disk path for long 4K renders. |
| `WIGNERF_EXPORT_ENCODER` | `auto` | mp4 encoder: `auto` \| `cpu` \| `nvenc`. `auto` uses the GPU `h264_nvenc` encoder if a runtime probe succeeds, else `libx264 -preset veryfast`. Overridable per job from the Export panel. |
| `WIGNERF_EXPORT_WORKERS` | `0` | Processes rendering export frames; `0` = auto (`min(cpu_count, 8)`). Rendering, not encoding, is the export bottleneck, so frames are spread over a spawn process pool while one ffmpeg encodes the ordered stream. |
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

`backend/tests/test_propagator2d.py` is the 2D anchor, and its strongest check
is a different one, because quantum ≡ classical *cannot* detect the classic
multi-D error. The Bopp shift must move every spatial argument of *U*
simultaneously; the plausible wrong version — a sum of two independent 1D
differences — agrees with the correct one for every quadratic potential, cross
terms included. What separates them is
`test_matches_an_independent_schroedinger_run`: a coupled potential evolved by
an ordinary split-operator TDSE, a different method sharing nothing but numpy's
FFT, compared against the streamed (x, y) plane cell by cell. The correct shift
converges as O(dt²) while the wrong one sits on a flat, 228× larger error — so
the assertion is on the **dt ratio**, not on a tolerance. The other 2D anchors
are separability against two 1D runs, ⟨L_z⟩ conservation under a central
potential, and reductions-vs-naive over all six planes and four marginals,
which is what catches 4-axis fftshift bookkeeping.

Other useful scripts — both take `--ndim 2`, which is the cheapest way to
exercise the 2D record path against a real server and to measure the per-worker
VRAM that decides whether a 2D session can start at all:

```sh
.venv/bin/python scripts/ws_smoke.py                  # streaming smoke test against a live server
.venv/bin/python scripts/ws_smoke.py --ndim 2
.venv/bin/python scripts/bench.py cpu cuda:1
.venv/bin/python scripts/bench.py --ndim 2 -N 32,48,64,80 cuda:1
.venv/bin/python scripts/bench.py --ndim 2 --regrid -N 32,64 cuda:1   # doubling cost
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
- `solve4D.py` — an older batch 4D solver, kept as a historical reference. The
  2D support here was written fresh rather than ported from it.

## .claude/projects/ (not tracked)

Claude Code keeps a project's session transcripts, subagent logs and project
memory in `<config dir>/projects/<slugified cwd>/`, which for this checkout
means `~/.claude/projects/-home-tigran-python-wignerf/`. Here that path is a
**symlink** into the repo:

```
~/.claude/projects/-home-tigran-python-wignerf
  -> /home/tigran/python/wignerf/.claude/projects/-home-tigran-python-wignerf
```

so the data sits beside the code it is about, and a copy of the working tree
takes the history with it. It is git-ignored (`.gitignore` ignores only
`.claude/projects/`, leaving room for a tracked project-scoped
`.claude/settings.json` later) — these are whole conversations, so `git push`
deliberately does **not** back them up. Anything that must survive a disk
failure needs a backup that copies the working tree.

There is no per-project setting for this, which is why it is a symlink. The
only relocation knob is `CLAUDE_CONFIG_DIR`, and it is the wrong instrument
twice over: it moves the entire `~/.claude` — credentials, settings, every
other project — and the OAuth credential name is derived from a hash of that
path, so setting it logs you out. The project path itself is built as
`configDir + "/projects/" + slug` and the resolved location is never stored,
so every absolute path already recorded in the transcripts keeps its
`~/.claude/…` spelling and resolves through the link. A setting that moved the
directory outright would instead leave all of those dangling.

**The failure mode to know about:** if the symlink is lost — a fresh machine,
or restoring `~/.claude` from a backup that predates it — Claude Code does not
error. It silently creates an empty real directory at that path, the repo-side
data is orphaned, and the project simply appears to have no history. Recreate
it with:

```sh
mv ~/.claude/projects/-home-tigran-python-wignerf{,-empty}   # only if one was recreated
ln -s /home/tigran/python/wignerf/.claude/projects/-home-tigran-python-wignerf \
      ~/.claude/projects/-home-tigran-python-wignerf
```

Note that `~/.claude/plans/`, `file-history/`, `tasks/` and the global
`history.jsonl` are keyed by session id rather than by project, so they stay
under `~/.claude` and are not covered by this.

## Status

A personal research tool, under active development.

2D space (4D phase space) landed in July 2026 as a first cut with four features
held back behind explicit refusals so the physics core could land verified. All
four are now in: the relativistic variants and float32 (2026-07-27, the latter
halving what a 2D worker holds), mp4 export of a 2D run (2026-07-28), and the
automatic regrid (2026-08-01, which needed a per-device memory guard on the
doubling and an allocation order that hands the old grid back before building
the new one). The last of the four, on 2026-08-02, collapsed the propagator's
two exponent slots into one by dividing each record into equal substeps instead
of walking at the adaptive step and clamping a straggler onto the record
boundary: −32 bytes per cell at both dimensionalities, 208 → 176 in 2D and
224 → 192 in 1D. Two further items are already scoped: cuts at fixed (y, p_y)
alongside the projections, which are exact for separable states but average away
the fringe contrast of precisely the entangled regime 2D exists to show; and
merging the inverse/forward FFT pair across step boundaries, worth about +50%
in 2D.

Beyond that: destructive forking (resume computation from any record, which
needs periodic float64 checkpoints alongside the quantized display history),
save/load of a whole simulation, then Lindblad dissipation.
