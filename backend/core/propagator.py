"""
Spectral split-operator propagator of 2nd order for the Wigner function
W(q, k, t), after Cabrera, Bondar, Jacobs, Rabitz (2015). Direct port of the
validated batch implementation in dynamics/solve.py, reshaped as a class
with mutable physics parameters for interactive use, and generic over ndim
(1D space -> W(x,p); 2D space -> W(x,y,px,py)) since 2026-07-26.

Variants: quantum/classical x relativistic/non-relativistic, selected per
instance. Units are Hartree atomic units (hbar = m_e = e = 1); c is a
parameter (C_AU by default, c=1 reproduces the old natural-unit runs) and
hbar_eff scales the quantum differential for classical-limit studies.

State convention: W is real of shape grid.N, fftshifted along every axis.
One step is the Strang splitting expT * expU * expT where dT already
carries the factor 1/2 (exactly as in solve.py).

THE MULTI-D BOPP SHIFT. The Bopp arguments are real-valued: qd() evaluates U
at q_i -+ hbar*theta_i/2 for EVERY spatial axis AT ONCE, with the same sign and
index-matched (theta_i is the dual of k_i). That single simultaneous shift is
what the 2D Moyal product gives; it is NOT the sum of two independent
one-dimensional shifts. The two agree for every quadratic U — so
quantum == classical cannot tell them apart — and first differ at third order
in mixed derivatives: for U = x^2*y they differ by exactly -2*a^2*b with
a = hbar*thetax/2, b = hbar*thetay/2. tests/test_propagator2d.py's independent
Schroedinger reference on a coupled potential is what pins this.

For real U the exponent dU is purely imaginary, so |expU| = 1 and the
evolution is unconditionally norm-stable; accuracy is governed by adjust_step.

That imaginary-ness is exact, not approximate, so the generators are STORED
as the real angular-rate meshes dU_im, dT_im (the generator being 1j times
them) — half the device bytes of the complex128 arrays they replace, and
exponents() reproduces the old phases bitwise. _rate_mesh refuses a generator
with a real part rather than let it silently break norm stability.

NO FULL-SHAPE ENERGY MESH. <H> is assembled from the reductions observables
already computes: int U*n_q over the spatial subspace plus int T*n_k over the
momentum one, where n_q = int W d^n k and n_k = int W d^n q. So U and T are
stored on their own SUB-shapes (U_mesh, T_mesh — kilobytes) rather than as one
float64 mesh of the full shape. That saves 8 B/cell and replaces 2*ndim + 2
full-array weighted sums per record with two tiny ones; at ndim=2 the full-shape
version would have been 134 MiB at 64^4 for a quantity read once per record.

PRECISION. The backend's complex_dtype ("float64" -> complex128, the default;
"float32" -> complex64, the opt-in preview mode) applies to exactly two things:
the spectral working array B in solve_spectral, and the phase arrays exponents()
hands it. Everything the exponents are BUILT from — the grid meshes, the two
qd() evaluations of U, the dU_im/dT_im rate meshes, U_mesh/T_mesh — stays
float64 in both modes. That split is not a compromise, it is the whole design:

  - it is REQUIRED. Relativistic dT built in float32 has max abs error 455
    against max |dT| = 228 (200%), because mc^2 cancels inside a difference of
    ~1.9e4-magnitude terms. Keeping construction double also keeps _rate_mesh's
    1e-13 real-part gate and the frozen-lattice regrid arithmetic exact, so
    neither needs a dtype-scaled tolerance.
  - and it is FREE. Measured on an RTX 3090, 2026-07-25: mixed (float64
    construction, complex64 stepping) runs at 3.72x/3.62x/3.22x the double
    speed at 1024^2/2048^2/4096^2, against 3.80x/3.69x/3.27x for float32
    everywhere. The FFTs are the cost; the meshes are built once per rebuild().

That split holds UNCHANGED at ndim=2 (milestone M1, 2026-07-27): the rate meshes
are bitwise identical between the two modes for all four variants at 4 axes just
as at 2, which is the property that had to be re-verified because the multi-D
Bopp shift moves every spatial argument together. Nothing here is
dimension-aware; only exponents() and solve_spectral() see the working dtype.

What single precision BUYS is much smaller in 2D, though, and worth knowing
before choosing it: measured 2.63x at 32^4, 2.09x at 48^4, 1.48x at 64^4 and
1.64x at 80^4, against 3.29x for a 1D 4096^2 run of the SAME 16.8M cell count on
the same card. The 2D step transforms two axes of a 4D array at a time
(fft_pair's fftn branch) where 1D transforms one axis of a 2D array, and cuFFT's
single-precision advantage for that strided layout is far smaller. So in 2D
float32 is chosen for MEMORY — 96 B/cell against 176, see config.py — and the
speed is a bonus.
"""

import logging

from .xp import C_AU

log = logging.getLogger(__name__)


class Propagator:
    def __init__(self, grid, *, quantum=True, relativistic=False,
                 mass=1.0, c=C_AU, hbar_eff=1.0, tol=1e-2, U=None, gradU=None):
        if U is None:
            raise ValueError("a U(*q) callable is required")
        if not quantum and gradU is None:
            raise ValueError("gradU (one dU/dq_i callable per spatial axis) is "
                             "required for the classical propagator")
        self.grid = grid
        self.ndim = grid.ndim
        self.backend = grid.backend
        self.xp = grid.backend.xp
        self.quantum = bool(quantum)
        self.relativistic = bool(relativistic)
        self.mass = float(mass)
        self.c = float(c)
        self.hbar_eff = float(hbar_eff)
        self.tol = float(tol)
        self.U = U
        self.gradU = tuple(gradU) if gradU is not None else None
        self._check_grad()

        self.cdtype = self.backend.complex_dtype
        self._plan(grid)
        self.rebuild()

    def _check_grad(self):
        if self.gradU is not None and len(self.gradU) != self.ndim:
            raise ValueError("gradU has %d callables, need one per spatial "
                             "axis (%d)" % (len(self.gradU), self.ndim))

    def _plan(self, grid):
        # Two multi-axis plan pairs, not 2*ndim single-axis ones: one Strang
        # step transforms all spatial axes together (x <-> lambda, for expT)
        # and all momentum axes together (p <-> theta, for expU). At ndim=1
        # fft_pair takes its one-dimensional entry points, so nothing about
        # the 1D path changes.
        self._fft_sp, self._ifft_sp = self.backend.fft_pair(
            grid.shape, grid.spatial_axes)
        self._fft_mo, self._ifft_mo = self.backend.fft_pair(
            grid.shape, grid.momentum_axes)

    # -- physics construction ---------------------------------------------

    def qd(self, f, xs, dxs):
        """Quantum differential of f at xs on the increments dxs (solve.py:102,
        generalized): ALL arguments are shifted together, same sign — see the
        multi-D Bopp note in the module docstring."""
        h = self.hbar_eff
        plus = f(*[x + 1j*h*d/2. for x, d in zip(xs, dxs)])
        minus = f(*[x - 1j*h*d/2. for x, d in zip(xs, dxs)])
        return (plus - minus)/(1j*h)

    def _kinetic(self):
        """(T(*k), (dT/dk_0, ...)) — isotropic in the momenta."""
        m, c, xp = self.mass, self.c, self.xp
        if not self.relativistic:
            def T(*k):
                return sum(ki**2 for ki in k)/(2.*m)

            def grad(i):
                return lambda *k: k[i]/m
        else:
            def T(*k):
                return c*xp.sqrt(sum(ki**2 for ki in k) + m**2*c**2)

            if m == 0.0:
                # T = c*|k|: the gradient is c times the UNIT vector k_i/|k|,
                # which the general form below evaluates as 0/0 at the origin —
                # and the origin IS a lattice point (a symmetric box with even N
                # puts an exact 0.0 on every axis), so this is reached, not
                # hypothetical. Define it as 0 there.
                #
                # That is not a new convention for 2D, it is the existing 1D one
                # spelled generically: at ndim=1 this returns c*sign(k0)
                # BITWISE, because sqrt(k0*k0) == |k0| exactly for every finite
                # lattice value and sign(0) is already 0. Pinned by
                # test_the_massless_gradient_reduces_to_the_1d_convention, so a
                # future change here cannot silently move the 1D path.
                #
                # Only the CLASSICAL variant reaches this: the quantum one
                # differentiates T through the Bopp difference qd(), which needs
                # no gradient and is untroubled by |k| being non-smooth at 0.
                def grad(i):
                    def g(*k):
                        r = xp.sqrt(sum(kj**2 for kj in k))
                        nz = r > 0.
                        return c*xp.where(nz, k[i]/xp.where(nz, r, 1.), 0.)
                    return g
            else:
                def grad(i):
                    return lambda *k: c*k[i]/xp.sqrt(
                        sum(kj**2 for kj in k) + m**2*c**2)
        return T, tuple(grad(i) for i in range(self.ndim))

    def rebuild(self):
        """(Re)build the exponent generators dU, dT and the sub-shape energy
        meshes. Called on construction and after any change to U/mass/c/
        hbar_eff."""
        g, xp = self.grid, self.xp
        T, dTdk = self._kinetic()
        if self.quantum:
            dU = self.qd(self.U, g.Q, [1j*t for t in g.Theta])
            dT = self.qd(T, g.K, [-1j*l for l in g.Lam])/2.
        else:
            dU = sum(gU(*g.Q)*1j*th for gU, th in zip(self.gradU, g.Theta))
            dT = -sum(dT_i(*g.K)*1j*lm
                      for dT_i, lm in zip(dTdk, g.Lam))/2.
        # Broadcast to the full shape so expU/expT multiplications are plain
        # elementwise products regardless of how U/gradU broadcast.
        self.dU_im = self._rate_mesh(dU, g.shape, "dU")
        self.dT_im = self._rate_mesh(dT, g.shape, "dT")
        # Energy on the shifted SUB-grids (display/observables): <H> is
        # int U*n_q + int T*n_k over the plane reductions, so neither term ever
        # needs the full shape. The rest energy m*c^2 cancels identically
        # inside dT (the kinetic term enters only as a difference) but
        # dominates <H>; observables subtract it.
        self.U_mesh = self._sub_mesh(self.U, g.spatial_axes)
        self.T_mesh = self._sub_mesh(T, g.momentum_axes)
        self.rest_energy = self.mass*self.c**2 if self.relativistic else 0.0

    def _sub_mesh(self, f, axs):
        """f evaluated on the shifted coordinate sub-grid spanned by `axs`,
        float64, with the exact shape a reduction over the other axes has."""
        xp = self.xp
        shape = tuple(self.grid.N[a] for a in axs)
        v = xp.asarray(f(*self.grid.sub_meshes(axs)))
        return xp.ascontiguousarray(
            xp.broadcast_to(v.real.astype(xp.float64), shape))

    def _rate_mesh(self, d, shape, name):
        """Store the exponent generator as the REAL angular-rate mesh w, where
        the generator is exactly 1j*w — half the bytes of the complex128 array
        it replaces, and exponents() rebuilds the phase from it unchanged.

        dU and dT are purely imaginary for every valid variant: U is required
        real (potential.py) so the quantum differential of a real function over
        a real Bopp increment is imaginary, and the classical branch is
        literally a sum of gradU*1j*Theta. The real part is not dropped
        silently, though — a nonzero one means |expU| != 1, an evolution that
        quietly gains or loses norm, which is precisely what
        test_exponents_unit_modulus pins. Refuse it here rather than let it
        decay a run."""
        xp = self.xp
        d = xp.asarray(d, dtype=xp.complex128)
        re_max = float(xp.max(xp.abs(d.real)))
        if re_max > 0.0:
            im_max = float(xp.max(xp.abs(d.imag)))
            if re_max > 1e-13*max(im_max, 1.0):
                raise ValueError(
                    "%s has a non-negligible real part (max|Re| = %.3g vs "
                    "max|Im| = %.3g): |exp| would differ from 1 and the "
                    "evolution would not conserve norm" % (name, re_max, im_max))
        return xp.ascontiguousarray(xp.broadcast_to(d.imag, shape))

    def set_grid(self, grid):
        """Adopt a regridded domain (auto-expand): swap the grid, rebuild
        the FFT plans only when the shape changed, rebuild the exponents.
        U/gradU callables are shape-agnostic closures — nothing to re-derive.

        RELEASE BEFORE ALLOCATE when the shape grows. rebuild() builds the new
        full-shape meshes while the old ones are still referenced by self, and a
        4D doubling makes that overlap the difference between fitting on a card
        and not: measured on a 3090, the switch peaked at 1.27x the NEW
        footprint with the old arrays merely dropped at the end. Dropping them
        first, and handing the blocks back to the driver rather than leaving
        them in the pool at the wrong size, is what lets the M3 regrid guard
        count what we already hold as available instead of hoping for it.

        A MOVE DELIBERATELY GETS NEITHER, and that is measured rather than
        assumed — the obvious symmetry argument (it rebuilds the same meshes, so
        holding both pairs must be 16 B/cell of pure overlap) predicts a saving
        that does not exist. `scripts/bench.py --ndim 2 --regrid move`, with the
        drop hoisted out of this branch and without, on a 3090 at 32^4/48^4/64^4:

            float64   peak/steady 1.000   driver +0 MiB      both orders
            float32   peak/steady 1.143   driver +16/82/256  both orders

        In float64 the pool is already holding blocks of every size the rebuild
        asks for (the arena high-water is set by adjust_step's two exponent pairs
        and qd()'s Bopp temporaries, both larger than anything here), so the card
        never sees the switch at all. In float32 the extra 16 B/cell is exactly
        ONE complex128 mesh pair over a 96 B/cell arena, and dropping the old
        pair first merely swaps which 16 B/cell it is: the pair overlap becomes
        rebuild()'s own float64 Bopp intermediate, since single precision leaves
        no double-sized free blocks to reuse. That row is also unreachable in a
        session — auto-expand is float64-only (MSG_EXPAND_F32) — so on the only
        path a move takes, its transient is zero. Which is the whole reason
        core/fit.py refuses a move for nothing.

        And _release_pool() on a move would make an OOM MORE likely, not less: it
        clears this thread's cuFFT plan cache, whose work area is real VRAM
        (~16 B/cell, see config.BYTES_PER_CELL_2D), and on an unchanged shape
        that plan is still VALID. Throwing a correctly-sized live allocation away
        to force a re-plan and a fresh cudaMalloc, at the instant the pool is
        most loaded, is the opposite of the trade it makes for a doubling — where
        the cached plan is for the old shape and is dead weight.

        This changes ALLOCATION ORDER ONLY — every array rebuilt here is a pure
        function of (grid, U, gradU, mass, c, hbar_eff), so ndim=1 results are
        bitwise unaffected. The FFT plans are rebuilt only on a shape change, as
        before; the difference is that the old ones are dropped first.
        """
        reshaped = grid.shape != self.grid.shape
        if reshaped:
            # Drop the old full-shape arrays and plans BEFORE the new ones are
            # built. Nothing below reads them: rebuild() derives everything from
            # the grid and the physics, and every outside reader (exponents,
            # worker._finite, observables.energy) runs after this returns. If
            # rebuild() then raises, this object is left unusable rather than
            # holding its previous meshes — which is the same outcome as before,
            # because the only caller (worker._apply_regrid) treats a failure
            # here as fatal by design: a per-worker rollback would desync the
            # lockstep geometry. NB set_physics() deliberately does NOT do this:
            # its caller rolls back and keeps evolving, which works only because
            # a failed rebuild there leaves the previous meshes in place.
            self.dU_im = self.dT_im = self.U_mesh = self.T_mesh = None
            self._fft_sp = self._ifft_sp = self._fft_mo = self._ifft_mo = None
            self._release_pool()
            self.grid = grid
            self._plan(grid)
        else:
            self.grid = grid
        self.rebuild()

    def _release_pool(self):
        """Return this device's free pool blocks to the driver, plus this
        thread's cuFFT plans (their work areas are real VRAM and the old
        shape's are dead). Same treatment worker._release_gpu_pool gives a
        closing session, for the same reason: free_all_blocks() returns only
        blocks that are FREE, so it is worth calling exactly when we have just
        dropped a multi-GiB set of them.

        THE BLAST RADIUS IS THE WHOLE PROCESS, not this session. The CuPy
        default pool is per process and nothing installs a per-backend allocator
        — that is exactly why routers/preview.py owns a `_pool` of its own — so
        this also returns the cached blocks of every OTHER session's workers on
        this device. Acceptable on these grounds and no others: it hands back
        only blocks that are already FREE, so the cost is a cudaMalloc for
        whoever asks next and never correctness; and it runs once per regrid,
        not once per IC keystroke, which is what made the preview's version of
        this the wrong trade. The pinned pool is deliberately left alone: it is
        host memory and a regrid does not churn it.
        """
        if not self.backend.is_gpu:
            return                     # host arrays: refcounting already did it
        # Best effort throughout: this only hands back memory that is already
        # free, so failing to do it costs headroom, never correctness — and it
        # must never be the thing that kills a worker mid-regrid.
        try:
            xp = self.xp
            try:
                xp.fft.config.get_plan_cache().clear()
            except Exception:          # pragma: no cover - cupy internals
                pass
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:              # pragma: no cover - never fatal
            pass

    def set_physics(self, *, U=None, gradU=None, mass=None, c=None,
                    hbar_eff=None, tol=None):
        if U is not None: self.U = U
        if gradU is not None: self.gradU = tuple(gradU)
        if mass is not None: self.mass = float(mass)
        if c is not None: self.c = float(c)
        if hbar_eff is not None: self.hbar_eff = float(hbar_eff)
        if tol is not None: self.tol = float(tol)
        self._check_grad()
        self.rebuild()

    # -- stepping -----------------------------------------------------------

    def exponents(self, dt):
        """exp(1j*dt*w) for the real rate meshes — bitwise what exp(dt*1j*w)
        gave when the generators were stored complex (exp of a zero real part
        is exactly 1.0, so the phase is untouched).

        The phase is always COMPUTED in double (dU_im/dT_im are float64) and
        then rounded to the working dtype. The cast is not optional in float32
        mode: `B *= expT` with B complex64 and expT complex128 is silently
        legal in both numpy and cupy, yielding a correct complex64 B via a full
        complex128 temporary — right answer, double the exponent-slot bytes,
        most of the speedup gone, and nothing in the physics to reveal it."""
        xp = self.xp
        # copy=False: xp.exp already returns a fresh complex128 mesh, so in
        # float64 mode the cast must be a no-op. Without it astype copies
        # regardless — 1 GiB allocated and 2.5 ms burned per mesh at 8192^2,
        # twice per call, in the DEFAULT precision, to produce the array we
        # already had.
        return (xp.exp(1j*(dt*self.dU_im)).astype(self.cdtype, copy=False),
                xp.exp(1j*(dt*self.dT_im)).astype(self.cdtype, copy=False))

    def solve_spectral(self, W, expU, expT):
        """One Strang step (solve.py:130-140). W in shifted order; returns a
        fresh real array (never a view into an FFT plan buffer). The result's
        dtype follows the working precision, so W is float32 from the first
        step of a float32 session.

        Axis transitions are spelled out per line: at four axes this is what
        makes the code auditable, and a wrong pairing here is silent."""
        xp = self.xp
        B = xp.asarray(W, dtype=self.cdtype)
        B = self._fft_sp(B)    # (q,k) -> (lambda,k)
        B *= expT
        B = self._ifft_sp(B)   # (lambda,k) -> (q,k)
        B = self._fft_mo(B)    # (q,k) -> (q,theta)
        B *= expU
        B = self._ifft_mo(B)   # (q,theta) -> (q,k)
        B = self._fft_sp(B)    # (q,k) -> (lambda,k)
        B *= expT
        B = self._ifft_sp(B)   # (lambda,k) -> (q,k)
        return xp.ascontiguousarray(B.real)

    def adjust_step(self, dt, W, maxtries=15):
        """Adaptive timestep control (solve.py:142-158): shrink |dt| until one
        full step and two half steps agree to relative tolerance tol.
        Returns (W_next, dt, expU, expT). Works for either sign of dt."""
        xp = self.xp
        tries = 0
        while True:
            tries += 1
            expU, expT = self.exponents(dt)
            W1 = self.solve_spectral(W, expU, expT)
            expUn, expTn = self.exponents(0.5*dt)
            W2 = self.solve_spectral(self.solve_spectral(W, expUn, expTn), expUn, expTn)
            # float64 accumulators: a no-op in double mode, but in float32 mode
            # these are sums over up to 16.7M elements and the controller's
            # decision must not be roundoff.
            rel = float(xp.sum(xp.abs(W1 - W2), dtype=xp.float64)
                        / xp.sum(xp.abs(W1), dtype=xp.float64))
            if rel < self.tol:
                break
            if tries > maxtries:
                log.warning("adjust_step: giving up after %d attempts (rel=%.3g)", maxtries, rel)
                break
            dt *= 0.7
        return W1, dt, expU, expT
