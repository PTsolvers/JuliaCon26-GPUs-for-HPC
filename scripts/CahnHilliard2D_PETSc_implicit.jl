#=
  CahnHilliard2D_PETSc.jl — implicit, parallel Cahn-Hilliard on a PETSc DMDA

  Parallel/implicit counterpart of `CahnHilliard2D.jl` (explicit, serial,
  CairoMakie).  Same physics, but discretised as a *mixed* (two-field) system so
  that the spatial stencil stays a 5-point STAR stencil and PETSc's DMDA
  machinery (ghost exchange, FD coloring, algebraic multigrid) applies directly.

  ── Equations ────────────────────────────────────────────────────────────────

  The Cahn-Hilliard equation

      ∂C/∂t = D ∇²μ,        μ = C³ − C − γ ∇²C

  is a 4th-order PDE in C.  Writing it as a system in (C, μ) keeps every
  operator second order, i.e. a 5-point stencil:

      R_C = (C − Cᵒˡᵈ)/dt − D ∇²μ         = 0
      R_μ = μ − (C³ − C) + γ ∇²C          = 0

  Time discretisation is backward Euler (fully implicit): every residual above
  is evaluated at the *new* time level, so there is no explicit stability limit
  dt ≲ dx⁴.  Each step is one nonlinear solve handled by SNES (Newton).

  ── Discretisation ───────────────────────────────────────────────────────────

  Cell-centred finite differences on a uniform nx × ny grid, 2 DOFs per node
  (C, μ), with zero-flux (Neumann) boundary conditions ∂C/∂n = ∂μ/∂n = 0 — the
  same boundary treatment as the explicit script.  ∂μ/∂n = 0 means no mass
  crosses the boundary, so the total mass ∫C is conserved to solver tolerance.

  ── Parallelism ──────────────────────────────────────────────────────────────

  The DMDA distributes the grid across MPI ranks; each rank evaluates the
  residual only on the nodes it owns, reading neighbour values from the ghost
  layer that `dm_global_to_local!` fills.

  ── Jacobian ─────────────────────────────────────────────────────────────────

  By default the Jacobian is written out **analytically**.  The system has a
  single nonlinear term (`C³ − C`, derivative `3C² − 1`); everything else is a
  constant-coefficient Laplacian, so the exact Jacobian is a short stencil
  formula.  Its COO sparsity pattern still comes from
  `PETSc.dmda_star_fd_coloring` (as in `examples/ex19.jl`), so assembly stays
  local and parallel.

  Pass `-fd_jacobian` to use finite-difference coloring instead.  That path
  needs `n_colors + 1 = 11` full residual evaluations per Jacobian, and is
  ~1.8× slower overall (257², 10 steps: 5.5 s analytic vs 9.5 s FD) while
  producing identical Newton iteration counts.  The two agree to
  `||J − Jfd||_F/||J||_F ≈ 4e-11` (`-snes_test_jacobian`).

  ── Running in parallel: mpiexecjl ───────────────────────────────────────────

  Do **not** use a system `mpiexec` — it will generally belong to a different
  MPI installation than the one MPI.jl and PETSc_jll are built against, and the
  ranks then fail to form a communicator (or crash outright, e.g. with a
  missing `libhwloc` on macOS).

  MPI.jl ships its own launcher for exactly this reason.  Install it once:

      julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'

  which writes `mpiexecjl` into `~/.julia/bin`.  It is a thin wrapper that sets
  the library paths for the MPI that MPI.jl is configured with, then calls that
  MPI's real `mpiexec`.  Add `~/.julia/bin` to your PATH, or call it by full
  path as below.

  Check which MPI binary MPI.jl is configured against with

      julia --project=. -e 'using MPI; println(MPI.MPIPreferences.binary)'

  In this project that reports `MPICH_jll`, which is also what PETSc_jll is
  built against — so `mpiexecjl` is the correct launcher.

  ── Usage ────────────────────────────────────────────────────────────────────

    # serial, default 128×128 grid
    julia --project=. scripts/CahnHilliard2D_PETSc.jl -snes_monitor -snes_converged_reason

    # 4 MPI ranks on a 256×256 grid, with solver output
    ~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/CahnHilliard2D_PETSc.jl \
        -da_grid_x 256 -da_grid_y 256 -snes_monitor -snes_converged_reason

    # algebraic multigrid on the coupled system (see "Linear solvers" below —
    # note this is *slower* than the default direct solve at these sizes)
    julia --project=. scripts/CahnHilliard2D_PETSc.jl \
        -da_grid_x 257 -da_grid_y 257 \
        -ksp_type fgmres -pc_type gamg -ksp_max_it 500 -ksp_rtol 1e-8 \
        -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 12 \
        -mg_levels_pc_type asm -mg_levels_sub_pc_type lu \
        -mg_coarse_pc_type lu

    # per-field multigrid via a Schur fieldsplit (fastest measured at 256²)
    julia --project=. scripts/CahnHilliard2D_PETSc.jl \
        -da_grid_x 257 -da_grid_y 257 \
        -snes_monitor -snes_converged_reason \
        -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur \
        -pc_fieldsplit_schur_fact_type full \
        -pc_fieldsplit_schur_precondition selfp \
        -fieldsplit_0_pc_type hypre -fieldsplit_1_pc_type hypre

  ── Linear solvers ───────────────────────────────────────────────────────────

  The default is a sparse direct LU (umfpack in serial, superlu_dist under
  MPI): robust at the grid sizes used here, but not scalable.  The mixed (C, μ)
  system is indefinite — the μ-equation has no time derivative, so its diagonal
  block is near-singular — which makes it awkward for black-box preconditioners.
  Measured behaviour (128²–256², serial):

    - default ILU, plain GAMG, plain hypre        → diverge immediately
    - Jacobi / SOR / point-block smoothers        → complete stall
    - GAMG + GMRES(8)/ILU(1) smoother             → 129² only; diverges at 257²
    - GAMG + GMRES(12)/ASM-LU, ksp_rtol 1e-8      → robust at 257², but ~13×
                                                    slower than direct LU
    - geometric MG (`-pc_type mg`) + same smoother→ works at 129², diverges 257²
    - fieldsplit *schur* + hypre blocks           → works, fastest iterative
                                                    option measured (256²: 18 s
                                                    vs 22 s for direct LU)
    - fieldsplit additive / multiplicative        → diverge (coupling dominates)

  The lesson: a Krylov-accelerated smoother is essential, and any field split
  must keep the C–μ coupling (i.e. Schur, not additive).  The script errors out
  rather than silently freezing if a chosen preconditioner diverges.

  Caveat: at the grid sizes used here the direct solver is still the one to
  beat.  The multigrid options above are included because they are what scales
  to much larger problems, not because they win at 257².

  Non-PETSc options handled by this script:
    -t_end <val>  physical end time               (default 2e-2; spinodal
                                                   decomposition starts ≈8e-3)
    -dt <val>     time step                       (default 2e-4, ~440× the
                                                   explicit stability limit)
    -nt <n>       step count; overrides the value derived from -t_end/-dt
    -nvis <n>     report/plot interval, 0 disables (default 10)
    -novis        disable output entirely

  Plotting is only done on rank 0 and only when running serially with GLMakie
  available; in parallel the run prints summary lines (mass and extrema)
  instead, which is what you want for a scaling run anyway.
=#

using MPI
using PETSc
using Printf
using Random

# ── Options ──────────────────────────────────────────────────────────────────
opts = PETSc.parse_options(ARGS)

MPI.Initialized() || MPI.Init()

# Start of the total-walltime clock, reported at the end of the run.  This is
# as early as we can measure: it covers PETSc initialisation, DMDA and coloring
# setup, JIT compilation of the callbacks, and the time loop itself.
t_script_start = MPI.Wtime()

# Default linear solver: a sparse direct LU.
#
# The mixed (C, μ) system is indefinite and strongly non-symmetric, and the
# usual black-box preconditioners (ILU, GAMG, hypre/BoomerAMG) all stall on it
# at the large time steps this example is built to take — the linear solve
# diverges and the state silently stops advancing.  A direct factorisation is
# robust and is perfectly adequate at the grid sizes used here; `superlu_dist`
# is a parallel direct solver, so this default also works under MPI.
#
# These are only *defaults*: anything passed on the command line wins, so you
# can still experiment with `-pc_type gamg` etc.
#
# The Newton defaults matter too: once sharp interfaces form, the C³ term makes
# the system stiff enough that an undamped Newton step can overshoot, so a
# backtracking line search (`bt`) is used and the iteration cap is raised.
opts = merge(
    (ksp_type = "preonly",
     pc_type  = "lu",
     pc_factor_mat_solver_type = MPI.Comm_size(MPI.COMM_WORLD) == 1 ?
                                 "umfpack" : "superlu_dist",
     snes_linesearch_type = "bt",
     snes_max_it = 50),
    opts,
)

petsclib = PETSc.getlib(; PetscScalar = Float64, PetscInt = Int64)
PETSc.initialize(petsclib, log_view=false)

const PetscScalar = petsclib.PetscScalar
const PetscInt    = petsclib.PetscInt

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# ── Physics (identical to CahnHilliard2D.jl) ─────────────────────────────────
# Grid units: dx = dy = 1 throughout, exactly as in CahnHilliard2D_plain.jl.
const lx, ly = 1.0, 1.0             # (unused in grid units; kept for the DMDA coordinate calls)
const D      = PetscScalar(1.0)     # mobility
const wcell  = 4.0                  # interface width, in cells -- resolve with >= 4
const γ      = PetscScalar(wcell^2 / 8)  # = 2
const C̄      = PetscScalar(0.0)     # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
const ampl   = PetscScalar(0.02)    # initial noise amplitude
const dx     = PetscScalar(1.0)     # grid units
const dy     = PetscScalar(1.0)

# ── Numerics ─────────────────────────────────────────────────────────────────
#
# The run is defined by a physical end time `t_end` and a step size `dt`; the
# step count follows.  `CahnHilliard2D.jl` uses the same `t_end`, so the two
# scripts simulate exactly the same problem and their walltimes are directly
# comparable — the only difference being that the explicit script's dt is
# pinned to its stability limit (≈4.6e-7 at 128², shrinking as dx⁴) whereas dt
# here is chosen for accuracy and is ~440× larger.
#
# The interesting physics needs t ≳ 1e-2: spinodal decomposition only becomes
# visible around t ≈ 8e-3, before which the field is still essentially the
# initial noise.  Hence the default t_end = 2e-2, safely past the onset.
#
# Passing `-nt` explicitly overrides the step count derived from `t_end`.
# In grid units κmax = 4/dx² + 4/dy² = 8, giving the explicit stability limit below.
# Being implicit we are free of it: dt defaults to `dt_factor` x that limit.
const κmax    = 4 / dx^2 + 4 / dy^2                    # = 8
const dt_expl = 2 / (D * κmax * (γ * κmax + 2)) / 2    # explicit 4th-order limit, safety 2
dt_factor = PetscScalar(PETSc.typedget(opts, :dt_factor, 100))
dt_0  = PetscScalar(PETSc.typedget(opts, :dt, dt_factor * dt_expl))
t_end = PetscScalar(PETSc.typedget(opts, :t_end, 40_000 * dt_expl))
nt    = Int(PETSc.typedget(opts, :nt, ceil(Int, t_end / dt_0)))
nvis  = Int(PETSc.typedget(opts, :nvis, 10))
haskey(opts, :novis) && (nvis = 0)

# ── DMDA: 2 DOFs per node (C, μ), 5-point STAR stencil ──────────────────────
#
# `DM_BOUNDARY_GHOSTED` allocates a ghost layer outside the physical domain
# that *we* fill, which is how the zero-flux (Neumann) boundary conditions are
# imposed in the residual below by mirroring the interior value.  This matches
# the boundary treatment of the explicit script `CahnHilliard2D.jl`.
#
# Note: `DM_BOUNDARY_PERIODIC` is *not* usable here — PETSc's
# `IS_COLORING_LOCAL` (which `dmda_star_fd_coloring` relies on) rejects a
# periodic direction whose two sides live on the same rank, which is always the
# case in serial and for small rank counts.
da = PETSc.DMDA(
    petsclib, comm,
    (PETSc.DM_BOUNDARY_GHOSTED, PETSc.DM_BOUNDARY_GHOSTED),
    (128, 128),               # default grid; override with -da_grid_x/-da_grid_y
    2,                        # DOFs per node: (C, μ)
    1,                        # stencil width
    PETSc.DMDA_STENCIL_STAR;
    opts...,
)

info = PETSc.getinfo(da)
nx   = Int(info.global_size[1])
ny   = Int(info.global_size[2])

# Same header line as CahnHilliard2D_plain.jl, plus the implicit-specific numbers.
if rank == 0
    @printf("PETSc-i nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)  %d rank(s)\n",
            nx, ny, γ, dt_0, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)), nranks)
    @printf("        implicit (backward Euler): dt = %.0fx the explicit limit %.5g, %d steps to t = %.2f\n",
            dt_0 / dt_expl, dt_expl, nt, nt * dt_0)
end

# ── SNES ─────────────────────────────────────────────────────────────────────
snes = PETSc.SNES(petsclib, comm; opts...)
PETSc.setDM!(snes, da)

# ── Solution vectors ─────────────────────────────────────────────────────────
# x      : current (unknown) state, 2 DOFs/node
# C_old  : concentration at the previous time level, stored *with ghosts* in
#          DMDA local layout so the residual can read it with the same indexing
#          as the ghosted solution.  (Only C enters the time derivative; μ has
#          no time derivative, it is a constraint.)
x     = PETSc.DMGlobalVec(da)
x_old = PETSc.DMGlobalVec(da)
l_old = PETSc.DMLocalVec(da)

# ── Initial condition: C̄ + white noise, μ consistent with it ────────────────
#
# Each rank seeds from its own rank id so the field is reproducible for a given
# rank count.  (The noise pattern therefore differs between rank counts; that is
# fine for a demo, and the mean is corrected globally below.)
Random.seed!(1234 + rank)

PETSc.withlocalarray!(x; read = false, write = true) do x_arr
    corners = PETSc.getcorners(da)
    nx_own = corners.upper[1] - corners.lower[1] + 1
    ny_own = corners.upper[2] - corners.lower[2] + 1
    xp = reshape(x_arr, 2, nx_own, ny_own)
    @views xp[1, :, :] .= C̄ .+ ampl .* randn(PetscScalar, nx_own, ny_own)
    @views xp[2, :, :] .= 0          # μ; the first Newton solve fixes it up
end

# Correct the mean of C globally so that ⟨C⟩ = C̄ exactly across all ranks.
# The Cahn-Hilliard equation conserves mass, so this sets the value the run
# will hold on to for all time.
local_sum, local_n = PETSc.withlocalarray!(x; read = true, write = false) do x_arr
    (sum(@view x_arr[1:2:end]), length(x_arr) ÷ 2)
end
global_sum = MPI.Allreduce(local_sum, +, comm)
global_n   = MPI.Allreduce(local_n, +, comm)
mean_shift = C̄ - global_sum / global_n
PETSc.withlocalarray!(x; read = true, write = true) do x_arr
    @views x_arr[1:2:end] .+= mean_shift
end

# ── Residual ─────────────────────────────────────────────────────────────────
#
# Evaluated on the locally owned nodes only.  Interior-neighbour values come
# from `l_x`, which `dm_global_to_local!` fills with the ghost layer from the
# neighbouring ranks; indices into the ghosted array are offset by (ox, oy)
# relative to the owned array, exactly as in ex19.jl.
#
#   R_C[i,j] = (C − Cᵒˡᵈ)/dt − D ∇²μ
#   R_μ[i,j] = μ − (C³ − C) + γ ∇²C
#
# Zero-flux (Neumann) conditions ∂C/∂n = ∂μ/∂n = 0 on the physical boundary are
# imposed by *mirroring*: outside the domain the neighbour value is taken equal
# to the value at the node itself, which makes the corresponding one-sided
# difference vanish.  `mirror` below does that, so the loop body itself has no
# boundary branches.  ∂μ/∂n = 0 makes the scheme exactly mass conserving: the
# Laplacian of μ is in divergence form and no flux crosses the boundary.
#
# `ig`/`jg` are the *global* 1-based node indices, used to detect which
# neighbours are outside the physical domain.
@inline function mirror(x_par, d, xi, xj, ni, nj, outside)
    return outside ? x_par[d, xi, xj] : x_par[d, ni, nj]
end

function cahn_hilliard_residual!(f_par, x_par, xold_par, nx_own, ny_own,
                                 ox, oy, xs, ys, mx, my, dt, D, γ, dx, dy)
    idx2 = one(PetscScalar) / dx^2
    idy2 = one(PetscScalar) / dy^2
    idt  = one(PetscScalar) / dt

    @inbounds for lj in 1:ny_own, li in 1:nx_own
        xi = li + ox        # ghost-array indices of this node
        xj = lj + oy
        ig = xs + li - 1    # global 1-based indices
        jg = ys + lj - 1

        # which neighbours fall outside the physical domain?
        out_w = ig == 1;   out_e = ig == mx
        out_s = jg == 1;   out_n = jg == my

        C = x_par[1, xi, xj]
        μ = x_par[2, xi, xj]

        # 5-point Laplacians, with mirrored values at the physical boundary
        Cw  = mirror(x_par, 1, xi, xj, xi-1, xj, out_w)
        Ce  = mirror(x_par, 1, xi, xj, xi+1, xj, out_e)
        Cs  = mirror(x_par, 1, xi, xj, xi, xj-1, out_s)
        Cn  = mirror(x_par, 1, xi, xj, xi, xj+1, out_n)
        ∇²C = (Cw - 2C + Ce) * idx2 + (Cs - 2C + Cn) * idy2

        μw  = mirror(x_par, 2, xi, xj, xi-1, xj, out_w)
        μe  = mirror(x_par, 2, xi, xj, xi+1, xj, out_e)
        μs  = mirror(x_par, 2, xi, xj, xi, xj-1, out_s)
        μn  = mirror(x_par, 2, xi, xj, xi, xj+1, out_n)
        ∇²μ = (μw - 2μ + μe) * idx2 + (μs - 2μ + μn) * idy2

        C_old = xold_par[1, xi, xj]

        # transport equation (backward Euler)
        f_par[1, li, lj] = (C - C_old) * idt - D * ∇²μ
        # chemical-potential constraint
        f_par[2, li, lj] = μ - (C^3 - C) + γ * ∇²C
    end
    return nothing
end

r = similar(x)

# ── Cᵒˡᵈ on each multigrid level ──────────────────────────────────────────────
#
# With `-pc_type mg`, PETSc evaluates the residual on coarsened DMs as well as
# the fine one, and the fine-grid `l_old` has the wrong length there.  For each
# coarse DM we therefore build (once) a `DMGlobalVec`/`DMLocalVec` pair plus the
# fine→coarse restriction (interpolation scaling), and re-restrict `x_old` into
# it whenever the timestep advances.
#
# Keyed by DM pointer.  `old_cache_stamp` tracks which timestep the cached
# coarse vectors were filled from, so they are refreshed exactly once per step.
const old_lvl_cache   = Dict{Ptr{Cvoid}, Any}()
const old_cache_stamp = Dict{Ptr{Cvoid}, Int}()
step_counter = 0

"""
    old_for_dm(da_) -> DMLocalVec

Ghosted `Cᵒˡᵈ` on the DM `da_`.  Returns the fine-grid `l_old` directly on the
fine level; on a coarse level it restricts `x_old` down and caches the result
for the current timestep.
"""
function old_for_dm(da_)
    da_.ptr == da.ptr && return l_old      # fine level: nothing to do

    key = da_.ptr
    entry = get(old_lvl_cache, key, nothing)
    if entry === nothing
        # Restriction from the fine DM to this coarse DM.  DMCreateInterpolation
        # returns the prolongation P together with its row-scaling vector; the
        # matching restriction is applied with MatRestrict.
        P, scale = PETSc.LibPETSc.DMCreateInterpolation(petsclib, da_, da)
        entry = (g = PETSc.DMGlobalVec(da_),
                 l = PETSc.DMLocalVec(da_),
                 P = P, scale = scale)
        old_lvl_cache[key] = entry
    end

    # Refresh only once per timestep.
    if get(old_cache_stamp, key, -1) != step_counter
        PETSc.LibPETSc.MatRestrict(petsclib, entry.P, x_old, entry.g)
        PETSc.LibPETSc.VecPointwiseMult(petsclib, entry.g, entry.g, entry.scale)
        PETSc.dm_global_to_local!(entry.g, entry.l, da_, PETSc.INSERT_VALUES)
        old_cache_stamp[key] = step_counter
    end
    return entry.l
end

PETSc.setfunction!(snes, r) do g_fx, snes_, g_x
    da_ = PETSc.getDM(snes_)

    l_x = PETSc.DMLocalVec(da_)
    PETSc.dm_global_to_local!(g_x, l_x, da_, PETSc.INSERT_VALUES)

    corners       = PETSc.getcorners(da_)
    ghost_corners = PETSc.getghostcorners(da_)

    xs, ys   = corners.lower[1], corners.lower[2]
    xe, ye   = corners.upper[1], corners.upper[2]
    xsg, ysg = ghost_corners.lower[1], ghost_corners.lower[2]
    xeg, yeg = ghost_corners.upper[1], ghost_corners.upper[2]

    nx_own = xe - xs + 1;   ny_own = ye - ys + 1
    nx_g   = xeg - xsg + 1; ny_g   = yeg - ysg + 1
    ox = xs - xsg;          oy = ys - ysg

    # Grid spacing from *this* DM, so the callback stays correct if PETSc hands
    # us a coarsened DM (e.g. under -pc_type mg).
    info_ = PETSc.getinfo(da_)
    mx_ = Int(info_.global_size[1])
    my_ = Int(info_.global_size[2])
    dx_ = PetscScalar(1.0)              # grid units (same on every MG level)
    dy_ = PetscScalar(1.0)

    # `Cᵒˡᵈ` on *this* DM.  Under geometric multigrid (`-pc_type mg`) PETSc
    # evaluates the residual on coarsened DMs too, and the fine-grid `l_old`
    # has the wrong size there — so restrict it onto the coarse DM once and
    # cache the result per DM.
    l_old_lvl = old_for_dm(da_)

    PETSc.withlocalarray!(g_fx, l_x, l_old_lvl;
                          read = (true, true, true),
                          write = (true, false, false)) do fx, lx, lold
        x_par    = reshape(lx,   2, nx_g,   ny_g)
        xold_par = reshape(lold, 2, nx_g,   ny_g)
        f_par    = reshape(fx,   2, nx_own, ny_own)
        cahn_hilliard_residual!(f_par, x_par, xold_par, nx_own, ny_own,
                                ox, oy, xs, ys, mx_, my_, dt_0, D, γ, dx_, dy_)
    end

    PETSc.destroy(l_x)
    return PetscInt(0)
end

# ── Jacobian: finite-difference coloring on the DMDA STAR stencil ────────────
#
# Same strategy as examples/ex19.jl: build the coloring and COO sparsity
# pattern once, then per Newton step perturb one color at a time and fill the
# COO value array with (F(x+h eₖ) − F(x))/h.  This gives a correct, fully
# parallel Jacobian without writing one by hand.
coloring = PETSc.dmda_star_fd_coloring(petsclib, da)

J = PETSc.LibPETSc.DMCreateMatrix(petsclib, da)
PETSc.LibPETSc.MatSetPreallocationCOOLocal(
    petsclib, J,
    PETSc.LibPETSc.PetscCount(coloring.nnz_coo),
    coloring.row_coo_local, coloring.col_coo_local,
)

val        = zeros(PetscScalar, coloring.nnz_coo)
x_pert_vec = PETSc.LibPETSc.VecDuplicate(petsclib, x)
f0_vec     = PETSc.LibPETSc.VecDuplicate(petsclib, x)
f1_vec     = PETSc.LibPETSc.VecDuplicate(petsclib, x)
h_eps      = PetscScalar(sqrt(eps(PetscScalar)))
inv_h      = one(PetscScalar) / h_eps

"""
    analytic_jacobian_values!(val, x_par, nx_own, ny_own, ox, oy, xs, ys,
                              mx, my, dt, D, γ, dx, dy)

Fill the COO value array `val` with the exact Jacobian of
`cahn_hilliard_residual!`, instead of rebuilding it by finite differences.

The system is nonlinear in exactly one term — the bulk free energy `C³ − C`,
whose derivative is `3C² − 1` — while every other contribution is a
constant-coefficient Laplacian.  Differentiating the two residuals

    R_C = (C − Cᵒˡᵈ)/dt − D ∇²μ
    R_μ = μ − (C³ − C) + γ ∇²C

gives, per node,

    ∂R_C/∂C = 1/dt              (diagonal only)
    ∂R_C/∂μ = −D  · [∇²]
    ∂R_μ/∂C = −(3C² − 1) + γ · [∇²]
    ∂R_μ/∂μ = 1                 (diagonal only)

where `[∇²]` is the 5-point stencil.  The zero-flux mirroring in the residual
means that when a neighbour lies outside the domain its stencil weight folds
onto the centre coefficient — handled by the `out_*` flags below, exactly
mirroring what `mirror` does in the residual.

`val` must be filled in the COO order established by `dmda_star_fd_coloring`:
nodes in j-major order, neighbours as self/left/right/bottom/top (each present
only when inside the domain), and a row-major 2×2 block per (row, col) pair.
"""
function analytic_jacobian_values!(val, x_par, nx_own, ny_own, ox, oy,
                                   xs, ys, mx, my, dt, D, γ, dx, dy)
    idx2 = one(PetscScalar) / dx^2
    idy2 = one(PetscScalar) / dy^2
    idt  = one(PetscScalar) / dt
    k = 1

    @inbounds for lj in 1:ny_own, li in 1:nx_own
        xi = li + ox
        xj = lj + oy
        ig = xs + li - 1
        jg = ys + lj - 1

        out_w = ig == 1;  out_e = ig == mx
        out_s = jg == 1;  out_n = jg == my

        C = x_par[1, xi, xj]

        # Centre weight of the mirrored 5-point Laplacian: a neighbour outside
        # the domain contributes its weight here instead (that is what the
        # mirroring does), so the corresponding off-diagonal simply vanishes.
        lap_c = -2idx2 - 2idy2 +
                (out_w ? idx2 : zero(idx2)) + (out_e ? idx2 : zero(idx2)) +
                (out_s ? idy2 : zero(idy2)) + (out_n ? idy2 : zero(idy2))

        # ── self block ────────────────────────────────────────────────────────
        val[k] = idt;              k += 1   # ∂R_C/∂C
        val[k] = -D * lap_c;       k += 1   # ∂R_C/∂μ
        val[k] = -(3C^2 - 1) + γ * lap_c; k += 1   # ∂R_μ/∂C
        val[k] = one(PetscScalar); k += 1   # ∂R_μ/∂μ

        # ── neighbours: self/left/right/bottom/top, matching the COO order ────
        # Only the ∇² terms couple to neighbours, so the (C,C) and (μ,μ) entries
        # of every off-diagonal block are zero.
        if ig > 1
            val[k] = zero(PetscScalar);   k += 1
            val[k] = out_w ? zero(PetscScalar) : -D * idx2; k += 1
            val[k] = out_w ? zero(PetscScalar) :  γ * idx2; k += 1
            val[k] = zero(PetscScalar);   k += 1
        end
        if ig < mx
            val[k] = zero(PetscScalar);   k += 1
            val[k] = out_e ? zero(PetscScalar) : -D * idx2; k += 1
            val[k] = out_e ? zero(PetscScalar) :  γ * idx2; k += 1
            val[k] = zero(PetscScalar);   k += 1
        end
        if jg > 1
            val[k] = zero(PetscScalar);   k += 1
            val[k] = out_s ? zero(PetscScalar) : -D * idy2; k += 1
            val[k] = out_s ? zero(PetscScalar) :  γ * idy2; k += 1
            val[k] = zero(PetscScalar);   k += 1
        end
        if jg < my
            val[k] = zero(PetscScalar);   k += 1
            val[k] = out_n ? zero(PetscScalar) : -D * idy2; k += 1
            val[k] = out_n ? zero(PetscScalar) :  γ * idy2; k += 1
            val[k] = zero(PetscScalar);   k += 1
        end
    end
    return nothing
end

# Set `-fd_jacobian` to fall back to the finite-difference coloring path (useful
# as a correctness check on the analytic derivatives above).
const use_fd_jacobian = haskey(opts, :fd_jacobian)

PETSc.setjacobian!(snes, J) do Jmat, snes_, g_x
    # Under geometric multigrid PETSc evaluates the Jacobian on coarse DMs too;
    # the coloring above belongs to the fine DM only, so defer to PETSc's own
    # FD coloring there.
    if PETSc.getDM(snes_).ptr != da.ptr
        PETSc.LibPETSc.SNESComputeJacobianDefaultColor(
            petsclib, snes_, g_x, Jmat, Jmat, C_NULL)
        return PetscInt(0)
    end

    if !use_fd_jacobian
        # ── Analytic Jacobian ─────────────────────────────────────────────────
        # One pass over the local grid, versus `n_colors + 1` full residual
        # evaluations for the FD path below.
        l_x = PETSc.DMLocalVec(da)
        PETSc.dm_global_to_local!(g_x, l_x, da, PETSc.INSERT_VALUES)

        corners       = PETSc.getcorners(da)
        ghost_corners = PETSc.getghostcorners(da)
        xs, ys   = corners.lower[1], corners.lower[2]
        xe, ye   = corners.upper[1], corners.upper[2]
        xsg, ysg = ghost_corners.lower[1], ghost_corners.lower[2]
        xeg, yeg = ghost_corners.upper[1], ghost_corners.upper[2]
        nx_own = xe - xs + 1;   ny_own = ye - ys + 1
        nx_g   = xeg - xsg + 1; ny_g   = yeg - ysg + 1

        PETSc.withlocalarray!(l_x; read = true, write = false) do lx
            x_par = reshape(lx, 2, nx_g, ny_g)
            analytic_jacobian_values!(val, x_par, nx_own, ny_own,
                                      xs - xsg, ys - ysg, xs, ys, nx, ny,
                                      dt_0, D, γ, dx, dy)
        end
        PETSc.destroy(l_x)

        PETSc.LibPETSc.MatSetValuesCOO(petsclib, Jmat, val, PETSc.LibPETSc.INSERT_VALUES)
        PETSc.LibPETSc.MatAssemblyBegin(petsclib, Jmat, PETSc.LibPETSc.MAT_FINAL_ASSEMBLY)
        PETSc.LibPETSc.MatAssemblyEnd(petsclib, Jmat, PETSc.LibPETSc.MAT_FINAL_ASSEMBLY)
        return PetscInt(0)
    end

    PETSc.LibPETSc.SNESComputeFunction(petsclib, snes_, g_x, f0_vec)
    PETSc.withlocalarray!(f0_vec; read = true, write = false) do f0
        for c in 1:coloring.n_colors
            cols = coloring.perturb_cols[c]
            isempty(cols) && continue

            PETSc.LibPETSc.VecCopy(petsclib, g_x, x_pert_vec)
            PETSc.withlocalarray!(x_pert_vec; read = true, write = true) do xp
                @inbounds for k in eachindex(cols)
                    xp[cols[k]] += h_eps
                end
            end

            PETSc.LibPETSc.SNESComputeFunction(petsclib, snes_, x_pert_vec, f1_vec)

            PETSc.withlocalarray!(f1_vec; read = true, write = false) do f1
                idxs = coloring.coo_idxs[c]
                rows = coloring.local_rows[c]
                @inbounds for k in eachindex(idxs)
                    val[idxs[k]] = (f1[rows[k]] - f0[rows[k]]) * inv_h
                end
            end
        end
    end

    PETSc.LibPETSc.MatSetValuesCOO(petsclib, Jmat, val, PETSc.LibPETSc.INSERT_VALUES)
    PETSc.LibPETSc.MatAssemblyBegin(petsclib, Jmat, PETSc.LibPETSc.MAT_FINAL_ASSEMBLY)
    PETSc.LibPETSc.MatAssemblyEnd(petsclib, Jmat, PETSc.LibPETSc.MAT_FINAL_ASSEMBLY)
    return PetscInt(0)
end

# ── Diagnostics / visualisation ──────────────────────────────────────────────
#
# Plotting requires the whole field on one rank, so it is only attempted in a
# serial run with GLMakie available.  In parallel we print conserved
# quantities instead, which is the useful thing to watch in a scaling run.
const CreatePlots = (nranks == 1) && (nvis > 0) && try
    using GLMakie
    true
catch
    false
end

# The field is held in an Observable so GLMakie's window updates live as the
# time loop runs (no re-`display` per frame).
fig     = nothing
C_obs   = nothing
title_obs = nothing
screen  = nothing
if CreatePlots
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    yc = LinRange(dy / 2, ly - dy / 2, ny)
    C_obs     = Observable(fill(Float32(C̄), nx, ny))
    title_obs = Observable("C,  t = 0")
    fig = Figure(; size = (700, 600))
    axs = Axis(fig[1, 1][1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y",
               title = title_obs)
    plt = heatmap!(axs, xc, yc, C_obs; colormap = :turbo, colorrange = (-1.1, 1.1))
    Colorbar(fig[1, 1][1, 2], plt)
    screen = display(fig)
end

"""
    check() -> (F, mass)

Free energy and total mass, matching `check` in `CahnHilliard2D_plain.jl`:

    F = Σ (C²−1)²/4 + γ/2 Σ (∇C)²

The gradient sum runs over interior faces, so it needs the ghost layer: each
rank adds the faces whose left/lower node it owns, counting every face once.
"""
function check()
    l_c = PETSc.DMLocalVec(da)
    PETSc.dm_global_to_local!(x, l_c, da, PETSc.INSERT_VALUES)
    corners       = PETSc.getcorners(da)
    ghost_corners = PETSc.getghostcorners(da)
    xs, ys   = corners.lower[1], corners.lower[2]
    xe, ye   = corners.upper[1], corners.upper[2]
    xsg, ysg = ghost_corners.lower[1], ghost_corners.lower[2]
    nx_g = ghost_corners.upper[1] - xsg + 1
    ny_g = ghost_corners.upper[2] - ysg + 1
    ox, oy = xs - xsg, ys - ysg

    lF, lm = PETSc.withlocalarray!(l_c; read = true, write = false) do lc
        xp = reshape(lc, 2, nx_g, ny_g)
        F = 0.0; m = 0.0
        @inbounds for lj in 1:(ye - ys + 1), li in 1:(xe - xs + 1)
            i, j = li + ox, lj + oy
            c = xp[1, i, j]
            F += ((c^2 - 1)^2) / 4
            m += c
            (xs + li - 1) < nx && (F += γ / 2 * (xp[1, i+1, j] - c)^2)
            (ys + lj - 1) < ny && (F += γ / 2 * (xp[1, i, j+1] - c)^2)
        end
        (F, m)
    end
    PETSc.destroy(l_c)
    return MPI.Allreduce(lF, +, comm), MPI.Allreduce(lm, +, comm)
end

"""
    report(it, t)

Print F and the mass drift, in the same format as `CahnHilliard2D_plain.jl`,
and update the GLMakie plot when running serially.
"""
function report(it, t)
    F, m = check()
    if rank == 0
        its = PETSc.LibPETSc.SNESGetIterationNumber(petsclib, snes)
        @printf("> step %6d, t = %8.2f, F = %.6g, Δmean = %+.2e, Newton its = %d\n",
                it, t, F, (m - m0) / (nx * ny), its)
    end

    if CreatePlots
        C = PETSc.withlocalarray!(x; read = true, write = false) do x_arr
            reshape(Float32.(@view x_arr[1:2:end]), nx, ny)
        end
        C_obs[]     = C                       # triggers a live redraw
        title_obs[] = @sprintf("C,  t = %.3e  (step %d)", t, it)
        yield()                               # let the GLMakie render task run
    end
    return nothing
end

# ── Time loop ────────────────────────────────────────────────────────────────
#
# One backward-Euler step = one SNES solve.  Before each solve, the previous
# solution is scattered into the ghosted local vector `l_old`, which the
# residual reads as Cᵒˡᵈ.
t = 0.0
PETSc.LibPETSc.VecCopy(petsclib, x, x_old)
PETSc.dm_global_to_local!(x_old, l_old, da, PETSc.INSERT_VALUES)

# Reference values: F must decrease monotonically and the mean must not drift.
F0, m0 = check()

nvis > 0 && report(0, t)

# Walltime of the time loop itself (setup, coloring and the initial report are
# excluded).  The barrier makes all ranks start the clock together, so the
# reduction below measures the actual wall-clock span rather than rank 0's
# head start.
MPI.Barrier(comm)
t_start = MPI.Wtime()

for it in 1:nt
    global t, step_counter

    # Marks the cached coarse-grid Cᵒˡᵈ vectors (if any) as stale, so the first
    # coarse residual evaluation of this step re-restricts x_old.
    step_counter = it

    # Solve the nonlinear system for the new time level.
    PETSc.solve!(x, snes)

    # A diverged solve must not pass silently: without this check the state
    # simply stops advancing and the run *looks* fine while producing a frozen,
    # patternless field.
    reason = PETSc.LibPETSc.SNESGetConvergedReason(petsclib, snes)
    if Int(reason) < 0
        error("SNES diverged at step $it (reason $reason). The default direct " *
              "solver was probably overridden with a preconditioner that cannot " *
              "handle this indefinite mixed system, or dt is too large.")
    end

    t += dt_0

    # Roll the solution forward: x -> x_old -> ghosted l_old.
    PETSc.LibPETSc.VecCopy(petsclib, x, x_old)
    PETSc.dm_global_to_local!(x_old, l_old, da, PETSc.INSERT_VALUES)

    if nvis > 0 && it % nvis == 0
        report(it, t)
    end
end

# The slowest rank sets the walltime, so reduce with `max`.
walltime = MPI.Allreduce(MPI.Wtime() - t_start, max, comm)

# Total script walltime: setup (PETSc init, DMDA, coloring, JIT) plus the time
# loop.  Measured here rather than after the plotting block below, so an
# interactive run does not include however long the window stays open.
total_walltime = MPI.Allreduce(MPI.Wtime() - t_script_start, max, comm)

if rank == 0
    println("done: $nt implicit steps, t = $(nt * dt_0)")
    @printf("walltime  : %.3f s  (%.4f s/step, %d ranks)  [time loop]\n",
            walltime, walltime / nt, nranks)
    @printf("total     : %.3f s  (incl. %.3f s setup + JIT)\n",
            total_walltime, total_walltime - walltime)
end

# Save the final frame and, when run as a script, keep the GLMakie window open
# until it is closed manually (otherwise the process would exit immediately).
if CreatePlots
    save("CahnHilliard2D_PETSc_implicit.png", fig)
    if !isinteractive()
        println("close the plot window to exit")
        wait(screen)
    end
end

# ── Cleanup ──────────────────────────────────────────────────────────────────
# The SNES holds an options database whose finalizer touches MPI; destroy it
# while PETSc is still alive (see the note in examples/ex19.jl).
if !isnothing(snes.opts)
    PETSc.destroy(snes.opts)
    snes.opts = nothing
end

GC.gc(true)
MPI.Barrier(comm)

PETSc.destroy(snes)
PETSc.destroy(J)
PETSc.destroy(x_pert_vec)
PETSc.destroy(f0_vec)
PETSc.destroy(f1_vec)
PETSc.destroy(l_old)
PETSc.destroy(x_old)
PETSc.destroy(x)
PETSc.destroy(r)
PETSc.destroy(da)
PETSc.finalize(petsclib)

MPI.Barrier(comm)
MPI.Finalize()

# On macOS ARM64 with MPICH ch4:ofi, MPICH's atexit handler can crash during
# teardown; quick_exit bypasses C atexit handlers (same workaround as ex19.jl).
if !isinteractive()
    ccall(:quick_exit, Cvoid, (Cint,), 0)
end
