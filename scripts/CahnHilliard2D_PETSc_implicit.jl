#=
  CahnHilliard2D_PETSc_implicit.jl — implicit (backward Euler) Cahn-Hilliard on a PETSc DMDA

  Implicit counterpart of `CahnHilliard2D_PETSc_explicit_2dof.jl`: same physics, same grid units
  (dx = dy = 1), same 2-DOF DMDA, same F/Δmean diagnostics and the same frames+gif output.  The
  only difference is *when* the right-hand side is evaluated.

  ── Equations ────────────────────────────────────────────────────────────────

  The Cahn-Hilliard equation

      ∂C/∂t = D ∇²μ,        μ = C³ − C − γ ∇²C

  is 4th order in C.  Writing it as a system in (C, μ) keeps every operator second order, i.e. a
  5-point stencil.  The explicit script evaluates the two relations in sequence; here they become
  residuals at the NEW time level, solved together:

      R_C = (C − Cᵒˡᵈ)/dt − D ∇²μ         = 0        (was:  C += dt·D·∇²μ)
      R_μ = μ − (C³ − C) + γ ∇²C          = 0        (was:  μ  = C³ − C − γ∇²C)

  That coupling is why C and μ must be solved for simultaneously, and hence why the DMDA carries
  2 DOFs per node.  Each step is one Newton solve (SNES).

  The payoff: no dt ≲ dx⁴ stability limit, so dt is set by accuracy.  The default is
  `dt_factor = 400` times the explicit limit, i.e. 100 steps where the explicit script needs
  40 000 to reach the same t_end ≈ 277.8 — and F ends within ~2% of the explicit result.

  ── Implicit scheme ──────────────────────────────────────────────────────────
  
  In the explicit scheme the new state is *computed*: every right-hand side is known from Cᵒˡᵈ,
  so each pass is an assignment.  Implicitly, the unknowns appear on both sides — C depends on
  ∇²μ, μ depends on C³ − C and ∇²C, all at the new time level — so a step is no longer an
  evaluation but the *solution* of a coupled nonlinear system R(x) = 0, with x = (C, μ) over the
  whole grid (2·n² unknowns).

  Newton solves that by repeated linearisation.  Expanding R about the current iterate,

      R(x + δx) ≈ R(x) + J(x) δx,        J = ∂R/∂x

  and asking for the update that zeroes the linear model gives

      J(x) δx = −R(x),        x ← x + δx

  where J(x) is a matrix and R(x) is a vector of the same size as x.  The linear system is solved    
  repeated until ‖R‖ is small.  So the Jacobian is not an optional extra: it *is* the linear
  operator of the system solved at every Newton iteration.  Its quality controls convergence
  (an exact J gives quadratic convergence — here 2–4 iterations per step), and its sparsity
  structure is what the linear solver and preconditioner actually see, which is why the block
  structure below matters as much as the values.

  PETSc needs J supplied in some form: written out analytically (what this script does), built by
  finite differences with colouring (`-fd_jacobian`), or approximated in a matrix-free manner. 

  ── Block structure of the Jacobian ──────────────────────────────────────────

  Newton needs J = ∂(R_C, R_μ)/∂(C, μ).  With 2 DOFs per node the unknowns are interleaved,
  x = [C₁ μ₁ C₂ μ₂ …], so J is naturally read as a matrix of 2x2 blocks — one block per pair of
  grid nodes.  Differentiating the two residuals gives, per node,

              ⎡ ∂R_C/∂C   ∂R_C/∂μ ⎤     ⎡  1/dt        −D ∇²   ⎤
      J   =   ⎢                   ⎥  =  ⎢                      ⎥
              ⎣ ∂R_μ/∂C   ∂R_μ/∂μ ⎦     ⎣ −(3C²−1) + γ∇²    1  ⎦

  where ∇² stands for the 5-point stencil.  Reading that block by block:

    - the DIAGONAL block (node with itself) is dense: all four entries are non-zero, and the
      only C-dependent one is −(3C²−1) + γ·(stencil centre).  Everything else is constant.
    - each OFF-DIAGONAL block (node with one of its 4 STAR neighbours) has only the two
      *anti-diagonal* entries filled, −D/dx² and +γ/dx², because C and μ couple to neighbours
      solely through the two Laplacians.  The (C,C) and (μ,μ) entries are zero there.
    - so J has block size 2 (PETSc reports `bs=2`) with 5 blocks per row: self + 4 neighbours.
      That is exactly the sparsity `PETSc.dmda_star_fd_coloring` enumerates, and what
      `analytic_jacobian_values!` fills in COO order (self, left, right, bottom, top).

  Two consequences worth noting.  The μ-row has **no 1/dt term** — the chemical-potential
  relation is a constraint, not an evolution equation — so the (μ,μ) block does not grow as
  dt → 0 and the whole system stays indefinite.  That is precisely what makes the linear solve
  hard, and why the preconditioner notes below matter.  It is also what `-pc_fieldsplit` exploits:
  the DMDA's 2 DOFs let PETSc split the C-field from the μ-field and treat the coupling with a
  Schur complement (see "Linear solvers").

  ── Discretisation ───────────────────────────────────────────────────────────

  Cell-centred finite differences, 2 DOFs per node (C, μ), zero-flux (Neumann) boundaries
  ∂C/∂n = ∂μ/∂n = 0 imposed by mirroring — the same treatment as the explicit scripts.
  ∂μ/∂n = 0 means no mass crosses the boundary, so ∫C is conserved to solver tolerance.

  ── Jacobian ─────────────────────────────────────────────────────────────────

  Written out **analytically**: the only nonlinear term is C³ − C (derivative 3C² − 1), everything
  else is a constant-coefficient Laplacian.  The COO sparsity pattern comes from
  `PETSc.dmda_star_fd_coloring` (as in PETSc.jl's examples/ex19.jl), so assembly stays local and
  parallel.  `-fd_jacobian` switches to finite-difference colouring instead: that needs
  n_colors + 1 = 11 residual evaluations per Jacobian and is ~1.8x slower, but gives identical
  Newton counts and agrees to ||J − Jfd||_F/||J||_F ≈ 4e-11 (check with -snes_test_jacobian).

  ── Running in parallel: mpiexecjl ───────────────────────────────────────────

  Do **not** use a system `mpiexec` — it will generally belong to a different MPI installation
  than the one MPI.jl and PETSc_jll are built against, and the ranks then fail to form a
  communicator (or crash outright, e.g. a missing libhwloc on macOS).  MPI.jl ships its own
  launcher; install it once with

      julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'

  which writes `mpiexecjl` into ~/.julia/bin.  Check that MPI.jl and PETSc agree with
  `julia --project=. -e 'using MPI; println(MPI.MPIPreferences.binary)'` — in this project both
  use MPICH_jll.

  If you want to run this on an HPC cluster, see the documentation of PETSc.jl on how to link the 
  package with MPI abi and the local MPI installation, which allows you to use precompiled binaries; 
  alternatively you can compile your own version of PETSc & link that to PETSc.jl 

  ── Usage ────────────────────────────────────────────────────────────────────

    # defaults: 513², dt = 400x the explicit limit, 100 steps.  NOTE this uses the *direct*
    # solver and is slow -- see "Linear solvers" for the fast invocations.
    julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl

    # smaller grid, solver output
    julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl -n 129 -snes_monitor -snes_converged_reason

    # iterative outer solver (FGMRES) with block-Jacobi preconditioner
    julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl -n 257  -snes_monitor -snes_converged_reason -ksp_type fgmres -pc_type bjacobi

    # RECOMMENDED -- geometric multigrid.  Converges in ~2 KSP iterations at ANY grid size,
    # which is the property to look for: the cost per step then grows only like the number of
    # unknowns.
    #
    # The default n = 513 is chosen for this: see the note on grid sizes below.
    julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl \
        -ksp_type fgmres -pc_type mg -pc_mg_levels 4 \
        -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 8 \
        -mg_levels_pc_type ilu -mg_levels_pc_factor_levels 1 \
        -mg_coarse_ksp_type preonly -mg_coarse_pc_type lu

    # ...and the same under MPI.  Two changes are needed, both because the sequential-only
    # factorisations above have no mpiaij implementation ("Could not locate a solver type for
    # factorization type ILU and matrix type mpiaij"):
    #   * the smoother's ILU must sit inside bjacobi (or asm), i.e. one ILU per rank;
    #   * the coarse solve must be a parallel LU.  `redundant` replicates the (tiny) coarse
    #     grid on every rank and factorises it locally; `-mg_coarse_pc_factor_mat_solver_type
    #     superlu_dist` is the true distributed alternative and works equally well here.
    ~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl \
        -ksp_type fgmres -pc_type mg -pc_mg_levels 4 \
        -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 8 \
        -mg_levels_pc_type bjacobi -mg_levels_sub_pc_type ilu \
        -mg_levels_sub_pc_factor_levels 1 \
        -mg_coarse_ksp_type preonly -mg_coarse_pc_type redundant \
        -mg_coarse_redundant_pc_type lu

    # alternative: additive Schwarz + ILU.  Simpler to write and slightly faster at these
    # sizes, but its iteration count grows with the rank count -- it is not grid-independent.
    ~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl \
        -ksp_type fgmres -pc_type asm -sub_pc_type ilu -sub_pc_factor_levels 3


  ── Linear solvers ───────────────────────────────────────────────────────────

  The default is a sparse direct LU (umfpack serial, superlu_dist under MPI): it never fails, and
  at the sizes people first try it is no slower than anything else.  It is nonetheless the worst
  choice asymptotically.  What makes this system awkward for the cheaper alternatives is that it
  is indefinite: the μ-equation has no time derivative, so its diagonal block does not grow as
  dt → 0.

  Two things are worth measuring, and they answer different questions.  Walltime answers "how long
  will this take", which is what you actually care about — but it is the noisier number: a
  background process or another job on the node can move it by a factor of several.  KSP iteration
  count answers "will this still work when the problem gets bigger", and is essentially immune to
  machine noise.  The iteration count tells you which walltime trend you are on.

  Measured serially, s/step (`-nt 5`), and mean KSP iterations per Newton step:

                    n=129    n=257    n=513   n=1025     KSP its (129 → 513)
      direct LU      0.62     1.04     3.93    22.73     — (no iterations)
      geometric MG   0.59     0.74     1.39     4.00     1.9  1.9  2.0   ← grid-independent
      ASM + ILU(3)   0.57     0.62     1.03     2.32     6.8  6.8  6.8   ← flat in n, grows w/ ranks

  The same walltimes as cost *factors* relative to n = 129.  Each row doubles n, so the number of
  unknowns N ~ (n-1)² grows 4x; an O(N) method — the best achievable, since every unknown must be
  touched at least once — would follow the "ideal" column:

      n        N/N₁₂₉   ideal O(N)     LU      MG     ASM
      129         1         1          1.0     1.0     1.0
      257         4         4          1.7     1.3     1.1
      513        16        16          6.4     2.4     1.8
      1025       64        64         36.8     6.8     4.1

  Every column beats "ideal", which is not a paradox: at n = 129 the per-step time is dominated by
  fixed overhead (residual and Jacobian assembly, PETSc bookkeeping) rather than by the linear
  solve, so the baseline is too expensive and every ratio is flattered.  Cost per unknown shows it
  directly — 37 / 36 / 35 ns at n = 129 for LU / MG / ASM, i.e. all three doing the same non-solve
  work, versus 22 / 3.8 / 2.2 ns at n = 1025 where the solve dominates.

  So read the asymptotic end.  Fitting t ~ N^p over the last refinement (513 → 1025):

      direct LU     p = 1.27    superlinear — the factorisation cost, as expected
      geometric MG  p = 0.76    ≈ O(N)
      ASM + ILU(3)  p = 0.59    ≈ O(N)

  Both iterative methods are at or below O(N) (p < 1 while the fixed overhead is still being
  amortised), whereas LU has clearly crossed into superlinear growth and keeps getting worse.  At
  129² all three are within 10% of each other — the direct solver is perfectly fine at small sizes,
  which is why it is a safe default; it is the *trend* that rules it out for large problems.

  MG and ASM are close in walltime, and ASM is even slightly ahead.  The iteration counts are what
  distinguish them: MG's ~2 is grid-independent by construction and stays ~2.8 on 4 ranks, whereas
  ASM is a one-level domain-decomposition method whose count grows with the number of subdomains —
  it diverged outright at 513² on 4 ranks in earlier testing.  MG is the method that keeps working
  as you scale up; ASM is the one that is quicker to type.

  (Timings were taken on a machine that was not perfectly idle; treat the ratios and exponents as
  meaningful and the absolute values as indicative.)

  A note on grid sizes.  The DMDA is vertex-centred: n points span n-1 intervals, and geometric
  multigrid halves the interval count at every level, so PETSc requires (n-1)/(n_coarse-1) to be
  an integer.  n = 513 gives 512 -> 256 -> 128 -> 64 intervals and coarsens cleanly; n = 512 gives
  511, which is odd, so even a single level fails. 
  That is why the default is n = 513 = 2^9 + 1 rather than 512.  Every other solver here works at
  any n; only -pc_type mg cares.  (The explicit scripts use 512 because they never coarsen.)

  Other options, measured:

    - geometric MG (`-pc_type mg`) + Krylov smoother  → best; see the Usage block for the exact
                                                        serial and MPI invocations
    - ASM + ILU(3)                                    → fastest at these sizes, simplest to write,
                                                        but not grid- or rank-independent
    - point-block Jacobi as the MG smoother           → converges, but slower than GMRES/ILU
    - fieldsplit *schur* + hypre blocks               → works; splits C from μ (see the Jacobian
                                                        block-structure section above)
    - fieldsplit additive / multiplicative            → diverge (the C–μ coupling dominates)
    - plain ILU / GAMG / hypre with default smoothers → diverge immediately

  The lesson: a Krylov-accelerated smoother is essential (plain Jacobi/SOR smoothers are too weak
  for an indefinite system), and any field split must keep the C–μ coupling — Schur, not additive.
  The script errors out rather than silently freezing if a chosen preconditioner diverges.

  NOTE for MPI: several of PETSc's factorisations are sequential-only.  `-pc_type ilu` and
  `-pc_type lu` have no `mpiaij` implementation, so under MPI they must be wrapped per rank
  (`-pc_type bjacobi -sub_pc_type ilu`) or replaced by a parallel solver (`redundant` + lu, or
  `-pc_factor_mat_solver_type superlu_dist`).  Otherwise PETSc aborts with
  "Could not locate a solver type for factorization type ILU and matrix type mpiaij".

  Keyword arguments (and their command-line equivalents):
    n=513          -n <size>          square grid size; use 2^k+1 for -pc_type mg
    dt_factor=400  -dt_factor <f>     dt as a multiple of the explicit stability limit
    nframes=40     -nframes <k>       number of frames/report lines over the whole run
    nt=nothing     -nt <n>            step count; overrides the value derived from t_end/dt
    nvis=nothing   -nvis <n>          fixed report interval, instead of the even frame schedule
    do_visu=true                      write frames + gif to output/<script name>/
    framerate=5                       gif framerate

  F and Δmean are printed whether or not do_visu is set; only the gather and rendering are
  skipped.  Rendering happens on rank 0 from a gathered field, so plotting works in parallel.
=#

using MPI
using PETSc
using Printf
using Random, Statistics, CairoMakie, OffsetArrays
include(joinpath(@__DIR__, "common.jl"))

# ── Pure kernels (no captured state) ─────────────────────────────────────────
# Arrays are indexed by GLOBAL indices (OffsetArray); no-flux via the mirror A[0]->A[1],
# A[n+1]->A[n].  The clamp also keeps us off the outer ghost ring, which PETSc never fills.
# Identical to `lap` in CahnHilliard2D_PETSc_explicit_2dof.jl.
Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[min(ix+1, nx), iy] - 2a + A[max(ix-1, 1), iy]) +
           (A[ix, min(iy+1, ny)] - 2a + A[ix, max(iy-1, 1)])
end

# The explicit script's two passes,
#     μ = C³ - C - γ∇²C          and          C += dt·D·∇²μ
# merged and evaluated at the NEW time level, as residuals Newton drives to zero.  Compare line
# for line with `chemical_potential!` / `update_concentration!` in the explicit 2-DOF file.
function residual!(R_C, R_μ, C, μ, Cold, γ, D, dt, nx, ny, xs, xe, ys, ye)
    @inbounds for iy in ys:ye, ix in xs:xe  # loop over global indices of the local block
        c = C[ix, iy]
        R_C[ix, iy] = (c - Cold[ix, iy]) / dt - D * lap(μ, ix, iy, nx, ny)
        R_μ[ix, iy] = μ[ix, iy] - (c * c * c - c) + γ * lap(C, ix, iy, nx, ny)
    end
    return
end

# Wrap DOF `d` globally-indexed: the vector is interleaved [C1 μ1 C2 μ2 ...], so reshape to
# (2, nx, ny) and view one field.  Same helper as the explicit 2-DOF script.
globalarray(a, c, d) =
    OffsetArray(view(reshape(a, 2, c.upper[1]-c.lower[1]+1, c.upper[2]-c.lower[2]+1), d, :, :),
                c.lower[1]:c.upper[1], c.lower[2]:c.upper[2])

# Run `f` on the given fields; a triple is (vector, corners, dof).  A vector appearing twice (both
# DOFs) is deduplicated before VecGetArray.
withvecs(f, triples...) = PETSc.withlocalarray!(unique(t[1] for t in triples)...) do arrays...
    vecs = unique(t[1] for t in triples)
    f(map(t -> globalarray(arrays[findfirst(==(t[1]), vecs)], t[2], t[3]), triples)...)
end

function analytic_jacobian_values!(val, x_par, nx_own, ny_own, ox, oy,
                                   xs, ys, mx, my, dt, D, γ, dx, dy)
    idx2 = one(dx) / dx^2
    idy2 = one(dy) / dy^2
    idt  = one(dt) / dt
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
        val[k] = one(eltype(val)); k += 1   # ∂R_μ/∂μ

        # ── neighbours: self/left/right/bottom/top, matching the COO order ────
        # Only the ∇² terms couple to neighbours, so the (C,C) and (μ,μ) entries
        # of every off-diagonal block are zero.
        if ig > 1
            val[k] = zero(eltype(val));   k += 1
            val[k] = out_w ? zero(eltype(val)) : -D * idx2; k += 1
            val[k] = out_w ? zero(eltype(val)) :  γ * idx2; k += 1
            val[k] = zero(eltype(val));   k += 1
        end
        if ig < mx
            val[k] = zero(eltype(val));   k += 1
            val[k] = out_e ? zero(eltype(val)) : -D * idx2; k += 1
            val[k] = out_e ? zero(eltype(val)) :  γ * idx2; k += 1
            val[k] = zero(eltype(val));   k += 1
        end
        if jg > 1
            val[k] = zero(eltype(val));   k += 1
            val[k] = out_s ? zero(eltype(val)) : -D * idy2; k += 1
            val[k] = out_s ? zero(eltype(val)) :  γ * idy2; k += 1
            val[k] = zero(eltype(val));   k += 1
        end
        if jg < my
            val[k] = zero(eltype(val));   k += 1
            val[k] = out_n ? zero(eltype(val)) : -D * idy2; k += 1
            val[k] = out_n ? zero(eltype(val)) :  γ * idy2; k += 1
            val[k] = zero(eltype(val));   k += 1
        end
    end
    return nothing
end

"""
    CahnHilliard2D_PETSc_implicit(; kwargs...)

Implicit (backward-Euler) Cahn-Hilliard on a PETSc DMDA.  Keyword arguments mirror the explicit
scripts; PETSc solver options are still taken from `ARGS`.
"""
function CahnHilliard2D_PETSc_implicit(; n=513, dt_factor=400, nframes=40, do_visu=true,
                                         framerate=5, nt=nothing, nvis=nothing)
    # `-n <size>` on the command line overrides the `n` keyword (square grid).
    n = Int(PETSc.typedget(PETSc.parse_options(ARGS), :n, n))

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

    PetscScalar = petsclib.PetscScalar
    PetscInt    = petsclib.PetscInt

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # ── Physics (identical to CahnHilliard2D.jl) ─────────────────────────────────
    # Grid units: dx = dy = 1 throughout, exactly as in CahnHilliard2D_plain.jl.
    lx, ly = 1.0, 1.0             # (unused in grid units; kept for the DMDA coordinate calls)
    D      = PetscScalar(1.0)     # mobility
    wcell  = 4.0                  # interface width, in cells -- resolve with >= 4
    γ      = PetscScalar(wcell^2 / 8)  # = 2
    C̄      = PetscScalar(0.0)     # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
    ampl   = PetscScalar(0.02)    # initial noise amplitude
    dx     = PetscScalar(1.0)     # grid units
    dy     = PetscScalar(1.0)

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
    κmax    = 4 / dx^2 + 4 / dy^2                    # = 8
    dt_expl = 2 / (D * κmax * (γ * κmax + 2)) / 2    # explicit 4th-order limit, safety 2
    dt_factor = PetscScalar(PETSc.typedget(opts, :dt_factor, dt_factor))
    dt_0  = PetscScalar(PETSc.typedget(opts, :dt, dt_factor * dt_expl))
    t_end = PetscScalar(PETSc.typedget(opts, :t_end, 40_000 * dt_expl))
    nt    = something(nt, Int(PETSc.typedget(opts, :nt, ceil(Int, t_end / dt_0))))

    # Frames are counted in *time*, not steps.  The explicit script writes a frame every 1000 of its
    # steps, i.e. 40 frames over t_end; with dt here being `dt_factor` times larger, the same 40 frames
    # need a proportionally smaller step interval.  Setting `-nframes` (or `-nvis` directly) overrides.
    nframes = Int(PETSc.typedget(opts, :nframes, nframes))
    nvis    = something(nvis, Int(PETSc.typedget(opts, :nvis, 0)))
    # Frame schedule: exactly `nframes` evenly spaced steps, ending on the last one.  A fixed
    # step interval cannot do this when nt is not a multiple of nframes (nt=100, nframes=40 gives
    # either 50 or 33 frames), so pick the steps directly.  `-nvis`/`nvis=` forces a plain interval.
    frame_steps = nvis > 0 ? Set(nvis:nvis:nt) :
                             Set(unique(round.(Int, range(nt ÷ nframes, nt, length = nframes))))
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
        (n, n),                   # grid size: `n=` keyword or -n; -da_grid_x/-da_grid_y also work
                                  # n = 2^k+1 (513 = 2^9+1) so geometric multigrid can coarsen
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
    old_lvl_cache   = Dict{Ptr{Cvoid}, Any}()
    old_cache_stamp = Dict{Ptr{Cvoid}, Int}()
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
        xs, ys = corners.lower[1], corners.lower[2]
        xe, ye = corners.upper[1], corners.upper[2]

        # Global size of *this* DM: under geometric multigrid (-pc_type mg) PETSc evaluates the
        # residual on coarsened DMs too, so nx/ny must come from the DM, not the outer scope.
        info_ = PETSc.getinfo(da_)
        mx_, my_ = Int(info_.global_size[1]), Int(info_.global_size[2])

        # `Cᵒˡᵈ` on this DM -- the fine-grid l_old has the wrong size on a coarse level, so it is
        # restricted down and cached per DM.
        l_old_lvl = old_for_dm(da_)

        # Same call shape as the explicit script: hand the kernel globally-indexed fields.
        withvecs((g_fx, corners, 1), (g_fx, corners, 2),
                 (l_x, ghost_corners, 1), (l_x, ghost_corners, 2),
                 (l_old_lvl, ghost_corners, 1)) do R_C, R_μ, C, μ, Cold
            residual!(R_C, R_μ, C, μ, Cold, γ, D, dt_0, mx_, my_, xs, xe, ys, ye)
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
    inv_h      = one(eltype(val)) / h_eps

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

    # Set `-fd_jacobian` to fall back to the finite-difference coloring path (useful
    # as a correctness check on the analytic derivatives above).
    use_fd_jacobian = haskey(opts, :fd_jacobian)

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
    # Same helpers, figure and output layout as the explicit scripts: frames + gif
    # written to output/<script name>/, rendered on rank 0 from a gathered field.
    CreatePlots = do_visu

    """
        gather_C() -> Matrix or nothing

    Collect the global concentration field on rank 0 for plotting: every rank sends
    its owned block and its bounds, rank 0 reassembles them.  `nothing` elsewhere.
    """
    function gather_C()
        corners = PETSc.getcorners(da)
        xs, ys = corners.lower[1], corners.lower[2]
        xe, ye = corners.upper[1], corners.upper[2]
        block = PETSc.withlocalarray!(x; read = true, write = false) do a
            Array(view(reshape(a, 2, xe - xs + 1, ye - ys + 1), 1, :, :))
        end
        nranks == 1 && return block
        bounds = MPI.Gather([xs, xe, ys, ye], 0, comm)
        blocks = MPI.gather(block, comm; root = 0)
        rank == 0 || return nothing
        Cg = Matrix{Float64}(undef, nx, ny)
        for r in 1:nranks
            Cg[bounds[4r-3]:bounds[4r-2], bounds[4r-1]:bounds[4r]] = blocks[r]
        end
        return Cg
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
            Cg = gather_C()
            if rank == 0
                push!(Fs, Point2f(t, F))
                axs[1].title = @sprintf("C   t = %.1f   F = %.4g", t, F)
                plt[1][3] = Cg                    # heatmap data
                plt[2][1] = Fs                    # line points
                recordframe!(vid)
                savepng(joinpath(dir, @sprintf("C_%06d.png", it)), fig)
            end
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

    # Figure setup (rank 0 only) -- same helper, colormap and inset as the explicit scripts.
    if CreatePlots
        C_init = gather_C()
        if rank == 0
            dir = outdir(@__FILE__); Fs = Point2f[]
            fig, axs, plt, vid = ch_figure(C_init, Fs, nt * dt_0, 1.05F0)
        end
    end

    # No report(0) here: `ch_figure` above already holds the initial field, and the explicit
    # scripts likewise write their first frame at step nvis, not at step 0.

    # Walltime of the time loop itself (setup, coloring and the initial report are
    # excluded).  The barrier makes all ranks start the clock together, so the
    # reduction below measures the actual wall-clock span rather than rank 0's
    # head start.
    MPI.Barrier(comm)
    t_start = MPI.Wtime()

    for it in 1:nt

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

        if it in frame_steps
            report(it, t)
        end
    end

    # The slowest rank sets the walltime, so reduce with `max`.
    walltime = MPI.Allreduce(MPI.Wtime() - t_start, max, comm)

    # Total script walltime: setup (PETSc init, DMDA, coloring, JIT) plus the time
    # loop.  Measured here rather than after the plotting block below, so an
    # interactive run does not include however long the window stays open.
    total_walltime = MPI.Allreduce(MPI.Wtime() - t_script_start, max, comm)

    # check() is collective, so every rank must call it -- not just rank 0.
    F_end, m_end = check()

    if rank == 0
        @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n",
                F0, F_end, (m_end - m0) / (nx * ny))
        @printf("walltime  : %.3f s  (%.4f s/step, %d ranks)  [time loop]\n",
                walltime, walltime / nt, nranks)
        @printf("total     : %.3f s  (incl. %.3f s setup + JIT)\n",
                total_walltime, total_walltime - walltime)
    end

    # Save the final frame and, when run as a script, keep the GLMakie window open
    # until it is closed manually (otherwise the process would exit immediately).
    if CreatePlots && rank == 0
        savegif(joinpath(dir, basename(dir) * ".gif"), vid; framerate = 5)
        println("frames + gif written to $dir")
    end

    # ── Cleanup ──────────────────────────────────────────────────────────────────
    # The SNES holds an options database whose finalizer touches MPI, so destroy it while PETSc
    # is still alive (see the note in PETSc.jl's examples/ex19.jl); then let any remaining
    # finalizers run before tearing the library down.  SNES must go before the objects it
    # references (J, da, r) so their reference counts drop first.
    isnothing(snes.opts) || (PETSc.destroy(snes.opts); snes.opts = nothing)
    GC.gc(true); MPI.Barrier(comm)
    foreach(PETSc.destroy, (snes, J, x_pert_vec, f0_vec, f1_vec, l_old, x_old, x, r, da))
    PETSc.finalize(petsclib)

    MPI.Barrier(comm)
    MPI.Finalize()

    # On macOS ARM64 with MPICH ch4:ofi, MPICH's atexit handler can crash during
    return
end

CahnHilliard2D_PETSc_implicit(do_visu=true)

# teardown; quick_exit bypasses C atexit handlers (same workaround as ex19.jl).
if !isinteractive()
    ccall(:quick_exit, Cvoid, (Cint,), 0)
end
