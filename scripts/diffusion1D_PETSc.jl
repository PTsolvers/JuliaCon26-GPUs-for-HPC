# 1D steady-state diffusion with variable coefficient -- serial, assembled by hand.
#
#     -d/dx ( k(x) du/dx ) = f(x)  on [0,1],    u(0) = u(1) = 0
#
# This is the smallest useful PETSc program: build a matrix, build a right-hand side, hand both
# to a Krylov solver (KSP), get a vector back.  No DMDA, no MPI decomposition -- those come in
# diffusion1D_PETSc_dmda.jl, which solves exactly the same equation in parallel.
#
# Discretisation (see the README for the derivation).  With n interior points, h = 1/(n+1) and
# x_i = i·h, the conductivity is evaluated at cell FACES, k_{i±1/2} = k(x_i ± h/2):
#
#     -k_{i-1/2} u_{i-1} + (k_{i-1/2} + k_{i+1/2}) u_i - k_{i+1/2} u_{i+1} = h² f_i
#
# so A is symmetric positive definite and tridiagonal.  Evaluating k on faces (rather than
# averaging k(x_i)) is what makes the scheme conservative: the flux leaving cell i through a face
# is exactly the flux entering cell i+1.
#
# Run with:
#     julia --project=. scripts/diffusion1D_PETSc.jl
#     julia --project=. scripts/diffusion1D_PETSc.jl -n 200 -ksp_type cg -pc_type jacobi -ksp_monitor
using MPI, PETSc, Printf, CairoMakie
include(joinpath(@__DIR__, "common.jl"))

# Variable conductivity
kfun(x) = 1.0 + 99.0 * (1 / (1 + exp(-(x - 0.5) / 0.02)))
ffun(x) = 1.0   # constant source

function diffusion1D_PETSc(; n = 100, do_visu = true)
    petsclib = PETSc.getlib(; PetscScalar = Float64, PetscInt = Int64)
    PETSc.initialize(petsclib, log_view=false)  # set `-log_view` on the command line to see PETSc's internal timings
    opts = PETSc.parse_options(ARGS)
    n    = Int(PETSc.typedget(opts, :n, n))

    h  = 1 / (n + 1)                       # n interior points, 2 Dirichlet boundaries
    x  = [i * h for i in 1:n]              # x_1 … x_n, excluding the boundaries

    # ── assemble A (tridiagonal, SPD) and the right-hand side ────────────────
    # MatSeqAIJ(petsclib, m, n, nnz_per_row): a sequential sparse matrix, 3 non-zeros per row.
    A = PETSc.MatSeqAIJ(petsclib, n, n, 3)
    b = PETSc.VecSeq(petsclib, zeros(n))
    for i in 1:n
        km = kfun(x[i] - h/2)              # conductivity on the left face  of point i
        kp = kfun(x[i] + h/2)              # ...and on the right face
        i > 1 && (A[i, i-1] = -km)
        A[i, i]             = km + kp
        i < n && (A[i, i+1] = -kp)
        b[i] = h^2 * ffun(x[i])            # u = 0 at both ends, so no boundary terms appear
    end
    PETSc.assemble!(A)                     # PETSc caches entries; this flushes them

    # ── solve ────────────────────────────────────────────────────────────────
    # The keyword options below are defaults; anything on the command line wins, so
    # `-ksp_type cg -pc_type gamg -ksp_monitor` changes the solver without touching this file.
    ksp = PETSc.KSP(A; opts...)
    u = ksp\b               # or: u   = similar(b); PETSc.solve!(u, ksp, b)

    its  = PETSc.LibPETSc.KSPGetIterationNumber(petsclib, ksp)
    uvec = u[:]
    @printf("n = %d,  KSP its = %d,  max(u) = %.6f at x = %.3f\n",
            n, its, maximum(uvec), x[argmax(uvec)])

    if do_visu
        dir = outdir(@__FILE__)
        diffusion1D_figure(x, uvec, kfun; path = joinpath(dir, "u.png"),
                           title = "1D diffusion, k contrast 100x  (n = $n)")
        println("plot written to ", joinpath(dir, "u.png"))
    end

    foreach(PETSc.destroy, (ksp, u, b, A))
    PETSc.finalize(petsclib)
    return x, uvec
end

x, u = diffusion1D_PETSc(do_visu=true)  # set to false to skip the plot
