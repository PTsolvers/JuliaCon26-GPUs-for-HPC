# 1D steady-state diffusion with variable coefficient -- MPI-parallel, via a DMDA.
#
#     -d/dx ( k(x) du/dx ) = f(x)  on [0,1],    u(0) = u(1) = 0
#
# Same equation and same discretisation as diffusion1D_PETSc.jl; the difference is that the grid
# is now owned by a DMDA, so the identical source runs on any number of ranks.  The DMDA does
# three things for us:
#
#   1. decides which points each rank owns (`getcorners`);
#   2. creates matrices and vectors with the matching parallel layout (`DMCreateMatrix`,
#      `DMGlobalVec`);
#   3. lets us assemble with GLOBAL row/column indices -- MatSetValuesStencil takes care of the
#      mapping, so the assembly loop below barely differs from the serial one.
#
# Run with:
#     julia --project=. scripts/diffusion1D_PETSc_dmda.jl
#     julia --project=. scripts/diffusion1D_PETSc_dmda.jl -n 200 -ksp_monitor
#     ~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/diffusion1D_PETSc_dmda.jl -n 1000
#
# Solvers are chosen entirely from the command line, e.g.
#     -ksp_type cg -pc_type jacobi          # CG + diagonal scaling (A is SPD)
#     -ksp_type cg -pc_type gamg            # algebraic multigrid -- iteration count
#                                           # independent of n (see the README)
#     -ksp_type preonly -pc_type lu         # direct solve (serial)
#     -ksp_view -ksp_converged_reason       # what did PETSc actually do?
using MPI, PETSc, Printf, CairoMakie
include(joinpath(@__DIR__, "common.jl"))

# Variable conductivity: a smooth 100x contrast between the left and right halves.
kfun(x) = 1.0 + 99.0 * (1 / (1 + exp(-(x - 0.5) / 0.02)))
ffun(x) = 1.0

function diffusion1D_PETSc_dmda(; n = 100, do_visu = true)
    petsclib = PETSc.getlib(; PetscScalar = Float64, PetscInt = Int64)
    PETSc.initialize(petsclib)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
    opts = PETSc.parse_options(ARGS)
    n    = Int(PETSc.typedget(opts, :n, n))

    # 1-D DMDA: n points, 1 DOF each, stencil width 1.  DM_BOUNDARY_NONE means the domain simply
    # ends -- points 1 and n are the Dirichlet boundary, handled explicitly below.
    da = PETSc.DMDA(petsclib, comm, (PETSc.DM_BOUNDARY_NONE,), (n,), 1, 1,
                    PETSc.DMDA_STENCIL_STAR; opts...)

    A = PETSc.LibPETSc.DMCreateMatrix(petsclib, da)   # parallel matrix, layout from the DMDA
    b = PETSc.DMGlobalVec(da)
    u = PETSc.DMGlobalVec(da)

    h = 1 / (n - 1)                        # here point i sits at x = (i-1)h, i = 1 … n
    corners = PETSc.getcorners(da)         # which points does THIS rank own?
    xs, xe  = corners.lower[1], corners.upper[1]

    # ── assemble, using global indices ───────────────────────────────────────
    # Each rank fills only its own rows; PETSc routes any off-rank entries during assemble!.
    PETSc.withlocalarray!(b; read = false, write = true) do bl
        for (li, i) in enumerate(xs:xe)
            xi = (i - 1) * h
            if i == 1 || i == n            # Dirichlet: u = 0
                # Scale the row like its neighbours (2k rather than 1), which is better for multigrid
                A[i, i] = 2 * kfun(xi)
                bl[li]  = 0.0
            else
                km = kfun(xi - h/2)        # conductivity on the faces either side of point i
                kp = kfun(xi + h/2)
                A[i, i-1] = -km
                A[i, i]   = km + kp
                A[i, i+1] = -kp
                bl[li]    = h^2 * ffun(xi)
            end
        end
    end
    PETSc.assemble!(A)                     # communicates any entries owned by another rank

    # ── solve ────────────────────────────────────────────────────────────────
    # Build the KSP from the *DM*, not from A: geometric multigrid (-pc_type mg) coarsens the DM
    # to build its grid hierarchy and interpolation operators, and cannot do that from a bare
    # matrix.  `KSPSetDMActive(false)` says "we assembled A ourselves, don't ask the DM for it".
    ksp = PETSc.KSP(da; opts...)
    PETSc.LibPETSc.KSPSetDMActive(petsclib, ksp, PETSc.LibPETSc.PETSC_FALSE)
    PETSc.LibPETSc.KSPSetOperators(petsclib, ksp, A, A)
    PETSc.solve!(u, ksp, b)

    # ── report: reduce over ranks, since each holds only its own slice ───────
    its = PETSc.LibPETSc.KSPGetIterationNumber(petsclib, ksp)
    lmax, largmax = PETSc.withlocalarray!(u; read = true, write = false) do ul
        isempty(ul) ? (-Inf, 0.0) : (maximum(ul), ((xs + argmax(ul) - 1) - 1) * h)
    end
    gmax = MPI.Allreduce(lmax, max, comm) # collect from all ranks
    # the x of the global maximum: only the rank that owns it contributes
    gx   = MPI.Allreduce(lmax == gmax ? largmax : 0.0, max, comm)
    rank == 0 && @printf("n = %d,  %d rank(s),  KSP its = %d,  max(u) = %.6f at x = %.3f\n",
                         n, nranks, its, gmax, gx)

    # ── plot: gather the distributed solution onto rank 0 ────────────────────
    if do_visu
        ul = PETSc.withlocalarray!(a -> Array(a), u; read = true, write = false)
        uall = MPI.gather(ul, comm; root = 0)
        if rank == 0
            uvec = reduce(vcat, uall)
            xall = [(i - 1) * h for i in 1:n]
            dir  = outdir(@__FILE__)
            diffusion1D_figure(xall, uvec, kfun; path = joinpath(dir, "u.png"),
                               title = "1D diffusion, k contrast 100x  (n = $n, $nranks rank(s))")
            println("plot written to ", joinpath(dir, "u.png"))
        end
    end

    foreach(PETSc.destroy, (ksp, u, b, A, da))
    PETSc.finalize(petsclib)
    return nothing
end

diffusion1D_PETSc_dmda()
