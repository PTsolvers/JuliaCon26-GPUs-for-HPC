# Cahn-Hilliard 2D -- explicit, MPI-parallel, with a *2-DOF* DMDA.
#
# Companion to CahnHilliard2D_PETSc_explicit.jl, which does the same thing with
# a 1-DOF DMDA.  The physics, the timestep and the kernels are identical; only
# the data layout differs.  
#
#   1-DOF (the other file):  two separate vectors, C and μ, each [C1 C2 C3 ...]
#   2-DOF (here):            one vector holding both, interleaved
#                            [C1 μ1 C2 μ2 C3 μ3 ...]
#
# The explicit scheme does NOT need 2 DOFs: μ is a temporary, computed from C
# and immediately consumed, never an unknown.  This file exists to show that the
# 2-DOF layout still works (nothing breaks -- F matches to the last digit) but
# costs performance, and to make concrete why the *implicit* solver
# (CahnHilliard2D_PETSc.jl) genuinely needs 2 DOFs: there C and μ are solved for
# simultaneously by Newton, so PETSc must know there are two unknowns per node
# in order to build a Jacobian's 2x2 blocks and to offer -pc_fieldsplit.
#
# Measured on a 512² grid, 40 000 steps (t_it, lower is better):
#
#     ranks     1 DOF      2 DOF
#       1      0.46 ms    0.61 ms     ~33% slower
#       2      0.33 ms    0.37 ms     ~11% slower
#       4      0.24 ms    0.28 ms     ~18% slower
#
# The penalty is memory traffic: each kernel touches only one of the two fields,
# but the interleaved layout drags the other one through cache on every access,
# and the halo exchange moves both fields even though only one is needed per
# pass.
#
# Run with:
#     julia --project=. scripts/CahnHilliard2D_PETSc_explicit_2dof.jl
#     ~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/CahnHilliard2D_PETSc_explicit_2dof.jl
using Random, Statistics, Printf, CairoMakie
using MPI, PETSc, OffsetArrays
include(joinpath(@__DIR__, "common.jl"))

# Arrays are indexed by GLOBAL indices (OffsetArray); no-flux via the mirror A[0]->A[1], A[n+1]->A[n]
Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[min(ix+1, nx), iy] - 2a + A[max(ix-1, 1), iy]) +
           (A[ix, min(iy+1, ny)] - 2a + A[ix, max(iy-1, 1)])
end

# pass 1: chemical potential μ = C³ - C - γ∇²C
function chemical_potential!(μ, C, γ, nx, ny, xs, xe, ys, ye)
    @inbounds for iy in ys:ye, ix in xs:xe  # loop over global indices of the local block
        c = C[ix, iy]
        μ[ix, iy] = c * c * c - c - γ * lap(C, ix, iy, nx, ny)
    end
    return
end

# pass 2: concentration update C += dt·D·∇²μ
function update_concentration!(C, μ, dtD, nx, ny, xs, xe, ys, ye)
    @inbounds for iy in ys:ye, ix in xs:xe
        C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)
    end
    return
end

# checks: F must decrease monotonically, mass must stay constant; summed in parallel.
function check(C, γ, nx, ny, xs, xe, ys, ye, comm)
    F = 0.0
    m = 0.0
    @inbounds for iy in ys:ye, ix in xs:xe
        c = C[ix, iy]
        F += ((c^2 - 1)^2) / 4
        m += c
        ix < nx && (F += γ / 2 * (C[ix+1, iy] - c)^2)
        iy < ny && (F += γ / 2 * (C[ix, iy+1] - c)^2)
    end
    return MPI.Allreduce(F, +, comm), MPI.Allreduce(m, +, comm)
end

# Wrap DOF `d` globally-indexed: the vector is interleaved [C1 μ1 C2 μ2 ...], so reshape to
# (2, nx, ny) and view one field -- the stride-2 view is where the 2-DOF layout loses performance.
globalarray(a, c, d) =
    OffsetArray(view(reshape(a, 2, c.upper[1]-c.lower[1]+1, c.upper[2]-c.lower[2]+1), d, :, :),
                c.lower[1]:c.upper[1], c.lower[2]:c.upper[2])

# Run `f` on the given fields; a triple is (vector, corners, dof).  Both fields share one vector,
# so it appears twice and is deduplicated before VecGetArray.
withvecs(f, triples...) = PETSc.withlocalarray!(unique(t[1] for t in triples)...) do arrays...
    vecs = unique(t[1] for t in triples)
    f(map(t -> globalarray(arrays[findfirst(==(t[1]), vecs)], t[2], t[3]), triples)...)
end

function CahnHilliard2D_PETSc_explicit_2dof(; n=512, nt=40_000, nvis=1000, do_visu=true, framerate=5)
    # physics (grid units, dx = dy = 1) -- identical to the plain script
    D     = 1.0
    wcell = 4.0                  # interface width, in cells -- resolve with >= 4
    γ     = wcell^2 / 8          # = 2
    C̄     = 0.4                  # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
    ampl  = 0.02                 # initial noise amplitude
    # numerics
    κmax  = 8.0                  # 4/dx² + 4/dy² with dx = dy = 1
    dt    = 2 / (D * κmax * (γ * κmax + 2)) / 2   # explicit 4th-order limit, safety 2
    dtD   = dt * D

    # ── PETSc setup: the only PETSc-specific part of this file ────────────────
    petsclib = PETSc.getlib(; PetscScalar = Float64, PetscInt = Int64)
    PETSc.initialize(petsclib, log_view = false)
    comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)

    # DMDA: owns the decomposition, ghost layer and stencil, and creates/exchanges the vectors.
    # 2 DOFs/node (C, μ), 1 ghost layer, 5-point STAR; GHOSTED pads outside the domain too.
    da = PETSc.DMDA(petsclib, comm,
                    (PETSc.DM_BOUNDARY_GHOSTED, PETSc.DM_BOUNDARY_GHOSTED),
                    (n, n), 2, 1, PETSc.DMDA_STENCIL_STAR)
    nx = ny = n

    # One vector holds BOTH fields interleaved, so a single global/local pair covers C and μ.
    g_x, l_x = PETSc.DMGlobalVec(da), PETSc.DMLocalVec(da)

    # Which nodes does this rank own?  (xs:xe, ys:ye) are global indices; ghost_corners adds the halo.
    corners, ghost_corners = PETSc.getcorners(da), PETSc.getghostcorners(da)
    xs, ys = corners.lower[1], corners.lower[2]
    xe, ye = corners.upper[1], corners.upper[2]
    nxl, nyl = xe - xs + 1, ye - ys + 1

    rank == 0 && @printf("2DOF    nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)  %d rank(s)\n",
                         nx, ny, γ, dt, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)), nranks)

    # ── initial condition: every rank draws the same global field and keeps its own slice, making
    # the result bit-identical to the plain script and independent of the rank count ─────────────
    Random.seed!(1234)
    C0 = C̄ .+ ampl .* randn(nx, ny)
    C0 .+= C̄ - mean(C0)              # pin the conserved mean exactly
    # Both sides indexed globally: "take my slice of the global field", no index translation.
    withvecs((g_x, corners, 1)) do C
        for iy in ys:ye, ix in xs:xe; C[ix, iy] = C0[ix, iy]; end
    end
    C0 = nothing

    # `check` needs the halo, so refresh it before the first measurement
    PETSc.dm_global_to_local!(g_x, l_x, da, PETSc.INSERT_VALUES)
    F0, m0 = withvecs(C -> check(C, γ, nx, ny, xs, xe, ys, ye, comm), (l_x, ghost_corners, 1))

    # visu -- rank 0 gathers the field and writes frames to output/
    gather_C() = gather_field(g_x, comm, rank, nranks, xs, xe, ys, ye, nx, ny, nxl, nyl)
    if do_visu && rank == 0
        dir = outdir(@__FILE__); Fs = Point2f[]
        fig, axs, plt, vid = ch_figure(gather_C(), Fs, nt*dt, 1.05F0)
    elseif do_visu
        gather_C()                       # non-root ranks must join the gather
    end

    # ── time loop ─────────────────────────────────────────────────────────────
    nwarm = 10; t_tic = 0.0; t_visu = 0.0
    for it = 1:nt
        it == nwarm+1 && (MPI.Barrier(comm); t_tic = MPI.Wtime())

        # pass 1: C (ghosted) -> μ (owned).  The exchange is the only line that differs from the
        # serial version; each pass sends the field it is about to read.
        PETSc.dm_global_to_local!(g_x, l_x, da, PETSc.INSERT_VALUES)
        withvecs((c, m) -> chemical_potential!(m, c, γ, nx, ny, xs, xe, ys, ye),
                 (l_x, ghost_corners, 1), (g_x, corners, 2))

        # pass 2: μ (ghosted) -> C (owned)
        PETSc.dm_global_to_local!(g_x, l_x, da, PETSc.INSERT_VALUES)
        withvecs((c, m) -> update_concentration!(c, m, dtD, nx, ny, xs, xe, ys, ye),
                 (g_x, corners, 1), (l_x, ghost_corners, 2))

        # F and Δmean print whether or not we plot; only the gather/rendering need do_visu.
        if it % nvis == 0
            t_visu_tic = MPI.Wtime()     # keep diagnostics out of the timing
            PETSc.dm_global_to_local!(g_x, l_x, da, PETSc.INSERT_VALUES)
            F, m = withvecs(C -> check(C, γ, nx, ny, xs, xe, ys, ye, comm), (l_x, ghost_corners, 1))
            rank == 0 && @printf("> step %6d, t = %8.2f, F = %.6g, Δmean = %+.2e\n",
                                 it, it*dt, F, (m - m0)/(nx*ny))
            if do_visu
                Cg = gather_C()
                if rank == 0
                    push!(Fs, Point2f(it*dt, F))
                    axs[1].title = @sprintf("C   t = %.1f   F = %.4g", it*dt, F)
                    plt[1][3] = Cg                      # heatmap data
                    plt[2][1] = Fs                      # line points
                    recordframe!(vid)
                    savepng(joinpath(dir, @sprintf("C_%06d.png", it)), fig)
                end
            end
            t_visu += MPI.Wtime() - t_visu_tic
        end
    end
    t_it = MPI.Allreduce((MPI.Wtime() - t_tic - t_visu) / (nt - nwarm), max, comm)

    # Memory traffic: the 1-DOF version's 5 accesses, but stride-2 halves each cache line (`2*`),
    # and every copy moves BOTH fields though a pass reads one (8, not 4).
    A_eff = (2*5 + 8) * nx * ny * sizeof(Float64)
    PETSc.dm_global_to_local!(g_x, l_x, da, PETSc.INSERT_VALUES)
    F, m = withvecs(C -> check(C, γ, nx, ny, xs, xe, ys, ye, comm), (l_x, ghost_corners, 1))
    if rank == 0
        @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n",
                t_it*1e3, A_eff/t_it/1e9, t_it*(nt - nwarm))
        @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n", F0, F, (m - m0)/(nx*ny))
        do_visu && (savegif(joinpath(dir, basename(dir) * ".gif"), vid; framerate);
                    println("frames + gif written to $dir"))
    end

    foreach(PETSc.destroy, (l_x, g_x, da))
    PETSc.finalize(petsclib)
    return
end

# Gather the owned blocks onto rank 0 for plotting (visualisation only; the solver never needs it).
function gather_field(g_x, comm, rank, nranks, xs, xe, ys, ye, nx, ny, nxl, nyl)
    block = PETSc.withlocalarray!(g_x; read = true, write = false) do a
        Array(view(reshape(a, 2, nxl, nyl), 1, :, :))
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

CahnHilliard2D_PETSc_explicit_2dof(do_visu=false)

# MPICH's atexit handler can crash during teardown on macOS; quick_exit skips it.
isinteractive() || ccall(:quick_exit, Cvoid, (Cint,), 0)
