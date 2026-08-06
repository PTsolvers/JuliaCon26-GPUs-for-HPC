# Cahn-Hilliard 2D -- explicit loops, distributed over MPI with a PETSc DMDA.
#
# Same physics as CahnHilliard2D_plain.jl but will run with MPI. 
using Random, Statistics, Printf, CairoMakie
using MPI, PETSc, OffsetArrays
include(joinpath(@__DIR__, "common.jl"))

# Every array below is indexed by GLOBAL grid indices, thanks to OffsetArray wrappers 
# no-flux (∂n = 0) through the ghost-node mirror A[0]->A[1], A[n+1]->A[n]
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

# checks: F must decrease monotonically, mass must stay constant.
# Same F as the plain script, but summed in parallel.
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

# Wrap a PETSc array as an OffsetArray indexed by GLOBAL grid indices, given the corners it spans:
# `getcorners(da)` for a global (owned-only) vector, `getghostcorners(da)` for a local one 
globalarray(a, c) = OffsetArray(reshape(a, c.upper[1]-c.lower[1]+1, c.upper[2]-c.lower[2]+1),
                                c.lower[1]:c.upper[1], c.lower[2]:c.upper[2])

# Run `f` on the given vectors, each wrapped as a globally-indexed array.  `withvecs(f, (v, c), ...)`
# pairs every vector with the corners describing it, so a call site stays one line.
withvecs(f, pairs...) = PETSc.withlocalarray!((p[1] for p in pairs)...) do arrays...
    f(map(globalarray, arrays, map(p -> p[2], pairs))...)
end

function CahnHilliard2D_PETSc_explicit(; n=512, nt=40_000, nvis=1000, do_visu=true, framerate=5)
    # physics (grid units, dx = dy = 1) -- identical to the plain script
    D     = 1.0
    wcell = 4.0                  # interface width, in cells -- resolve with >= 4
    γ     = wcell^2 / 8          # = 2
    C̄     = 0.0                  # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
    ampl  = 0.02                 # initial noise amplitude
    # numerics
    κmax  = 8.0                  # 4/dx² + 4/dy² with dx = dy = 1
    dt    = 2 / (D * κmax * (γ * κmax + 2)) / 2   # explicit 4th-order limit, safety 2
    dtD   = dt * D

    # ── PETSc setup: the only PETSc-specific part of this file ────────────────
    petsclib = PETSc.getlib(; PetscScalar = Float64, PetscInt = Int64)
    PETSc.initialize(petsclib, log_view = false)
    comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)

    # Setup a DMDA grid that has info about the parallel decomposition, the ghost layer and the stencil.  The DMDA will create the global and local vectors for us, and handle the ghost exchange.
    # 1 DOF per node (C), 1 ghost layer, 5-point (STAR) stencil.  GHOSTED gives a ghost layer outside the domain too; we never read it (the mirror in `lap` avoids it), but it keeps the local array rectangular.
    da = PETSc.DMDA(petsclib, comm,
                    (PETSc.DM_BOUNDARY_GHOSTED, PETSc.DM_BOUNDARY_GHOSTED),
                    (n, n), 1, 1, PETSc.DMDA_STENCIL_STAR)
    nx = ny = n

    # global vecs hold owned nodes only (the state); local vecs add the ghost layer the kernels read
    g_C, l_C = PETSc.DMGlobalVec(da), PETSc.DMLocalVec(da)
    g_μ, l_μ = PETSc.DMGlobalVec(da), PETSc.DMLocalVec(da)

    # Which nodes does this rank own?  (xs:xe, ys:ye) are global indices; ghost_corners adds the halo.
    corners, ghost_corners = PETSc.getcorners(da), PETSc.getghostcorners(da)
    xs, ys = corners.lower[1], corners.lower[2]
    xe, ye = corners.upper[1], corners.upper[2]
    nxl, nyl = xe - xs + 1, ye - ys + 1

    rank == 0 && @printf("PETSc-x nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)  %d rank(s)\n",
                         nx, ny, γ, dt, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)), nranks)

    # ── initial condition ─────────────────────────────────────────────────────
    # Every rank draws the same global field and keeps its own slice, so the result is
    # bit-identical to the plain script and independent of the number of ranks.
    Random.seed!(1234)
    C0 = C̄ .+ ampl .* randn(nx, ny)
    C0 .+= C̄ - mean(C0)              # pin the conserved mean exactly
    # Both sides indexed globally: "take my slice of the global field", no index translation.
    withvecs((g_C, corners)) do C
        for iy in ys:ye, ix in xs:xe; C[ix, iy] = C0[ix, iy]; end
    end
    C0 = nothing

    # `check` needs the halo, so refresh it before the first measurement
    PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
    F0, m0 = withvecs(C -> check(C, γ, nx, ny, xs, xe, ys, ye, comm), (l_C, ghost_corners))

    # visu -- rank 0 gathers the field and writes frames to output/
    gather_C() = gather_field(g_C, comm, rank, nranks, xs, xe, ys, ye, nx, ny, nxl, nyl)
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

        # pass 1: C (ghosted) -> μ (owned).  The exchange is the *only* line that differs from
        # the serial two-pass update; each pass sends the field it is about to read.
        PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
        withvecs((c, m) -> chemical_potential!(m, c, γ, nx, ny, xs, xe, ys, ye),
                 (l_C, ghost_corners), (g_μ, corners))

        # pass 2: μ (ghosted) -> C (owned)
        PETSc.dm_global_to_local!(g_μ, l_μ, da, PETSc.INSERT_VALUES)
        withvecs((c, m) -> update_concentration!(c, m, dtD, nx, ny, xs, xe, ys, ye),
                 (g_C, corners), (l_μ, ghost_corners))

        # F and Δmean are printed whether or not we plot -- they are the cheap
        # correctness check (F must decrease, mass must not drift); only the
        # gather and the rendering below are skipped when do_visu = false.
        if it % nvis == 0
            t_visu_tic = MPI.Wtime()     # keep diagnostics out of the timing
            PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
            F, m = withvecs(C -> check(C, γ, nx, ny, xs, xe, ys, ye, comm), (l_C, ghost_corners))
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

    # Memory traffic per step: the same 5 kernel accesses as the plain script -- (read C, write μ)
    # + (read μ, read C, write C) -- plus 4 for the two global->local copies a distributed run pays and a serial one does not. 
    A_eff = (5 + 4) * nx * ny * sizeof(Float64)
    PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
    F, m = withvecs(C -> check(C, γ, nx, ny, xs, xe, ys, ye, comm), (l_C, ghost_corners))
    if rank == 0
        @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n",
                t_it*1e3, A_eff/t_it/1e9, t_it*(nt - nwarm))
        @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n", F0, F, (m - m0)/(nx*ny))
        do_visu && (savegif(joinpath(dir, basename(dir) * ".gif"), vid; framerate);
                    println("frames + gif written to $dir"))
    end

    foreach(PETSc.destroy, (l_μ, g_μ, l_C, g_C, da))
    PETSc.finalize(petsclib)
    return
end

# Gather the owned blocks onto rank 0 for plotting: every rank sends its block and its bounds,
# rank 0 reassembles them.  Purely for visualisation -- the solver never needs the global field.
function gather_field(g_C, comm, rank, nranks, xs, xe, ys, ye, nx, ny, nxl, nyl)
    block = PETSc.withlocalarray!(a -> Array(reshape(a, nxl, nyl)), g_C; read = true, write = false)
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

CahnHilliard2D_PETSc_explicit(n=1024, do_visu=true)

# MPICH's atexit handler can crash during teardown on macOS; quick_exit skips it.
isinteractive() || ccall(:quick_exit, Cvoid, (Cint,), 0)
