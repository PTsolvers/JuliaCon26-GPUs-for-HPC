# Cahn-Hilliard 2D -- explicit loops, distributed over MPI with a PETSc DMDA.
#
# Same physics as CahnHilliard2D_plain.jl but will run with MPI. 
using Random, Statistics, Printf, CairoMakie
using MPI, PETSc
include(joinpath(@__DIR__, "common.jl"))

# no-flux (∂n = 0) through the ghost-node mirror A[0]->A[1], A[n+1]->A[n].
# Identical to the plain version, except that the clamp is applied in *global* indices: (gx, gy) is this node's position in the full grid and (nx, ny) the full grid size, so the mirror only kicks in at the true domain boundary.  In the interior the neighbour is a ghost node supplied by dm_global_to_local!, and the offsets below are the plain ±1 -- exactly as in the serial code.
Base.@propagate_inbounds function lap(A, ix, iy, gx, gy, nx, ny)
    a = A[ix, iy]
    return (A[ix + (gx < nx), iy] - 2a + A[ix - (gx > 1), iy]) +
           (A[ix, iy + (gy < ny)] - 2a + A[ix, iy - (gy > 1)])
end

# pass 1: chemical potential μ = C³ - C - γ∇²C
# C is ghosted (reads the halo), μ is owned-only; (ox, oy) is the offset from owned to ghosted indices and (xs, ys) the global index of the first owned node.
function chemical_potential!(μ, C, γ, ox, oy, xs, ys, nx, ny)
    nxl, nyl = size(μ)
    @inbounds for iy in 1:nyl, ix in 1:nxl
        c = C[ix + ox, iy + oy]
        μ[ix, iy] = c * c * c - c -
                    γ * lap(C, ix + ox, iy + oy, xs + ix - 1, ys + iy - 1, nx, ny)
    end
    return
end

# pass 2: concentration update C += dt·D·∇²μ
# Now μ is the ghosted field and C the owned one -- the two passes swap roles, which is why each needs its own global->local exchange in the time loop.
function update_concentration!(C, μ, dtD, ox, oy, xs, ys, nx, ny)
    nxl, nyl = size(C)
    @inbounds for iy in 1:nyl, ix in 1:nxl
        C[ix, iy] += dtD * lap(μ, ix + ox, iy + oy, xs + ix - 1, ys + iy - 1, nx, ny)
    end
    return
end

# checks: F must decrease monotonically, mass must stay constant.
# Same F as the plain script, but summed in parallel: each rank adds the faces whose left/lower node it owns, so every interior face is counted exactly once.
function check(C_ghost, γ, ox, oy, xs, ys, nxl, nyl, nx, ny, comm)
    F = 0.0
    m = 0.0
    @inbounds for iy in 1:nyl, ix in 1:nxl
        i, j = ix + ox, iy + oy
        c = C_ghost[i, j]
        F += ((c^2 - 1)^2) / 4
        m += c
        (xs + ix - 1) < nx && (F += γ / 2 * (C_ghost[i+1, j] - c)^2)
        (ys + iy - 1) < ny && (F += γ / 2 * (C_ghost[i, j+1] - c)^2)
    end
    return MPI.Allreduce(F, +, comm), MPI.Allreduce(m, +, comm)
end

function CahnHilliard2D_PETSc_explicit(; n=512, nt=40_000, nvis=1000,
                                         do_visu=true, framerate=5)
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
    PETSc.initialize(petsclib)
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # 1 DOF per node (C), 1 ghost layer, 5-point (STAR) stencil.  GHOSTED gives
    # a ghost layer outside the domain too; we never read it (the mirror in
    # `lap` avoids it), but it keeps the local array rectangular.
    da = PETSc.DMDA(petsclib, comm,
                    (PETSc.DM_BOUNDARY_GHOSTED, PETSc.DM_BOUNDARY_GHOSTED),
                    (n, n), 1, 1, PETSc.DMDA_STENCIL_STAR)
    nx = ny = n

    g_C = PETSc.DMGlobalVec(da)  # owned nodes only -- the state vector
    l_C = PETSc.DMLocalVec(da)   # owned + ghosts   -- what the kernels read
    g_μ = PETSc.DMGlobalVec(da)
    l_μ = PETSc.DMLocalVec(da)

    # Which nodes does this rank own, and where do they sit in the local array?
    corners       = PETSc.getcorners(da)
    ghost_corners = PETSc.getghostcorners(da)
    xs, ys = corners.lower[1], corners.lower[2]         # global index of first owned node
    xe, ye = corners.upper[1], corners.upper[2]
    nxl, nyl = xe - xs + 1, ye - ys + 1                 # owned size
    nxg = ghost_corners.upper[1] - ghost_corners.lower[1] + 1   # ghosted size
    nyg = ghost_corners.upper[2] - ghost_corners.lower[2] + 1
    ox = xs - ghost_corners.lower[1]                    # owned -> ghosted offset
    oy = ys - ghost_corners.lower[2]

    rank == 0 && @printf("PETSc-x nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)  %d rank(s)\n",
                         nx, ny, γ, dt, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)), nranks)

    # ── initial condition ─────────────────────────────────────────────────────
    # Every rank draws the same global field and keeps its own slice, so the
    # result is bit-identical to the plain script and independent of nranks.
    Random.seed!(1234)
    C0 = C̄ .+ ampl .* randn(nx, ny)
    C0 .+= C̄ - mean(C0)              # pin the conserved mean exactly
    PETSc.withlocalarray!(g_C; read = false, write = true) do a
        reshape(a, nxl, nyl) .= @view C0[xs:xe, ys:ye]
    end
    C0 = nothing

    # `check` needs the halo, so refresh it before the first measurement
    PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
    F0, m0 = PETSc.withlocalarray!(l_C; read = true, write = false) do a
        check(reshape(a, nxg, nyg), γ, ox, oy, xs, ys, nxl, nyl, nx, ny, comm)
    end

    # visu -- rank 0 gathers the field and writes frames to output/
    if do_visu
        gather_C() = gather_field(g_C, comm, rank, nranks, xs, xe, ys, ye, nx, ny, nxl, nyl)
        Cg = gather_C()
        if rank == 0
            dir = outdir(@__FILE__)
            Fs  = Point2f[]
            fig, axs, plt, vid = ch_figure(Cg, Fs, nt*dt, 1.05F0)
        end
    end

    # ── time loop ─────────────────────────────────────────────────────────────
    nwarm = 10; t_tic = 0.0; t_visu = 0.0
    for it = 1:nt
        it == nwarm+1 && (MPI.Barrier(comm); t_tic = MPI.Wtime())

        # pass 1: C (ghosted) -> μ (owned).  The exchange is the *only* line
        # that differs from the serial two-pass update.
        PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
        PETSc.withlocalarray!(l_C, g_μ; read = (true, false), write = (false, true)) do c, m
            chemical_potential!(reshape(m, nxl, nyl), reshape(c, nxg, nyg),
                                γ, ox, oy, xs, ys, nx, ny)
        end

        # pass 2: μ (ghosted) -> C (owned)
        PETSc.dm_global_to_local!(g_μ, l_μ, da, PETSc.INSERT_VALUES)
        PETSc.withlocalarray!(g_C, l_μ; read = (true, true), write = (true, false)) do c, m
            update_concentration!(reshape(c, nxl, nyl), reshape(m, nxg, nyg),
                                  dtD, ox, oy, xs, ys, nx, ny)
        end

        if do_visu && it % nvis == 0
            t_visu_tic = MPI.Wtime()     # keep diagnostics out of the timing
            PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
            F, m = PETSc.withlocalarray!(l_C; read = true, write = false) do a
                check(reshape(a, nxg, nyg), γ, ox, oy, xs, ys, nxl, nyl, nx, ny, comm)
            end
            Cg = gather_C()
            if rank == 0
                @printf("> step %6d, t = %8.2f, F = %.6g, Δmean = %+.2e\n",
                        it, it*dt, F, (m - m0)/(nx*ny))
                push!(Fs, Point2f(it*dt, F))
                axs[1].title = @sprintf("C   t = %.1f   F = %.4g", it*dt, F)
                plt[1][3] = Cg                          # heatmap data
                plt[2][1] = Fs                          # line points
                recordframe!(vid)
                savepng(joinpath(dir, @sprintf("C_%06d.png", it)), fig)
            end
            t_visu += MPI.Wtime() - t_visu_tic
        end
    end
    t_it = MPI.Allreduce((MPI.Wtime() - t_tic - t_visu) / (nt - nwarm), max, comm)

    # 5 array accesses per step: (read C, write μ) + (read μ, read C, write C)
    A_eff = 5 * nx * ny * sizeof(Float64)
    PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)
    F, m = PETSc.withlocalarray!(l_C; read = true, write = false) do a
        check(reshape(a, nxg, nyg), γ, ox, oy, xs, ys, nxl, nyl, nx, ny, comm)
    end
    if rank == 0
        @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n",
                t_it*1e3, A_eff/t_it/1e9, t_it*(nt - nwarm))
        @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n",
                F0, F, (m - m0)/(nx*ny))
        if do_visu
            savegif(joinpath(dir, basename(dir) * ".gif"), vid; framerate)
            println("frames + gif written to $dir")
        end
    end

    foreach(PETSc.destroy, (l_μ, g_μ, l_C, g_C, da))
    PETSc.finalize(petsclib)
    return
end

# Gather the owned blocks onto rank 0 for plotting.
function gather_field(g_C, comm, rank, nranks, xs, xe, ys, ye, nx, ny, nxl, nyl)
    block = PETSc.withlocalarray!(g_C; read = true, write = false) do a
        Array(reshape(a, nxl, nyl))
    end
    nranks == 1 && return block
    bounds = MPI.Gather([xs, xe, ys, ye], 0, comm)
    blocks = MPI.gather(block, comm; root = 0)
    rank == 0 || return nothing
    Cg = Matrix{Float64}(undef, nx, ny)
    for r in 1:nranks
        bx, ex, by, ey = bounds[4r-3], bounds[4r-2], bounds[4r-1], bounds[4r]
        Cg[bx:ex, by:ey] = blocks[r]
    end
    return Cg
end

CahnHilliard2D_PETSc_explicit()

# MPICH's atexit handler can crash during teardown on macOS; quick_exit skips it.
isinteractive() || ccall(:quick_exit, Cvoid, (Cint,), 0)
