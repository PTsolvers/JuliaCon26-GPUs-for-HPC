# Cahn-Hilliard 2D -- plain Julia, explicit loops.
using Random, Statistics, Printf, CairoMakie
include(joinpath(@__DIR__, "common.jl"))

# no-flux (∂n = 0) through the ghost-node mirror A[0]->A[1], A[n+1]->A[n]
# @propagate_inbounds, to skip bounds check on lap.
Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[min(ix+1, nx), iy] - 2a + A[max(ix-1, 1), iy]) +
           (A[ix, min(iy+1, ny)] - 2a + A[ix, max(iy-1, 1)])
end

# pass 1: chemical potential μ = C³ - C - γ∇²C
function chemical_potential!(μ, C, γ)
    nx, ny = size(C)
    @inbounds for iy in 1:ny, ix in 1:nx
        c = C[ix, iy]
        μ[ix, iy] = c * c * c - c - γ * lap(C, ix, iy, nx, ny)
    end
    return
end

# pass 2: concentration update C += dt·D·∇²μ
function update_concentration!(C, μ, dtD)
    nx, ny = size(C)
    @inbounds for iy in 1:ny, ix in 1:nx
        C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)
    end
    return
end

function CahnHilliard2D_plain(; n=512, nt=40_000, nvis=1000, do_visu=true, framerate=5)
    nx = ny = n                  # square domain
    # physics (grid units, dx = dy = 1)
    D     = 1.0
    wcell = 4.0                  # interface width, in cells -- resolve with >= 4
    γ     = wcell^2 / 8          # = 2
    C̄     = 0.0                  # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
    ampl  = 0.02                 # initial noise amplitude
    # numerics
    κmax  = 8.0                  # 4/dx² + 4/dy² with dx = dy = 1
    dt    = 2 / (D * κmax * (γ * κmax + 2)) / 2   # explicit 4th-order limit, safety 2
    dtD   = dt * D
    @printf("plain   nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)\n",
            nx, ny, γ, dt, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)))
    # initial condition
    Random.seed!(1234)
    C      = C̄ .+ ampl .* randn(nx, ny)
    C    .+= C̄ - mean(C)             # pin the conserved mean exactly
    μ      = zeros(nx, ny)
    F0, m0 = check(C, γ)
    # visu -- frames are written to output/, never displayed
    if do_visu
        dir = outdir(@__FILE__)
        Fs  = Point2f[]
        fig, axs, plt, vid = ch_figure(C, Fs, nt*dt, 1.05F0)
    end
    # time loop
    nwarm = 10; t_tic = 0.0; t_visu = 0.0
    for it = 1:nt
        it == nwarm+1 && (t_tic = time())
        chemical_potential!(μ, C, γ)
        update_concentration!(C, μ, dtD)
        if do_visu && it % nvis == 0
            t_visu_tic = time()          # keep diagnostics out of the timing
            F, m = check(C, γ)
            @printf("> step %6d, t = %8.2f, F = %.6g, Δmean = %+.2e\n",
                    it, it*dt, F, (m - m0)/(nx*ny))
            push!(Fs, Point2f(it*dt, F))
            axs[1].title = @sprintf("C   t = %.1f   F = %.4g", it*dt, F)
            plt[1][3] = C                           # heatmap data
            plt[2][1] = Fs                          # line points
            recordframe!(vid)
            savepng(joinpath(dir, @sprintf("C_%06d.png", it)), fig)
            t_visu += time() - t_visu_tic
        end
    end
    t_it = (time() - t_tic - t_visu) / (nt - nwarm)
    # 5 array accesses per step: (read C, write μ) + (read μ, read C, write C)
    A_eff = 5 * nx * ny * sizeof(eltype(C))
    @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n",
            t_it*1e3, A_eff/t_it/1e9, t_it*(nt - nwarm))
    F, m = check(C, γ)
    @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n", F0, F, (m - m0)/(nx*ny))
    if do_visu
        savegif(joinpath(dir, basename(dir) * ".gif"), vid; framerate)
        println("frames + gif written to $dir")
    end
    return
end

CahnHilliard2D_plain()
