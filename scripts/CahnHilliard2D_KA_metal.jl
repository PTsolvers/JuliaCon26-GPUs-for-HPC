# Cahn-Hilliard 2D on the GPU -- Metal variant (KernelAbstractions.jl).
# Grid units, single temporary, diagnostics: see README. CPU twin: CahnHilliard2D_plain.jl
#
# `fast=true` swaps in kernels that decompose the linear index by hand in Int32 with shift/mask.
# Needs nx a power of two and nx*ny divisible by 256 (unsafe_indices).
using KernelAbstractions
using Random, Statistics, Printf, CairoMakie
include(joinpath(@__DIR__, "common.jl"))

# ---- backend: uncomment one ----
# const backend = CPU();                        const FT = Float64  # CPU reference
using Metal;  const backend = MetalBackend(); const FT = Float32  # Apple GPU (no Float64)

# no-flux (∂n = 0) through the ghost-node mirror A[0]->A[1], A[n+1]->A[n].
# min/max make it branchless, so no warp divergence at the boundaries.
# @propagate_inbounds, to inherit the kernel's `inbounds = true`.
Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[min(ix+1, nx), iy] - 2a + A[max(ix-1, 1), iy]) +
           (A[ix, min(iy+1, ny)] - 2a + A[ix, max(iy-1, 1)])
end

# kernel 1: chemical potential μ = C³ - C - γ∇²C   (∇²C stays in registers)
@kernel inbounds = true function k_chemical_potential!(μ, C, γ)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    c = C[ix, iy]
    μ[ix, iy] = c * c * c - c - γ * lap(C, ix, iy, nx, ny)
end

# kernel 2: concentration update C += dt·D·∇²μ     (∇²μ stays in registers)
@kernel inbounds = true function k_update_concentration!(C, μ, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)
end

# ---- Int32 shift/mask indexing (Metal workaround, see header) ----
@kernel inbounds = true unsafe_indices = true function k_chemical_potential_fast!(μ, C, γ, mask::Int32, sh::Int32, nx::Int32, ny::Int32)
    I  = Int32(@index(Global, Linear)) - Int32(1)
    ix = (I & mask) + Int32(1)
    iy = (I >> sh)  + Int32(1)
    c  = C[ix, iy]
    μ[ix, iy] = c * c * c - c - γ * lap(C, ix, iy, nx, ny)
end

@kernel inbounds = true unsafe_indices = true function k_update_concentration_fast!(C, μ, dtD, mask::Int32, sh::Int32, nx::Int32, ny::Int32)
    I  = Int32(@index(Global, Linear)) - Int32(1)
    ix = (I & mask) + Int32(1)
    iy = (I >> sh)  + Int32(1)
    C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)
end

# checks: F must decrease monotonically, mass must stay constant
@views function check(C, γ)
    F = sum(@. ((C^2 - 1)^2) / 4) + γ / 2 * (sum(@. (C[2:end, :] - C[1:end-1, :])^2)
                                           + sum(@. (C[:, 2:end] - C[:, 1:end-1])^2))
    return F, sum(C)
end

function CahnHilliard2D_KA(; n=512, nt=40_000, nvis=1000,
                           do_visu=true, fast=true, verbose=true, framerate=5)
    nx = ny = n                      # square domain
    # physics (grid units)
    D     = FT(1.0)
    wcell = FT(4.0)                  # interface width, in cells -- resolve with >= 4
    γ     = wcell^2 / 8              # = 2
    C̄     = FT(0.0)                  # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
    ampl  = FT(0.02)                 # initial noise amplitude
    # numerics
    κmax  = FT(8.0)                  # 4/dx² + 4/dy² with dx = dy = 1
    dt    = 2 / (D * κmax * (γ * κmax + 2)) / 2   # explicit 4th-order limit, safety 2
    dtD   = dt * D
    verbose && @printf("%s / %s   nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)\n",
                       backend_info(backend), FT, nx, ny, γ, dt, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)))
    # initial condition on the host, then upload
    Random.seed!(1234)
    C_h    = FT.(C̄ .+ ampl .* randn(nx, ny))
    C_h  .+= C̄ - mean(C_h)              # pin the conserved mean exactly
    C      = KernelAbstractions.allocate(backend, FT, nx, ny); copyto!(C, C_h)
    μ      = KernelAbstractions.zeros(backend, FT, nx, ny)
    F0, m0 = check(C_h, γ)
    # kernel objects -- `fast` swaps in the Int32 shift/mask indexing (see above)
    if fast
        ispow2(nx)              || error("fast=true needs nx a power of two (got nx=$nx)")
        (nx*ny) % 256 == 0      || error("fast=true needs nx*ny divisible by 256")
        msk, sh    = Int32(nx - 1), Int32(trailing_zeros(nx))
        nx32, ny32 = Int32(nx), Int32(ny)
        kμ = k_chemical_potential_fast!(backend, 256, (nx*ny,))     # static ndrange
        kC = k_update_concentration_fast!(backend, 256, (nx*ny,))
        stepμ! = () -> kμ(μ, C, γ,   msk, sh, nx32, ny32)
        stepC! = () -> kC(C, μ, dtD, msk, sh, nx32, ny32)
    else
        kμ = k_chemical_potential!(backend, 256, (nx, ny))          # static ndrange
        kC = k_update_concentration!(backend, 256, (nx, ny))
        stepμ! = () -> kμ(μ, C, γ)
        stepC! = () -> kC(C, μ, dtD)
    end
    # visu -- frames are written to output/, never displayed
    if do_visu
        dir = outdir(@__FILE__)
        C_v = Float64.(C_h) # needed else Makie silently fails
        Fs  = Point2f[]
        fig, axs, plt, vid = ch_figure(C_v, Fs, nt*dt, 1.05F0)
    end
    # time loop
    KernelAbstractions.synchronize(backend)
    nwarm = 10; t_tic = 0.0; t_visu = 0.0
    for it = 1:nt
        it == nwarm+1 && (KernelAbstractions.synchronize(backend); t_tic = time())
        stepμ!()
        stepC!()
        if do_visu && it % nvis == 0
            KernelAbstractions.synchronize(backend)
            t_visu_tic = time()          # keep diagnostics out of the timing
            copyto!(C_h, C)
            F, m = check(C_h, γ)
            @printf("> step %6d, t = %8.2f, F = %.6g, Δmean = %+.2e\n",
                    it, it*dt, F, (m - m0)/(nx*ny))
            C_v .= C_h                              # Float32 solver -> Float64 plot buffer
            push!(Fs, Point2f(it*dt, F))
            axs[1].title = @sprintf("C   t = %.1f   F = %.4g", it*dt, F)
            plt[1][3] = C_v                         # heatmap data
            plt[2][1] = Fs                          # line points
            recordframe!(vid); save(joinpath(dir, @sprintf("C_%06d.png", it)), fig)
            t_visu += time() - t_visu_tic
        end
    end
    KernelAbstractions.synchronize(backend)
    t_it = (time() - t_tic - t_visu) / (nt - nwarm)
    # 5 array accesses per step: (read C, write μ) + (read μ, read C, write C)
    T_eff = Teff(5, nx, ny, t_it, FT)
    copyto!(C_h, C); F, m = check(C_h, γ)
    Δmean = (m - m0) / (nx*ny)
    if verbose
        @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n",
                t_it*1e3, T_eff, t_it*(nt - nwarm))
        @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n", F0, F, Δmean)
    end
    if do_visu
        savegif(joinpath(dir, basename(dir) * ".gif"), vid; framerate)
        verbose && println("frames + gif written to $dir")
    end
    return (; t_it, T_eff, F0, F, Δmean)
end

# weak-scaling test: dt is independent of nx in grid units, so growing nx enlarges the domain.
# `fast` also makes a good A/B here -- on Metal it is worth ~3x, on CUDA nothing.
function scaling_test(; ns=2 .^ (9:12), nt=1000, fast=true)
    @printf("%s / %s   scaling test, nt=%d, indexing: %s\n\n", backend_info(backend), FT, nt,
            fast ? "Int32 shift/mask" : "2D NTuple")
    @printf("%-8s %-10s %-12s %-14s %-14s %-12s\n",
            "n", "mem [MB]", "t_it [ms]", "T_eff [GB/s]", "t_it/cell [ns]", "Δmean")
    for n in ns
        r = CahnHilliard2D_KA(; n, nt, fast, do_visu=false, verbose=false)
        @printf("%-8d %-10.1f %-12.4f %-14.1f %-14.3f %-12.2e\n",
                n, 2*n*n*sizeof(FT)/2^20, r.t_it*1e3, r.T_eff,
                r.t_it/(n*n)*1e9, r.Δmean)
    end
end

CahnHilliard2D_KA()
# CahnHilliard2D_KA(; n=4096, nt=34_000, do_visu=false)
# scaling_test()
# scaling_test(; fast=false)   # 2D NTuple indexing, for the ~3x comparison
