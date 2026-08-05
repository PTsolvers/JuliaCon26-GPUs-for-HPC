# Cahn-Hilliard 2D on the GPU -- backend-agnostic kernels (KernelAbstractions.jl).
# Grid units, single temporary, diagnostics: see README. CPU twin: CahnHilliard2D_plain.jl
using KernelAbstractions
using Random, Statistics, Printf, CairoMakie

# ---- backend: uncomment one ----
using Metal;  const backend = MetalBackend(); const FT = Float32  # Apple GPU (no Float64)
# using CUDA;   const backend = CUDABackend();  const FT = Float64  # NVIDIA
# const backend = CPU();                      const FT = Float64  # CPU reference

# no-flux (∂n = 0) through the ghost-node mirror A[0]->A[1], A[n+1]->A[n].
# min/max make it branchless, so no warp divergence at the boundaries.
@inline function lap(A, ix, iy, nx, ny)
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

# KA's CPU backend threads over workgroups, so the thread count matters there
backend_info() = backend isa CPU ? "CPU ($(Threads.nthreads()) threads)" :
                                   string(nameof(typeof(backend)))

# checks: F must decrease monotonically, mass must stay constant
@views function check(C, γ)
    F = sum(@. ((C^2 - 1)^2) / 4) + γ / 2 * (sum(@. (C[2:end, :] - C[1:end-1, :])^2)
                                           + sum(@. (C[:, 2:end] - C[:, 1:end-1])^2))
    return F, sum(C)
end

function CahnHilliard2D_KA(; nx=512, ny=512, nt=34_000, nvis=1000, do_visu=true)
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
    @printf("%s / %s   nx=%d ny=%d   γ=%.4g  dt=%.5g  Λ=%.1f cells (~%.0f features)\n",
            backend_info(), FT, nx, ny, γ, dt, 2π*sqrt(2γ), nx/(2π*sqrt(2γ)))
    # initial condition on the host, then upload
    Random.seed!(1234)
    C_h    = FT.(C̄ .+ ampl .* randn(nx, ny))
    C_h  .+= C̄ - mean(C_h)              # pin the conserved mean exactly
    C      = KernelAbstractions.allocate(backend, FT, nx, ny); copyto!(C, C_h)
    μ      = KernelAbstractions.zeros(backend, FT, nx, ny)
    F0, m0 = check(C_h, γ)
    # kernel objects
    kμ = k_chemical_potential!(backend, 256, (nx, ny))    # static ndrange
    kC = k_update_concentration!(backend, 256, (nx, ny))
    # visu
    if do_visu
        fig = Figure(; size=(600, 800))
        axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="C"),
               Axis(fig[2, 1]; xlabel="t", ylabel="F", limits=(0, nt*dt, 0, 1.05F0)))
        C_v = Float64.(C_h) # needed else Makie silently fails
        Fs  = Point2f[]
        plt = (heatmap!(axs[1], 1:nx, 1:ny, C_v; colormap=:balance, colorrange=(-1, 1)),
               lines!(axs[2], Fs; color=:crimson, linewidth=2))
        cbs = Colorbar(fig[1, 1][1, 2], plt[1])
    end
    # time loop
    KernelAbstractions.synchronize(backend)
    nwarm = 10; t_tic = 0.0; t_visu = 0.0
    for it = 1:nt
        it == nwarm+1 && (KernelAbstractions.synchronize(backend); t_tic = time())
        kμ(μ, C, γ)
        kC(C, μ, dtD)
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
            display(fig)
            t_visu += time() - t_visu_tic
        end
    end
    KernelAbstractions.synchronize(backend)
    t_it = (time() - t_tic - t_visu) / (nt - nwarm)
    # 5 array accesses per step: (read C, write μ) + (read μ, read C, write C)
    A_eff = 5 * nx * ny * sizeof(FT)
    @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n",
            t_it*1e3, A_eff/t_it/1e9, t_it*(nt - nwarm))
    copyto!(C_h, C); F, m = check(C_h, γ)
    @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n", F0, F, (m - m0)/(nx*ny))
    return
end

CahnHilliard2D_KA()
