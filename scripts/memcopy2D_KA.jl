# 2D memory-throughput reference + linear diffusion (KernelAbstractions.jl).
# T_eff accounting and the μ = C reduction to diffusion: see README.
using KernelAbstractions, Printf, Random, CairoMakie
include(joinpath(@__DIR__, "common.jl"))

# ---- backend: uncomment one ----
# const backend = CPU();                        const FT = Float64  # CPU reference
using CUDA;   const backend = CUDABackend();  const FT = Float64  # NVIDIA
# using AMDGPU; const backend = ROCBackend();  const FT = Float64  # AMD GPU

# no-flux (∂n = 0) ghost-node mirror -- identical to CahnHilliard2D_KA.jl
# @propagate_inbounds to propagate `inbounds = true`.
Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[min(ix+1, nx), iy] - 2a + A[max(ix-1, 1), iy]) +
           (A[ix, min(iy+1, ny)] - 2a + A[ix, max(iy-1, 1)])
end

# ---- standard 2D indexing ----
@kernel inbounds = true function k_memcopy!(A, B)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy]
end

@kernel inbounds = true function k_saxpy!(A, B, C, s)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy] + s * C[ix, iy]
end

@kernel inbounds = true function k_diffusion!(C2, C, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    C2[ix, iy] = C[ix, iy] + dtD * lap(C, ix, iy, nx, ny)
end

# build the three kernels for a given n, with static ndrange
function kernels(n)
    km = k_memcopy!(backend, 256, (n, n))
    ks = k_saxpy!(backend, 256, (n, n))
    kd = k_diffusion!(backend, 256, (n, n))
    return (km, ks, kd)
end

function memcopy_bench(; ns=2 .^ (9:15))   # 512 … 32768, 3 arrays => 25.8 GB at 32768²
    @printf("%s / %s\n\n", backend_info(backend), FT)
    @printf("%-8s %-20s %-12s %-14s\n", "n", "kernel [arrays]", "t_it [ms]", "T_eff [GB/s]")
    for n in ns
        A = KernelAbstractions.allocate(backend, FT, n, n); rand!(A)
        B = KernelAbstractions.allocate(backend, FT, n, n); rand!(B)
        C = KernelAbstractions.allocate(backend, FT, n, n); rand!(C)
        km, ks, kd = kernels(n)
        # copyto! is the vendor memcpy -- the achievable ceiling, not a spec number
        for (nm, k, args, narr) in (("copyto!   [2]", copyto!, (A, B),      2),
                                    ("memcopy   [2]", km, (A, B),            2),
                                    ("saxpy     [3]", ks, (A, B, C, FT(2)),  3),
                                    ("diffusion [2]", kd, (A, B, FT(0.125)), 2))
            t = bench(backend, k, args)
            @printf("%-8s %-20s %-12.4f %-14.1f\n",
                    nm == "copyto!   [2]" ? string(n) : "", nm, t*1e3, Teff(narr, n, n, t, FT))
        end
        println()
    end
end

function diffusion2D(; n=8192, nt=50, do_visu=false)
    nx = ny = n                      # square domain
    D  = FT(1.0)
    dt = FT(1.0) / D / FT(4.1)          # dx² = 1
    @printf("%s / %s   nx=%d ny=%d   dt=%.5g\n", backend_info(backend), FT, nx, ny, dt)
    # initial condition: a Gaussian blob
    xc, yc, w = nx/2, ny/2, nx/16
    C_h = FT[exp(-((ix - xc)^2 + (iy - yc)^2) / w^2) for ix in 1:nx, iy in 1:ny]
    C   = KernelAbstractions.allocate(backend, FT, nx, ny); copyto!(C, C_h)
    C2  = KernelAbstractions.zeros(backend, FT, nx, ny)
    m0  = sum(C_h)                                # no-flux BCs => mass is conserved
    _, _, kdiff = kernels(nx)
    # time loop
    KernelAbstractions.synchronize(backend)
    nwarm = 10; t_tic = 0.0
    for it = 1:nt
        it == nwarm+1 && (KernelAbstractions.synchronize(backend); t_tic = time())
        kdiff(C2, C, dt * D)
        C, C2 = C2, C                 # swap
    end
    KernelAbstractions.synchronize(backend)
    t_it  = (time() - t_tic) / (nt - nwarm)
    T_eff = Teff(2, nx, ny, t_it, FT)
    @printf("t_it = %.3f ms   T_eff = %.1f GB/s\n", t_it*1e3, T_eff)
    copyto!(C_h, C)
    @printf("Δmean = %+.2e (no-flux => conserved)\n", (sum(C_h) - m0)/(nx*ny))
    # visu -- written to output/, never displayed
    if do_visu
        dir = outdir(@__FILE__)
        C_v = Float64.(C_h)   # Float32 heatmaps silently ignore later updates
        fig, ax, plt = field_figure(C_v; title=@sprintf("C   t = %.1f", nt*dt))
        save(joinpath(dir, @sprintf("diffusion_%05d.png", nt)), fig)
        println("figure written to $dir")
    end
    return
end

memcopy_bench()
# diffusion2D()
