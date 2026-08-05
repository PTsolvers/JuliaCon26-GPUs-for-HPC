# 2D memory-throughput reference + linear diffusion (KernelAbstractions.jl).
# T_eff accounting and the μ = C reduction to diffusion: see README.
using KernelAbstractions, Printf, CairoMakie

# ---- backend: uncomment one ----
const backend = CPU()
# using CUDA; const backend = CUDABackend()

# no-flux (∂n = 0) ghost-node mirror -- identical to CahnHilliard2D_KA.jl
@inline function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[min(ix+1, nx), iy] - 2a + A[max(ix-1, 1), iy]) +
           (A[ix, min(iy+1, ny)] - 2a + A[ix, max(iy-1, 1)])
end

@kernel inbounds = true function memcopy!(A, B)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy]
end

@kernel inbounds = true function saxpy!(A, B, C, s)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy] + s * C[ix, iy]
end

@kernel inbounds = true function diffusion!(C2, C, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    C2[ix, iy] = C[ix, iy] + dtD * lap(C, ix, iy, nx, ny)
end

# best-of-ntrial mean over nrep launches, so scheduling hiccups drop out
function bench(k, args; nrep=50, ntrial=5)
    k(args...)
    KernelAbstractions.synchronize(backend)
    best = Inf
    for _ in 1:ntrial
        KernelAbstractions.synchronize(backend); t0 = time()
        for _ in 1:nrep
            k(args...)
        end
        KernelAbstractions.synchronize(backend)
        best = min(best, (time() - t0) / nrep)
    end
    return best
end

Teff(narr, nx, ny, t) = narr * nx * ny * sizeof(Float64) / t / 1e9

backend_info() = backend isa CPU ? "CPU ($(Threads.nthreads()) threads)" :
                                   string(nameof(typeof(backend)))

function memcopy_bench(; ns=(512, 1024, 2048))
    @printf("%s\n\n", backend_info())
    @printf("%-8s %-20s %-12s %-14s\n", "n", "kernel [arrays]", "t_it [ms]", "T_eff [GB/s]")
    for n in ns
        A = KernelAbstractions.zeros(backend, Float64, n, n)
        B = KernelAbstractions.zeros(backend, Float64, n, n)
        C = KernelAbstractions.zeros(backend, Float64, n, n)
        for (nm, k, args, narr) in (
                ("memcopy   [2]", memcopy!(backend, 256, (n, n)),   (A, B),         2),
                ("saxpy     [3]", saxpy!(backend, 256, (n, n)),     (A, B, C, 2.0), 3),
                ("diffusion [2]", diffusion!(backend, 256, (n, n)), (A, B, 0.125),  2))
            t = bench(k, args)
            @printf("%-8s %-20s %-12.4f %-14.1f\n",
                    nm == "memcopy   [2]" ? string(n) : "", nm, t*1e3, Teff(narr, n, n, t))
        end
        println()
    end
end

function diffusion2D(; nx=512, ny=512, nt=2000, do_visu=true)
    D  = 1.0
    dt = 1.0 / D / 4.1
    @printf("%s   nx=%d ny=%d   dt=%.5g\n", backend_info(), nx, ny, dt)
    # initial condition: a Gaussian blob
    xc, yc, w = nx/2, ny/2, nx/16
    C_h = [exp(-((ix - xc)^2 + (iy - yc)^2) / w^2) for ix in 1:nx, iy in 1:ny]
    C   = KernelAbstractions.allocate(backend, Float64, nx, ny); copyto!(C, C_h)
    C2  = KernelAbstractions.zeros(backend, Float64, nx, ny)
    m0  = sum(C_h)                              # no-flux BCs => mass is conserved
    kdiff = diffusion!(backend, 256, (nx, ny))  # static ndrange
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
    A_eff = 2 * nx * ny * sizeof(Float64)
    @printf("t_it = %.3f ms   T_eff = %.1f GB/s\n", t_it*1e3, A_eff/t_it/1e9)
    copyto!(C_h, C)
    @printf("Δmass/mass = %+.2e (no-flux => conserved)\n", (sum(C_h) - m0)/m0)
    # visu
    if do_visu
        fig = Figure(; size=(600, 500))
        ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y",
                   title=@sprintf("C   t = %.1f", nt*dt))
        plt = heatmap!(ax, 1:nx, 1:ny, C_h; colormap=:turbo)
        Colorbar(fig[1, 1][1, 2], plt)
        display(fig)
    end
    return
end

memcopy_bench()
# diffusion2D()
