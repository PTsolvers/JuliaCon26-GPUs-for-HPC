# # Part 2: KernelAbstractions
#
# by Collin Wittenstein: [cwittens.github.io](https://cwittens.github.io/)
#
# We build the Cahn-Hilliard solver from Part 1 again, this time with KernelAbstractions:
#
# 1. Write the kernels.
# 2. Look at how a kernel gets launched, and what a sloppy launch costs.
# 3. Build ∂C/∂t out of them, in two versions, and benchmark both.
# 4. Hand it to OrdinaryDiffEq and run the simulation.
# 5. Time the whole solve and compare with step 3.
#
# The equation is still the same as in Part 1:
#
# ```
# ∂C/∂t = D ∇²μ ,    μ = C³ - C - γ ∇²C
# ```
#
# No-flux boundaries, grid units (`dx = dy = 1`).

# ## Setup
using KernelAbstractions
using CUDA
using Adapt: adapt
using OrdinaryDiffEqStabilizedRK
using BenchmarkTools, Random, Statistics, Printf
using CairoMakie

backend = CUDABackend()
## backend = ROCBackend()      # AMD
## backend = MetalBackend()    # Apple, Float32 only
## backend = CPU()             # no GPU needed, also good for debugging

T = Float64;

# One only has to change the `backend` line to switch vendors. The rest of the code is completely portable!
#
# Benchmark helper function to time the kernel and compare against the expected
# memory traffic. This is Part 1's `T_eff`: the arrays a kernel has to move,
# divided by the time it took, so a rate in GB/s that can be held against what
# the hardware can deliver.

function bench(f, backend, bytes; reps = 10)
    f(); KernelAbstractions.synchronize(backend)   # compile + warm up
    t = @belapsed begin
        for _ in 1:$reps
            $f()
        end
        KernelAbstractions.synchronize($backend)
    end
    return bytes * reps / t / 1e9   # GB/s
end;

# ## 1. Kernel syntax
#
# The plain-Julia first pass from Part 1, computing the chemical potential:
#
# ```julia
# function chemical_potential!(μ, C, γ)
#     nx, ny = size(C)
#     @inbounds for iy in 1:ny, ix in 1:nx
#         c = C[ix, iy]
#         μ[ix, iy] = c * c * c - c - γ * lap(C, ix, iy, nx, ny)
#     end
# end
# ```
#
# Same physics as a KA kernel. The Laplacian is written out inline here instead
# of through a `lap` helper, and the grid spacing is carried in `ax`, `ay`. (ax = 1/dx²)
# Getting rid of `lap` here is just personal taste.

@kernel inbounds = true function kernel_potential!(Dst, @Const(C), Nx, Ny, ax, ay, γ)
    i, j = @index(Global, NTuple)

    ## no-flux (∂n = 0) via ghost-node mirror
    idx_left  = max(i - 1, 1)
    idx_right = min(i + 1, Nx)
    jdx_down  = max(j - 1, 1)
    jdx_up    = min(j + 1, Ny)

    ## μ = C³ - C - γ ∇²C
    Dst[i, j] = C[i, j] * (C[i, j]^2 - 1) - γ * (
        (C[idx_left, j] - 2 * C[i, j] + C[idx_right, j]) * ax +
        (C[i, jdx_down] - 2 * C[i, j] + C[i, jdx_up]) * ay
    )
end

# The arithmetic is unchanged. The loop is gone: a kernel describes what one
# work item does (the inner part of the loop basically), and `@index(Global, NTuple)` supplies
# the `(i, j)` the loop used to provide.
#
# The four annotations:
#
# - `@kernel`: marks the function as a kernel. Returns nothing, writes into its
#   arguments.
# - `@index(Global, NTuple)`: index tuple for this work item. Use
#   `@index(Global, Linear)` for a single integer on 1D arrays.
# - `@Const(C)`: promises nothing writes to `C`. Information the compiler can use to optimize.
# - `inbounds = true`: applies `@inbounds` to the whole kernel body. It stops at
#   function calls, so any helper the kernel calls needs
#   `Base.@propagate_inbounds`.
#
# More kernels:

@kernel inbounds = true function kernel_concentration!(Dst, @Const(μ), Nx, Ny, ax, ay, D)
    i, j = @index(Global, NTuple)

    idx_left  = max(i - 1, 1)
    idx_right = min(i + 1, Nx)
    jdx_down  = max(j - 1, 1)
    jdx_up    = min(j + 1, Ny)

    ## ∂C/∂t = D ∇²μ
    Dst[i, j] = D * (
        (μ[idx_left, j] - 2 * μ[i, j] + μ[idx_right, j]) * ax +
        (μ[i, jdx_down] - 2 * μ[i, j] + μ[i, jdx_up]) * ay
    )
end

# Generic Laplacian, needed below:

@kernel inbounds = true function kernel_diffusion!(Dst, @Const(u), Nx, Ny, ax, ay)
    i, j = @index(Global, NTuple)

    idx_left  = max(i - 1, 1)
    idx_right = min(i + 1, Nx)
    jdx_down  = max(j - 1, 1)
    jdx_up    = min(j + 1, Ny)

    Dst[i, j] = (u[idx_left, j] - 2 * u[i, j] + u[idx_right, j]) * ax +
                (u[i, jdx_down] - 2 * u[i, j] + u[i, jdx_up]) * ay
end

# Now that we have written the kernels, we want to launch them. This is a little bit different than a normal function call and it takes two steps. First, we instantiate the kernel for a backend, then we call it with an `ndrange`:
#
# ```julia
# diffusion = kernel_diffusion!(backend)
# diffusion(dst, src, Nx, Ny, ax, ay, ndrange = size(src))
# ```
#
# Mind that we not only pass the arguments to the kernel, but also the `ndrange` argument.
#
# If you want a more detailed introduction into KernelAbstractions.jl, you can check out this tutorial I wrote a few months ago:
# [github.com/cwittens/A_KernelAbstractions_Tutorial](https://github.com/cwittens/A_KernelAbstractions_Tutorial/)

# ## 2. Launch configuration
#
# What we just did, instantiating as `diffusion = kernel_diffusion!(backend)` leaves the workgroup size and the `ndrange` unknown until call time, which means the compiler can not optimise for it. Either or both can be fixed at specialisation:
#
# ```julia
# k = kernel_diffusion!(backend)                    # both dynamic
# k = kernel_diffusion!(backend, (128, 2))          # static workgroup size
# k = kernel_diffusion!(backend, (128, 2), ndr)     # both static
# ```
# Here in this example the specific workgroup size used (e.g. (32, 8), (64, 4), (128, 2) or (256, 1)) does matter significantly less than not giving one at all.
#
# We can see this in the following example of running a really simple custom copy kernel.

@kernel inbounds = true function kernel_copy!(Dst, @Const(Src))
    i, j = @index(Global, NTuple)
    Dst[i, j] = Src[i, j]
end

n   = 4096
ndr = (n, n)
wgs1 = (32, 8)
wgs2 = (128, 2)
wgs3 = (256, 1)
nb  = sizeof(T) * n * n          # bytes in one array

u  = adapt(backend, rand(T, n, n))
du = similar(u)

k_dyn  = kernel_copy!(backend)               # dynamic WG, dynamic ndrange
k_swg  = kernel_copy!(backend, wgs2)         # static  WG, dynamic ndrange
k_stat1 = kernel_copy!(backend, wgs1, ndr)   # static  WG, static  ndrange
k_stat2 = kernel_copy!(backend, wgs2, ndr)   # static  WG, static  ndrange
k_stat3 = kernel_copy!(backend, wgs3, ndr)   # static  WG, static  ndrange

copy_results = [
    ("copyto! (vendor memcpy)",      bench(() -> copyto!(du, u), backend, 2nb)),
    ("dynamic WG / dynamic ndrange", bench(() -> k_dyn(du, u, ndrange = ndr), backend, 2nb)),
    ("static  WG $wgs2 / dynamic ndrange", bench(() -> k_swg(du, u, ndrange = ndr), backend, 2nb)),
    ("static  WG $wgs1 / static  ndrange", bench(() -> k_stat1(du, u), backend, 2nb)),
    ("static  WG $wgs2 / static  ndrange", bench(() -> k_stat2(du, u), backend, 2nb)),
    ("static  WG $wgs3 / static  ndrange", bench(() -> k_stat3(du, u), backend, 2nb)),
]

for (name, r) in copy_results
    @printf("%-38s %7.0f GB/s\n", name, r)
end

# Why are the static versions faster? With the sizes known at compile time, the
# compiler can specialise the index arithmetic. Comparing the generated code
#
#     @device_code_llvm debuginfo=:none k_dyn(du, u, ndrange = ndr)
#     @device_code_llvm debuginfo=:none k_stat2(du, u)
#
# shows four differences:
#
# - Integer division. Turning the flat thread and block ids into `(i, j)` needs
#   two divisions. Dynamic sizes make these runtime divisions, which GPUs do not
#   have in hardware and emulate in software. Static sizes make the divisors
#   compile-time constants, and here powers of two, so they become shifts.
# - Bounds check. KA checks every work item against the ndrange, because the
#   last workgroup may be partial. Dynamic needs several runtime comparisons;
#   static reduces this to a single comparison against a constant. The check is
#   reduced, not removed. `@kernel unsafe_indices=true` opts out entirely.
# - Kernel arguments. Dynamic passes the ndrange and workgroup size to the
#   kernel. Static puts them in the type, so the kernel takes only its data.
# - Launch bounds. A static workgroup size lets CUDA.jl tell the compiler the
#   maximum thread count, which helps register allocation.
#
# None of this changes the DRAM traffic, so none of it shows up in a memory
# counter.
#
# Practical rule: specialise once, outside the hot loop, with both sizes fixed.
# For a PDE solver the grid size is constant for the whole run.
#
# Now we do the same measurement on the three kernels from section 1. All of them read
# one array and write one, the same as the copy:

ax = ay = one(T)
D, γ = 1.0, 2.0
wgs = (128, 2)
k_diff_dyn  = kernel_diffusion!(backend)
k_diff_stat = kernel_diffusion!(backend, wgs, ndr)
k_pot_stat  = kernel_potential!(backend, wgs, ndr)
k_con_stat  = kernel_concentration!(backend, wgs, ndr)

GC.gc(true)                                                         #hide
CUDA.reclaim()                                                      #hide

diff_results = [
    ("copy          (static)",  bench(() -> k_stat2(du, u), backend, 2nb)),
    ("diffusion     (dynamic)", bench(() -> k_diff_dyn(du, u, n, n, ax, ay, ndrange = ndr), backend, 2nb)),
    ("diffusion     (static)",  bench(() -> k_diff_stat(du, u, n, n, ax, ay), backend, 2nb)),
    ("potential     (static)",  bench(() -> k_pot_stat(du, u, n, n, ax, ay, γ), backend, 2nb)),
    ("concentration (static)",  bench(() -> k_con_stat(du, u, n, n, ax, ay, D), backend, 2nb)),
]

ref = diff_results[1][2]
for (name, r) in diff_results
    @printf("%-22s %7.0f GB/s   %5.1f%% of copy kernel\n", name, r, 100r / ref)
end


# `kernel_potential!` adds a cubic on top of the same stencil and costs about
# the same as `kernel_diffusion!`. The stencils are memory bound, so a few
# extra flops per cell are free.

# ## 3. Assembling the time derivative
#
# In order to perform the time integration, OrdinaryDiffEq wants a function of the form `f!(du, u, p, t)`.
#
# Version 1: generic Laplacian kernel, broadcast for the nonlinearity,
# Laplacian again, scaling. Simple reusable pieces but nothing specialised, including the
# launch.

function rhs_naive!(dC, C, cache, t)
    (; D, γ, ax, ay, μ) = cache
    Nx, Ny = size(C)
    backend = get_backend(C)

    diffusion = kernel_diffusion!(backend)                       # both dynamic

    diffusion(μ, C, Nx, Ny, ax, ay, ndrange = (Nx, Ny))          # μ  = ∇²C
    @. μ = C * (C^2 - 1) - γ * μ                                 # μ  = C³ - C - γ∇²C
    diffusion(dC, μ, Nx, Ny, ax, ay, ndrange = (Nx, Ny))         # dC = ∇²μ
    @. dC = D * dC

    return nothing
end;

# The two broadcasts are written exactly as they would be on the CPU, and they
# run on the GPU because `C` and `μ` are `CuArray`s: GPUArrays.jl implements
# broadcasting for GPU arrays, so each `@.` line compiles and launches a kernel
# of its own. The same holds for `sum`, `mean`, `maximum`, `cumsum`, `map`,
# `reduce` and most of the standard library. Nothing here is GPU-specific code.
#
# So this version launches four kernels per call, two written by hand and two
# generated by the broadcast machinery.
#
# Getting this for free is a large part of why writing GPU code in Julia is
# pleasant, and it is the fastest way to a working solver. It also means the
# launches are easy to lose track of, which is what the next version addresses.

# Version 2: the two problem-specific kernels, both sizes fixed.

function rhs_tuned!(dC, C, cache, t)
    (; D, γ, ax, ay, μ) = cache
    Nx, Ny = size(C)
    backend = get_backend(C)

    potential     = kernel_potential!(backend, (128, 2), (Nx, Ny))
    concentration = kernel_concentration!(backend, (128, 2), (Nx, Ny))

    potential(μ, C, Nx, Ny, ax, ay, γ)                           # μ  = C³ - C - γ∇²C
    concentration(dC, μ, Nx, Ny, ax, ay, D)                      # dC = D ∇²μ

    return nothing
end;

# Two kernel launches instead of four: the nonlinearity and the scaling now
# happen inside the stencil kernels, at the cost of writing two problem-specific
# kernels instead of reusing one generic Laplacian.
#
# Specialising inside the right-hand side costs nothing at runtime: the
# workgroup size and `ndrange` live in the type, so the object is built at
# compile time.
#
# ### Array count
#
# Let's count how many reads and writes each version does.
#
# `rhs_naive!`:
#
# | | reads | writes | arrays |
# |---|---|---|---|
# | `diffusion(μ, C)`             | `C` | `μ` | 2 |
# | `@. μ = C*(C^2-1) - γ*μ`      | `C`, `μ` | `μ` | 3 |
# | `diffusion(dC, μ)`            | `μ` | `dC` | 2 |
# | `@. dC = D * dC`              | `dC` | `dC` | 2 |
# | | | | **9** |
#
# `rhs_tuned!`:
#
# | | reads | writes | arrays |
# |---|---|---|---|
# | `potential(μ, C, …)`      | `C` | `μ` | 2 |
# | `concentration(dC, μ, …)` | `μ` | `dC` | 2 |
# | | | | **4** |
#
# The predicted ratio just from the traffic is that the tuned version is 9/4 = 2.25 times faster. On top of that, `rhs_naive!` pays the dynamic-launch penalty measured in section 2.

# Predicted from traffic alone, the tuned version is 9/4 = 2.25 times faster.
# The launch configuration adds to that. The two Laplacians in `rhs_naive!`
# move 4 of the 9 arrays and run at 72% of the copy rate instead of 95%, so
# roughly 1.3 times slower than their static counterparts. Weighting the
# two contributions by traffic puts the prediction at about 2.55.

GC.gc(true)                                                         #hide
CUDA.reclaim()                                                      #hide

for N in (512, 1024, 2048, 4096, 8192)
    C  = adapt(backend, rand(T, N, N))
    dC = similar(C)
    cache = (; D, γ, ax, ay, μ = similar(C))

    ## compile both before timing either
    rhs_naive!(dC, C, cache, 0.0)
    rhs_tuned!(dC, C, cache, 0.0)
    KernelAbstractions.synchronize(backend)

    t_naive = @belapsed CUDA.@sync rhs_naive!($dC, $C, $cache, 0.0)
    t_tuned = @belapsed CUDA.@sync rhs_tuned!($dC, $C, $cache, 0.0)

    @printf("N = %5d   naive %8.3f ms   tuned %8.3f ms   ratio %5.2fx\n",
            N, 1e3t_naive, 1e3t_tuned, t_naive / t_tuned)
end

# The measured ratio matches the prediction once the arrays are large enough for
# the kernels to be bandwidth bound. At small N the launch overhead dominates,
# the extra launches in `rhs_naive!` are cheaper than their traffic suggests,
# and the ratio falls short.


# ## 4. Running the simulation

Random.seed!(1234)
n = 4096;

# ### Setting up physics
wcell = 4.0
γ     = wcell^2 / 8
D     = 1.0
C̄     = 0.0              # 0 -> bicontinuous, ±0.4 -> droplets
ampl  = 0.02

C_cpu = C̄ .+ ampl .* randn(T, n, n);
C_cpu .+= C̄ - mean(C_cpu);   # pin the conserved mean exactly

C0 = adapt(backend, C_cpu);  # <- the only line that knows about the device

# `adapt` converts a CPU array to the backend's array type: `CuArray` on CUDA,
# `ROCArray` on AMD, plain `Array` on `CPU()`. Reverse direction: `Array(...)`
# or `adapt(CPU(), ...)`.
#
# The setup above is ordinary Julia. `randn`, a broadcast, a `mean`. None of it
# is GPU code and none of it needs to be, because it runs once.
#
# Initialising directly on the device is also possible and worth it for very
# large grids or expensive setups, since it avoids the host allocation and the
# transfer. For one-off setup code, plain CPU code is easier to write and test,
# and the transfer cost is amortised over the whole run.

# ### Setting up simulation parameters
ax = ay = one(T)   # grid units: dx = dy = 1
cache = (; D, γ, ax, ay, μ = similar(C0))

tspan = (0.0, 1200.0)
prob_tuned = ODEProblem(rhs_tuned!, C0, tspan, cache);
prob_naive = ODEProblem(rhs_naive!, C0, tspan, cache);

# ### Timestepping
#
# Our right-hand sides compute ∂C/∂t. They say nothing about how to step forward in
# time. Part 1 did that with explicit Euler:
#
# ```julia
# C .+= dt * D * ∇²μ
# ```
#
# If you have worked with stability limits before: Cahn-Hilliard is fourth order
# in space, so the largest stable explicit step scales like `dt ∝ dx⁴`.
# Halving the grid spacing costs 16 times more
# steps on top of 4 times more cells.
#
# If that means nothing to you: just know that explicit Euler is the simplest method to
# implement, but not a good one for this problem. Everything up to here has been
# about how the code is written, fusing kernels and fixing launch parameters.
# The other large factor in computational science is the mathematics, and
# switching to a method suited to the problem can often be the bigger win of the two.
#
# This is what the `f!(du, u, p, t)` form buys. The problem is now an
# `ODEProblem`, so the whole OrdinaryDiffEq.jl method library applies to it, one
# of the most complete collections of timesteppers in any language. This means we
# can just try different methods (aka different mathematical algorithms) with only
# one changed line. The kernels do not change.
#
# In this case we use `ROCK2()`, a stabilised explicit method that can take much larger steps than explicit Euler on a diffusion-dominated problem. It stays explicit, so it does not need to form a Jacobian or solve linear systems.

# ### Now let's run the simulation
#
# We only save the first and last time step of our solution, so we don't measure any additional time for saving the solution. If we however want to make a gif of the solution, we can set it to e.g. `saveat = logrange(1, 1200, length = 120)`.
saveat = [tspan[1], tspan[2]]

sol = solve(prob_tuned, ROCK2(), saveat = saveat);

heatmap(Array(sol.u[end]); colormap = :viridis, colorrange = (-1, 1),
        axis = (; aspect = DataAspect(), title = "t = $(sol.t[end])",
                xticksvisible = false, yticksvisible = false,
                xticklabelsvisible = false, yticklabelsvisible = false))

# ## 5. Time to solution
#
# Before, we compared just the kernels and tried to optimize them as much as possible and got about 2.5x speedup.
# Now we want to see how this translates to the total wall time for the whole simulation.

solve(prob_naive, ROCK2(), saveat = saveat); # compile before measure
solve(prob_tuned, ROCK2(), saveat = saveat); # compile before measure

GC.gc(true)                                                         #hide
CUDA.reclaim()                                                      #hide

@time solve(prob_naive, ROCK2(), saveat = saveat);
@time solve(prob_tuned, ROCK2(), saveat = saveat);

# 11 s against 6.3 s, a factor of about 1.75.
# The kernels were 2.5x faster but the simulation only got about 1.75x faster, because evaluating ∂C/∂t is not the only thing
# happening.
#
# `@time` gives wall clock and allocations, but not where the time went. For a
# breakdown by kernel, CUDA.jl has a profiler that reports device time per
# launch:

CUDA.@profile solve(prob_tuned, ROCK2(), saveat = saveat); # compile @profile
txt = CUDA.@profile solve(prob_tuned, ROCK2(), saveat = saveat)

# Small caveat: this profiler is CUDA-only for now. AMDGPU.jl has no built-in equivalent, but
# there is an [open PR](https://github.com/JuliaGPU/AMDGPU.jl/pull/695) which still needs some work,
# if anyone is interested in contributing.

# But we can see that only about 44% of the time is spent in our two kernels, and the rest is `ROCK2` doing its own work. That means even if we made our kernels calculate instantly, the solve would still take about 3.5 s, or roughly 3x faster than the naive version.

# This means that in this case the method mattered more than the perfect GPU kernels did. The naive version, with a
# generic Laplacian, four kernel launches and a fully dynamic launch
# configuration, still solves the same problem in about 11 s, or about 4x faster than
# the explicit Euler implementation from Part 1
# ([`CahnHilliard2D_KA.jl`](https://github.com/PTsolvers/JuliaCon26-GPUs-for-HPC/blob/main/scripts/CahnHilliard2D_KA.jl)),
# with carefully tuned kernels running at a significantly higher `T_eff`. The unoptimized code with the better method wins, because
# `ROCK2` needs far fewer evaluations of ∂C/∂t by being able to take larger time steps `dt`.

# So in conclusion, `T_eff` is a good metric to know that you are using clever coding (and LLMs are getting better and better at that), but in computational science, it's equally important to sometimes take a step back and think about what is the right math (/method) to use for the problem at hand. And an advantage of Julia is that a change of method is often significantly less work than in other languages.