using Chmy
using KernelAbstractions
using CairoMakie, Statistics, Printf, Random

# uncomment for GPU
using CUDA

include("common.jl")
include("chmy_helpers.jl")

function solve!(Cⁿ⁺¹, Cⁿ, nt, γ, C, exprs, params, do_visu, nvis)
    backend = get_backend(Cⁿ)
    update! = update_kernel!(backend, 256)
    dims    = size(Cⁿ)
    ranges  = make_ranges(dims)
    offsets = make_offsets(Val(ndims(Cⁿ)))
    KernelAbstractions.synchronize(backend)
    r = 0.05 / params.γ / ndims(Cⁿ)
    F0, m0 = check(Cⁿ, params.γ)
    # visualisation
    if do_visu
        fig, plt = makefigure(Cⁿ)
        display(fig)
    end
    # time loop
    nwarm  = 10; t_tic  = 0.0; t_visu = 0.0
    for it in 1:nt
        it == nwarm+1 && (KernelAbstractions.synchronize(backend); t_tic=time())
        bnd = Binding(γ => params.γ, C => Cⁿ)
        update_elements!(update!, Cⁿ⁺¹, Cⁿ, r, bnd, exprs, ranges, offsets)
        Cⁿ⁺¹, Cⁿ = Cⁿ, Cⁿ⁺¹
        if do_visu && it % nvis == 0
            KernelAbstractions.synchronize(backend)
            t_visu_tic = time() # keep diagnostics out of the timing
            F, m = check(Cⁿ, params.γ)
            @printf("> step %6d, t = %8.2f, F = %.6g, Δmean = %+.2e\n", it, it*r, F, (m - m0)/prod(dims))
            updatefigure!(plt, Cⁿ)
            display(fig)
            t_visu += time() - t_visu_tic
        end
    end
    KernelAbstractions.synchronize(backend)
    t_it = (time() - t_tic - t_visu) / (nt - nwarm)
    # 2 array accesses per step: (read Cⁿ, write Cⁿ⁺¹)
    T_eff = Teff(2, prod(dims), t_it, eltype(Cⁿ))
    F, m = check(Cⁿ, params.γ)
    Δmean = (m - m0) / prod(dims)
    @printf("\nt_it = %.3f ms   T_eff = %.1f GB/s   total %.1f s\n", t_it*1e3, T_eff, t_it*(nt - nwarm))
    @printf("F: %.6g -> %.6g (must decrease)   Δmean = %+.2e\n", F0, F, Δmean)
    return
end

function CahnHilliardND_Chmy(dims::Vararg{Int,N}; do_visu=true, backend=CPU(), nt=40_000, nvis=1000) where {N}
    # parameters
    params = (C̄=0.0, ampl=0.02, γ=1.0)
    # physics
    @scalars @uniform(γ) C
    # operators
    divg = Divergence(StaggeredCentralDifference())
    grad = Gradient(StaggeredCentralDifference())
    # equations
    ∇C    = grad(C)
    μ     = C^3 - C - γ * divg(∇C)
    q     = -grad(μ)
    ∂C_∂t = -divg(q)
    # locations and indices
    loc  = ntuple(_ -> 𝓈, Val(N))
    inds = ntuple(SIndex, Val(N))
    # boundary conditions and simplified expressions
    bcs   = boundary_conditions(q, ∇C, Val(N))
    exprs = make_expressions(normalize(∂C_∂t[loc...][inds...]), bcs)
    # allocate arrays
    Random.seed!(1234)
    Cⁿₕ = params.C̄ .+ 2 .* params.ampl .* (rand(dims...) .- 0.5)
    Cⁿ  = KernelAbstractions.allocate(backend, Float64, dims);
    copyto!(Cⁿ, Cⁿₕ)
    Cⁿ⁺¹ = copy(Cⁿ)
    # time loop
    solve!(Cⁿ⁺¹, Cⁿ, nt, γ, C[loc...], exprs, params, do_visu, nvis)
    return
end

CahnHilliardND_Chmy(64, 64, 64; do_visu=true, backend=CPU(), nt=50_000)
