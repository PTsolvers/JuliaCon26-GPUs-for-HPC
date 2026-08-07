using Chmy
using KernelAbstractions
using CairoMakie
using Statistics
using Printf
using Random
using CUDA

const 𝓅 = Chmy.Point()
const 𝓈 = Chmy.Segment()

struct NormalizeIndicesRule <: AbstractRule end

function (rule::NormalizeIndicesRule)(expr::SExpr{Ind})
    inds = map(simplify, indices(expr))
    return SExpr(Ind(), argument(expr), inds...)
end

normalize(expr::STerm) = Postwalk(NormalizeIndicesRule())(expr)

merge_groups(a, b) = map((x, y) -> (x..., y...), a, b)

function boundary_conditions(q, ∇C, ::Val{N}) where {N}
    return ntuple(Val(N)) do D
        lower = ((no_flux(q, Val(N), Val(D), SLiteral(0)), neumann(∇C, Val(N), Val(D), SLiteral(0))...), neumann(∇C, Val(N), Val(D), SLiteral(-1)))
        upper = ((no_flux(q, Val(N), Val(D), SLiteral(1)), neumann(∇C, Val(N), Val(D), SLiteral(1))...), neumann(∇C, Val(N), Val(D), SLiteral(2)))
        return (lower, upper)
    end
end

function make_expressions(expr::STerm, ::Tuple{}, kvs::Tuple)
    rules = map(Chmy.SubsRule, kvs)
    return ((simplify(Prewalk(Chain(rules))(expr)),),)
end
function make_expressions(expr::STerm, bcs::Tuple, kvs::Tuple)
    bc    = first(bcs)
    bcs   = Base.tail(bcs)
    bulk  = make_expressions(expr, bcs, kvs)
    lower = merge_groups(make_expressions(expr, bcs, (kvs..., bc[1][1]...)), make_expressions(expr, bcs, (kvs..., bc[1][2]...)))
    upper = merge_groups(make_expressions(expr, bcs, (kvs..., bc[2][1]...)), make_expressions(expr, bcs, (kvs..., bc[2][2]...)))
    return (bulk..., lower..., upper...)
end
make_expressions(expr, bcs) = make_expressions(expr, bcs, ())

make_ranges(dims) = make_ranges(dims, ())
make_ranges(::Tuple{}, ranges) = (CartesianIndices(ranges),)
function make_ranges(dims, ranges)
    n = first(dims)
    dims = Base.tail(dims)
    return (make_ranges(dims, (ranges..., 3:(n-2)))..., make_ranges(dims, (ranges..., 1:1))..., make_ranges(dims, (ranges..., n:n))...)
end

make_offsets(::Val{N}) where {N} = make_offsets(ntuple(_ -> 0, Val(N)), ())
make_offsets(::Tuple{}, offsets) = ((CartesianIndex(offsets),),)
function make_offsets(dims::Tuple, offsets)
    dims = Base.tail(dims)
    bulk = make_offsets(dims, (offsets..., 0))
    lower = merge_groups(make_offsets(dims, (offsets..., 0)), make_offsets(dims, (offsets..., 1)))
    upper = merge_groups(make_offsets(dims, (offsets..., 0)), make_offsets(dims, (offsets..., -1)))
    return (bulk..., lower..., upper...)
end

@inline update_values!(C2, C1, r, ::Tuple{}, bnd, ::Tuple{}, I) = nothing
@inline function update_values!(C2, C1, r, exprs, bnd, offsets, I)
    J = I + first(offsets)
    @inbounds C2[J] = C1[J] + r * compute(first(exprs), bnd, Tuple(J)...)
    return update_values!(C2, C1, r, Base.tail(exprs), bnd, Base.tail(offsets), I)
end

@kernel inbounds=true function update_kernel!(C2, C1, r, exprs, bnd, offsets, origin)
    I = @index(Global, Cartesian)
    I += origin
    update_values!(C2, C1, r, exprs, bnd, offsets, I)
end

@inline update_elements!(kernel, C2, C1, r, bnd, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function update_elements!(kernel, C2, C1, r, bnd, exprs, ranges, offsets)
    range = first(ranges)
    origin = first(range) - oneunit(first(range))
    kernel(C2, C1, r, first(exprs), bnd, first(offsets), origin; ndrange=size(range))
    return update_elements!(kernel, C2, C1, r, bnd, Base.tail(exprs), Base.tail(ranges), Base.tail(offsets))
end

function solve!(C2, C1, nt, γ, C, exprs, ranges)
    backend = get_backend(C2)
    update! = update_kernel!(backend, 256)
    offsets = make_offsets(Val(ndims(C2)))
    KernelAbstractions.synchronize(backend)
    @time begin
        for _ in 1:nt
            bnd = Binding(γ => 2.0, C => C1)
            update_elements!(update!, C2, C1, 0.005, bnd, exprs, ranges, offsets)
            C2, C1 = C1, C2
        end
        KernelAbstractions.synchronize(backend)
    end
    return C2, C1
end

function neumann(∇C, ::Val{N}, ::Val{D}, Δ) where {N,D}
    loc = ntuple(i -> i == D ? 𝓅 : 𝓈, Val(N))
    return ntuple(Val(2N - 1)) do k
        tangential = k == 1 ? 0 : (k - 2) ÷ 2 + 1
        axis = tangential < D ? tangential : tangential + 1
        offset = SLiteral(k == 1 ? 0 : iseven(k) ? -1 : 1)
        ids = ntuple(Val(N)) do i
            i == D ? SIndex(i) + Δ : i == axis ? SIndex(i) + offset : SIndex(i)
        end
        normalize(∇C[D][loc...][ids...]) => SLiteral(0)
    end
end

function no_flux(q, ::Val{N}, ::Val{D}, Δ) where {N,D}
    loc = ntuple(i -> i == D ? 𝓅 : 𝓈, Val(N))
    ids = ntuple(i -> i == D ? SIndex(i) + Δ : SIndex(i), Val(N))
    return normalize(q[D][loc...][ids...]) => SLiteral(0)
end

function CahnHilliardND_Chmy(dims::Vararg{Int,N}; do_visu=true, backend=CPU(), nt=40_000) where {N}
    # parameters
    C̄    = 0.4
    ampl = 0.02
    # physics
    Chmy.@uniform @scalars γ
    @scalars C
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
    bcs    = boundary_conditions(q, ∇C, Val(N))
    exprs  = make_expressions(normalize(∂C_∂t[loc...][inds...]), bcs)
    ranges = make_ranges(dims)
    # allocate arrays
    Random.seed!(1234)
    Cⁿₕ   = C̄ .+ ampl .* randn(dims)
    Cⁿₕ .+= C̄ - mean(Cⁿₕ)
    Cⁿ    = KernelAbstractions.allocate(backend, Float64, dims); copyto!(Cⁿ, Cⁿₕ)
    Cⁿ⁺¹  = copy(Cⁿ)
    # time loop
    Cⁿ⁺¹, Cⁿ = solve!(Cⁿ⁺¹, Cⁿ, nt, γ, C[loc...], exprs, ranges)
    # plot
    do_visu && visme(Cⁿ⁺¹)
    return
end

function visme(C::AbstractVector)
    fig = Figure()
    ax  = Axis(fig[1, 1])
    lines!(ax, Array(C))
    display(fig)
    return
end
function visme(C::AbstractMatrix)
    fig = Figure()
    ax  = Axis(fig[1, 1]; aspect=DataAspect())
    hm  = heatmap!(ax, Array(C))
    Colorbar(fig[1, 2], hm)
    display(fig)
    return
end
visme(C::AbstractArray) = visme(C[:, :, end÷2])

CahnHilliardND_Chmy(32768, 32768; do_visu=false, backend=CUDABackend(), nt=100)
