const 𝓅 = Chmy.Point()
const 𝓈 = Chmy.Segment()

struct NormalizeIndicesRule <: AbstractRule end

function (rule::NormalizeIndicesRule)(expr::SExpr{Ind})
    inds = map(simplify, indices(expr))
    return SExpr(Ind(), argument(expr), inds...)
end

normalize(expr::STerm) = Postwalk(NormalizeIndicesRule())(expr)

merge_groups(a, b) = map((x, y) -> (x..., y...), a, b)

function no_flux(q, ::Val{N}, ::Val{D}, Δ) where {N,D}
    loc = ntuple(i -> i == D ? 𝓅 : 𝓈, Val(N))
    ids = ntuple(i -> i == D ? SIndex(i) + Δ : SIndex(i), Val(N))
    return normalize(q[D][loc...][ids...]) => SLiteral(0)
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
    bc = first(bcs)
    bcs = Base.tail(bcs)
    bulk = make_expressions(expr, bcs, kvs)
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

function makefigure(C::AbstractVector)
    fig = Figure()
    ax  = Axis(fig[1, 1])
    plt = lines!(ax, Array(C))
    return fig, plt
end
function makefigure(C::AbstractMatrix)
    fig = Figure()
    ax  = Axis(fig[1, 1]; aspect=DataAspect())
    plt = heatmap!(ax, Array(C); colorrange=(-1, 1))
    Colorbar(fig[1, 2], plt)
    return fig, plt
end
makefigure(C::AbstractArray) = makefigure(C[:, :, end÷2])

function updatefigure!(plt, C)
    plt[1] = Array(C)
    return
end
updatefigure!(plt, C::AbstractArray{<:Any,3}) = updatefigure!(plt, C[:, :, end÷2])
