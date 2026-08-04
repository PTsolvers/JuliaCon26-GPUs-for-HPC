using CairoMakie
using Random, Statistics
Random.seed!(1234)

@views function diffusion_step!(∂²Cx, ∂²Cy, ∂²Cz, ∇²C, μ, qCx, qCy, qCz, C, γ, D, dt, dx, dy, dz)
    # potentials x-dir
    @. ∂²Cx[2:end-1, :, :] = C[3:end, :, :] - 2.0C[2:end-1, :, :] + C[1:end-2, :, :]
    @. ∂²Cx[1  , :, :] = -C[1    , :, :] + C[2  , :, :]
    @. ∂²Cx[end, :, :] =  C[end-1, :, :] - C[end, :, :]
    # potentials y-dir
    @. ∂²Cy[:, 2:end-1, :] = C[:, 3:end, :] - 2.0C[:, 2:end-1, :] + C[:, 1:end-2, :]
    @. ∂²Cy[:, 1  , :] = -C[:, 1    , :] + C[:, 2  , :]
    @. ∂²Cy[:, end, :] =  C[:, end-1, :] - C[:, end, :]
    # potentials z-dir
    @. ∂²Cz[:, :, 2:end-1] = C[:, :, 3:end] - 2.0C[:, :, 2:end-1] + C[:, :, 1:end-2]
    @. ∂²Cz[:, :, 1  ] = -C[:, :, 1    ] + C[:, :, 2  ]
    @. ∂²Cz[:, :, end] =  C[:, :, end-1] - C[:, :, end]
    # Laplacian
    @. ∇²C = ∂²Cx / dx^2 + ∂²Cy / dy^2 + ∂²Cz / dz^2
    @. μ   = C^3 - C - γ * ∇²C
    # update fluxes
    @. qCx[2:end-1, :, :] = -(μ[2:end, :, :] - μ[1:end-1, :, :]) / dx
    @. qCy[:, 2:end-1, :] = -(μ[:, 2:end, :] - μ[:, 1:end-1, :]) / dy
    @. qCz[:, :, 2:end-1] = -(μ[:, :, 2:end] - μ[:, :, 1:end-1]) / dz
    # update conentration
    @. C -= dt * D * ((qCx[2:end, :, :]-qCx[1:end-1, :, :]) / dx
                    + (qCy[:, 2:end, :]-qCy[:, 1:end-1, :]) / dy
                    + (qCz[:, :, 2:end]-qCz[:, :, 1:end-1]) / dz)
    return
end

function CahnHilliard3D()
    # physics
    lx, ly, lz = 1.0, 1.0, 1.0
    D          = 1.0
    w          = 0.03           # interface width  ->  ~11 features across the box
    γ          = w^2 / 8
    C̄          = 0.0            # conserved mean: 0 -> bicontinuous, ±0.4 -> droplets
    ampl       = 0.02           # initial noise amplitude
    # numerics
    nx, ny, nz = 64, 64, 64
    nt         = 30_000
    nvis       = 500
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    κmax       = 4/dx^2 + 4/dy^2
    dt = 2.0 / (D * κmax * (γ*κmax + 2.0)) / 2.0
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # array init
    ∂²Cx = zeros(nx, ny, nz)
    ∂²Cy = zeros(nx, ny, nz)
    ∂²Cz = zeros(nx, ny, nz)
    ∇²C  = zeros(nx, ny, nz)
    μ    = zeros(nx, ny, nz)
    qCx  = zeros(nx+1, ny, nz)
    qCy  = zeros(nx, ny+1, nz)
    qCz  = zeros(nx, ny, nz+1)
    # initial condition
    C    = C̄ .+ ampl .* randn(nx, ny, nz)
    C  .+= C̄ - mean(C)
    # visu
    sl = ceil(Int, ny/2)
    fig = Figure(; size=(600, 500))
    axs = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="C")
    plt = heatmap!(axs, xc, zc, C[:, sl, :]; colormap=:turbo)
    cbs = Colorbar(fig[1, 1][1, 2], plt)
    # time loop
    for it = 1:nt
        diffusion_step!(∂²Cx, ∂²Cy, ∂²Cz, ∇²C, μ, qCx, qCy, qCz, C, γ, D, dt, dx, dy, dz)
        if it % nvis == 0
            println("> step $it")
            plt[3] = C[:, sl, :]
            display(fig)
        end
    end
    return
end

CahnHilliard3D()
