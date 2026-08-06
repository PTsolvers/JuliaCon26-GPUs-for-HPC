# JuliaCon26-GPUs-for-HPC

Hands-on with Julia for HPC on GPUs workshop at JuliaCon 2026.

## Getting started

> **TODO** — access to the Otus cluster at PC2: account request, login, module setup, how to grab an H100 node, and the Julia environment to instantiate.

## Workshop outline

1. **Performance basics** — what limits a stencil code, and how to measure it
2. **KernelAbstractions in depth** — portable kernels, and composing with the wider ecosystem (e.g. other timestepper instead of hand-rolled explicit Euler)
3. **Chmy.jl (+ KA)** — the same equations, expressed at a higher level using dimensions-agnostic DSL
4. **PETSc.jl** — the same equations again, via PETSc
5. **Reactant.jl** — if time permits

# Part 1: Performance basics

## Why bother with GPUs

PDEs get expensive fast. Cahn-Hilliard is fourth order, so an explicit timestep is bounded by `dt ∝ dx⁴`: refining the mesh by 2 costs 16× more steps *and* 4× more cells. Useful resolutions are large — at **8192²** a domain holds ~650 features, enough for coarsening statistics not dominated by the box size.

That run is 40 000 steps over 67 M cells:

| | per cell | 8192², 40 000 steps |
|---|---|---|
| Apple M2 laptop, 8 threads | 0.72 ns | ~32 min |
| Grace CPU, 36 of 72 cores | 0.125 ns | 5.6 min |
| GH200 144G HBM3e | 0.012 ns | **31 s** |

The Grace row sits on the *same node* as the GPU, so it is the like-for-like comparison: **10.8× slower**, with the same code, the same `Float64` and the same `Δmean = 1e-20`. The difference is memory bandwidth rather than arithmetic speed — quantified below.

## The model

The [Cahn–Hilliard equation](https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation), phase separation of a binary mixture:

```
∂C/∂t = D ∇²μ ,    μ = C³ - C - γ ∇²C
```

with no-flux boundaries on both `C` and `μ`, imposed by a ghost-node mirror (`A[0] -> A[1]`), which makes the scheme exactly mass-conserving.

**The conserved mean `C̄` selects the regime.** It is fixed by the initial condition and never moves, so it acts as a model parameter:

| `C̄ = 0` — bicontinuous | `C̄ = 0.4` — droplets |
|---|---|
| ![](assets/CahnHilliard2D_KA_C0.gif) | ![](assets/CahnHilliard2D_KA_C04.gif) |

Uniform states are unstable only for `|C̄| < 1/√3 ≈ 0.577`; beyond that the mixture is metastable and separation requires nucleation. Growth also slows as `|C̄|` rises — the rate goes as `(1-3C̄²)²`, so `C̄ = 0.4` takes ~3.7× longer to separate than `C̄ = 0`.

**Two invariants worth asserting on.** Both are cheap, and together they catch sign errors, wrong boundary conditions and unstable timesteps:

- free energy `F = ∫[(C²-1)²/4 + γ/2|∇C|²]` decreases monotonically
- `mean(C)` is constant — machine zero in Float64, ~1e-11 in Float32

## A first implementation

[`scripts/CahnHilliard2D_plain.jl`](scripts/CahnHilliard2D_plain.jl) is the solver in plain Julia — explicit loops, no abstractions, ~95 lines including plotting. The physics is two passes over the grid:

```julia
# no-flux (∂n = 0) through the ghost-node mirror A[0]->A[1], A[n+1]->A[n]
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
end

# pass 2: concentration update C += dt·D·∇²μ
function update_concentration!(C, μ, dtD)
    nx, ny = size(C)
    @inbounds for iy in 1:ny, ix in 1:nx
        C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)
    end
end
```

The `min`/`max` in `lap` apply the ghost-node mirror without branching. Only two arrays are ever written: `μ` has to be a real array because pass 2 reads its *neighbours*, while `∇²C`, the fluxes `qCx`, `qCy` and the other intermediates collapse into registers, since

```
∂C/∂t = D ∇²μ = -D ∇·q        with     q = -∇μ
```

i.e. the flux divergence *is* the Laplacian of `μ`. Working set: 2 arrays rather than 7.

This runs fine at 512². Reaching 8192² takes two further things — scaling that keeps the constants well-conditioned, and hardware that can move the memory — which are the subject of the rest of Part 1.

## Scaling

**Grid units (`dx = dy = 1`).** Every constant becomes O(1) — `γ = 2`, `dt = 7e-3`, `D = 1` — instead of spanning `1e-7 … 1e4`, which is also what makes a Float32-only backend safe. Physical units for a cell size `L` follow from `γ_phys = γ·L²`, `t_phys = t·L²/D`, `w_phys = w·L`.

A useful consequence: the interface width `w = √(8γ)` and the fastest-growing wavelength `Λ = 2π√(2γ)` are both measured *in cells* and independent of `n`. Growing `n` therefore enlarges the domain at fixed resolution-per-feature, and **`dt` does not shrink** — weak scaling, which is what makes 8192² affordable. Resolving the interface better instead means raising `wcell`, at a cost of `nt ∝ wcell⁴`.

## Memory bound, and what to measure

A 5-point Laplacian does a handful of flops per cell and moves several arrays. Current hardware manages ~50 flops in the time it takes to fetch one number from main memory, so the bytes dominate and FLOP/s is not a useful metric.

Wall time is the end-goal number, but it is *relative* — it compares two machines rather than saying whether a kernel is good on either. Something closer to absolute is more useful: a fraction of what the hardware can deliver.

The two measurements above make the point. Grace sustains 321 GB/s on this problem and the GH200 3460 GB/s, a ratio of **10.8**; the wall-time ratio was also **10.8×**. To two digits the speedup is the bandwidth ratio.

The CPU thread sweep shows the same from the other side — Cahn-Hilliard at 8192² on Grace:

| threads | `T_eff` | speedup | parallel efficiency |
|---|---|---|---|
| 1 | 16.8 GB/s | 1.0× | 100% |
| 4 | 65.6 | 3.9× | 98% |
| 8 | 130.3 | 7.7× | 97% |
| 18 | 252.3 | 15.0× | 83% |
| 36 | **320.7** | 19.1× | 53% |
| 72 | 287.0 | 17.1× | 24% |

Near-linear to 8 threads, rolling off after, saturating at 36, and slower on 72 cores than on 36. The memory bus saturates before the cores do.

Hence counting the arrays each kernel must move, ignoring cache:

```
memcopy    A = B              -> 2 arrays (1 read  + 1 write)
saxpy      A = B + s*C        -> 3 arrays (2 reads + 1 write)
diffusion  C2 = C + dt·D·∇²C  -> 2 arrays (1 read  + 1 write)
Cahn-Hilliard, per step       -> 5 arrays (read C, write μ | read μ, read C, write C)

T_eff = n_arrays · nx·ny·sizeof(eltype) / t_it
```

`T_eff` assumes *perfect* caching of stencil halos, so it charges only the minimum traffic. A kernel reaching the memcopy rate is doing about as well as the hardware allows.

## Step 1 - measure the ceiling

[`scripts/memcopy2D_KA.jl`](scripts/memcopy2D_KA.jl) runs memcopy, saxpy and diffusion (Laplacian) across resolutions:

```julia
julia> include("scripts/memcopy2D_KA.jl")   # memcopy_bench()
```

GH200 144G HBM3e, Float64. Device-reported peak is **4917 GB/s** (3.201 GHz × 6144 bit); `copyto!` — NVIDIA's own tuned copy — reaches **4308 GB/s**, i.e. 88% of it.

| n | memcopy [2] | saxpy [3] | diffusion [2] |
|---|---|---|---|
| 512 | 1294 | 1729 | 1249 |
| 1024 | 3480 | 5055 | 2589 |
| 2048 | 3242 | 3822 | 2805 |
| 4096 | 3791 | 4251 | 3206 |
| 8192 | 3947 | 4414 | 3334 |
| 16384 | 3984 | 4433 | 3346 |
| 32768 | **3988** | **4457** | **3269** |

All four below at 16384², so they are directly comparable:

| | `T_eff` | of peak | of `copyto!` | of memcopy |
|---|---|---|---|---|
| `copyto!` (vendor, 1:1) | 4308 | 88% | 100% | 108% |
| saxpy (KA, 2:1) | 4433 | 90% | 103% | 111% |
| memcopy (KA, 1:1) | 3984 | 81% | 92% | 100% |
| diffusion (KA) | 3346 | 68% | 78% | 84% |

Percentages below are against **KA memcopy** unless stated otherwise: it is written with the same tools as the solver, so it is the like-for-like reference. The other columns answer different questions — `copyto!` for what the abstraction costs (8%), peak for what the silicon could do in principle.

Only the bottom of the size sweep measures bandwidth. 512² is launch-bound at a third of the rate; 1024² saxpy reports 5055 GB/s, above hardware peak, because the working set is L2-resident; 2048² sits in the L2-boundary dip. From 8192² up the numbers are flat to 1%.

A 2:1 read:write ratio also beats any 1:1 copy — saxpy is 11% above memcopy and 3% above the vendor copy, since it costs fewer DRAM bus turnarounds per byte. This is the same effect that puts STREAM Triad above STREAM Copy on most GPUs, and a reminder that the achievable rate depends on the access pattern as well as the hardware.

Diffusion — one Laplacian on top of a copy — holds **82–85% of memcopy** at every converged size. The added flops are essentially free; the cost is the stencil's halo traffic.

## Step 2 - Cahn-Hilliard

```julia
julia> include("scripts/CahnHilliard2D_KA.jl")   # or scaling_test()
```

| n | `T_eff` | of saxpy | of memcopy | ns/cell |
|---|---|---|---|---|
| 512 | 1249 | 28% | 31% | 0.032 |
| 1024 | 3242 | 73% | 81% | 0.012 |
| 2048 | 3230 | 72% | 81% | 0.012 |
| 4096 | 3546 | 80% | 89% | 0.011 |
| 8192 | 3460 | 78% | 87% | 0.012 |
| 16384 | 3406 | 76% | 85% | 0.012 |
| 32768 | 3376 | 76% | 85% | 0.012 |

Two Laplacians and two passes, at **~85% of memcopy**. With the ceiling measured, that number says how much room is left — here, not much.

The last column is worth a look as well. From 1024² up the cost per cell is flat at ~0.012 ns across a **1024× range in problem size**, 16 MB to 16 GB of arrays: the weak scaling from earlier, where a bigger run is a bigger domain at constant cost per cell rather than a longer one. `Δmean` stays at ~1e-20 throughout. Only 512² falls off, being too little work to fill the device. Repeat runs agree to within 1.8% at every size.

The remaining ~15% is not waste. `ncu` shows diffusion moving *identical* DRAM bytes to memcopy at 8192² (536.9 MB read / 517 MB written — the 2-array charge is exact, and L2 absorbs the halo re-reads) while issuing 5.25× the L1 load sectors. The stencil is L1/LSU-request-bound, which cache blocking would not change.

## Same code, two machines

The identical scripts on the Grace CPU of the same node (36 threads, 8192², Float64):

| | Grace, 36 cores | GH200 | ratio |
|---|---|---|---|
| memcopy [2] | 398 GB/s | 3947 | 9.9× |
| saxpy [3] | 391 | 4414 | 11.3× |
| diffusion [2] | 371 | 3334 | 9.0× |
| **Cahn-Hilliard [5]** | **321** | **3460** | **10.8×** |
| | | | |
| CH as % of own memcopy | 81% | 88% | |

The solver sits at a similar fraction of its machine's copy rate on both — 81% on Grace, 88% on Hopper. What changed between them is the ceiling, not the quality of the code, which is the practical use of `T_eff`: it reports how much room is left on a single machine, without needing a second one for comparison.

One asymmetry: on the GPU saxpy is 12% above memcopy, on Grace 2% below it. The bus-turnaround advantage of a 2:1 read:write ratio is a GPU effect that a CPU's caches and prefetchers largely hide, so saxpy is the better reference on GPUs and memcopy on Grace.

## The route there

Each step goes plain Julia → KernelAbstractions → GPU. [`CahnHilliard2D_plain.jl`](scripts/CahnHilliard2D_plain.jl) and [`CahnHilliard2D_KA.jl`](scripts/CahnHilliard2D_KA.jl) are deliberately line-for-line comparable — same `lap`, same two passes, same diagnostics — so the diff isolates what moving to a GPU costs in source terms.

| script | |
|---|---|
| [`memcopy2D_KA.jl`](scripts/memcopy2D_KA.jl) | memcopy / saxpy / diffusion, `T_eff` reference |
| [`CahnHilliard2D_plain.jl`](scripts/CahnHilliard2D_plain.jl) | plain Julia, explicit loops, CPU |
| [`CahnHilliard2D_KA.jl`](scripts/CahnHilliard2D_KA.jl) | KernelAbstractions twin, CPU / CUDA |
| [`CahnHilliard2D_KA_metal.jl`](scripts/CahnHilliard2D_KA_metal.jl), [`memcopy2D_KA_metal.jl`](scripts/memcopy2D_KA_metal.jl) | Apple GPU variants — see below |
| [`common.jl`](scripts/common.jl) | shared helpers: `outdir`, `T_eff`, `bench`, figures |

## Benchmarking traps

The streaming kernels reproduce to ±0.1%. The stencil is more sensitive, landing anywhere in 2620–3130 GB/s on the same hardware depending on how it is measured:

- **`zeros` is not representative data.** All-zero data toggles far fewer HBM lines; on a power-capped GH200 the SM clock sat at ~1720 MHz on zeros versus ~1355 MHz on real data, worth ~5% to the stencil. Fill with `rand!`.
- **Burst ≠ sustained.** A 50-launch burst runs ~11% faster than a multi-second loop.
- **Small `n` is not a bandwidth measurement.** ≤1024² is L2-resident and can report above hardware peak, ≤512² is launch-bound, 2048² sits in the L2-boundary dip. 4096² is still 5% low.
- **`@inbounds` does not cross a function call.** `@kernel inbounds = true` covers only the kernel body, so `lap` needs `Base.@propagate_inbounds`. Worth 9%, and invisible in every memory metric — the only tell is registers/thread.

Tried on CUDA without benefit: workgroup shape, `@Const`, hoisting loads, interior-only guards. Numbers in `CLAUDE.md`.

## Running on an Apple laptop

The Metal backend is Float32-only, which the grid units already accommodate. The `*_metal.jl` variants are preferable to switching the backend line in the NVIDIA scripts: on Apple GPUs KA rebuilds `(ix,iy)` from a 1D dispatch in 64-bit arithmetic, emulated in software, costing ~5× on any 2D kernel ([Metal.jl#910](https://github.com/JuliaGPU/Metal.jl/issues/910)). The `_metal` scripts carry an `Int32` shift/mask workaround that recovers it.

`scaling_test(; fast=false)` versus `fast=true` on the same machine:

| n | `fast=true` | `fast=false` |
|---|---|---|
| 512 | 44.3 GB/s | 22.6 |
| 1024 | 65.7 | 24.1 |
| 2048 | 69.9 | 23.0 |

The slow path is flat at ~0.87 ns/cell at every resolution, being bound by per-thread index arithmetic rather than by grid size. On NVIDIA the same 2D penalty exists but is mild, and largely removed by the static workgroup size and static `ndrange` these scripts already use.

## Further reading

- [PDEs on GPUs](https://pde-on-gpu.vaw.ethz.ch) — the full course this material condenses
