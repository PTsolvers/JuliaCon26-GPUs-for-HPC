# JuliaCon26-GPUs-for-HPC

Hands-on with Julia for HPC on GPUs workshop at JuliaCon 2026.

## Getting started

> **TODO** — access to the Otus cluster at PC2: account request, login, module setup, how to grab an H100 node, and the Julia environment to instantiate.

## Workshop outline

1. **Performance basics** — what limits a stencil code, and how to measure it
2. **KernelAbstractions in depth** — portable kernels, and composing with the wider ecosystem (e.g. other timestepper instead of hand-rolled explicit Euler)
3. **Chmy.jl (+ KA)** — the same equations, expressed at a higher level using dimensions-agnostic DSL
4. **PETSc.jl** — the same equations again, via PETSc but on (parallel) CPU's
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



# Part 4: Using PETSc.jl


In this part of workshop we will look at the same equations once more, but this time through PETSc — on parallel CPUs rather than GPUs. The goal of this part is not to make Cahn-Hilliard faster, but to show what a library like PETSc buys you: MPI decomposition you do not have to write, and a menu of solvers you can change from the command line without touching your code. That second point is what makes *implicit* timestepping along with multigrid preconditioners practical, which is where the section ends.

## What is PETSc?

[PETSc](https://petsc.org/) (Portable, Extensible Toolkit for Scientific Computation) is a long-established (and massive!) C library for solving PDEs in parallel, from laptops to the largest machines. It is used as a solver in many existing codes (such as FENICS) or finite element packages. 

Think of it as a parallel library to solve (non)linear system of equations. At the same time, it also has powerful timestepping algorithms.

It provides the pieces you would otherwise hand-roll:

| | |
|---|---|
| `Vec`, `Mat` | distributed vectors and sparse matrices |
| `DM` / `DMDA` | the grid: which rank owns what, and the ghost exchange between them (also provides support for finite elements, AMR, ...)|
| `KSP` | Krylov linear solvers (CG, GMRES, …) plus preconditioners (ILU, multigrid, direct, …) |
| `SNES` | Newton solvers for nonlinear systems, built on KSP |
| `TS` | timesteppers, built on SNES |

The layering is the point: each level uses the one below, so an implicit timestep is a `TS` calling a `SNES` calling a `KSP`. You supply a *residual function* (with the physics) — and PETSc supplies everything else. Crucially, **every solver choice is a runtime option**, so the same binary can run a direct solve on a laptop and multigrid on 10 000 cores.

## What about PETSc.jl?
Obviously, it would be great to combined PETSc with the existing Julia ecosystem. There are a number of julia packages that do this; arguably [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl) is the most feature-complete at the moment (specially after a big releases earlier this year).

It provides:
- a **high-level interface** that feels like Julia — `KSP(A)`, `solve!(x, ksp, b)` — covering the most-used parts (or at least those parts that the `PETSc.jl` developers are interested in);
- a **low-level interface** (`PETSc.LibPETSc.*`) that mirrors the C API almost one-for-one, for everything not yet wrapped. It has over 3000 functions.

There are ofcourse many practical advantages over using the C version of PETSc. It ships **pre-built binaries**, so `] add PETSc` gives you a working parallel PETSc with MUMPS, SuperLU_DIST and HYPRE on Linux, macOS and Windows — no build step. And because residual routines are ordinary Julia functions, you can use automatic differentiation for Jacobians, or write them with KernelAbstractions and run them on a GPU.

On linux and mac, it will also work in parallel with:
```julia
using MPI, PETSc
MPI.Initialized() || MPI.Init()
petsclib = PETSc.getlib(; PetscScalar = Float64, PetscInt = Int64)
PETSc.initialize(petsclib, log_view=false) # set to true to display log info @ the end
# ... your code ...
PETSc.finalize(petsclib)
```

`petsclib` selects a build: PETSc.jl ships `Float64`/`Float32`, real/complex and `Int32`/`Int64` variants, and you pick the combination you want.

## Solve a 1D steady-state diffusion equation

Lets start with a simple example that still has all the pieces: a 1D steady-state diffusion example with  **variable coefficients**:

$$-\frac{d}{dx}\left( k(x) \frac{du}{dx} \right) = f(x)
\qquad \text{on } [0,1], \qquad u(0) = u(1) = 0$$

With $n$ points, spacing $h$, and $x_i = (i-1)h$, the conservative discretisation evaluates the conductivity on cell **faces**, $k_{i\pm1/2} = k(x_i \pm h/2)$:

$$-k_{i-1/2} u_{i-1} + \left(k_{i-1/2} + k_{i+1/2}\right) u_i - k_{i+1/2} u_{i+1} = h^2 f_i$$

Evaluating $k$ at faces rather than averaging $k(x_i)$ is what makes the scheme conservative: the flux leaving cell $i$ through a face is exactly the flux entering cell $i+1$. The resulting coefficient matrix $A$ is tridiagonal and symmetric positive definite, so CG is the natural Krylov method.

The two inputs are a **uniform source** $f(x) = 1$ and a smooth **100× contrast** in conductivity, $k(x) = 1 + 99\sigma\left((x-0.5)/0.02\right)$ with $\sigma$ the logistic function — so $k \approx 1$ on the left half and $\approx 100$ on the right.

Both matter for the shape of the answer. With a source, $u$ must curve: integrating once gives $-k(x)\,u'(x) = x - C$, so the flux grows linearly as it picks up source along the way. For *constant* $k$ that integrates to the symmetric parabola $u = x(1-x)/2k$, peaking at $x = 0.5$. (Without a source, $f = 0$, $u$ would just be a straight line — here identically zero, since both ends are pinned at zero.) With variable $k$, $u'(x) = (C-x)/k(x)$ drops by 100× across the jump, which is what produces the kink, the steep left flank, the near-flat right one, and a maximum pushed left to $x \approx 0.21$ instead of $0.5$.

### The grid, and the linear system

The unknowns live on the grid points, the conductivities on the faces between them:

```
   i-1                 i                 i+1        <- unknowns u_i on the points
    o ------- x ------- o ------- x ------- o
              ^                   ^
          k_{i-1/2}           k_{i+1/2}             <- conductivity on the faces
    |<------ h ------>|
```

Each stencil row uses one point either side and the two faces adjacent to it. Since $u_{i-1}, u_i, u_{i+1}$ enter linearly, stacking all $n$ equations gives

$$A\,\mathbf{u} = \mathbf{b}, \qquad d_i = k_{i-1/2} + k_{i+1/2}, \qquad
A = \begin{pmatrix}
d_1      & -k_{3/2} &            &            &  \\
-k_{3/2} & d_2      & -k_{5/2}   &            &  \\
         & \ddots   & \ddots     & \ddots     &  \\
         &          & -k_{n-3/2} & d_{n-1}    & -k_{n-1/2} \\
         &          &            & -k_{n-1/2} & d_n
\end{pmatrix}$$

with $\mathbf{b}$ holding $h^2 f_i$. $A$ is **tridiagonal** (three non-zeros per row, so $\approx 3n$ entries instead of $n^2$), **symmetric** (the coefficient linking $i$ to $i+1$ and $i+1$ to $i$ are both $-k_{i+1/2}$ — the same face, a direct consequence of evaluating $k$ there), and **positive definite**. Symmetric positive definite is the class conjugate gradient is built for, hence `-ksp_type cg` below.

Solving the PDE is now just: build $A$ and $\mathbf{b}$ & hand them to a linear solver.

[`scripts/diffusion1D_PETSc.jl`](scripts/diffusion1D_PETSc.jl) is the whole program — assemble, solve, done:

```julia
A = PETSc.MatSeqAIJ(petsclib, n, n, 3)      # sequential sparse matrix, 3 non-zeros per row
b = PETSc.VecSeq(petsclib, zeros(n))
for i in 1:n
    km = kfun(x[i] - h/2)                   # conductivity on the left face of point i
    kp = kfun(x[i] + h/2)                   # ...and the right face
    i > 1 && (A[i, i-1] = -km)
    A[i, i]             = km + kp
    i < n && (A[i, i+1] = -kp)
    b[i] = h^2 * ffun(x[i])
end
PETSc.assemble!(A)                          # PETSc caches entries; this flushes them

ksp = PETSc.KSP(A; opts...)
u   = ksp \ b                              # or, in-place: PETSc.solve!(u, ksp, b)
```

```bash
$ julia --project=. scripts/diffusion1D_PETSc.jl
n = 100,  KSP its = 1,  max(u) = 0.022606 at x = 0.208
plot written to output/diffusion1D_PETSc/u.png
```

"KSP its = 1" because the default for a small system is a direct LU factorisation — one application, no iteration. This is already a complete PETSc program, but it runs on one core.

![1D diffusion with a 100x conductivity contrast](assets/diffusion1D_PETSc.png)

The plot is the check that the physics is right. With constant `k` the solution would be a symmetric parabola peaking at `x = 0.5`; here the conductivity jumps by 100x at mid-domain (lower panel, log scale), so heat escapes easily to the right and the maximum is pushed left to `x ≈ 0.21`. The kink in `u` sits exactly where `k` changes, and `u` is much flatter on the conductive side — a small gradient suffices to carry the same flux.

## Doing this in parallel

### First: getting an `mpiexec` that matches

Launch MPI jobs with **`mpiexecjl`**, from the command-line as this is compatible with  `MPI.jl`. To do this, install it once:

```bash
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'
```

That writes `mpiexecjl` to `~/.julia/bin` — add it to your `PATH`, or call it by full path as below. (If it is already there you will get an error saying so; add `force=true` to overwrite.) To check which MPI you are actually on: `julia --project=. -e 'using MPI; println(MPI.MPIPreferences.binary)'`. On a cluster you would instead point MPI.jl at the system MPI — see [Using PETSc.jl on very large HPC systems](#using-petscjl-on-very-large-hpc-systems) at the end.

### The DMDA

[`scripts/diffusion1D_PETSc_dmda.jl`](scripts/diffusion1D_PETSc_dmda.jl) solves the identical 1D elliptic equation on any number of ranks. The only change is that a **DMDA** needs to be specified, which has info about the grid and is used to distribute it in parallel:

```julia
da = PETSc.DMDA(petsclib, comm, (PETSc.DM_BOUNDARY_NONE,), (n,), 1, 1,
                PETSc.DMDA_STENCIL_STAR; opts...)   # n points, 1 DOF, stencil width 1

A = PETSc.LibPETSc.DMCreateMatrix(petsclib, da)     # matrix with the DMDA's parallel layout
b = PETSc.DMGlobalVec(da)

corners = PETSc.getcorners(da)                      # which points does THIS rank own?
xs, xe  = corners.lower[1], corners.upper[1]
for i in xs:xe
    ...                                             # same stencil, GLOBAL indices
end
PETSc.assemble!(A)                                  # routes any off-rank entries
```

There are 3 things the DMDA does: it decides the parallel decomposition, it creates matrices and vectors with the matching layout, and it lets you assemble using **global indices** — so the loop is the serial loop with `1:n` replaced by `xs:xe`. Nothing else changes.

Once this is done 
```
$ julia --project=. scripts/diffusion1D_PETSc_dmda.jl
n = 100,  1 rank(s),  KSP its = 1,  max(u) = 0.022618 at x = 0.212

$ ~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/diffusion1D_PETSc_dmda.jl
n = 100,  4 rank(s),  KSP its = 7,  max(u) = 0.022618 at x = 0.212
```

Same answer on 1, 2 and 4 ranks — but note **the iteration count changed**: 1 → 3 → 7. The default preconditioner is block Jacobi with one block per rank, so it gets weaker as the domain is cut into more pieces. That is a genuine property of the method, not a bug, and it is exactly the kind of thing you want to be able to change without editing code.

### Changing the solver from the command line

Every PETSc option can be passed on the command line, so the same script becomes a different solver without an edit or a recompile:

```bash
... diffusion1D_PETSc_dmda.jl -n 1000 -ksp_type cg -pc_type jacobi    # KSP its = 995
... diffusion1D_PETSc_dmda.jl -n 1000 -ksp_type cg -pc_type bjacobi   # KSP its = 7
... diffusion1D_PETSc_dmda.jl -n 1000 -ksp_type cg -pc_type gamg      # KSP its = 6
```

Useful options to know:

| option | |
|---|---|
| `-ksp_type cg \| gmres \| fgmres \| preonly` | Krylov method (`preonly` = apply the PC once, for direct solves) |
| `-pc_type jacobi \| bjacobi \| ilu \| lu \| gamg \| mg \| hypre` | preconditioner |
| `-ksp_monitor`, `-ksp_converged_reason` | watch the residual / find out why it stopped |
| `-ksp_view` | print the entire solver configuration PETSc actually built |
| `-ksp_rtol 1e-10` | tolerance |
| `-help` | every option the current program understands |

`-ksp_view` is the one to reach for when something is unexpectedly slow: it shows the solver PETSc assembled, including defaults you never set.

### Multigrid

Refine the grid and the picture becomes sharper. The number to watch is the **KSP iteration count as a function of `n`** — it says whether a method will still work when the problem gets bigger, and unlike a timing it is immune to what else the machine is doing:

| n | `-pc_type jacobi` | `-pc_type gamg` | `-pc_type mg` (Galerkin) |
|---|---|---|---|
| 513 | 510 | 6 | **2** |
| 2049 | 2041 | 6 | **2** |
| 8193 | 8172 | 6 | **1** |
| 32769 | 10000 (did not converge) | 8 | **1** |

Diagonal scaling (jacobi) costs `O(n)` iterations — each one `O(n)` work, so `O(n²)` overall, and by 32769 it does not converge at all within the iteration limit. Both multigrid variants are essentially **flat**: the work per solve grows only in proportion to the number of unknowns. That is the defining property of a multigrid method, and the reason it is the standard tool for elliptic problems.

The two multigrid flavours differ in where the hierarchy comes from:

```bash
# algebraic multigrid: builds its own coarse grids by inspecting A.  Needs nothing but the matrix.
... -n 2049 -ksp_type cg -pc_type gamg

# geometric multigrid: coarsens the DMDA, and forms the coarse operators as A_c = RᵀAR
# ("Galerkin"), so no re-discretisation on each level is needed.
... -n 2049 -ksp_type cg -pc_type mg -pc_mg_levels 4 -pc_mg_galerkin both -mg_coarse_pc_type lu

# ...and the same in parallel (the coarse LU must be a parallel one)
~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/diffusion1D_PETSc_dmda.jl -n 8193 \
    -ksp_type cg -pc_type mg -pc_mg_levels 4 -pc_mg_galerkin both \
    -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type lu
```

Geometric multigrid is the stronger of the two here because the DMDA already knows the grid hierarchy — it does not have to guess one from the matrix. `-pc_mg_galerkin both` is what makes it convenient: without it, PETSc would ask you to supply a freshly discretised operator on every level via a callback, whereas Galerkin builds them from the fine matrix we already assembled (so the code is slightly more complex).

**One trap worth knowing.** Dirichlet boundaries are often imposed with an identity row (`A[i,i] = 1`). With a 100× contrast in `k`, that row is ~100× smaller in value than its neighbours, and Galerkin coarsening inherits the mismatch. That's no good for convergence. Scaling the boundary rows to match the interior — `A[i,i] = 2k(x_i)`, restores a clean 2 iterations at every level. Badly scaled rows are invisible to a direct solver and fatal to multigrid.

## Customizing and timing runs


`PETSc.initialize(petsclib, log_view = true)` (or `-log_view`) turns on PETSc's profiler. At the end of the run it prints a table of every operation — `KSPSolve`, `MatMult`, `PCSetUp`, `VecScatterBegin` — with call counts, time, flop rates and MPI message volumes.

This is the right tool for "where is the time going" (also on large parallel HPC machines)
.

## Cahn-Hilliard with explicit timestepping

Now back to the workshop's equation. The [plain-Julia](/scripts/CahnHilliard2D_plain.jl) Cahn-Hilliard solver from Part 1 becomes MPI-parallel with essentially no change to the physics — the DMDA supplies the ghost exchange that a distributed stencil needs.

[`scripts/CahnHilliard2D_PETSc_explicit.jl`](scripts/CahnHilliard2D_PETSc_explicit.jl) uses PETSc for exactly three things:

1. **DMDA** — decomposes the `nx × ny` grid across ranks;
2. **global and local vectors** — the global vector holds owned points, the local one adds a layer of ghost points;
3. **`dm_global_to_local!`** — fills that ghost layer from the neighbours.

The kernels are the plain Julia ones, unchanged:

```julia
PETSc.dm_global_to_local!(g_C, l_C, da, PETSc.INSERT_VALUES)   # the only new line
withvecs((c, m) -> chemical_potential!(m, c, γ, nx, ny, xs, xe, ys, ye),
         (l_C, ghost_corners), (g_μ, corners))
```

The one subtlety is indexing. In parallel, every rank only owns part of the domain and halos are used to transfre info. To make the code similar to the serial plain julia versions, arrays are wrapped as `OffsetArray`s carrying **global** indices, so `lap` is identical to the serial version and each rank simply loops over `xs:xe, ys:ye` instead of `1:nx, 1:ny`. Whether a neighbour is an owned point or a ghost is thus invisible in the kernel.

```bash
julia --project=. scripts/CahnHilliard2D_PETSc_explicit.jl
~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/CahnHilliard2D_PETSc_explicit.jl
```

`F` and `Δmean` are identical on any number of ranks, which is the check that the halo exchange is right.

### 1 DOF or 2?

[`CahnHilliard2D_PETSc_explicit_2dof.jl`](scripts/CahnHilliard2D_PETSc_explicit_2dof.jl) is the same solver but rather than holding only `C` at the DMDA, we use two degrees of freedom at every point  `C` and `μ` interleaved in one vector, `[C₁ μ₁ C₂ μ₂ …]`, rather than two separate vectors. The physics is identical — `F: 65902.1 -> 65180.1` from both, at n = 512 over 4000 steps — but it is **25% slower** (0.63 vs 0.50 ms/step), because each kernel touches one field while the interleaved layout drags the other through cache, and each ghost exchange moves both fields when only one is needed.

So the explicit scheme should use 1 DOF. The reason the comparison is worth making is that the *implicit* scheme, discussed in the next section, needs 2.

## Cahn-Hilliard with implicit timestepping

Explicit Cahn-Hilliard is bounded by $\Delta t \propto \Delta x^4$, which is brutal: the 512² run needs 40 000 steps. Implicit timestepping removes the stability limit, so $\Delta t$ is set by accuracy instead.

Writing both relations at the **new** time level and moving everything to one side gives residuals that must vanish:

$$\begin{aligned}
R_C &= \frac{C - C^{\text{old}}}{\Delta t} - D\nabla^2 \mu &&= 0 \\
R_\mu &= \mu - (C^3 - C) + \gamma\nabla^2 C &&= 0
\end{aligned}$$

These are the *same expressions* as the explicit passes, rearranged — but now $C$ and $\mu$ appear on both sides, so a timestep is no longer an evaluation but the solution of a coupled nonlinear system $R(x) = 0$ with $x = (C, \mu)$ over the whole grid. Newton linearisation gives:

$$\mathbf{J}(x) \delta x = -R(x), \qquad x \leftarrow x + \alpha \delta x,
\qquad \mathbf{J} = \frac{\partial R}{\partial x}$$

so **every timestep requires a Jacobian and a linear solve**. That is the price for the larger $\Delta t$.

With 2 DOFs per node the unknowns interleave, $x = [C_1, \mu_1, C_2, \mu_2, \dots]$, and $\mathbf{J}$ reads as a matrix of $2\times 2$ blocks — one per pair of grid nodes:

$$\mathbf{J} =
\begin{pmatrix}
\partial R_C/\partial C & \partial R_C/\partial \mu \\
\partial R_\mu/\partial C & \partial R_\mu/\partial \mu
\end{pmatrix}
=
\begin{pmatrix}
1/\Delta t & -D\nabla^2 \\
-(3C^2 - 1) + \gamma\nabla^2 & 1
\end{pmatrix}$$

The diagonal block (a node with itself) is dense; each off-diagonal block (a node with one of its 4 neighbours) has only the two anti-diagonal entries, since `C` and `μ` couple to neighbours purely through the Laplacians. So `J` has block size 2 with 5 blocks per row — and *this* is why the DMDA must carry 2 DOFs: PETSc needs to know there are two unknowns per node to build those blocks, and to offer field-splitting.

Note the $\mu$ row has **no $1/\Delta t$ term** — the chemical potential is a constraint, not an evolution equation. The system is therefore indefinite, which is what makes the linear solve interesting (or more challenging, some would say).

[`scripts/CahnHilliard2D_PETSc_implicit.jl`](scripts/CahnHilliard2D_PETSc_implicit.jl) supplies the residual (the code above, one loop) and the Jacobian analytically; `SNES` does the rest (solving the nonlinear system, linesearch to find the optimal $\alpha$, ...).

### Running it, and watching Newton converge

```bash
julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl -n 129 -nt 2 -snes_monitor -snes_converged_reason
```

`-snes_monitor` prints the nonlinear residual $\|R\|$ at each Newton iteration, and `-snes_converged_reason` says why it stopped:

```
  0 SNES Function norm 2.074291063508e+01
  1 SNES Function norm 6.983171234290e-03
  2 SNES Function norm 3.337947518888e-09
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
> step      1, t =     2.78, F = 4160.15, Δmean = +2.57e-19, Newton its = 2
```

Look at the norms: `2e+01 → 7e-03 → 3e-09`, roughly squaring the error each time. That is **quadratic convergence**, and it is the signature of a correct Jacobian — a wrong or approximate one may still converges, but linearly, taking many more iterations. `-snes_test_jacobian` checks it directly against finite differences (here they agree to $\sim\!10^{-10}$).

Because every Newton step contains a linear solve, the two monitors nest. Adding `-ksp_monitor`:

```
  0 SNES Function norm 2.074291063508e+01     <- Newton iteration 0
    0 KSP Residual norm 2.074291063508e+01    <-   linear solve for δx
    1 KSP Residual norm 1.829713716018e-02
    2 KSP Residual norm 9.705965522197e-04
    3 KSP Residual norm 5.585927073168e-05
  1 SNES Function norm 6.982811325128e-03     <- Newton iteration 1
    0 KSP Residual norm 6.982811325128e-03
    ...
  2 SNES Function norm 1.233011679467e-08
```

This is the `SNES → KSP → PC` layering made visible, and it is the first thing to look at when an implicit solve is slow: if the Newton counts are small but each linear solve takes hundreds of iterations, the preconditioner is the problem, not the physics.

Useful SNES options, alongside the KSP ones from the 1D example:

| option | |
|---|---|
| `-snes_monitor` | residual norm per Newton iteration |
| `-snes_converged_reason` | why it stopped (or why it failed) |
| `-snes_view` | the full nonlinear solver configuration, KSP and PC included |
| `-snes_test_jacobian` | compare the supplied Jacobian against finite differences |
| `-snes_linesearch_type bt \| basic` | backtracking line search (the default here) or plain full steps |
| `-snes_max_it`, `-snes_rtol` | iteration cap and tolerance |
| `-log_view` | the profiler: time in `SNESSolve` vs `KSPSolve` vs `MatMult` |

### How the solvers scale
The disadvantage of the implicit solve is that each step is much slower than the explicit solve, as we need to solve a nonlinear system of equations. The advantage is that we can use much larger timestes and that it is the more classical way to solve such equations, which are resolved to computer precision.

Yet, using scalable preconditioners is crucial. You will quickly run into limitations if you want to solve it with direct solvers. Below some timings (performed on a MacBook M4 Max), in s/step:

| | n=129 | n=257 | n=513 | n=1025 | KSP its |
|---|---|---|---|---|---|
| direct LU | 0.62 | 1.04 | 3.93 | 22.73 | — |
| geometric MG | 0.59 | 0.74 | 1.39 | 4.00 | ~2, grid-independent |
| ASM + ILU(3) | 0.57 | 0.62 | 1.03 | 2.32 | 6.8 |

Fitting $t \sim N^p$ over the last refinement: LU $p = 1.27$ (superlinear, as a factorisation must be), MG $0.76$, ASM $0.59$. At 129² all three are within 10% — the direct solver is fine at small sizes; it is the *trend* that rules it out. Multigrid's ~2 iterations are independent of grid size, which is the property to look for.

```bash
# recommended: geometric multigrid (needs n = 2^k+1, hence the default 513)
~/.julia/bin/mpiexecjl -n 4 julia --project=. scripts/CahnHilliard2D_PETSc_implicit.jl \
    -ksp_type fgmres -pc_type mg -pc_mg_levels 4 \
    -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 8 \
    -mg_levels_pc_type bjacobi -mg_levels_sub_pc_type ilu \
    -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type lu
```

With `dt` 400× the explicit limit, the run needs **100 steps instead of 40 000** and lands within ~2% of the explicit `F`. The script's header documents the solver options in more detail.

## Using PETSc.jl on very large HPC systems

The pre-built binaries use their own MPI, which is fine on a laptop but not on a cluster, where you must use the system MPI to get the interconnect. Two routes, both documented in the [PETSc.jl docs](https://github.com/JuliaParallel/PETSc.jl):

1. **MPI ABI** — configure [MPIPreferences](https://juliaparallel.org/MPI.jl/stable/configuration/) to point MPI.jl at the system MPI, and use a PETSc binary built against the same ABI. You keep the pre-built PETSc.
2. **Your own PETSc** — build (or load) PETSc on the machine and point PETSc.jl at it via preferences. That is more work, but it lets you enable exactly the external packages and options you need (including compiling it with GPU support).

Check what you are actually using with `julia -e 'using MPI; println(MPI.MPIPreferences.binary)'` before submitting a large job.

## PETSc and GPU support

PETSc has gained substantial GPU support in recent years. If built with CUDA/HIP/Kokkos, vectors and matrices can live on the device — usually a command-line option such as `-dm_vec_type cuda -dm_mat_type aijcusparse` — and nearly all the iterative solvers and preconditioners then run there.

The catch is data movement: PETSc keeps host and device copies and synchronises them, so a residual evaluated on the CPU forces a transfer every iteration and can easily cost more than it saves. For good performance the **residual routine should run on the GPU too** — which is exactly what KernelAbstractions is for, and why the KA kernels from Part 1 compose well with this. PETSc.jl's `examples/ex19.jl` shows a fully GPU-resident residual and FD-coloring Jacobian.


Note that **the pre-built PETSc_jll binaries currently ship without GPU support**, so this route needs a custom PETSc build for now. We would really appreciate help with this!
