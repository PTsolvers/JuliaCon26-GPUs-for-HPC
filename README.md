# JuliaCon26-GPUs-for-HPC

Hands-on with Julia for HPC on GPUs workshop at JuliaCon 2026.

## Getting started

> **TODO** — access to the Otus cluster at PC2: account request, login, module setup, how to grab an H100 node, and the Julia environment to instantiate.

## Workshop outline

**1. Performance basics — what limits a stencil code, and how to measure it**

Where the time actually goes, from CPU to GPU. Finite-difference stencil codes are overwhelmingly *memory bound*: the few flops in a Laplacian are essentially free next to the cost of moving the arrays. So wall time alone is not a useful target — we introduce **memcopy** as the achievable-bandwidth reference and **`T_eff`** as the metric to report alongside it.

Cahn-Hilliard comes in here as motivation: a PDE worth solving, stiff enough that explicit timestepping needs many small steps, which is exactly where GPU throughput pays off.

The progression makes the memory-bound claim concrete:

- **memcopy / saxpy** — measure the ceiling on this machine
- **2D diffusion** — one Laplacian; should land close to memcopy `T_eff`, since the added flops are "free"
- **Cahn-Hilliard** — two Laplacians and a second pass; see what that costs

Each step goes plain Julia → KernelAbstractions → GPU. `CahnHilliard2D_plain.jl` and `CahnHilliard2D_KA.jl` are deliberately line-for-line comparable: same `lap`, same two passes, same diagnostics. Only the loop wrappers differ.

**2. KernelAbstractions in depth** — writing portable kernels, and composing with the wider ecosystem (e.g. a more capable timestepper rather than hand-rolled explicit Euler).

**3. Chmy.jl (+ KA)** — the same equations, expressed at a higher level.

**4. PETSc.jl** — the same equations again, via PETSc.

**5. Reactant.jl** — if time permits.
