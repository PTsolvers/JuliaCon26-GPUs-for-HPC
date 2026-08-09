#!/usr/bin/env bash
# This is to setup PETSc on a node of the HPC System Otus in Paderborn
# Note that this will not allow runing PETSc on multiple nodes (needs more
# compicated setup, see e.g. PETSc.jl docs)
#
# run this with "source setup_julia_petsc_local.sh" to setup the environment for PETSc.jl

set -euo pipefail

module load lang/JuliaHPC/1.12.6-foss-2025a

# Slurm leaks PMIX_*/SLURM_PMIX_* into every process in this job step.
# MPICH_jll's Hydra doesn't speak PMIx, and package extensions like
# PETScCUDAExt do a singleton MPI_Init at precompile time -> strip these
# so that init just falls back to plain singleton mode instead of
# aborting. (mpiexecjl multi-rank runs need a *different* fix -
# MPIR_CVAR_PMI_VERSION=2 - handled separately in run_mpi.sh.)
unset $(env | grep -oE '^PMIX?_[A-Za-z0-9_]*' | tr '\n' ' ')
unset $(env | grep -oE '^SLURM_PMIX?_?[A-Za-z0-9_]*' | tr '\n' ' ')
export MPIR_CVAR_PMI_VERSION=2

julia --project=. -e '
    using MPIPreferences
    MPIPreferences.use_jll_binary("MPICH_jll")
    Pkg.instantiate()

    using MPI
    MPI.install_mpiexecjl(; force=true)
'