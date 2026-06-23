# Development Overview

This document summarizes the development and maintenance history of the `CPP-ML-Interface` plugin, `SmartSim` integrations, and the `terrain_solver` mini-app. It serves as a unified reference for ongoing improvements, bug fixes, and performance tuning.

## 1. Deep Learning Backend Coupling (`PhyDLL` and `SmartSim`)

The core goal of this repository is to couple an MPI-parallel physics application (`terrain_solver`) with Deep Learning inference engines directly on HPC clusters. 

### Key Achievements:
- **Provider Abstraction**: Replaced hardcoded RedisAI calls with a generalized `CPP-ML-Interface` architecture allowing `SmartSim`, `PhyDLL`, and `AIxelerator` to be used interchangeably via TOML config files.
- **Dynamic Tensor Shapes**: Rather than hardcoding dimensions for the DL clients (e.g. `[1x18]`), the physical solvers now broadcast a runtime metadata header (`BcastMetaHeader`) using out-of-band communication before inference begins. DL clients (C++ and Python) intercept this, calculate expected strides, and shape the tensors exactly as required by the model architecture without changing core physics code.

## 2. Stability and Scaling Optimizations

When deploying on HPC hardware (e.g. running across 96 MPI ranks on Slurm), several backend-specific limitations were identified and resolved:

### PhyDLL Scaling
- **OOM Prevention (Inference Chunking)**: When running massive single-layer Multi-Layer Perceptrons (e.g. width `4096`), passing 37 million samples synchronously through PyTorch exhausted GPU memory (`CUDA out of memory`).
  - **Solution**: We introduced a `batch_size` (chunk size) parameter to the PhyDLL DL clients (`phydll_dl_client.py` and `dl_client.cpp`). The clients now slice massive received physics tensors into multiple smaller batches along the primary axis, execute the forward pass iteratively inside `torch::no_grad()` (Python) and `torch::NoGradGuard` (C++), and copy the slices back into the output MPI buffer before sending the final buffer back to the solver.
- **Stack Buffer Overflows**: Rewrote string initialization in `kernel.c` (switching from static `256` char arrays to dynamic `malloc`) to prevent crashes when serializing arguments for >24 ranks.
- **CPU Over-subscription Constraints**: Implemented `$MLCOUPLING_INTRA_OP_THREADS` and `mpirun --bind-to none` parameters to override Slurm's cgroup CPU bindings so PyTorch can fully utilize multithreading.
- **Finalization Sequence Ordering**: Resolved a fatal exit crash (`MPI_Bcast called after MPI_FINALIZE`) by explicitly invoking `ml_coupling.reset()` prior to `MPI_Finalize()` in `terrain_solver.cpp`. This forces the provider's destructor (and thus `phydll_finalize()`) to execute and broadcast shutdown signals while the MPI environment is still alive.

### SmartSim Bugs
- **Empty Tensor / Batch Limitations**: Passing extremely large flattened vectors resulted in a `tensor key is empty` error from RedisAI. The CMI override code in `terrain_solver` previously restricted RedisAI's model batch size configuration artificially. Setting `batch_size = 0` resolved the constraint, properly dispatching arbitrarily large batches to the DB.

## 3. Parallel Build Infrastructure

Compiling complex C++ code across different sub-projects (`mini_app`, `CPP-ML-Interface`) requires a highly coordinated build environment inside Slurm.

- **Dynamic Resource Utilization**: The `cmake --build` scripts dynamically scale thread limits up to `96` inside Slurm jobs using `$SLURM_CPUS_ON_NODE`. The same scripts fall back to a safe thread limit (4 or 8) when run on a login shell or developer workstation.
- **Self-Submitting Jobs (`slurm_build.sh`)**: Every C++ sub-project ships a `slurm_build.sh` entry point that detects if it is running inside an existing Slurm job. If not, it re-executes itself under `srun --partition=devel --cpus-per-task=96` so the actual build runs on a 96-core compute node. The same script is also compatible with `sbatch` for asynchronous submission. The four copies of this wrapper live at:
  * `CPP-ML-Interface/slurm_build.sh`
  * `cpu_benchmark/provider_bench/slurm_build.sh`
  * `mini_app/slurm_build.sh`
  * `module_test/slurm_build.sh`
- **Script Sandboxing Bug**: Previously, executing `source install.sh` inside `mini_app/slurm_build.sh` unintentionally overwrote the `SCRIPT_DIR` environment variable, silently skipping the main executable's compilation. Always scope shell variables tightly (e.g., `MINI_APP_DIR` instead of generic `SCRIPT_DIR`) when sourcing scripts from dependent submodules.
- **GPU-Less H100 Targeting**: The `devel` partition has no GPUs, so `TORCH_CUDA_ARCH_LIST=9.0` is hardcoded in every `build.sh` to force H100-compatible SM_90 code generation. Without this, PyTorch's host auto-detection defaults to older capabilities and the build hangs or produces binaries that don't load on compute nodes.
