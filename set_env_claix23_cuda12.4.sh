#!/bin/bash

# Initialize Lmod module command for non-interactive shells
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# Disable Lmod pagination and interactive warning prompts
export TERM=dumb
export LMOD_PAGER=cat
export PAGER=cat

# SmartSim requirements:
# - Python 3.9-3.11, pip
# - CMake >= 3.13
# - C/C++ compilers, GNU Make > 4.0, git
# CUDA 12.3 path used here:
# - GCC < 13
# - CUDA 12.3
# - cuDNN 8.9-compatible family

# CUDA/12.4.0  GCCcore/11.3.0  Clang/15.0.5  GCC/11.3.0  OpenMPI/4.1.4  FFTW.MPI/3.3.10  HDF5/1.12.2  PnetCDF/1.12.3  cuDNN/8.9.7.29-CUDA-12.4.0  imkl/2024.2.0

module_names="OpenSSL/1.1 CUDA/12.4.0 GCCcore/11.3.0 Clang/15.0.5 GCC/11.3.0 OpenMPI/4.1.4 FFTW.MPI/3.3.10 HDF5/1.12.2 PnetCDF/1.12.3 cuDNN/8.9.7.29-CUDA-12.4.0 imkl/2024.2.0"
if [[ "${USE_SCOREP:-}" == "1" ]]; then
    module_names="$module_names Score-P/8.4 PAPI/7.0.0"
fi

echo "Loading required modules..."
for module in $module_names; do
    module load "$module" >/dev/null 2>&1 || true
done

# Keep OpenSSL runtime path so Python SSL remains consistent.
export LD_LIBRARY_PATH="$EBROOTOPENSSL/lib:$EBROOTCLANG/lib:$LD_LIBRARY_PATH"
export LIBCLANG_PATH="${EBROOTCLANG}/lib"
