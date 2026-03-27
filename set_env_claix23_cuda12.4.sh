#!/usr/local_rwth/bin/zsh

# SmartSim requirements:
# - Python 3.9-3.11, pip
# - CMake >= 3.13
# - C/C++ compilers, GNU Make > 4.0, git
# CUDA 12.3 path used here:
# - GCC < 13
# - CUDA 12.3
# - cuDNN 8.9-compatible family

# CUDA/12.4.0  GCCcore/11.3.0  Clang/15.0.5  GCC/11.3.0  OpenMPI/4.1.4  FFTW.MPI/3.3.10  HDF5/1.12.2  PnetCDF/1.12.3  cuDNN/8.9.7.29-CUDA-12.4.0  imkl/2024.2.0

module_names=(
    OpenSSL/1.1
    CUDA/12.4.0
    GCCcore/11.3.0
    Clang/15.0.5
    GCC/11.3.0
    OpenMPI/4.1.4
    FFTW.MPI/3.3.10
    HDF5/1.12.2
    PnetCDF/1.12.3
    cuDNN/8.9.7.29-CUDA-12.4.0
    imkl/2024.2.0
)

any_load_required=false

for module in "${module_names[@]}"; do
    if ! module is-loaded "$module" &> /dev/null; then
        any_load_required=true
        echo "Module $module is not loaded. Purging and loading required modules."
        break
    fi
done

if [ "$any_load_required" = true ]; then
    echo "Purging and loading required modules for SmartSim..."
    module purge
    echo "Loading required modules..."
    for module in "${module_names[@]}"; do
        module load "$module"
    done
fi

# Keep OpenSSL runtime path so Python SSL remains consistent.
export LD_LIBRARY_PATH="$EBROOTOPENSSL/lib:$LD_LIBRARY_PATH"
