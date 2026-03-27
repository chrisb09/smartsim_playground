#!/usr/local_rwth/bin/zsh

# https://www.craylabs.org/docs/installation_instructions/basic.html
# Python 3.9-3.11
# Pip
# Cmake 3.13.x (or later)
# C compiler
# C++ compiler
# GNU Make > 4.0
# git



#module_names=(
#    "foss/2023b"
#    "OpenSSL/1.1"
#    "Python/3.11.5"
#    "CMake/3.29.3"
#    "CUDA/12.3.0"
#    "cuDNN/8.9.7.29-CUDA-12.3.0"
#    "Clang/18.1.2-CUDA-12.3.0"
#    "GCCcore/12.3.0"
#)

# this loads gcc 12.3, cmake 3.26, openssl 1.1, python 3.11.3
module_names=(
    "GCC/12.3.0" 
    "Clang/16.0.6"
    "CUDA/12.3.0"
    "cuDNN/8.9.7.29-CUDA-12.3.0"
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
    echo "Loading required modules for SmartSim..."
    for module in "${module_names[@]}"; do
        module load "$module"
    done
fi

# Apparently, python 3.11 is linked against OpenSSL 1.1, which is not in the default library path in foss/2023b, since it uses OpenSSL 3 by default.

# Add OpenSSL 1.1 library path for Python's SSL module
export LD_LIBRARY_PATH="$EBROOTOPENSSL/lib:$LD_LIBRARY_PATH"