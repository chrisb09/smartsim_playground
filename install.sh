#!/bin/sh

# Set temporary directory to writable location to avoid /tmp issues with CUDA
# Specifically, we might have insufficient permissions to create files in /tmp
export TMPDIR="${PWD}/tmp"
mkdir -p "$TMPDIR"

. ./set_env_claix23_cuda12.4.sh 

DEFAULT_RUNTIME_ROOT="/home/thes2181/python"
RUNTIME_ROOT="${SMARTSIM_RUNTIME_ROOT:-$DEFAULT_RUNTIME_ROOT}"
RUNTIME_TAR_DIR="${SMARTSIM_RUNTIME_TAR_DIR:-$RUNTIME_ROOT}"
CREATE_RUNTIME_TARS="${SMARTSIM_CREATE_RUNTIME_TARS:-1}"
CREATE_RUNTIME_SYMLINK="${SMARTSIM_CREATE_RUNTIME_SYMLINK:-1}"
FORCE_RUNTIME_TAR_REBUILD="${SMARTSIM_FORCE_RUNTIME_TAR_REBUILD:-0}"
VERIFY_RUNTIME_TAR_FRESHNESS="${SMARTSIM_VERIFY_RUNTIME_TAR_FRESHNESS:-0}"
RUNTIME_LIBSTDCPP_GCCCORE_VERSION="${SMARTSIM_RUNTIME_LIBSTDCPP_GCCCORE_VERSION:-13.2.0}"
RUNTIME_LIBSTDCPP_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/GCCcore/${RUNTIME_LIBSTDCPP_GCCCORE_VERSION}/lib64/libstdc++.so.6"
RUNTIME_LIBGCC_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/GCCcore/${RUNTIME_LIBSTDCPP_GCCCORE_VERSION}/lib64/libgcc_s.so.1"

if [ ! -d "$RUNTIME_ROOT" ]; then
    mkdir -p "$RUNTIME_ROOT" 2>/dev/null || true
fi
if [ ! -w "$RUNTIME_ROOT" ]; then
    echo "Warning: runtime root '$RUNTIME_ROOT' is not writable. Falling back to '${PWD}/python'."
    RUNTIME_ROOT="${PWD}/python"
    RUNTIME_TAR_DIR="${SMARTSIM_RUNTIME_TAR_DIR:-$RUNTIME_ROOT}"
    mkdir -p "$RUNTIME_ROOT"
fi

mkdir -p "$RUNTIME_TAR_DIR"
echo "Using SmartSim runtime root: $RUNTIME_ROOT"
echo "Using SmartSim runtime tar dir: $RUNTIME_TAR_DIR"

if [ "$CREATE_RUNTIME_SYMLINK" = "1" ]; then
    if [ ! -L "${PWD}/python" ]; then
        if [ -d "${PWD}/python" ]; then
            echo "Keeping existing '${PWD}/python' directory (not replacing with symlink)."
        else
            ln -snf "$RUNTIME_ROOT" "${PWD}/python" 2>/dev/null || true
        fi
    fi
fi

# options: cpu,rocm-64,cuda-11,cuda-12
devices="cpu cuda-12"

# Optional override: source ./install.sh cuda-12
# If provided, install only for the requested device(s).
if [ -n "${1:-}" ]; then
    devices="$1"
    echo "Overriding devices via argument: $devices"
fi

# options: torch tensorflow onnxruntime
backends="torch tensorflow onnxruntime"

CHANGED_RUNTIME_DEVICES=""

for device in $devices; do

    echo ""
    echo "=== Installing SmartSim for device: $device ==="

    # See smart build --help for all options

    # TODO: if cuda-12 is selected, we want to disable torch installation by smartsim, and instead install it using pip3 directly so we can omit the uninstall of pytorch-cu121 which is not perfectly compatible with CUDA 12.4 (and neither 12.3).

    all_backends="torch onnxruntime tensorflow"
    backend_opts=""
    for backend in $all_backends; do
        if echo "$backends" | grep -qw "$backend"; then
            echo "Including backend: $backend"
        else
            echo "Excluding backend: $backend"
            backend_opts="$backend_opts --skip-$backend"
        fi
        if [ "$device" = "cuda-12" ] && [ "$backend" = "torch" ]; then
            echo "Excluding torch backend for device $device to avoid CUDA 12.4 compatibility issues with PyTorch. We will install it separately using pip after the smartsim build."
            backend_opts="$backend_opts --skip-torch"
        fi
    done

    # Force GCC for Redis build to avoid Clang linker plugin issues
    export CC=gcc
    export CXX=g++
    export LD=ld
    
    smartsim_build_cmd="smart build --device $device$backend_opts"

    echo "SmartSim build command: $smartsim_build_cmd"

    python_env="$RUNTIME_ROOT/smartsim_$device/"

    echo "Install location: $python_env"

    base_dir="${PWD}/"

    echo "Base directory for SmartSim installation: $base_dir"

    env_created=false
    runtime_changed=false

    if [ ! -d "$python_env" ]; then
        echo "Creating Python virtual environment at $python_env"
        python -m venv $python_env
        env_created=true
        runtime_changed=true
    else
        echo "Python virtual environment already exists at $python_env"
    fi

    # Check if virtual environment is already activated and it's the correct one
    if [ -z "${VIRTUAL_ENV:-}" ] || [ "${VIRTUAL_ENV:-}" != "$(realpath "$python_env")" ]; then
        # Check if device is the last in the list, only activate virtual environment if it's the last device to avoid activating multiple virtual environments in the same shell session
        if [ "$device" = "${devices##* }" ] || [ "$env_created" = true ]; then
            echo "Activating Python virtual environment for device: $device"
            . "$python_env/bin/activate"
        else
            echo "Not activating virtual environment for device: $device to avoid pointless long loading times. Please activate it manually if you want to use it."
            continue
        fi
    else
        echo "Python virtual environment is already activated. (VIRTUAL_ENV=$VIRTUAL_ENV)"
    fi

    # Check if smartsim is already installed
    if python -c "import smartsim" >/dev/null 2>&1; then
        echo "smartsim is already installed in the virtual environment."
    else
        # Installing smartsim python package
        echo "Installing smartsim python package..."
        pip install smartsim
        runtime_changed=true
    fi

    echo "Checking SmartSim build status..."

    install_full=false

    output=$(smart info 2>/dev/null)
    install_status=$?

    if [ "$install_status" -ne 0 ]; then
        echo "command failed"
        install_full=true
    else
        echo "command succeeded"

        if echo "$output" | grep -q "Installed.*REDIS"; then
            echo "SmartSim' redis is already built and available."
        else
            echo "SmartSim is missing redis."
            install_full=true
        fi
        
        if echo "$output" | grep -q "Tensorflow.*True"; then
            echo "SmartSim' has TensorFlow support."
        elif echo "$output" | grep -q "Torch.*True"; then
            echo "SmartSim' has Torch support."
        elif echo "$output" | grep -q "ONNX.*True"; then
            echo "SmartSim' has ONNXRuntime support."
        else
            echo "SmartSim is missing a ml backend."
            install_full=true
        fi
    fi

    if [ "$install_full" = true ]; then
        echo "Cleaning previous SmartSim build..."
        #smart clean
        echo "Building full SmartSim installation"
        $smartsim_build_cmd
        runtime_changed=true

        # Verify the build
        echo "Verifying SmartSim build..."
        smart validate        

        echo "Final SmartSim installation info:"

        smart info

        if [ "$device" = "cuda-12" ]; then
            echo "Running custom command for device $device..."
            echo "Installing PyTorch for CUDA 12.4..."
            #pip uninstall torch torchvision torchaudio -y && \
            pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 && \
            echo "PyTorch installation for CUDA 12.4 complete." && \
            smart info
            runtime_changed=true
        elif [ "$device" = "cpu" ]; then
            echo "No custom command defined for device $device. Skipping."
        else
            echo "No custom command defined for device $device. Skipping."
        fi
    else
        echo "SmartSim is already fully built with all backends."
    fi

    runtime_lib_dir="$python_env/runtime_libs"
    mkdir -p "$runtime_lib_dir"
    if [ -f "$RUNTIME_LIBSTDCPP_SOURCE" ]; then
        if strings "$RUNTIME_LIBSTDCPP_SOURCE" 2>/dev/null | grep -q GLIBCXX_3.4.32; then
            :
        else
            echo "Warning: bundled runtime libstdc++ does not report GLIBCXX_3.4.32 (source: $RUNTIME_LIBSTDCPP_SOURCE)"
        fi
        if [ ! -f "$runtime_lib_dir/libstdc++.so.6" ] || ! cmp -s "$RUNTIME_LIBSTDCPP_SOURCE" "$runtime_lib_dir/libstdc++.so.6"; then
            cp -f "$RUNTIME_LIBSTDCPP_SOURCE" "$runtime_lib_dir/libstdc++.so.6"
            echo "Bundled runtime libstdc++ from GCCcore/${RUNTIME_LIBSTDCPP_GCCCORE_VERSION} into $runtime_lib_dir"
            runtime_changed=true
        else
            echo "Runtime libstdc++ already up-to-date in $runtime_lib_dir"
        fi
    else
        echo "Warning: compatible runtime libstdc++ source not found at $RUNTIME_LIBSTDCPP_SOURCE"
    fi
    if [ -f "$RUNTIME_LIBGCC_SOURCE" ]; then
        if [ ! -f "$runtime_lib_dir/libgcc_s.so.1" ] || ! cmp -s "$RUNTIME_LIBGCC_SOURCE" "$runtime_lib_dir/libgcc_s.so.1"; then
            cp -f "$RUNTIME_LIBGCC_SOURCE" "$runtime_lib_dir/libgcc_s.so.1"
            echo "Bundled runtime libgcc_s from GCCcore/${RUNTIME_LIBSTDCPP_GCCCORE_VERSION} into $runtime_lib_dir"
            runtime_changed=true
        fi
    fi

    if [ "$runtime_changed" = true ]; then
        CHANGED_RUNTIME_DEVICES="$CHANGED_RUNTIME_DEVICES $device"
    fi

done

if [ "$CREATE_RUNTIME_TARS" = "1" ]; then
    echo ""
    echo "--- Creating SmartSim runtime tar bundles ---"
    for device in $devices; do
        runtime_dir="$RUNTIME_ROOT/smartsim_$device"
        runtime_tar="$RUNTIME_TAR_DIR/smartsim_$device.tar"
        if [ -d "$runtime_dir" ]; then
            rebuild_tar=0
            if [ "$FORCE_RUNTIME_TAR_REBUILD" = "1" ]; then
                rebuild_tar=1
                echo "Forcing tar rebuild for device '$device' via SMARTSIM_FORCE_RUNTIME_TAR_REBUILD=1"
            elif [ ! -f "$runtime_tar" ]; then
                rebuild_tar=1
                echo "Creating missing runtime tar: $runtime_tar"
            elif echo " $CHANGED_RUNTIME_DEVICES " | grep -qw "$device"; then
                rebuild_tar=1
                echo "Runtime for '$device' changed during this run; rebuilding: $runtime_tar"
            elif [ "$VERIFY_RUNTIME_TAR_FRESHNESS" = "1" ] && find "$runtime_dir" -type f -newer "$runtime_tar" -print -quit | grep -q .; then
                rebuild_tar=1
                echo "Runtime changed since last tar (freshness scan enabled); rebuilding: $runtime_tar"
            fi

            if [ "$rebuild_tar" = "1" ]; then
                echo "Packing $runtime_dir -> $runtime_tar"
                tar -cf "$runtime_tar" -C "$RUNTIME_ROOT" "smartsim_$device"
            else
                echo "Runtime tar is up-to-date; skipping: $runtime_tar"
            fi
        else
            echo "Skipping tar for device '$device' (runtime dir missing: $runtime_dir)"
        fi
    done
fi

echo "Installation of the smartsim 'backend' complete. This includes the SmartRedis Python client. The Fortran/C/C++ clients can be built from source as needed."

echo ""

echo "--- Installing SmartRedis C/C++ client ---"

# This entails multiple bugfixes that have been merged into develop but that for some reason
# have not resulted in a new tagged release.
# Official Repo:
#smartsim_repo="https://github.com/CrayLabs/SmartRedis.git"
#smartsim_branch="develop"
#smartsim_commit="552d05b3a59bdd79fc119c25eb75126ebba41b14"

# My fork
smartsim_repo="https://github.com/chrisb09/SmartRedis.git"
smartsim_branch="develop"
smartsim_commit="a5e235564f5d3912eec9541ea3ebce9dd42eccf7"

compile=false

if [ -d "SmartRedis" ]; then
    echo "SmartRedis directory already exists. Skipping clone."

    # Check the existing repository state without relying on shell cwd.
    current_branch=$(git -C SmartRedis rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "$smartsim_branch" ]; then
        echo "Current branch is $current_branch, expected $smartsim_branch. Attempting to switch branches..."
        git -C SmartRedis checkout $smartsim_branch || { echo "Failed to switch to branch $smartsim_branch. Please check the SmartRedis directory."; exit 1; }
        compile=true
    else
        echo "Already on the correct branch: $current_branch"
    fi

    # Check if the existing SmartRedis directory is at the correct commit
    current_commit=$(git -C SmartRedis rev-parse HEAD)
    if [ "$current_commit" != "$smartsim_commit" ]; then
        echo "Current commit is $current_commit, expected $smartsim_commit. Attempting to reset to the correct commit..."
        git -C SmartRedis reset --hard $smartsim_commit || { echo "Failed to reset to commit $smartsim_commit. Please check the SmartRedis directory."; exit 1; }
        compile=true
    else
        echo "Already at the correct commit: $current_commit"
    fi
else
    short_commit=$(echo "$smartsim_commit" | head -c 8)
    echo "Cloning SmartRedis repository (${smartsim_branch} branch, commit ${short_commit})..."

    git clone "${smartsim_repo}" && \
    cd SmartRedis && \
    git checkout "${smartsim_commit}" && \
    cd .. || { echo "Failed to clone and checkout SmartRedis."; exit 1; }
    compile=true
fi

installed_lib="SmartRedis/install/lib64/libsmartredis.so"
if [ ! -f "$installed_lib" ]; then
    echo "SmartRedis installed library is missing. A rebuild is required."
    compile=true
elif find SmartRedis/include SmartRedis/src SmartRedis/CMakeLists.txt -newer "$installed_lib" -print -quit | grep -q .; then
    echo "SmartRedis installed library is older than the source tree. A rebuild is required."
    compile=true
fi

if [ "$compile" = true ]; then

    cd SmartRedis
# Debug: Print current environment and compiler info
    echo ""
    echo "=== SmartRedis Build Environment ==="
    echo "Current GCC version:"
    gcc --version | head -n 1
    echo "CC=$CC"
    echo "CXX=$CXX"
    echo "LD=$LD"
    echo "Loaded modules:"
    module -t list 
    echo "=================================="
    echo ""

    make lib -j 4
    cd ..
fi


export smartredis_DIR=${PWD}/SmartRedis/cmake

echo "Current directory: $(pwd)"

. ./env.sh

#mkdir -p build && \
#cd build && \
#cmake .. -DCMAKE_INSTALL_PREFIX="${base_dir}/SmartRedis/install" -DBUILD_FORTRAN=OFF #-DBUILD_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DPEDANTIC=OFF #-DRETAIN_RPATH=ON && \
#make -j 4 && \
#make install
