# Project Status: SmartSim CPP-ML-Interface Build

## Goals
- Successfully build the `CPP-ML-Interface` with support for both SmartSim (SmartRedis) and AIxelerator (LibTorch).
- Ensure binary compatibility between all components (ABI alignment).
- Enable CUDA/GPU support for inference on HPC nodes.
- Maintain a clean and reproducible build environment using the cluster's module system.

## Progress & Successes
- **ABI Mismatch Resolved**: Identified that PyTorch wheels (ABI 0) were conflicting with SmartRedis (ABI 1). Fixed by configuring `AIxeleratorService` to download and use the standalone **LibTorch (cxx11-abi)** version 2.4.0.
- **Python Environment Isolation**: Resolved `ImportError` and `ModuleNotFoundError` issues by explicitly setting `Python3_EXECUTABLE` to the user's Python 3.9 environment and removing global `PYTHONPATH` pollution.
- **CUPTI Linking**: Fixed missing `cuptiActivityEnableDriverApi` symbols by prioritizing the full CUDA Toolkit paths on `/cvmfs` over the minimal libraries bundled in Python wheels.
- **Core Build Success**: The main library (`libcpp_ml_interface_library.so`) and the executable (`cpp_ml_interface_executable`) now compile and link successfully with `WITH_AIX=ON`.

## Challenges & Insights
### 1. The ABI "Tug of War"
- **Insight**: PyTorch's `find_package(Torch)` aggressively sets `_GLIBCXX_USE_CXX11_ABI=0`. This "pollutes" the entire CMake project, making it impossible to link against modern libraries like SmartRedis (ABI 1).
- **Solution**: Never link high-performance C++ apps against Python site-packages `torch` if you need ABI 1. Always use the LibTorch `cxx11-abi` distribution.

### 2. CUDA Library Priority
- **Insight**: Python wheels (e.g., `nvidia-cuda-cupti-cu12`) often contain "stripped" shared libraries that lack specific versioned symbols required by PyTorch's Kineto profiler.
- **Solution**: Prepend the System/CVMFS CUDA `extras/CUPTI/lib64` path to `LD_LIBRARY_PATH` and `LIBRARY_PATH` to ensure the "complete" versions are found first.

### 3. MPI recommendation logic
- **Insight**: PyTorch includes logic to recommend the MPI implementation it was built with (e.g., OpenMPI). This can cause CMake to switch MPI vendors mid-configuration if not careful.

## Current Hurdles
- **MPI Symbol Errors**: Some AIxelerator internal tests (`testAIxeleratorLib.x`, `testAIxeleratorService_interfaceC.x`) are currently failing to link with `undefined reference to ompi_mpi_comm_world`. This indicates a mismatch between the MPI headers used during compilation and the MPI libraries found during linking.
- **Module Test Build**: We are transitioning to building the `module_test` subdirectory, which requires a "merged" registry generation.

## Future Work
- Align the MPI environment to ensure AIxelerator tests link correctly.
- Complete the `module_test` build.
- Verify GPU inference performance on a compute node.
