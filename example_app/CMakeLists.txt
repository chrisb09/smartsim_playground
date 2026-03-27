# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(DummySolver)


add_executable(dummy_solver dummy_solver.cpp)

# Find HDF5 library
find_package(HDF5 REQUIRED)



# Manually set SmartRedis include and library paths
target_include_directories(dummy_solver PRIVATE /hpcwork/ro092286/smartsim/SmartRedis/include)
target_include_directories(dummy_solver PRIVATE /hpcwork/ro092286/smartsim/SmartRedis/install/include)
target_link_directories(dummy_solver PRIVATE /hpcwork/ro092286/smartsim/SmartRedis/install/lib64)

# Add Intel compiler libraries path for SmartRedis dependencies
target_link_directories(dummy_solver PRIVATE /cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/imkl/2024.2.0/compiler/2024.2/lib)

# Link against GCC's libstdc++
target_link_directories(dummy_solver PRIVATE /cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/GCCcore/11.3.0/lib64)

target_link_libraries(dummy_solver PRIVATE smartredis ${HDF5_LIBRARIES} stdc++)
