cmake_minimum_required(VERSION 3.14)

project(PhydllTests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# Specify the CPP-ML-Interface source directory
set(CPP_ML_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CPP-ML-Interface")
set(SMARTREDIS_DIR "${CPP_ML_INTERFACE_DIR}/extern/SmartRedis")
set(CUSTOM_CORE_DEPENDENCIES_DIR "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software")

# Enable WITH_PHYDLL for the subproject build
set(WITH_PHYDLL ON CACHE BOOL "Enable PhyDLL support" FORCE)

# Correctly import CPP-ML-Interface as a sub-build
add_subdirectory("${CPP_ML_INTERFACE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build" EXCLUDE_FROM_ALL)

# Add our executable
add_executable(test_phydll_physics test_phydll_physics.cpp)

# Include directories for SmartRedis
target_include_directories(test_phydll_physics PRIVATE 
    ${SMARTREDIS_DIR}/include 
    ${SMARTREDIS_DIR}/install/include
)

# Link directories for SmartRedis and compiler libraries
target_link_directories(test_phydll_physics PRIVATE 
    ${SMARTREDIS_DIR}/install/lib64
    ${CUSTOM_CORE_DEPENDENCIES_DIR}/imkl/2024.2.0/compiler/2024.2/lib
    ${CUSTOM_CORE_DEPENDENCIES_DIR}/GCCcore/11.3.0/lib64
)

# Link against the imported headers, library, and dependencies
target_link_libraries(test_phydll_physics PRIVATE 
    MPI::MPI_CXX
    cpp_ml_interface_headers
    cpp_ml_interface_library
    smartredis
)

target_compile_definitions(test_phydll_physics PRIVATE WITH_PHYDLL)
