cmake_minimum_required(VERSION 3.16)
project(batch_memory_benchmark LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

set(CPPML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CPP-ML-Interface")
set(SMARTREDIS_DIR "${CPPML_DIR}/extern/SmartRedis")
set(AIX_DIR "${CPPML_DIR}/extern/AIxeleratorService")
set(SMARTSIM_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../python/smartsim_cuda-12/bin/python")

if(NOT EXISTS "${SMARTSIM_PYTHON}")
    set(SMARTSIM_PYTHON "python3")
endif()

option(WITH_PHYDLL "Enable PhyDLL provider support through CPP-ML-Interface" ON)
set(CPPML_DISABLE_REGISTRY_GEN ON CACHE BOOL "Disable CPP-ML-Interface standalone registry generation" FORCE)

add_subdirectory("${CPPML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build" EXCLUDE_FROM_ALL)

set(CPPML_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.hpp")
set(CPPML_LIBRARY_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build/generated_registry.hpp")
set(TEMP_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.tmp.hpp")

file(GLOB_RECURSE CPPML_HEADERS "${CPPML_DIR}/include/**/*.hpp")
file(GLOB_RECURSE BENCHMARK_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
set(ALL_HEADERS ${CPPML_HEADERS} ${BENCHMARK_HEADERS})
list(REMOVE_DUPLICATES ALL_HEADERS)

set(BASE_CLASSES "MLCouplingLibrary,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication")
set(CLANG_ARGS "-std=c++17 -I${CPPML_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${SMARTREDIS_DIR}/include -I${SMARTREDIS_DIR}/install/include -xc++")

add_custom_command(
    OUTPUT "${CPPML_GENERATED_REGISTRY}" "${CPPML_LIBRARY_GENERATED_REGISTRY}"
    COMMAND ${CMAKE_COMMAND} -E env "CLANG_ARGS=${CLANG_ARGS}"
            "${SMARTSIM_PYTHON}" "${CPPML_DIR}/scripts/generate_registry.py"
            "${TEMP_GENERATED_REGISTRY}" "${BASE_CLASSES}" ${ALL_HEADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${CPPML_GENERATED_REGISTRY}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${CPPML_LIBRARY_GENERATED_REGISTRY}"
    DEPENDS ${ALL_HEADERS} "${CPPML_DIR}/scripts/generate_registry.py"
    WORKING_DIRECTORY "${CPPML_DIR}"
    COMMENT "Generating registry for batch_memory_benchmark..."
)

add_custom_target(benchmark_registry DEPENDS "${CPPML_GENERATED_REGISTRY}" "${CPPML_LIBRARY_GENERATED_REGISTRY}")

add_executable(fake_solver src/fake_solver.cpp)
add_dependencies(fake_solver benchmark_registry)
add_dependencies(cpp_ml_interface_library benchmark_registry)

target_include_directories(cpp_ml_interface_library PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(fake_solver PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CPPML_DIR}/include"
    "${SMARTREDIS_DIR}/include"
    "${SMARTREDIS_DIR}/install/include"
)

target_link_directories(fake_solver PRIVATE
    "${SMARTREDIS_DIR}/install/lib64"
    "${SMARTREDIS_DIR}/install/lib"
)

target_link_libraries(fake_solver PRIVATE
    MPI::MPI_CXX
    smartredis
    cpp_ml_interface_headers
    cpp_ml_interface_library
)
