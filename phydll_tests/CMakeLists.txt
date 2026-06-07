cmake_minimum_required(VERSION 3.14)

project(PhydllTests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# Need to point to CPP-ML-Interface and Phydll libraries
set(CPP_ML_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CPP-ML-Interface")
include_directories(${CPP_ML_INTERFACE_DIR}/include)
include_directories(${CPP_ML_INTERFACE_DIR}/extern/phydll/src/core)

add_executable(test_phydll_physics test_phydll_physics.cpp)
target_link_libraries(test_phydll_physics PRIVATE MPI::MPI_CXX)
# Wait, we need to link the actual cpp_ml_interface library and phydll.
# We can link them directly from the build folder if they are built
target_link_directories(test_phydll_physics PRIVATE ${CPP_ML_INTERFACE_DIR}/extern/phydll/build/lib)
target_link_libraries(test_phydll_physics PRIVATE ${CPP_ML_INTERFACE_DIR}/build/libcpp_ml_interface_library.so phydll)
target_compile_definitions(test_phydll_physics PRIVATE WITH_PHYDLL)
