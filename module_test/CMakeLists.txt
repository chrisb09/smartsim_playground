cmake_minimum_required(VERSION 3.16)
project(module_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# check if DEBUG is set in the environment, and if so, add -DDEBUG to the compile definitions
if(DEFINED ENV{DEBUG} AND NOT "$ENV{DEBUG}" STREQUAL "")
	message(STATUS "DEBUG environment variable is set, enabling debug mode")
	add_compile_definitions(DEBUG_MODE)
else()
	message(STATUS "DEBUG environment variable is not set, building in release mode")
endif()

set(CPP_MODULE_DIR "/hpcwork/ro092286/smartsim/CPP-ML-Interface")
set(SMARTREDIS_DIR "/hpcwork/ro092286/smartsim/SmartRedis")
set(SMARTSIM_PYTHON "/hpcwork/ro092286/smartsim/python/smartsim_cuda-12/bin/python"
	CACHE FILEPATH "Python interpreter with clang module available")

if(NOT EXISTS "${SMARTSIM_PYTHON}")
	message(FATAL_ERROR "SMARTSIM_PYTHON not found: ${SMARTSIM_PYTHON}")
endif()

set(GENERATED_REGISTRY "${CPP_MODULE_DIR}/include/generated_registry.hpp")
set(TEMP_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.tmp.hpp")

file(GLOB_RECURSE CPP_MODULE_HEADERS
	"${CPP_MODULE_DIR}/include/application/*.hpp"
	"${CPP_MODULE_DIR}/include/behavior/*.hpp"
	"${CPP_MODULE_DIR}/include/normalization/*.hpp"
	"${CPP_MODULE_DIR}/include/provider/*.hpp"
)
file(GLOB_RECURSE MODULE_TEST_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
set(ALL_HEADERS ${CPP_MODULE_HEADERS} ${MODULE_TEST_HEADERS})
list(REMOVE_DUPLICATES ALL_HEADERS)

set(BASE_CLASSES "MLCouplingProvider,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication")
set(CLANG_ARGS "-std=c++17 -I${CPP_MODULE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${SMARTREDIS_DIR}/include -I${SMARTREDIS_DIR}/install/include -I${CPP_MODULE_DIR}/build/_deps/tomlplusplus-src/include -xc++")

add_custom_command(
	OUTPUT "${GENERATED_REGISTRY}"
	COMMAND ${CMAKE_COMMAND} -E env "CLANG_ARGS=${CLANG_ARGS}"
			"${SMARTSIM_PYTHON}" "${CPP_MODULE_DIR}/scripts/generate_registry.py"
			"${TEMP_GENERATED_REGISTRY}" "${BASE_CLASSES}" ${ALL_HEADERS}
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${GENERATED_REGISTRY}"
	DEPENDS ${ALL_HEADERS} "${CPP_MODULE_DIR}/scripts/generate_registry.py"
	WORKING_DIRECTORY "${CPP_MODULE_DIR}"
	COMMENT "Generating merged registry (CPP module + module_test headers)..."
)

add_custom_target(module_test_registry DEPENDS "${GENERATED_REGISTRY}")

add_executable(module_test_solver solver.cpp)
add_dependencies(module_test_solver module_test_registry)

target_include_directories(module_test_solver PRIVATE
	"${CPP_MODULE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${SMARTREDIS_DIR}/include"
	"${SMARTREDIS_DIR}/install/include"
	"${CPP_MODULE_DIR}/build/_deps/tomlplusplus-src/include"
)

target_compile_definitions(module_test_solver PRIVATE SMARTSIM_ENABLED WITH_SMARTSIM)

target_link_directories(module_test_solver PRIVATE "${SMARTREDIS_DIR}/install/lib64")
target_link_libraries(module_test_solver PRIVATE smartredis)
