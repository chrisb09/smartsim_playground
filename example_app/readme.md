This is supposed to be a crude self-made example/hello world application to check if smartsim via c++ api is working and for me to figure out how to use it.

### From the smartsim install readme:

```
Most applications should be able to incorporate the following into their CMakeLists.txt to

include(ExternalProject)
ExternalProject_Add(smartredis
    GIT_REPOSITORY https://github.com/CrayLabs/SmartRedis.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/external
               -DBUILD_FORTRAN=on # For Fortran applications
    PREFIX ${CMAKE_BINARY_DIR}/external
)
ExternalProject_Get_Property(smartredis binary_dir source_dir)

add_library(libsmartredis SHARED IMPORTED)
add_dependencies(libsmartredis smartredis)
set_target_properties(libsmartredis PROPERTIES
    IMPORTED_LOCATION ${binary_dir}/libsmartredis.so
    INTERFACE_INCLUDE_DIRECTORIES  $<INSTALL_INTERFACE:${CMAKE_BINARY_DIR}/external/include}>
)

# Optional, only for Fortran applications
add_library(libsmartredis-fortran SHARED IMPORTED)
add_dependencies(libsmartredis-fortran smartredis)
set_target_properties(libsmartredis-fortran PROPERTIES
    IMPORTED_LOCATION ${binary_dir}/libsmartredis-fortran.so
    INTERFACE_INCLUDE_DIRECTORIES  $<INSTALL_INTERFACE:${CMAKE_BINARY_DIR}/external/include}>
)

# ... define the example_target executable here

target_include_directories(example_target PRIVATE ${CMAKE_BINARY_DIR}/external/include)
target_link_libraries(example_target libsmartredis)
# Optional, only for Fortran applcations
target_link_libraries(example_target libsmartredis-fortran)
```