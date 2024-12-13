# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(FluidSim CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add options to enable/disable LERP, DIFFUSE, SCALE, and SIMD
option(LERP "Interpolate between cell indexes" ON)
option(DIFFUSE "Viscous diffusion" ON)
option(SCALE "Scale the velocity from 0 to 255 when displaying" ON)
option(SIMD "Use SIMD instructions if available" ON)

# Source file
set(SOURCE_FILE fluid.cpp)

# Compiler flags
add_compile_options(
    -O3
    -funroll-loops
    -D_REENTRANT
)

# Add SIMD-related flags only if SIMD option is enabled
if(SIMD)
    add_compile_options(
        -mavx
        -march=native
    )
endif()

# Include directories
include_directories(/usr/include/SDL2)

# Library dependencies
set(LIBS SDL2 SDL2_image m)

# Preprocessor definitions for LERP, SCALE, and DIFFUSE
if(LERP)
    add_compile_definitions(LERP)
endif()

if(SCALE)
    add_compile_definitions(SCALE)
endif()

if(DIFFUSE)
    add_compile_definitions(DIFFUSE)
endif()

# Add the executable target
add_executable(fluid ${SOURCE_FILE})

# Link libraries
target_link_libraries(fluid ${LIBS})
