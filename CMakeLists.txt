# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(FluidSim CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add options to enable/disable LERP and SCALE
option(LERP "interpolate between cell indexes" ON)
option(DIFFUSE "Viscuous diffusion" ON)
option(SCALE "scale the velocity from 0 to 255 when displaying" ON)

# Source file
set(SOURCE_FILE fluid.cpp)

# Compiler flags
add_compile_options(
    -O3
    -mavx
    -march=native
    -funroll-loops
    -D_REENTRANT
)

# Include directories
include_directories(/usr/include/SDL2)

# Library dependencies
set(LIBS SDL2 SDL2_image m)

# Preprocessor definitions for LERP and SCALE
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
