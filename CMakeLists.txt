cmake_minimum_required(VERSION 3.18)
project(bill_accelerator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Set up Position Independent Code for Python extension
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch Pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Add source files
set(SOURCES
    src/image_core.cu
    src/bindings.cpp
)

# Create Python extension module
pybind11_add_module(bill_cuda ${SOURCES})

# Include directories
target_include_directories(bill_cuda PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# CUDA specific flags
target_compile_options(bill_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
