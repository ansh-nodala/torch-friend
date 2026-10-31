cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_friend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CUDA "Enable CUDA acceleration for NVIDIA GPUs" OFF)

if(MSVC)
    string(REGEX REPLACE "/RTC(su|[1s])" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

# 1. Find Torch and Python3
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)

# Locate libtorch_python.so
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

set(TORCH_TOOLS_SOURCES src/rope.cpp)

# 2. Conditional CUDA Support
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
        list(APPEND TORCH_TOOLS_SOURCES src/rope_cuda.cu)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA enabled and compiler found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(FATAL_ERROR "USE_CUDA=ON was requested, but no CUDA compiler (nvcc) was found!")
    endif()
else()
    message(STATUS "Building for CPU-only execution.")
endif()

# 3. Core Static Library
add_library(torch_tools STATIC ${TORCH_TOOLS_SOURCES})
set_target_properties(torch_tools PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(torch_tools PRIVATE ${TORCH_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(torch_tools PRIVATE OpenMP::OpenMP_CXX)
endif()

# 4. Python Shared Extension Module
add_library(torch_tools_cpp MODULE binding/bind.cpp)
target_compile_definitions(torch_tools_cpp PRIVATE TORCH_EXTENSION_NAME=torch_tools_cpp)

target_link_libraries(torch_tools_cpp PRIVATE
    torch_tools
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(torch_tools_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(NOT DEFINED PYTHON_EXT_SUFFIX)
    set(PYTHON_EXT_SUFFIX ".so")
endif()

set_target_properties(torch_tools_cpp PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../torch/lib;${TORCH_INSTALL_PREFIX}/lib"
)
