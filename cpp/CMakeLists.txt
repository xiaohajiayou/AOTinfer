cmake_minimum_required(VERSION 3.18)
project(gpu_hello LANGUAGES CXX)

# TorchConfig 会尝试启用 CUDA，如果默认路径没有 nvcc 会报错。
# 明确告诉 CMake 使用当前环境里的 nvcc。
if (NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CUDA_NVCC_EXECUTABLE nvcc)
    if (CUDA_NVCC_EXECUTABLE)
        message(STATUS "Found nvcc at ${CUDA_NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE} CACHE FILEPATH "nvcc path" FORCE)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

add_executable(gpu_hello gpu_hello.cpp)
target_link_libraries(gpu_hello "${TORCH_LIBRARIES}")
set_property(TARGET gpu_hello PROPERTY CXX_STANDARD 17)
