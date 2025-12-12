cmake_minimum_required(VERSION 3.18)
project(gpu_hello LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}../../../thirdparty/libtorch")
find_package(Torch REQUIRED)

add_executable(gpu_hello gpu_hello.cpp)
target_link_libraries(gpu_hello "${TORCH_LIBRARIES}")
set_property(TARGET gpu_hello PROPERTY CXX_STANDARD 17)
