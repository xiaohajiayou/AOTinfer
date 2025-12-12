cmake_minimum_required(VERSION 3.18)
project(aoti_greedy_demo LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}../../../thirdparty/libtorch")
find_package(Torch REQUIRED)

add_executable(aoti_greedy_demo 
                main.cpp
                model_runner.cpp
)
target_link_libraries(aoti_greedy_demo "${TORCH_LIBRARIES}")
set_property(TARGET aoti_greedy_demo PROPERTY CXX_STANDARD 17)