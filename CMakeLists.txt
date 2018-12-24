cmake_minimum_required(VERSION 3.12)
project(test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/home/leleyu/software/libtorch")

find_package(Torch REQUIRED)

add_executable(graph main.cpp)

target_link_libraries(graph ${TORCH_LIBRARIES})
