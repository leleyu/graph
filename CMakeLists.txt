cmake_minimum_required(VERSION 3.10)
project(graph)

set(CMAKE_CXX_STANDARD 14)
set(TORCH_HOME "/Users/leleyu/software/libtorch")
set(CMAKE_PREFIX_PATH ${TORCH_HOME})

find_package(Torch REQUIRED)

add_subdirectory(single)
add_subdirectory(distributed)