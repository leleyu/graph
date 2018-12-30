cmake_minimum_required(VERSION 3.12)
project(test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/Users/leleyu/software/libtorch")

find_package(Torch REQUIRED)

include_directories("include/")

add_executable(graph main.cpp include/graph/utils.h utils.cpp include/graph/libsvm.h include/graph/model.h)
add_executable(test test.cpp include/graph/model.h)

target_link_libraries(graph ${TORCH_LIBRARIES})
target_link_libraries(test ${TORCH_LIBRARIES})
