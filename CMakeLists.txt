cmake_minimum_required(VERSION 3.10)
project(graph)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/Users/leleyu/software/libtorch")

find_package(Torch REQUIRED)

include_directories("include/")

set(HEADER include/graph/dataset.h
        include/graph/model.h
        include/graph/mean.h
        include/graph/graphsage.h)

set(SRC src/dataset.cpp src/mean.cpp src/graphsage.cpp)

add_executable(graph src/main.cpp ${HEADER} ${SRC} )
add_executable(test test.cpp ${HEADER} ${SRC})

target_link_libraries(graph ${TORCH_LIBRARIES})
target_link_libraries(test ${TORCH_LIBRARIES})
