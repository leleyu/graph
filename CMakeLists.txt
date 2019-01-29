cmake_minimum_required(VERSION 3.10)
project(graph)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/Users/leleyu/software/libtorch")

find_package(Torch REQUIRED)

include_directories("include/")

set(HEADER include/graph/dataset.h
        include/graph/model.h
        include/graph/layer.h
        include/graph/metric.h
        include/graph/graphsage.h
        include/graph/sampler.h
        include/graph/graph.h)

set(SRC src/dataset.cpp
        src/layer.cpp
        src/graphsage.cpp
        src/metric.cpp
        src/sampler.cpp
        src/graph.cc)

add_executable(supervised src/supervised.cpp ${HEADER} ${SRC})

add_executable(unsupervised src/unsupervised.cpp ${HEADER} ${SRC})
add_executable(mlp src/mlp.cpp ${HEADER} ${SRC})

target_link_libraries(supervised ${TORCH_LIBRARIES})

target_link_libraries(unsupervised ${TORCH_LIBRARIES})
target_link_libraries(mlp ${TORCH_LIBRARIES})
