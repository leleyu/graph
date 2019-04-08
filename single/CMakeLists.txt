cmake_minimum_required(VERSION 3.10)
project(graph)

include_directories("include")
set(HEADER include/graph/dataset.h
        include/graph/graphsage.h
        include/graph/graph.h
        include/graph/layer.h
        include/graph/metric.h
        include/graph/sampler.h)

set(SRC src/dataset.cpp
        src/graphsage.cc
        src/graph.cc
        src/layer.cc
        src/metric.cc
        src/sampler.cc)

add_executable(supervised src/supervised.cc ${HEADER} ${SRC})
add_executable(unsupervised src/unsupervised.cc ${HEADER} ${SRC})

target_link_libraries(supervised ${TORCH_LIBRARIES})
target_link_libraries(unsupervised ${TORCH_LIBRARIES})