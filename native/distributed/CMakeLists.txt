cmake_minimum_required(VERSION 3.10)
project(graph)


find_package(JNI REQUIRED)

include_directories(${JAVA_INCLUDE_PATH})
include_directories(${JAVA_INCLUDE_PATH}/linux)
include_directories("include")
include_directories("../single/include")
include_directories(${TORCH_HOME}/include/torch/csrc/api/include)
include_directories(${TORCH_HOME}/include)

set(HEADER include/angel/graph/interface/graphsage_jni.h
        include/angel/graph/interface/supervised_graphsage_jni.h
        include/angel/commons.h
        include/angel/graph/model/graphsage.h
        include/angel/graph/data/subgraph.h
        include/angel/graph/layer/mean.h)

set(SRC src/angel/graph/model/graphsage.cc
        src/angel/graph/layer/mean.cc
        src/angel/graph/interface/graphsage_jni.cc
        src/angel/graph/interface/supervised_graphsage_jni.cc
        src/angel/graph/data/subgraph.cc)

set(PYTORCH_HEADER include/angel/pytorch/lr.h
        include/angel/pytorch/lr_jni.h
        include/angel/commons.h)

set(PYTORCH_SRC src/angel/pytorch/lr.cc
        src/angel/pytorch/lr_jni.cc)

add_library(torch_graph SHARED ${SRC} ${HEADER})
target_link_libraries(torch_graph ${TORCH_LIBRARIES})

add_library(torch_angel SHARED ${PYTORCH_SRC} ${PYTORCH_HEADER})
target_link_libraries(torch_angel ${TORCH_LIBRARIES})