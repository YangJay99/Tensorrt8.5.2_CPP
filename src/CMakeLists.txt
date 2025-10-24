cmake_minimum_required(VERSION 3.10)
project(trt_inference_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find TensorRT and CUDA (adjust HINTS/paths on your system)
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_library(CUDA_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

message("TENSORRT_LIB: ${TENSORRT_LIB}")
message("CUDA_LIB: ${CUDA_LIB}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(/usr/include/aarch64-linux-gnu)
include_directories(/usr/local/cuda-11.4/targets/aarch64-linux/include)

# Source files
set(SRC_FILES
    tensorrt_inference.cpp
    utils.cpp
)

# Build shared library
add_library(trt_inference SHARED ${SRC_FILES})


# 链接依赖库
target_link_libraries(trt_inference
    ${TENSORRT_LIB}
    ${CUDA_LIB}
)
