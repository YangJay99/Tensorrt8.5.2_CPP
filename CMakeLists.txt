cmake_minimum_required(VERSION 3.10)
project(trt_infer_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add include path
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-11.4/targets/aarch64-linux/include)

# build the aiisp library (subdir)
add_subdirectory(src)

# main executable
add_executable(trt_infer_app main.cpp)

# link against aiisp_inference shared lib and needed system libs
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_library(CUDA_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

# link to our shared lib
target_link_libraries(trt_infer_app PRIVATE trt_inference ${TENSORRT_LIB} ${CUDA_LIB})

# install rules (optional)
install(TARGETS trt_infer_app DESTINATION bin)
install(TARGETS trt_inference DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
