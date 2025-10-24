#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

#include "tensorrt_inference.h"
#include "utils.h"
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <engine_file>\n";
        return -1;
    }

    std::string enginePath = argv[1];


    TRTInfer trt_infer;
    if (trt_infer.init(enginePath.c_str())) {
        std::cerr << "init failed\n";
        return -1;
    }
    cudaStream_t stream = trt_infer.get_stream();
    for (size_t i = 0; i < 20; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        int ret = trt_infer.infer();
        cudaStreamSynchronize(stream);
        if (ret != 0) {
            std::cerr << "inference failed  " << i << "\n";
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "inference time: " << duration.count() << " ms\n";
    }

    printf("warmup done\n");
    float total_infer_time = 0.0f;
    int valid_count = 100;
    for (size_t i = 0; i < valid_count; ++i) {
        float infer_time = 0.0f;

        auto start = std::chrono::high_resolution_clock::now();
        int ret = trt_infer.infer();
        cudaStreamSynchronize(stream);
        if (ret != 0) {
            std::cerr << "inference failed  " << i << "\n";
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "inference time: " << duration.count() << " ms\n";
        total_infer_time += duration.count();
    }
    std::cout << "Average inference time: " << total_infer_time / valid_count << " ms\n";
    


    trt_infer.destroy();
    std::cout << "All done.\n";
    return 0;
}
