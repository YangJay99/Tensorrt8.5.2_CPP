#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

class TRTInfer {
public:
    TRTInfer();
    ~TRTInfer();

    // 初始化 engine（engineFile 为已序列化的 .engine 路径）
    // 同时注册输入输出内存地址到 GPU
    int init(const std::string& engineFile);

    // 推理接口：无需再拷贝数据
    int infer();
    
    // 获取 CUDA stream
    cudaStream_t get_stream();

    // 释放资源
    void destroy();

private:
    IRuntime* runtime_ = nullptr;
    IExecutionContext* context_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    cudaStream_t stream_ = nullptr;

    void* fusion_dev = nullptr;
    size_t singleBytes_ = 0;
    std::vector<void*> allocated_ptrs_;
};
