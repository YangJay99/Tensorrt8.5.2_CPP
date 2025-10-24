#include "tensorrt_inference.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>

using namespace nvinfer1;

#define CUDA_CHECK(status) \
    do { cudaError_t err = status; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; return false; } } while(0)

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kVERBOSE)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;


inline static const char* dataTypeToString(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kHALF:  return "FP16";
        case nvinfer1::DataType::kINT8:  return "INT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kBOOL:  return "BOOL";
        case nvinfer1::DataType::kUINT8: return "UINT8"; 
        default: return "UNKNOWN";
    }
}

TRTInfer::TRTInfer() {
}

TRTInfer::~TRTInfer() {
    destroy();
}

#define CUDA_CHECK_RET(status, rc) \
    do { cudaError_t err = status; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        return (rc); } } while(0)

// helper to get bytes per element for DataType
static inline size_t dtype_size_bytes(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 1;
    }
}

int TRTInfer::init(const std::string& engineFile) {
    // clear any prior state
    allocated_ptrs_.clear();
    runtime_ = createInferRuntime(gLogger);
    if (!runtime_) {
        std::cerr << "Failed to create runtime\n";
        return 1001;
    }

    std::ifstream ifs(engineFile, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open engine file: " << engineFile << std::endl;
        return 1002;
    }
    ifs.seekg(0, std::ios::end);
    size_t fsize = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<char> data(fsize);
    ifs.read(data.data(), fsize);

    engine_ = runtime_->deserializeCudaEngine(data.data(), fsize);
    if (!engine_) {
        std::cerr << "Failed to deserialize engine\n";
        return 1003;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context\n";
        return 1004;
    }

    CUDA_CHECK_RET(cudaStreamCreate(&stream_), 1005);

    printf("------------------ Tensorrt model Input/Output Tensor Info ------------------ \n");
    int nbIOTensors = engine_->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (name == nullptr) {
            fprintf(stderr, "Warning: Tensor index %d has no name\n", i);
            continue;
        }

        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        const char* modeStr = (mode == nvinfer1::TensorIOMode::kINPUT) ? "Input" : "Output";

        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::DataType dtype = engine_->getTensorDataType(name);

        // compute element count and bytes
        int64_t elem_count = 1;
        bool dynamic_dim = false;
        for (int d = 0; d < dims.nbDims; ++d) {
            int dim = dims.d[d];
            if (dim <= 0) { // -1 or 0 indicate dynamic/unknown
                dynamic_dim = true;
            } else {
                elem_count *= static_cast<int64_t>(dim);
            }
        }

        size_t bytes = 0;
        if (dynamic_dim) {
            // can't allocate fixed size for dynamic dims; user must set binding dims before enqueue
            std::cerr << "Warning: tensor " << name << " has dynamic/unknown dims. You must set binding dimensions before enqueue.\n";
            // as fallback allocate a small buffer to avoid null pointer but inform user
            elem_count = std::max<int64_t>(elem_count, 1);
        }
        bytes = static_cast<size_t>(elem_count) * dtype_size_bytes(dtype);

        // Logging
        std::cout << "Name: "<< name << "(" << modeStr << ")" << "\nDataType: "<< dataTypeToString(dtype) << "\nDims:  [";
        for (int d = 0; d < dims.nbDims; ++d) {
            if (d) printf(",");
            printf("%d", dims.d[d]);
        }
        printf("]\n");

        // allocate device memory of correct byte size
        void *data_dev = nullptr;
        CUDA_CHECK_RET(cudaMalloc(&data_dev, bytes), 1006);
        allocated_ptrs_.push_back(data_dev);

        printf("Tensor %s allocated at %p, size: %zu bytes (%lld elements x %zu bytes/elem)\n",
               name, data_dev, bytes, (long long)elem_count, dtype_size_bytes(dtype));

        // set tensor address (if dynamic dims exist you still need to set binding dims before enqueue)
        if (!context_->setTensorAddress(name, data_dev)) {
            std::cerr << "Warning: setTensorAddress failed for " << name << std::endl;
            // continue - but user should investigate
        }
    }
    printf("------------------------------------------------------------------------------------------ \n");
    std::cout << "Host memory mapped successfully. Zero-copy ready." << std::endl;
    return 0;
}

int TRTInfer::infer() {
    if (!context_) {
        std::cerr << "Invalid context\n";
        return 2001;
    }

    // if engine uses dynamic shapes, ensure shapes are set:
    // e.g. context_->setBindingDimensions(bindIdx, dims)  before enqueueV3

    if (!context_->enqueueV3(stream_)) {
        std::cerr << "enqueueV3 failed\n";
        return 2002;
    }
    // optional: synchronize to measure time outside
    CUDA_CHECK_RET(cudaStreamSynchronize(stream_), 2003);
    return 0;
}

cudaStream_t TRTInfer::get_stream(){
    return stream_;
}

void TRTInfer::destroy() {
    // free allocated device buffers
    for (void* p : allocated_ptrs_) {
        if (p) {
            cudaFree(p);
        }
    }
    allocated_ptrs_.clear();

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_)  { engine_->destroy(); engine_ = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
}
