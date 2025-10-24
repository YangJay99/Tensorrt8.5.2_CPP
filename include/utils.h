#pragma once
#include <string>
#include <vector>
#include <cstdint>

#define CUDA_CHECK(status) \
    do { cudaError_t err = status; if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; return false; } } while(0)
        
namespace utils {

// read raw into provided vector (size must match)
void readRawFileToBuffer(const std::string& filename, std::vector<uint16_t>& buffer);

// save raw vector<uint16_t> to file
void saveRawFile(const std::string& filename, const std::vector<uint16_t>& data);

// save float buffer to binary
void saveBinFile(const std::string& filename, const std::vector<float>& data);

// clamp helper
template<typename T>
T clamp_val(T v, T lo, T hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

} // namespace utils
