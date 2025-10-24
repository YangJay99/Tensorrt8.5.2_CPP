#include "utils.h"
#include <fstream>
#include <stdexcept>

namespace utils {

void readRawFileToBuffer(const std::string& filename, std::vector<uint16_t>& buffer) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open raw file: " + filename);
    size_t bytes = buffer.size() * sizeof(uint16_t);
    file.read(reinterpret_cast<char*>(buffer.data()), bytes);
    if (!file) throw std::runtime_error("Failed to read expected bytes from: " + filename);
}

void saveRawFile(const std::string& filename, const std::vector<uint16_t>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write file: " + filename);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint16_t));
    if (!file) throw std::runtime_error("Failed to write file: " + filename);
}

void saveBinFile(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write file: " + filename);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    if (!file) throw std::runtime_error("Failed to write file: " + filename);
}

} // namespace utils
