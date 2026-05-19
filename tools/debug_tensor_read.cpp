#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// Debug exact tensor info reading as minllama does
int main(int argc, char** argv) {
    std::cout << "=== minllama Tensor Reading Debug ===" << std::endl;
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::cout << "Debugging tensor reading: " << model_path << std::endl;
    
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        std::cout << "❌ Cannot open file" << std::endl;
        return 1;
    }
    
    // Read header
    std::array<unsigned char, 24> header = {};
    if (!file.read(reinterpret_cast<char*>(header.data()), 24)) {
        std::cout << "❌ Failed to read header" << std::endl;
        return 1;
    }
    
    // Parse header
    std::string magic(header.begin(), header.begin() + 4);
    uint32_t version = *reinterpret_cast<uint32_t*>(header.data() + 4);
    uint64_t n_tensors = *reinterpret_cast<uint64_t*>(header.data() + 8);
    uint64_t n_kv = *reinterpret_cast<uint64_t*>(header.data() + 16);
    
    std::cout << "Header:" << std::endl;
    std::cout << "  Magic: " << magic << std::endl;
    std::cout << "  Version: " << version << std::endl;
    std::cout << "  Tensors: " << n_tensors << std::endl;
    std::cout << "  KV pairs: " << n_kv << std::endl;
    
    // Skip KV section
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t key_len = 0;
        file.read(reinterpret_cast<char*>(&key_len), 8);
        std::vector<char> key(key_len);
        file.read(key.data(), key_len);
        
        uint32_t val_type = 0;
        file.read(reinterpret_cast<char*>(&val_type), 4);
        
        if (val_type == 8) {  // String
            uint64_t str_len = 0;
            file.read(reinterpret_cast<char*>(&str_len), 8);
            std::vector<char> str_data(str_len);
            file.read(str_data.data(), str_len);
        } else {
            // Skip other types (assume 8 bytes)
            std::vector<char> skip_data(8);
            file.read(skip_data.data(), 8);
        }
    }
    
    std::cout << "\n=== Reading First Tensor ===" << std::endl;
    
    // Read tensor info
    uint32_t name_len = 0;
    if (!file.read(reinterpret_cast<char*>(&name_len), 4)) {
        std::cout << "❌ Failed to read tensor name length" << std::endl;
        return 1;
    }
    std::cout << "Tensor name length: " << name_len << std::endl;
    
    if (name_len == 0) {
        std::cout << "❌ Zero name length" << std::endl;
        return 1;
    }
    
    std::vector<char> name(name_len);
    if (!file.read(name.data(), name_len)) {
        std::cout << "❌ Failed to read tensor name" << std::endl;
        return 1;
    }
    
    std::string tensor_name(name.data(), name_len);
    std::cout << "Tensor name: '" << tensor_name << "'" << std::endl;
    
    // Read dimensions
    uint32_t n_dims = 0;
    if (!file.read(reinterpret_cast<char*>(&n_dims), 4)) {
        std::cout << "❌ Failed to read dimension count" << std::endl;
        return 1;
    }
    std::cout << "Number of dimensions: " << n_dims << std::endl;
    
    for (uint32_t d = 0; d < n_dims; ++d) {
        uint64_t dim = 0;
        if (!file.read(reinterpret_cast<char*>(&dim), 8)) {
            std::cout << "❌ Failed to read dimension " << d << std::endl;
            return 1;
        }
        std::cout << "Dimension " << d << ": " << dim << std::endl;
    }
    
    // Read tensor type
    uint32_t tensor_type = 0;
    if (!file.read(reinterpret_cast<char*>(&tensor_type), 4)) {
        std::cout << "❌ Failed to read tensor type" << std::endl;
        return 1;
    }
    std::cout << "Tensor type: " << tensor_type << std::endl;
    
    // Read offset
    uint64_t offset = 0;
    if (!file.read(reinterpret_cast<char*>(&offset), 8)) {
        std::cout << "❌ Failed to read tensor offset" << std::endl;
        return 1;
    }
    std::cout << "Tensor offset: " << offset << std::endl;
    
    // Get current position and file size
    std::streampos current_pos = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    
    std::cout << "Current position: " << current_pos << std::endl;
    std::cout << "File size: " << file_size << std::endl;
    std::cout << "Remaining bytes: " << (file_size - current_pos) << std::endl;
    
    return 0;
}