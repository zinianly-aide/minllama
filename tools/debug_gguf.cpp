#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <gguf_file>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "❌ Cannot open file: " << filename << std::endl;
        return 1;
    }
    
    std::cout << "=== GGUF File Debug Tool ===" << std::endl;
    std::cout << "File: " << filename << std::endl;
    
    // Read header
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "GGUF") {
        std::cout << "❌ Invalid magic: " << std::string(magic, 4) << std::endl;
        return 1;
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    std::cout << "Version: " << version << std::endl;
    
    uint64_t n_tensors;
    file.read(reinterpret_cast<char*>(&n_tensors), 8);
    std::cout << "Tensors: " << n_tensors << std::endl;
    
    // Skip tensor info
    file.seekg(n_tensors * 16, std::ios::cur);
    
    // Read KV pairs
    std::cout << "\nKV Pairs:" << std::endl;
    while (file.peek() != EOF) {
        uint64_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), 8);
        if (key_len == 0) break;
        
        std::vector<char> key(key_len);
        file.read(key.data(), key_len);
        std::string key_str(key.data(), key_len);
        
        uint32_t type;
        file.read(reinterpret_cast<char*>(&type), 4);
        std::cout << "  Key: '" << key_str << "' (type: " << type << ")";
        
        uint64_t value_len;
        file.read(reinterpret_cast<char*>(&value_len), 8);
        
        if (key_str == "general.architecture" || key_str == "general.name" || key_str == "general.quantization_version") {
            std::vector<char> value(value_len);
            file.read(value.data(), value_len);
            std::string value_str(value.data(), value_len);
            std::cout << " -> '" << value_str << "'";
        } else {
            file.seekg(value_len, std::ios::cur);
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Debug Complete ===" << std::endl;
    return 0;
}