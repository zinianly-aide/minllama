#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <array>

// Simulate minllama's read functions
bool read_exact(std::ifstream &file, void *data, std::size_t size) {
    file.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(size));
    bool result = file.gcount() == static_cast<std::streamsize>(size);
    std::cout << "[read_exact] size=" << size << ", read=" << file.gcount() << ", result=" << (result ? "OK" : "FAIL") << std::endl;
    return result;
}

bool read_u64_le(std::ifstream &file, std::uint64_t *out) {
    std::array<unsigned char, 8> bytes = {};
    if (!read_exact(file, bytes.data(), bytes.size())) {
        return false;
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    *out = value;
    return true;
}

bool read_u32_le(std::ifstream &file, std::uint32_t *out) {
    std::array<unsigned char, 4> bytes = {};
    if (!read_exact(file, bytes.data(), bytes.size())) {
        return false;
    }
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint32_t>(bytes[i]) << (8 * i);
    }
    *out = value;
    return true;
}

bool read_gguf_string(std::ifstream &file, std::string *out) {
    std::uint64_t size = 0;
    if (!read_u64_le(file, &size)) {
        return false;
    }
    std::cout << "[read_gguf_string] string size: " << size << std::endl;
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    std::string value(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !read_exact(file, value.data(), value.size())) {
        return false;
    }
    *out = value;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== minllama GGUF Reading Debug Tool ===" << std::endl;
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::cout << "Debugging: " << model_path << std::endl;
    
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        std::cout << "❌ Cannot open file" << std::endl;
        return 1;
    }
    
    // Read header
    std::array<unsigned char, 24> header = {};
    if (!read_exact(file, header.data(), header.size())) {
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
    
    if (version != 3) {
        std::cout << "❌ Wrong version, expecting 3, got " << version << std::endl;
        return 1;
    }
    
    // Read KV pairs
    std::cout << "\n=== Reading KV Pairs ===" << std::endl;
    
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::cout << "\nKV[" << i << "]: ";
        
        // Read key
        std::string key;
        if (!read_gguf_string(file, &key)) {
            std::cout << "❌ Failed to read key" << std::endl;
            break;
        }
        std::cout << "Key: '" << key << "'" << std::endl;
        
        // Read value type
        uint32_t type_raw = 0;
        if (!read_u32_le(file, &type_raw)) {
            std::cout << "❌ Failed to read value type" << std::endl;
            break;
        }
        std::cout << "Type: " << type_raw << std::endl;
        
        // Skip value (simulate what minllama does)
        if (type_raw == 8) {  // String
            uint64_t str_size = 0;
            if (!read_u64_le(file, &str_size)) {
                std::cout << "❌ Failed to read string size" << std::endl;
                break;
            }
            std::cout << "String length: " << str_size << std::endl;
            if (str_size > 0) {
                // Skip the string data
                std::vector<char> str_data(str_size);
                if (!read_exact(file, str_data.data(), str_data.size())) {
                    std::cout << "❌ Failed to skip string data" << std::endl;
                    break;
                }
            }
        } else {
            // Skip other types (assume 8 bytes)
            std::vector<char> skip_data(8);
            if (!read_exact(file, skip_data.data(), skip_data.size())) {
                std::cout << "❌ Failed to skip value data" << std::endl;
                break;
            }
        }
        
        std::cout << "✅ KV[" << i << "] completed" << std::endl;
    }
    
    std::cout << "\n=== Position after KV reading ===" << std::endl;
    std::streampos pos = file.tellg();
    std::cout << "Current position: " << pos << std::endl;
    
    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos end_pos = file.tellg();
    std::cout << "File size: " << end_pos << std::endl;
    
    if (pos == end_pos) {
        std::cout << "❌ File ended exactly during KV reading!" << std::endl;
    } else if (pos > end_pos) {
        std::cout << "❌ Reading position beyond file size!" << std::endl;
    } else {
        std::cout << "✅ Still have data remaining" << std::endl;
        std::cout << "Remaining bytes: " << (end_pos - pos) << std::endl;
        
        // Try to read first tensor
        std::cout << "\n=== Reading first tensor ===" << std::endl;
        
        // Read name
        std::string tensor_name;
        if (!read_gguf_string(file, &tensor_name)) {
            std::cout << "❌ Failed to read tensor name" << std::endl;
        } else {
            std::cout << "Tensor name: '" << tensor_name << "'" << std::endl;
        }
    }
    
    return 0;
}