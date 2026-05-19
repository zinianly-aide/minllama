#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// Create a minimal valid GGUF v3 file for testing
int main() {
    std::cout << "Creating minimal test GGUF v3 file..." << std::endl;
    
    std::ofstream file("/tmp/test_minimal.gguf", std::ios::binary);
    if (!file) {
        std::cout << "❌ Cannot create file" << std::endl;
        return 1;
    }
    
    // Write header: GGUF, version 3, 1 tensor, 1 KV pair
    // Magic
    file.write("GGUF", 4);
    // Version (3)
    uint32_t version = 3;
    file.write(reinterpret_cast<const char*>(&version), 4);
    // Tensor count (1)
    uint64_t n_tensors = 1;
    file.write(reinterpret_cast<const char*>(&n_tensors), 8);
    // KV count (1)
    uint64_t n_kv = 1;
    file.write(reinterpret_cast<const char*>(&n_kv), 8);
    
    // Write KV pair: general.architecture = llama
    // Key: "general.architecture" (length 20)
    uint64_t key_len = 20;
    file.write(reinterpret_cast<const char*>(&key_len), 8);
    file.write("general.architecture", 20);
    // Value type: 8 (string)
    uint32_t val_type = 8;
    file.write(reinterpret_cast<const char*>(&val_type), 4);
    // Value: "llama" (length 5)
    uint64_t val_len = 5;
    file.write(reinterpret_cast<const char*>(&val_len), 8);
    file.write("llama", 5);
    
    // Write tensor info
    // Tensor name: "token_embd.weight" (length 18)
    uint32_t name_len = 18;
    file.write(reinterpret_cast<const char*>(&name_len), 4);
    file.write("token_embd.weight", 18);
    // Dimensions: 32000 (1 dimension)
    uint32_t n_dims = 1;
    file.write(reinterpret_cast<const char*>(&n_dims), 4);
    uint64_t dim = 32000;
    file.write(reinterpret_cast<const char*>(&dim), 8);
    // Type: 1 (F16)
    uint32_t tensor_type = 1;
    file.write(reinterpret_cast<const char*>(&tensor_type), 4);
    // Offset: after header + KV + tensor info (24 + 8+20+4+8+5 + 4+18+4+8+4) = 73
    uint64_t offset = 73;
    file.write(reinterpret_cast<const char*>(&offset), 8);
    
    // Write tensor data (dummy F16 data)
    std::vector<uint16_t> dummy_data(32000 / 2, 0);  // 16000 F16 values
    file.write(reinterpret_cast<const char*>(dummy_data.data()), dummy_data.size() * 2);
    
    file.close();
    
    std::cout << "✅ Created minimal test GGUF file" << std::endl;
    std::cout << "File size: " << (73 + dummy_data.size() * 2) << " bytes" << std::endl;
    
    return 0;
}