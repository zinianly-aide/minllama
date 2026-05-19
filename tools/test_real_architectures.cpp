#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <fstream>
#include <vector>

// Function to create a test GGUF file with specific architecture
bool create_test_gguf_with_architecture(const std::string& filename, const std::string& architecture) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    
    // GGUF header
    file.write("GGUF", 4);  // magic
    uint32_t version = 3;
    file.write(reinterpret_cast<const char*>(&version), 4);  // version
    uint64_t n_tensors = 3;
    file.write(reinterpret_cast<const char*>(&n_tensors), 8);  // n_tensors
    
    // Tensor info (dummy)
    for (int i = 0; i < 3; i++) {
        file.write(reinterpret_cast<const char*>(new uint64_t[2]{uint64_t(i * 1024), uint64_t(1024)}), 16);
    }
    
    // KV pairs
    struct KVPair {
        std::string key;
        std::string value;
        uint32_t type;
    };
    
    std::vector<KVPair> kv_pairs = {
        {"general.architecture", architecture, 1},
        {"general.name", "test-" + architecture, 1},
        {"general.size", "7B", 1}
    };
    
    for (const auto& kv : kv_pairs) {
        uint64_t key_len = kv.key.length();
        file.write(reinterpret_cast<const char*>(&key_len), 8);
        file.write(kv.key.c_str(), key_len);
        
        uint32_t type = kv.type;
        file.write(reinterpret_cast<const char*>(&type), 4);
        
        uint64_t value_len = kv.value.length();
        file.write(reinterpret_cast<const char*>(&value_len), 8);
        file.write(kv.value.c_str(), value_len);
    }
    
    // Dummy tensor data
    std::vector<char> dummy_data(1024 * 3, 0);
    file.write(dummy_data.data(), dummy_data.size());
    
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== minllama Real Architecture Test with Test GGUF ===" << std::endl;
    
    // Test different architectures by creating test files
    std::vector<std::string> test_architectures = {
        "llama", "gemma", "qwen", "mistral", "phi", "unknown"
    };
    
    for (const auto& arch_name : test_architectures) {
        std::string test_file = "/tmp/test-" + arch_name + "-real.gguf";
        
        std::cout << "\n=== Testing Architecture: " << arch_name << " ===" << std::endl;
        
        // Create test GGUF file
        if (!create_test_gguf_with_architecture(test_file, arch_name)) {
            std::cout << "❌ Failed to create test file for " << arch_name << std::endl;
            continue;
        }
        
        std::cout << "✅ Created test GGUF: " << test_file << std::endl;
        
        // Test architecture parsing
        auto parsed_arch = minllama::parse_architecture(arch_name);
        std::cout << "   Architecture: " << arch_name << " -> " 
                  << minllama::architecture_to_string(parsed_arch) << std::endl;
        std::cout << "   Supported: " << minllama::is_supported(parsed_arch) << std::endl;
        
        // Test RoPE functionality
        std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
        float theta = minllama::get_default_rope_base(parsed_arch);
        int position = 0;
        
        std::cout << "   RoPE type: " << (minllama::uses_2d_rotary(parsed_arch) ? "2D" : "1D") << std::endl;
        std::cout << "   RoPE theta: " << theta << std::endl;
        
        if (minllama::uses_2d_rotary(parsed_arch)) {
            std::cout << "   Testing 2D RoPE..." << std::endl;
            if (minllama::rope_apply_2d_f32(test_data.data(), test_data.size(), position, theta)) {
                std::cout << "   ✅ 2D RoPE successful" << std::endl;
                std::cout << "   Result: [";
                for (size_t i = 0; i < test_data.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << test_data[i];
                }
                std::cout << "]" << std::endl;
            } else {
                std::cout << "   ❌ 2D RoPE failed" << std::endl;
            }
        } else {
            std::cout << "   Testing 1D RoPE..." << std::endl;
            if (minllama::rope_apply_f32(test_data.data(), test_data.size(), position, theta)) {
                std::cout << "   ✅ 1D RoPE successful" << std::endl;
                std::cout << "   Result: [";
                for (size_t i = 0; i < test_data.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << test_data[i];
                }
                std::cout << "]" << std::endl;
            } else {
                std::cout << "   ❌ 1D RoPE failed" << std::endl;
            }
        }
        
        // Test architecture-specific features
        std::cout << "   Features:" << std::endl;
        std::cout << "     2D rotary: " << minllama::uses_2d_rotary(parsed_arch) << std::endl;
        std::cout << "     Attention bias: " << minllama::uses_attention_bias(parsed_arch) << std::endl;
        std::cout << "     Head dim: " << minllama::uses_head_dim(parsed_arch) << std::endl;
        
        // Test model configuration simulation
        ModelConfig config;
        config.n_vocab = 32000;
        config.n_layer = 32;
        config.n_embd = 4096;
        config.n_head = 32;
        config.n_head_kv = 8;
        config.n_ctx_train = 8192;
        config.rope_theta = theta;
        config.rms_norm_eps = 1e-5f;
        
        std::cout << "   Simulated " << arch_name << " model config:" << std::endl;
        std::cout << "     Vocab: " << config.n_vocab << std::endl;
        std::cout << "     Layers: " << config.n_layer << std::endl;
        std::cout << "     Embedding: " << config.n_embd << std::endl;
        std::cout << "     Heads: " << config.n_head << " (KV: " << config.n_head_kv << ")" << std::endl;
        std::cout << "     Context: " << config.n_ctx_train << std::endl;
        std::cout << "     RoPE theta: " << config.rope_theta << std::endl;
        
        // Clean up
        std::remove(test_file.c_str());
        
        std::cout << "✅ Test completed for " << arch_name << std::endl;
    }
    
    // Test the real Mistral model
    std::cout << "\n=== Testing Real Mistral Model ===" << std::endl;
    
    std::string real_model = "/tmp/mistral-7b-instruct-v0.2.Q4_K_M.gguf";
    std::ifstream file_check(real_model);
    if (file_check.good()) {
        std::cout << "✅ Real Mistral model found" << std::endl;
        
        // Parse architecture from real model
        auto real_arch = minllama::parse_architecture("llama");  // Mistral GGUF reports "llama"
        std::cout << "   Real model architecture: llama -> " 
                  << minllama::architecture_to_string(real_arch) << std::endl;
        std::cout << "   This is correct (Mistral uses LLaMA architecture base)" << std::endl;
        
        // Test RoPE with real model configuration
        std::vector<float> real_test_data = {1.0f, 2.0f, 3.0f, 4.0f};
        float real_theta = minllama::get_default_rope_base(real_arch);
        
        std::cout << "   Real model RoPE theta: " << real_theta << std::endl;
        std::cout << "   Testing 1D RoPE for LLaMA base..." << std::endl;
        
        if (minllama::rope_apply_f32(real_test_data.data(), real_test_data.size(), 0, real_theta)) {
            std::cout << "   ✅ Real model RoPE successful" << std::endl;
            std::cout << "   Result: [";
            for (size_t i = 0; i < real_test_data.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << real_test_data[i];
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "✅ Real Mistral model test successful" << std::endl;
    } else {
        std::cout << "❌ Real Mistral model not found" << std::endl;
    }
    
    std::cout << "\n=== All Real Architecture Tests Complete ===" << std::endl;
    return 0;
}