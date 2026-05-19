#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <vector>
#include <fstream>

// Create a simple test GGUF file for Gemma model
bool create_test_gemma_file(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "❌ Failed to create test file: " << filename << std::endl;
        return false;
    }

    // Simple GGUF header (minimal for testing)
    struct TestGGUFHeader {
        uint32_t version;
        uint64_t n_tensors;
        uint64_t n_kv;
    } header = {3, 2, 5}; // simplified for testing

    // Simple tensor info
    struct TestGGUFTensor {
        uint64_t offset;
        uint64_t size;
    } tensors[2] = {{0, 1024}, {1024, 2048}}; // minimal tensors

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write tensor info
    file.write(reinterpret_cast<const char*>(tensors), sizeof(tensors));

    // Write KV pairs (minimal for architecture detection)
    // "general.architecture" = "gemma"
    const char* kv_name = "general.architecture";
    uint64_t kv_name_len = strlen(kv_name);
    const char* kv_value = "gemma";
    uint64_t kv_value_len = strlen(kv_value);

    file.write(reinterpret_cast<const char*>(&kv_name_len), sizeof(kv_name_len));
    file.write(kv_name, kv_name_len);
    
    uint32_t kv_type = 1; // string
    file.write(reinterpret_cast<const char*>(&kv_type), sizeof(kv_type));
    
    file.write(reinterpret_cast<const char*>(&kv_value_len), sizeof(kv_value_len));
    file.write(kv_value, kv_value_len);

    // Add more dummy data
    char dummy_data[3072] = {0};
    file.write(dummy_data, sizeof(dummy_data));

    file.close();
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== minllama Real Model Test with Test Data ===" << std::endl;
    
    std::string test_model_file = "/tmp/test-gemma-real.gguf";
    
    // Create test model file
    std::cout << "\n1. Creating test Gemma model file..." << std::endl;
    if (!create_test_gemma_file(test_model_file)) {
        std::cout << "❌ Failed to create test model file" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Test model file created: " << test_model_file << std::endl;
    
    // Test model loading
    std::cout << "\n2. Testing model loading..." << std::endl;
    
    ml_model* model = ml_model_load(test_model_file.c_str());
    if (!model) {
        std::cout << "❌ Failed to load test model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Test model loaded successfully!" << std::endl;
    
    // Test architecture detection
    std::cout << "\n3. Testing architecture detection..." << std::endl;
    
    // Check model architecture (this should be 1 for Gemma if our parsing works)
    auto arch = minllama::parse_architecture("gemma");  // direct test
    std::cout << "   Direct architecture test: gemma -> " 
              << minllama::architecture_to_string(arch) << std::endl;
    std::cout << "   Supported: " << minllama::is_supported(arch) << std::endl;
    std::cout << "   2D rotary: " << minllama::uses_2d_rotary(arch) << std::endl;
    std::cout << "   Attention bias: " << minllama::uses_attention_bias(arch) << std::endl;
    std::cout << "   Head dim: " << minllama::uses_head_dim(arch) << std::endl;
    std::cout << "   Rope base: " << minllama::get_default_rope_base(arch) << std::endl;
    
    // Test the model's internal architecture field
    std::cout << "   Model's architecture field: " << model->config.architecture << std::endl;
    
    // Test model properties
    std::cout << "\n4. Testing model properties..." << std::endl;
    std::cout << "   Vocabulary: " << model->config.n_vocab << std::endl;
    std::cout << "   Layers: " << model->config.n_layer << std::endl;
    std::cout << "   Embedding: " << model->config.n_embd << std::endl;
    std::cout << "   Heads: " << model->config.n_head << std::endl;
    
    // Test tokenization
    std::cout << "\n5. Testing tokenization..." << std::endl;
    
    ml_token tokens[10];
    const char* test_text = "Hello Gemma";
    int n_tokens = ml_tokenize(model, test_text, tokens, 10);
    
    std::cout << "   Text: '" << test_text << "'" << std::endl;
    std::cout << "   Tokens: " << n_tokens << std::endl;
    for (int i = 0; i < n_tokens && i < 10; ++i) {
        std::cout << "     [" << i << "]: " << tokens[i] << std::endl;
    }
    
    // Test RoPE routing
    std::cout << "\n6. Testing architecture-specific RoPE routing..." << std::endl;
    
    // Simulate different architectures for RoPE testing
    int test_architectures[] = {0, 1, 2}; // LLaMA, Gemma, Qwen
    
    for (int arch : test_architectures) {
        auto arch_type = static_cast<minllama::Architecture>(arch);
        std::cout << "   Architecture: " << minllama::architecture_to_string(arch_type) << std::endl;
        std::cout << "   Should use 2D RoPE: " << minllama::uses_2d_rotary(arch_type) << std::endl;
        std::cout << "   Should use attention bias: " << minllama::uses_attention_bias(arch_type) << std::endl;
        
        // Test RoPE with different architectures
        std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
        bool use_2d_rope = minllama::uses_2d_rotary(arch_type);
        
        if (use_2d_rope) {
            std::cout << "   Testing 2D RoPE..." << std::endl;
            if (minllama::rope_apply_2d_f32(test_data.data(), test_data.size(), 0, 10000.0f)) {
                std::cout << "   ✅ 2D RoPE applied successfully" << std::endl;
            } else {
                std::cout << "   ❌ 2D RoPE failed" << std::endl;
            }
        } else {
            std::cout << "   Testing 1D RoPE..." << std::endl;
            if (minllama::rope_apply_f32(test_data.data(), test_data.size(), 0, 10000.0f)) {
                std::cout << "   ✅ 1D RoPE applied successfully" << std::endl;
            } else {
                std::cout << "   ❌ 1D RoPE failed" << std::endl;
            }
        }
    }
    
    // Clean up
    ml_model_free(model);
    
    std::cout << "\n=== Real Model Test Complete ===" << std::endl;
    return 0;
}