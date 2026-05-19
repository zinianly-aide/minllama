#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "=== minllama Architecture Test ===" << std::endl;
    
    // Test 1: Architecture parsing
    std::cout << "\n1. Testing architecture parsing:" << std::endl;
    std::vector<std::string> test_archs = {"llama", "gemma", "qwen", "unknown", "GEMMA", "QWEN"};
    
    for (const auto& name : test_archs) {
        auto arch = minllama::parse_architecture(name);
        std::cout << "  " << name << " -> " << minllama::architecture_to_string(arch)
                  << " (supported: " << minllama::is_supported(arch) << ")"
                  << " (2D rotary: " << minllama::uses_2d_rotary(arch) << ")"
                  << " (attention bias: " << minllama::uses_attention_bias(arch) << ")"
                  << " (head_dim: " << minllama::uses_head_dim(arch) << ")"
                  << " (rope_base: " << minllama::get_default_rope_base(arch) << ")"
                  << std::endl;
    }
    
    // Test 2: Test with actual model file
    std::cout << "\n2. Testing model file support:" << std::endl;
    
    const char* test_files[] = {
        "/tmp/test-llama-smollm.gguf",
        "nonexistent.gguf"
    };
    
    for (const char* filename : test_files) {
        std::cout << "  Testing: " << filename << std::endl;
        
        auto model = ml_model_load(filename);
        if (model) {
            std::cout << "    ✅ Loaded successfully" << std::endl;
            std::cout << "    Architecture: " << model->config.architecture 
                      << " (" << minllama::architecture_to_string(static_cast<minllama::Architecture>(model->config.architecture)) << ")" << std::endl;
            std::cout << "    Supported: " << ml_model_is_supported(model) << std::endl;
            std::cout << "    Vocab: " << model->config.n_vocab 
                      << ", Layers: " << model->config.n_layer
                      << ", Heads: " << model->config.n_head << std::endl;
            ml_model_free(model);
        } else {
            std::cout << "    ❌ Failed to load" << std::endl;
        }
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
