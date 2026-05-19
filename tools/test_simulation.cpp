#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== minllama Architecture Simulation Test ===" << std::endl;
    
    // Test 1: Architecture properties simulation
    std::cout << "\n1. Testing architecture properties simulation:" << std::endl;
    
    std::vector<minllama::Architecture> test_archs = {
        minllama::Architecture::LLaMA,
        minllama::Architecture::Gemma,
        minllama::Architecture::Qwen
    };
    
    for (auto arch : test_archs) {
        std::cout << "  " << minllama::architecture_to_string(arch) << ":" << std::endl;
        std::cout << "    Supported: " << minllama::is_supported(arch) << std::endl;
        std::cout << "    2D rotary: " << minllama::uses_2d_rotary(arch) << std::endl;
        std::cout << "    Attention bias: " << minllama::uses_attention_bias(arch) << std::endl;
        std::cout << "    Head dim: " << minllama::uses_head_dim(arch) << std::endl;
        std::cout << "    Rope base: " << minllama::get_default_rope_base(arch) << std::endl;
        std::cout << std::endl;
    }
    
    // Test 2: Simulate RoPE behavior
    std::cout << "2. Simulating RoPE behavior differences:" << std::endl;
    
    // Create test data for RoPE
    std::vector<float> test_data_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> test_data_2d = {1.0f, 2.0f, 3.0f, 4.0f};
    
    float theta = 10000.0f;
    int position = 0;
    
    std::cout << "  Input data: [";
    for (size_t i = 0; i < test_data_1d.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << test_data_1d[i];
    }
    std::cout << "]" << std::endl;
    
    // Test 1D RoPE (LLaMA)
    if (minllama::rope_apply_f32(test_data_1d.data(), test_data_1d.size(), position, theta)) {
        std::cout << "  1D RoPE result (LLaMA): [";
        for (size_t i = 0; i < test_data_1d.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << test_data_1d[i];
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "  ❌ 1D RoPE failed" << std::endl;
    }
    
    // Test 2D RoPE (Gemma)
    if (minllama::rope_apply_2d_f32(test_data_2d.data(), test_data_2d.size(), position, theta)) {
        std::cout << "  2D RoPE result (Gemma): [";
        for (size_t i = 0; i < test_data_2d.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << test_data_2d[i];
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "  ❌ 2D RoPE failed" << std::endl;
    }
    
    // Test 3: Architecture detection simulation
    std::cout << "\n3. Simulating architecture detection from GGUF:" << std::endl;
    
    std::vector<std::string> gguf_architectures = {
        "llama", "gemma", "qwen", "unknown", "GEMMA", "QWEN"
    };
    
    for (const auto& arch_name : gguf_architectures) {
        auto detected_arch = minllama::parse_architecture(arch_name);
        std::cout << "  GGUF reports: '" << arch_name << "' -> " 
                  << minllama::architecture_to_string(detected_arch) << " (supported: " 
                  << minllama::is_supported(detected_arch) << ")" << std::endl;
    }
    
    // Test 4: Model configuration simulation
    std::cout << "\n4. Simulating model configuration:" << std::endl;
    
    // Simulate a model configuration
    ModelConfig config;
    config.n_vocab = 32000;
    config.n_layer = 28;
    config.n_embd = 2048;
    config.n_head = 16;
    config.n_head_kv = 16;
    config.n_ctx_train = 2048;
    config.rope_theta = 10000.0f;
    config.rms_norm_eps = 1e-6f;
    
    // Test different architectures
    for (auto arch : test_archs) {
        std::cout << "  " << minllama::architecture_to_string(arch) << " model:" << std::endl;
        std::cout << "    Vocab: " << config.n_vocab << std::endl;
        std::cout << "    Layers: " << config.n_layer << std::endl;
        std::cout << "    Embedding: " << config.n_embd << std::endl;
        std::cout << "    Heads: " << config.n_head << std::endl;
        std::cout << "    2D RoPE: " << minllama::uses_2d_rotary(arch) << std::endl;
        std::cout << "    Attention bias: " << minllama::uses_attention_bias(arch) << std::endl;
        std::cout << "    Head dim: " << minllama::uses_head_dim(arch) << std::endl;
        std::cout << "    Rope base: " << minllama::get_default_rope_base(arch) << std::endl;
        std::cout << std::endl;
    }
    
    // Test 5: Performance impact simulation
    std::cout << "5. Simulating performance impact:" << std::endl;
    
    // Simulate different model sizes
    std::vector<std::pair<std::string, int>> model_sizes = {
        {"LLaMA 7B", 7},
        {"Gemma 2B", 2},
        {"Qwen 1.5B", 1}
    };
    
    for (const auto& [model_name, size] : model_sizes) {
        auto arch = (model_name == "Gemma 2B") ? minllama::Architecture::Gemma : 
                   (model_name == "Qwen 1.5B") ? minllama::Architecture::Qwen : 
                   minllama::Architecture::LLaMA;
        
        int overhead = minllama::uses_2d_rotary(arch) ? 15 : 0; // 15% overhead for 2D RoPE
        int bias_cost = minllama::uses_attention_bias(arch) ? 5 : 0; // 5% cost for attention bias
        
        std::cout << "  " << model_name << " (" << size << "B):" << std::endl;
        std::cout << "    Base inference: 100%" << std::endl;
        std::cout << "    2D RoPE overhead: " << overhead << "%" << std::endl;
        std::cout << "    Attention bias: " << bias_cost << "%" << std::endl;
        std::cout << "    Total overhead: " << (overhead + bias_cost) << "%" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "=== Simulation Test Complete ===" << std::endl;
    return 0;
}