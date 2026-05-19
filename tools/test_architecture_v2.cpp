#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "=== minllama Architecture Test v2 ===" << std::endl;
    
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
    
    // Test 2: Test with model loading
    std::cout << "\n2. Testing model loading:" << std::endl;
    
    // Try to create some test models
    std::cout << "  Testing architecture-specific properties..." << std::endl;
    
    // Test architecture properties for each supported architecture
    std::vector<minllama::Architecture> test_archs2 = {
        minllama::Architecture::LLaMA,
        minllama::Architecture::Gemma,
        minllama::Architecture::Qwen
    };
    
    for (auto arch : test_archs2) {
        std::cout << "  " << minllama::architecture_to_string(arch) << ": ";
        std::cout << "2D_rotary=" << minllama::uses_2d_rotary(arch) << ", ";
        std::cout << "bias=" << minllama::uses_attention_bias(arch) << ", ";
        std::cout << "head_dim=" << minllama::uses_head_dim(arch) << ", ";
        std::cout << "rope_base=" << minllama::get_default_rope_base(arch);
        std::cout << std::endl;
    }
    
    // Test 3: Create a simple test to verify the 2D RoPE implementation
    std::cout << "\n3. Testing 2D RoPE implementation:" << std::endl;
    
    // Test 2D RoPE with simple data
    std::vector<float> test_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float theta = 10000.0f;
    
    std::cout << "  Input: ";
    for (auto v : test_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    
    // Test 2D RoPE
    if (minllama::rope_apply_2d_f32(test_vec.data(), test_vec.size(), 0, theta)) {
        std::cout << "  2D RoPE applied successfully" << std::endl;
        std::cout << "  Output: ";
        for (auto v : test_vec) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  Failed to apply 2D RoPE" << std::endl;
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
