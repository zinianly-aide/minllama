#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    std::cout << "=== minllama Ollama Models Simple Test ===" << std::endl;
    
    // Test Ollama models
    std::vector<std::pair<std::string, std::string>> ollama_models = {
        {"Gemma4", "/Users/anshi/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"},
        {"Qwen3.5", "/Users/anshi/.ollama/models/blobs/sha256-59bb50d8116b6a1f9bfbb940d6bb946a05554e591e30c8c2429ed6c854867ecb"}
    };
    
    for (const auto& [name, path] : ollama_models) {
        std::cout << "\n=== Testing " << name << " ===" << std::endl;
        
        // Check if file exists
        std::ifstream file_check(path);
        if (!file_check.good()) {
            std::cout << "❌ File not found: " << path << std::endl;
            continue;
        }
        
        // Get file size
        file_check.seekg(0, std::ios::end);
        size_t file_size = file_check.tellg();
        std::cout << "✅ File size: " << file_size << " bytes" << std::endl;
        
        // Read first few KB to get architecture info
        std::ifstream file(path, std::ios::binary);
        std::vector<char> buffer(8192);
        file.read(buffer.data(), 8192);
        
        // Check GGUF magic
        if (buffer.size() >= 4 && std::string(buffer.data(), 4) == "GGUF") {
            std::cout << "✅ Valid GGUF file detected" << std::endl;
            
            // Read version (bytes 4-8)
            if (buffer.size() >= 8) {
                uint32_t version = *reinterpret_cast<uint32_t*>(buffer.data() + 4);
                std::cout << "✅ GGUF Version: " << version << std::endl;
                
                if (version == 3) {
                    std::cout << "✅ Compatible version (v3)" << std::endl;
                } else {
                    std::cout << "❌ Incompatible version (v" << version << ")" << std::endl;
                }
            }
            
            // Search for architecture string in the first 8KB
            std::string buffer_str(buffer.data(), buffer.size());
            size_t arch_pos = buffer_str.find("general.architecture");
            
            if (arch_pos != std::string::npos) {
                // Find the architecture value after the key
                size_t value_start = buffer_str.find("\"", arch_pos + 20) + 1;
                size_t value_end = buffer_str.find("\"", value_start);
                
                if (value_end != std::string::npos) {
                    std::string architecture = buffer_str.substr(value_start, value_end - value_start);
                    std::cout << "✅ Architecture from GGUF: " << architecture << std::endl;
                    
                    // Parse with minllama
                    auto arch = minllama::parse_architecture(architecture);
                    std::cout << "🔧 Parsed: " << minllama::architecture_to_string(arch) << std::endl;
                    std::cout << "🔧 Supported: " << minllama::is_supported(arch) << std::endl;
                    
                    // Test features
                    std::cout << "🔧 Features:" << std::endl;
                    std::cout << "  2D rotary: " << minllama::uses_2d_rotary(arch) << std::endl;
                    std::cout << "  Attention bias: " << minllama::uses_attention_bias(arch) << std::endl;
                    std::cout << "  Head dim: " << minllama::uses_head_dim(arch) << std::endl;
                    std::cout << "  Rope base: " << minllama::get_default_rope_base(arch) << std::endl;
                    
                    // Test RoPE
                    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
                    float theta = minllama::get_default_rope_base(arch);
                    
                    std::cout << "\n🧪 Testing RoPE (theta=" << theta << "):" << std::endl;
                    if (minllama::uses_2d_rotary(arch)) {
                        bool success = minllama::rope_apply_2d_f32(test_data.data(), test_data.size(), 0, theta);
                        std::cout << "  2D RoPE: " << (success ? "✅ Success" : "❌ Failed") << std::endl;
                    } else {
                        bool success = minllama::rope_apply_f32(test_data.data(), test_data.size(), 0, theta);
                        std::cout << "  1D RoPE: " << (success ? "✅ Success" : "❌ Failed") << std::endl;
                    }
                    
                } else {
                    std::cout << "❌ Could not parse architecture value" << std::endl;
                }
            } else {
                std::cout << "❌ Could not find architecture info in first 8KB" << std::endl;
            }
            
        } else {
            std::cout << "❌ Not a GGUF file" << std::endl;
        }
        
        // Simulate model configuration
        std::cout << "\n🔧 Simulating " << name << " configuration:" << std::endl;
        
        ModelConfig config;
        if (name == "Gemma4") {
            config.n_vocab = 256000;
            config.n_layer = 42;
            config.n_embd = 4096;
            config.n_head = 32;
            config.n_head_kv = 32;
            config.n_ctx_train = 8192;
        } else if (name == "Qwen3.5") {
            config.n_vocab = 152064;
            config.n_layer = 32;
            config.n_embd = 4096;
            config.n_head = 32;
            config.n_head_kv = 8;
            config.n_ctx_train = 32768;
        }
        
        config.rope_theta = minllama::get_default_rope_base(minllama::parse_architecture("gemma"));
        config.rms_norm_eps = 1e-5f;
        
        std::cout << "  Vocab: " << config.n_vocab << std::endl;
        std::cout << "  Layers: " << config.n_layer << std::endl;
        std::cout << "  Embedding: " << config.n_embd << std::endl;
        std::cout << "  Heads: " << config.n_head << " (KV: " << config.n_head_kv << ")" << std::endl;
        std::cout << "  Context: " << config.n_ctx_train << std::endl;
        std::cout << "  RoPE theta: " << config.rope_theta << std::endl;
        
        std::cout << "✅ " << name << " test completed" << std::endl;
    }
    
    std::cout << "\n=== Ollama Models Tests Complete ===" << std::endl;
    return 0;
}