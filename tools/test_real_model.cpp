#include "minllama_internal.h"
#include "architecture.h"
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "=== minllama Real Model Test ===" << std::endl;
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_file>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // Check if file exists
    std::ifstream file(model_path);
    if (!file.good()) {
        std::cout << "❌ Model file not found: " << model_path << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model file exists: " << model_path << std::endl;
    
    // Test 1: Try to load the model
    std::cout << "\n1. Testing model loading..." << std::endl;
    
    ml_model* model = ml_model_load(model_path.c_str());
    if (!model) {
        std::cout << "❌ Failed to load model (expected for small test file)" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model loaded successfully!" << std::endl;
    
    // Test 2: Check model properties
    std::cout << "\n2. Testing architecture detection..." << std::endl;
    
    auto arch = minllama::parse_architecture(std::to_string(model->config.architecture));
    std::cout << "   Model architecture: " << model->config.architecture << " -> " 
              << minllama::architecture_to_string(arch) << std::endl;
    std::cout << "   Supported: " << minllama::is_supported(arch) << std::endl;
    std::cout << "   2D rotary: " << minllama::uses_2d_rotary(arch) << std::endl;
    std::cout << "   Attention bias: " << minllama::uses_attention_bias(arch) << std::endl;
    std::cout << "   Head dim: " << minllama::uses_head_dim(arch) << std::endl;
    std::cout << "   Rope base: " << minllama::get_default_rope_base(arch) << std::endl;
    
    // Test 3: Test tokenization
    std::cout << "\n3. Testing tokenization..." << std::endl;
    
    // Test token encoding/decoding
    std::vector<ml_token> tokens;
    const char* test_text = "Hello world";
    
    if (ml_tokenize(model, test_text, tokens.data(), tokens.size()) > 0) {
        std::cout << "✅ Text encoded successfully!" << std::endl;
        std::cout << "   Tokens: [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << tokens[i];
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "❌ Failed to encode text" << std::endl;
    }
    
    // Clean up
    ml_model_free(model);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}