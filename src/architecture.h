#pragma once

#include <string>
#include <cstdint>

namespace minllama {

/**
 * Supported model architectures
 */
enum class Architecture {
    LLaMA,
    Gemma,
    Qwen,
    // Future: Mistral, Phi, etc.
};

/**
 * Get human-readable name for architecture
 */
const char* architecture_to_string(Architecture arch);

/**
 * Parse architecture from string
 * Returns Architecture::LLaMA if unknown
 */
Architecture parse_architecture(const std::string& name);

/**
 * Check if architecture is supported
 */
bool is_supported(Architecture arch);

/**
 * Get default RoPE base for architecture
 * Most models use 10000, but some use different values
 */
int64_t get_default_rope_base(Architecture arch);

/**
 * Check if architecture uses 2D rotary (Gemma)
 */
bool uses_2d_rotary(Architecture arch);

/**
 * Check if architecture uses attention bias (Gemma)
 */
bool uses_attention_bias(Architecture arch);

/**
 * Check if architecture uses head_dim (Gemma)
 */
bool uses_head_dim(Architecture arch);

} // namespace minllama
