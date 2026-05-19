#include "architecture.h"
#include <cstring>
#include <cstdlib>

namespace minllama {

namespace {
    struct ArchitectureInfo {
        const char* name;
        bool supported;
        bool uses_2d_rotary;
        bool uses_attention_bias;
        bool uses_head_dim;
        int64_t default_rope_base;
    };

    static const ArchitectureInfo ARCH_INFOS[] = {
        { "llama", true,  false, false, false, 10000 },
        { "gemma", true,  true,  true,  true,  10000 },
        { "qwen",  true,  false, false, false, 10000 },
        // Future additions here
    };

    static const int ARCH_COUNT = sizeof(ARCH_INFOS) / sizeof(ARCH_INFOS[0]);
}

const char* architecture_to_string(Architecture arch) {
    int arch_int = static_cast<int>(arch);
    if (arch_int < 0 || static_cast<size_t>(arch_int) >= ARCH_COUNT) {
        return "unknown";
    }
    return ARCH_INFOS[arch_int].name;
}

Architecture parse_architecture(const std::string& name) {
    for (int i = 0; i < ARCH_COUNT; ++i) {
        if (strcasecmp(name.c_str(), ARCH_INFOS[i].name) == 0) {
            return static_cast<Architecture>(i);
        }
    }
    // Unknown architecture defaults to LLaMA for compatibility
    return Architecture::LLaMA;
}

bool is_supported(Architecture arch) {
    int arch_int = static_cast<int>(arch);
    if (arch_int < 0 || static_cast<size_t>(arch_int) >= ARCH_COUNT) {
        return false;
    }
    return ARCH_INFOS[arch_int].supported;
}

int64_t get_default_rope_base(Architecture arch) {
    int arch_int = static_cast<int>(arch);
    if (arch_int < 0 || static_cast<size_t>(arch_int) >= ARCH_COUNT) {
        return 10000;
    }
    return ARCH_INFOS[arch_int].default_rope_base;
}

bool uses_2d_rotary(Architecture arch) {
    int arch_int = static_cast<int>(arch);
    if (arch_int < 0 || static_cast<size_t>(arch_int) >= ARCH_COUNT) {
        return false;
    }
    return ARCH_INFOS[arch_int].uses_2d_rotary;
}

bool uses_attention_bias(Architecture arch) {
    int arch_int = static_cast<int>(arch);
    if (arch_int < 0 || static_cast<size_t>(arch_int) >= ARCH_COUNT) {
        return false;
    }
    return ARCH_INFOS[arch_int].uses_attention_bias;
}

bool uses_head_dim(Architecture arch) {
    int arch_int = static_cast<int>(arch);
    if (arch_int < 0 || static_cast<size_t>(arch_int) >= ARCH_COUNT) {
        return false;
    }
    return ARCH_INFOS[arch_int].uses_head_dim;
}

} // namespace minllama
