# P3-2: Model Compatibility Extension

## Goal
Extend minllama to support additional model architectures beyond LLaMA.

## Status
🚧 STARTED (2026-05-19)

---

## Architecture Limitations

### Current State
- Only supports: `llama` architecture
- Explicit check at line 298-302 in `src/gguf.cpp`

### Known Unsupported Models
- ❌ Gemma (Google)
- ❌ Mistral (Mistral AI)
- ❌ Qwen (Alibaba)
- ❌ Phi (Microsoft)
- ❌ TinyLlama (not a distinct architecture, but popular)

---

## Strategy

### Phase 1: Add Architecture Enumeration
1. Define architecture IDs/strings for supported models
2. Remove hard `architecture != "llama"` check
3. Add per-architecture config validation

### Phase 2: Gemma Support (Priority 1)
- Research Gemma architecture details
- Add Gemma-specific config parsing
- Implement Gemma-2D RoPE (if different)
- Test with Gemma models

### Phase 3: Qwen Support (Priority 2)
- Research Qwen architecture details
- Add Qwen-specific config parsing
- Handle Qwen's rotary position embedding differences
- Test with Qwen models

### Phase 4: Other Models (Priority 3)
- Mistral, Phi, TinyLlama, etc.
- Tokenizer differences
- Attention variants

---

## Implementation Plan

### Step 1: Architecture Registry
Create `src/architecture.h`:
```cpp
namespace minllama {
    enum class Architecture {
        LLaMA,
        Gemma,
        Qwen,
        // Future: Mistral, Phi, etc.
    };

    const char* architecture_to_string(Architecture arch);
    Architecture parse_architecture(const std::string& name);
    bool is_supported(Architecture arch);
}
```

### Step 2: Remove Hard Check
Update `src/gguf.cpp` line 298-302:
```cpp
// OLD:
if (architecture != "llama") {
    error... return false;
}

// NEW:
auto arch = parse_architecture(architecture);
if (!is_supported(arch)) {
    error... return false;
}
```

### Step 3: Architecture-Specific Config
Add architecture-specific config structs:
```cpp
struct GemmaConfig {
    int64_t attention_bias = 0;  // Gemma has bias in attention
    int64_t attention_head_dim = 0;  // Gemma uses head_dim
};

struct QwenConfig {
    int64_t rotary_base = 10000;
    int64_t rotary_dim = 0;
    // ... other Qwen-specific params
};
```

### Step 4: Implementation per Architecture
Create architecture-specific loaders in `src/architecture.cpp`:
- `load_gemma_config()`
- `load_qwen_config()`
- Handle differences in:
  - Attention implementation (bias, head_dim)
  - RoPE (base, dimension)
  - Layer norms
  - Rotary position embeddings

---

## Research Tasks

### Gemma Architecture
- ✅ Paper: Google Gemma: Open Models Based on Gemini Technology
- Architecture details from GGUF specs
- Attention: 2D rotary + bias
- Normalization: RMS + bias (in some versions)

### Qwen Architecture
- ✅ Paper: Qwen: Introduction to a Strong, Scalable, and Affordable Multimodal LLM
- Qwen2 specifics (rotary base, dimension)
- Attention: GQA standard
- RoPE: custom base values

---

## Testing Strategy

### Test Models
1. **Gemma 2B / 9B** (small models for testing)
   - Download from HuggingFace
   - Test basic forward pass
   - Compare with reference implementation

2. **Qwen 1.5 / 2**
   - Test quantization support
   - Verify RoPE handling

3. **TinyLlama**
   - Verify tokenizer compatibility
   - Confirm no architecture changes needed

---

## Success Criteria

- [ ] Gemma models load successfully
- [ ] Basic generation works
- [ ] Tokenizer works (or use GGUF-provided tokenizer)
- [ ] Performance reasonable (vs llama.cpp reference)
- [ ] Tests pass

---

## Files to Create/Modify

**New**:
- `src/architecture.h`
- `src/architecture.cpp`
- `docs/p3-2-gemma.md` (research notes)
- `docs/p3-2-qwen.md` (research notes)
- `tests/test_architecture.cpp` (tests)

**Modify**:
- `src/gguf.cpp` (remove hard check, add architecture dispatch)
- `src/model.cpp` (pass architecture to config struct)
- `CMakeLists.txt` (add new source files)

---

## Timeline

**Week 1**: Gemma implementation
- Research & design
- Implement architecture registry
- Add Gemma config parsing
- Test with Gemma 2B

**Week 2**: Qwen implementation
- Research & design
- Add Qwen config parsing
- Test with Qwen models

**Week 3**: Other models & cleanup
- Add remaining models
- Consolidate tests
- Update documentation

---

## Notes

- Gemma uses different RoPE: 2D rotary position embeddings
- Qwen has specific rotary_base (default 10000) and rotary_dim
- Many models share same attention type (GQA), so may be minimal changes
- Tokenizer: use GGUF's tokenizer if available, or map from tokenizer.ggml.*

---

## References

- [Gemma Paper](https://arxiv.org/abs/2408.00118)
- [Qwen Paper](https://arxiv.org/abs/2309.16609)
- [GGUF Specification](https://github.com/ggerganov/gguf)
- [llama.cpp Architecture Support](https://github.com/ggerganov/llama.cpp)

---

**Created**: 2026-05-19
**Last Updated**: 2026-05-19
