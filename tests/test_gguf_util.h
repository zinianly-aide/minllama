#ifndef MINLLAMA_TEST_GGUF_UTIL_H
#define MINLLAMA_TEST_GGUF_UTIL_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace minllama_test {

enum class GgufValueType : std::uint32_t {
    Uint32 = 4,
    Float32 = 6,
    String = 8,
    Array = 9,
    Uint64 = 10,
};

inline void write_u32_le(std::ofstream &out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.put(static_cast<char>((value >> (8 * i)) & 0xffu));
    }
}

inline void write_u64_le(std::ofstream &out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.put(static_cast<char>((value >> (8 * i)) & 0xffu));
    }
}

inline void write_f32_le(std::ofstream &out, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float32 is required");
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32_le(out, bits);
}

inline void write_f16_le(std::ofstream &out, std::uint16_t value) {
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8) & 0xffu));
}

inline void write_gguf_string(std::ofstream &out, const std::string &value) {
    write_u64_le(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

inline void write_kv_header(std::ofstream &out, const std::string &key, GgufValueType type) {
    write_gguf_string(out, key);
    write_u32_le(out, static_cast<std::uint32_t>(type));
}

inline void write_metadata_string(std::ofstream &out, const std::string &key, const std::string &value) {
    write_kv_header(out, key, GgufValueType::String);
    write_gguf_string(out, value);
}

inline void write_metadata_u32(std::ofstream &out, const std::string &key, std::uint32_t value) {
    write_kv_header(out, key, GgufValueType::Uint32);
    write_u32_le(out, value);
}

inline void write_metadata_f32(std::ofstream &out, const std::string &key, float value) {
    write_kv_header(out, key, GgufValueType::Float32);
    write_f32_le(out, value);
}

inline void write_metadata_array_string(std::ofstream &out,
                                        const std::string &key,
                                        const std::vector<std::string> &values) {
    write_kv_header(out, key, GgufValueType::Array);
    write_u32_le(out, static_cast<std::uint32_t>(GgufValueType::String));
    write_u64_le(out, static_cast<std::uint64_t>(values.size()));
    for (const std::string &value : values) {
        write_gguf_string(out, value);
    }
}

struct TensorInfoSpec {
    std::string name;
    std::uint32_t n_dims = 0;
    std::vector<std::uint64_t> dims;
    std::uint32_t gguf_type = 0;
    std::uint64_t offset = 0;
};

inline void write_llama_metadata(std::ofstream &out, const std::string &architecture) {
    write_metadata_string(out, "general.architecture", architecture);
    write_metadata_u32(out, "llama.embedding_length", 4096);
    write_metadata_u32(out, "llama.block_count", 32);
    write_metadata_u32(out, "llama.attention.head_count", 32);
    write_metadata_u32(out, "llama.attention.head_count_kv", 32);
    write_metadata_u32(out, "llama.context_length", 4096);
    write_metadata_f32(out, "llama.rope.freq_base", 10000.0f);
    write_metadata_f32(out, "llama.attention.layer_norm_rms_epsilon", 0.000001f);
    write_metadata_array_string(out, "tokenizer.ggml.tokens", {"<unk>", "<s>", "</s>"});
    write_metadata_string(out, "tokenizer.ggml.model", "llama");
    write_metadata_u32(out, "tokenizer.ggml.bos_token_id", 1);
    write_metadata_u32(out, "tokenizer.ggml.eos_token_id", 2);
}

inline void write_tensor_info(std::ofstream &out, const TensorInfoSpec &tensor) {
    write_gguf_string(out, tensor.name);
    write_u32_le(out, tensor.n_dims);
    for (std::uint32_t i = 0; i < tensor.n_dims; ++i) {
        write_u64_le(out, tensor.dims[i]);
    }
    write_u32_le(out, tensor.gguf_type);
    write_u64_le(out, tensor.offset);
}

inline void write_tensor_data_padding(std::ofstream &out, std::uint64_t size) {
    constexpr char zero = '\0';
    for (std::uint64_t i = 0; i < size; ++i) {
        out.put(zero);
    }
}

inline void write_alignment_padding(std::ofstream &out, std::uint32_t alignment = 32) {
    const std::streamoff pos = out.tellp();
    if (pos < 0 || alignment == 0) {
        return;
    }
    const std::uint64_t rem = static_cast<std::uint64_t>(pos) % alignment;
    if (rem == 0) {
        return;
    }
    write_tensor_data_padding(out, alignment - rem);
}

inline void write_tensor_payload_f32(std::ofstream &out, const std::vector<float> &values) {
    for (float value : values) {
        write_f32_le(out, value);
    }
}

inline void write_tensor_payload_f16(std::ofstream &out, const std::vector<std::uint16_t> &values) {
    for (std::uint16_t value : values) {
        write_f16_le(out, value);
    }
}

inline bool write_tensor_payload_q4_0_block(std::ofstream &out,
                                            std::uint16_t scale_f16,
                                            const std::vector<int> &quantized_values) {
    if (quantized_values.size() != 32) {
        return false;
    }

    write_f16_le(out, scale_f16);
    // GGML Q4_0: byte[i] = [high_nibble = value[i*2] | low_nibble = value[i*2+1]]
    for (std::size_t i = 0; i < 16; ++i) {
        const int high = quantized_values[i * 2] + 8;
        const int low  = quantized_values[i * 2 + 1] + 8;
        if (low < 0 || low > 15 || high < 0 || high > 15) {
            return false;
        }
        out.put(static_cast<char>((high << 4) | low));
    }
    return static_cast<bool>(out);
}

inline bool write_tensor_payload_q8_0_block(std::ofstream &out,
                                            std::uint16_t scale_f16,
                                            const std::vector<int> &quantized_values) {
    if (quantized_values.size() != 32) {
        return false;
    }
    write_f16_le(out, scale_f16);
    for (int q : quantized_values) {
        if (q < -128 || q > 127) {
            return false;
        }
        const std::int8_t v = static_cast<std::int8_t>(q);
        out.put(static_cast<char>(v));
    }
    return static_cast<bool>(out);
}

inline bool write_fake_gguf_header(const std::string &path,
                                   const char magic[4] = "GGUF",
                                   std::uint32_t version = 3,
                                   std::uint64_t n_tensors = 0,
                                   std::uint64_t n_kv = 0) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(magic, 4);
    write_u32_le(out, version);
    write_u64_le(out, n_tensors);
    write_u64_le(out, n_kv);
    return static_cast<bool>(out);
}

inline bool write_fake_llama_gguf(const std::string &path,
                                  const std::string &architecture = "llama") {
    constexpr std::uint64_t n_kv = 12;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write("GGUF", 4);
    write_u32_le(out, 3);
    write_u64_le(out, 0);
    write_u64_le(out, n_kv);

    write_llama_metadata(out, architecture);
    return static_cast<bool>(out);
}

inline bool write_fake_llama_gguf_with_tensors(const std::string &path,
                                               const std::vector<TensorInfoSpec> &tensors,
                                               const std::string &architecture = "llama",
                                               std::uint64_t tensor_data_padding = 0) {
    constexpr std::uint64_t n_kv = 12;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write("GGUF", 4);
    write_u32_le(out, 3);
    write_u64_le(out, static_cast<std::uint64_t>(tensors.size()));
    write_u64_le(out, n_kv);

    write_llama_metadata(out, architecture);
    for (const TensorInfoSpec &tensor : tensors) {
        write_tensor_info(out, tensor);
    }
    write_alignment_padding(out, 32);
    write_tensor_data_padding(out, tensor_data_padding);
    return static_cast<bool>(out);
}

} // namespace minllama_test

#endif
