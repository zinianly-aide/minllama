#include "minllama_internal.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace minllama {

namespace {
constexpr std::array<unsigned char, 4> kGgufMagic = {'G', 'G', 'U', 'F'};
constexpr std::uint32_t kSupportedGgufVersion = 3;
constexpr std::uint64_t kMinimalHeaderSize = 24;
constexpr std::uint32_t kGgmlTypeF32 = 0;
constexpr std::uint32_t kGgmlTypeF16 = 1;
constexpr std::uint32_t kGgmlTypeQ4_0 = 2;
constexpr std::uint64_t kQ4_0BlockSize = 32;
constexpr std::uint64_t kQ4_0TypeSize = 18;

enum class GgufValueType : std::uint32_t {
    Uint32 = 4,
    Float32 = 6,
    String = 8,
    Array = 9,
    Uint64 = 10,
};

struct MetadataSeen {
    bool architecture = false;
    bool n_vocab = false;
    bool n_layer = false;
    bool n_embd = false;
    bool n_head = false;
    bool n_head_kv = false;
    bool n_ctx_train = false;
    bool rope_theta = false;
    bool rms_norm_eps = false;
};

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const unsigned char *>(&value) == 1;
}

std::uint32_t read_u32_le(const std::array<unsigned char, kMinimalHeaderSize> &bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t read_u64_le(const std::array<unsigned char, kMinimalHeaderSize> &bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8 * i);
    }
    return value;
}

bool read_exact(std::ifstream &file, void *data, std::size_t size) {
    file.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(size));
    return file.gcount() == static_cast<std::streamsize>(size);
}

bool read_u32_le(std::ifstream &file, std::uint32_t *out) {
    std::array<unsigned char, 4> bytes = {};
    if (!read_exact(file, bytes.data(), bytes.size())) {
        return false;
    }
    *out = static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_u16_le(std::ifstream &file, std::uint16_t *out) {
    std::array<unsigned char, 2> bytes = {};
    if (!read_exact(file, bytes.data(), bytes.size())) {
        return false;
    }
    *out = static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(bytes[1] << 8);
    return true;
}

bool read_u64_le(std::ifstream &file, std::uint64_t *out) {
    std::array<unsigned char, 8> bytes = {};
    if (!read_exact(file, bytes.data(), bytes.size())) {
        return false;
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    *out = value;
    return true;
}

bool seek_abs(std::ifstream &file, std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    return static_cast<bool>(file);
}

bool read_f32_le(std::ifstream &file, float *out) {
    std::uint32_t bits = 0;
    if (!read_u32_le(file, &bits)) {
        return false;
    }
    static_assert(sizeof(float) == sizeof(bits), "float32 is required");
    std::memcpy(out, &bits, sizeof(bits));
    return true;
}

bool read_gguf_string(std::ifstream &file, std::string *out) {
    std::uint64_t size = 0;
    if (!read_u64_le(file, &size)) {
        return false;
    }
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    std::string value(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !read_exact(file, value.data(), value.size())) {
        return false;
    }
    *out = value;
    return true;
}

bool read_config_u32(std::ifstream &file, GgufValueType type, std::uint32_t *out) {
    if (type == GgufValueType::Uint32) {
        return read_u32_le(file, out);
    }
    if (type == GgufValueType::Uint64) {
        std::uint64_t value = 0;
        if (!read_u64_le(file, &value) || value > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        *out = static_cast<std::uint32_t>(value);
        return true;
    }
    return false;
}

bool skip_value(std::ifstream &file, GgufValueType type, std::uint64_t *array_len = nullptr) {
    switch (type) {
    case GgufValueType::Uint32: {
        std::uint32_t ignored = 0;
        return read_u32_le(file, &ignored);
    }
    case GgufValueType::Uint64: {
        std::uint64_t ignored = 0;
        return read_u64_le(file, &ignored);
    }
    case GgufValueType::Float32: {
        float ignored = 0.0f;
        return read_f32_le(file, &ignored);
    }
    case GgufValueType::String: {
        std::string ignored;
        return read_gguf_string(file, &ignored);
    }
    case GgufValueType::Array: {
        std::uint32_t elem_type_raw = 0;
        std::uint64_t count = 0;
        if (!read_u32_le(file, &elem_type_raw) || !read_u64_le(file, &count)) {
            return false;
        }
        if (array_len) {
            *array_len = count;
        }
        const auto elem_type = static_cast<GgufValueType>(elem_type_raw);
        // Current stage intentionally supports only array[string], enough for
        // tokenizer.ggml.tokens. Other GGUF array types are rejected for now.
        if (elem_type != GgufValueType::String) {
            return false;
        }
        for (std::uint64_t i = 0; i < count; ++i) {
            std::string ignored;
            if (!read_gguf_string(file, &ignored)) {
                return false;
            }
        }
        return true;
    }
    }
    return false;
}

bool parse_metadata_value(std::ifstream &file,
                          const std::string &key,
                          GgufValueType type,
                          ModelConfig *config,
                          MetadataSeen *seen) {
    if (key == "general.architecture") {
        if (type != GgufValueType::String) {
            return false;
        }
        std::string architecture;
        if (!read_gguf_string(file, &architecture) || architecture != "llama") {
            return false;
        }
        seen->architecture = true;
        return true;
    }
    if (key == "llama.embedding_length") {
        seen->n_embd = true;
        return read_config_u32(file, type, &config->n_embd);
    }
    if (key == "llama.block_count") {
        seen->n_layer = true;
        return read_config_u32(file, type, &config->n_layer);
    }
    if (key == "llama.attention.head_count") {
        seen->n_head = true;
        return read_config_u32(file, type, &config->n_head);
    }
    if (key == "llama.attention.head_count_kv") {
        seen->n_head_kv = true;
        return read_config_u32(file, type, &config->n_head_kv);
    }
    if (key == "llama.context_length") {
        seen->n_ctx_train = true;
        return read_config_u32(file, type, &config->n_ctx_train);
    }
    if (key == "llama.rope.freq_base") {
        if (type != GgufValueType::Float32) {
            return false;
        }
        seen->rope_theta = true;
        return read_f32_le(file, &config->rope_theta);
    }
    if (key == "llama.attention.layer_norm_rms_epsilon") {
        if (type != GgufValueType::Float32) {
            return false;
        }
        seen->rms_norm_eps = true;
        return read_f32_le(file, &config->rms_norm_eps);
    }
    if (key == "tokenizer.ggml.tokens") {
        if (type != GgufValueType::Array) {
            return false;
        }
        std::uint64_t token_count = 0;
        if (!skip_value(file, type, &token_count) ||
            token_count > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        config->n_vocab = static_cast<std::uint32_t>(token_count);
        seen->n_vocab = true;
        return true;
    }

    // Tokenizer details are accepted only as metadata to skip at this stage;
    // tokenizer construction is deliberately deferred.
    return skip_value(file, type);
}

bool has_required_metadata(const MetadataSeen &seen) {
    return seen.architecture &&
           seen.n_vocab &&
           seen.n_layer &&
           seen.n_embd &&
           seen.n_head &&
           seen.n_head_kv &&
           seen.n_ctx_train &&
           seen.rope_theta &&
           seen.rms_norm_eps;
}

bool checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *out) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool checked_add_u64(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *out) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool tensor_element_count(const TensorInfo &info, std::uint64_t *out) {
    if (info.n_dims == 0 || info.n_dims > ML_MAX_TENSOR_DIMS) {
        return false;
    }

    std::uint64_t count = 1;
    for (std::uint32_t dim = 0; dim < info.n_dims; ++dim) {
        if (info.dims[dim] == 0 || !checked_mul_u64(count, info.dims[dim], &count)) {
            return false;
        }
    }

    *out = count;
    return true;
}

bool tensor_byte_size(const TensorInfo &info, std::uint64_t *out) {
    std::uint64_t elements = 0;
    if (!tensor_element_count(info, &elements)) {
        return false;
    }

    switch (info.gguf_type) {
    case kGgmlTypeF32:
        return checked_mul_u64(elements, 4, out);
    case kGgmlTypeF16:
        return checked_mul_u64(elements, 2, out);
    case kGgmlTypeQ4_0:
        if (elements % kQ4_0BlockSize != 0) {
            return false;
        }
        return checked_mul_u64(elements / kQ4_0BlockSize, kQ4_0TypeSize, out);
    default:
        return false;
    }
}

bool parse_tensor_infos(std::ifstream &file, std::uint64_t n_tensors, TensorIndex *out_tensor_index) {
    TensorIndex index;
    index.tensors.reserve(static_cast<std::size_t>(n_tensors));
    index.by_name.reserve(static_cast<std::size_t>(n_tensors));

    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        if (!read_gguf_string(file, &info.name) ||
            !read_u32_le(file, &info.n_dims) ||
            info.n_dims > ML_MAX_TENSOR_DIMS) {
            return false;
        }

        for (std::uint32_t dim = 0; dim < info.n_dims; ++dim) {
            if (!read_u64_le(file, &info.dims[dim])) {
                return false;
            }
        }

        if (!read_u32_le(file, &info.gguf_type) || !read_u64_le(file, &info.offset)) {
            return false;
        }

        if (index.by_name.find(info.name) != index.by_name.end()) {
            return false;
        }

        const std::size_t pos = index.tensors.size();
        index.tensors.push_back(std::move(info));
        index.by_name.emplace(index.tensors.back().name, pos);
    }

    *out_tensor_index = std::move(index);
    return true;
}

bool file_size(std::ifstream &file, std::uint64_t *out) {
    const std::streampos current = file.tellg();
    if (current < 0) {
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streampos end = file.tellg();
    if (end < 0) {
        return false;
    }
    file.seekg(current);
    *out = static_cast<std::uint64_t>(end);
    return true;
}

bool build_tensor_views(std::uint64_t data_offset, std::uint64_t file_bytes, TensorIndex *tensor_index) {
    tensor_index->views.clear();
    tensor_index->views.reserve(tensor_index->tensors.size());

    for (const TensorInfo &info : tensor_index->tensors) {
        std::uint64_t byte_size = 0;
        std::uint64_t data_begin = 0;
        std::uint64_t data_end = 0;
        if (!tensor_byte_size(info, &byte_size) ||
            !checked_add_u64(data_offset, info.offset, &data_begin) ||
            !checked_add_u64(data_begin, byte_size, &data_end) ||
            data_end > file_bytes) {
            return false;
        }

        tensor_index->views.push_back(TensorView{data_begin, byte_size, data_end});
    }

    return true;
}

bool find_tensor_for_read(const TensorIndex &tensor_index,
                          const std::string &name,
                          const TensorInfo **out_info,
                          const TensorView **out_view,
                          std::uint64_t *out_elements) {
    const TensorInfo *info = tensor_index.find(name);
    const TensorView *view = tensor_index.find_view(name);
    if (!info || !view) {
        return false;
    }

    std::uint64_t elements = 0;
    std::uint64_t expected_bytes = 0;
    if (!tensor_element_count(*info, &elements) ||
        !tensor_byte_size(*info, &expected_bytes) ||
        expected_bytes != view->byte_size ||
        view->data_end < view->data_begin) {
        return false;
    }

    *out_info = info;
    *out_view = view;
    *out_elements = elements;
    return true;
}

bool resize_float_output(std::uint64_t elements, std::vector<float> *out) {
    if (elements > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    out->assign(static_cast<std::size_t>(elements), 0.0f);
    return true;
}

bool load_tensor_q4_0_as_f32(const char *path,
                             const TensorIndex &tensor_index,
                             const std::string &name,
                             std::vector<float> *out) {
    const TensorInfo *info = nullptr;
    const TensorView *view = nullptr;
    std::uint64_t elements = 0;
    if (!find_tensor_for_read(tensor_index, name, &info, &view, &elements) ||
        info->gguf_type != kGgmlTypeQ4_0 ||
        elements % kQ4_0BlockSize != 0 ||
        view->byte_size != (elements / kQ4_0BlockSize) * kQ4_0TypeSize ||
        !resize_float_output(elements, out)) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file || !seek_abs(file, view->data_begin)) {
        return false;
    }

    // Correctness-first reference Q4_0 decode. ggml Q4_0 stores one fp16
    // scale followed by 16 bytes. Low nibbles decode elements 0..15, high
    // nibbles decode elements 16..31, with a -8 zero point.
    std::array<unsigned char, 16> packed = {};
    for (std::uint64_t block = 0; block < elements / kQ4_0BlockSize; ++block) {
        std::uint16_t scale_bits = 0;
        if (!read_u16_le(file, &scale_bits) || !read_exact(file, packed.data(), packed.size())) {
            out->clear();
            return false;
        }

        const float scale = ml_fp16_to_fp32(scale_bits);
        const std::size_t base = static_cast<std::size_t>(block * kQ4_0BlockSize);
        for (std::size_t i = 0; i < packed.size(); ++i) {
            const int low = static_cast<int>(packed[i] & 0x0fu) - 8;
            const int high = static_cast<int>((packed[i] >> 4) & 0x0fu) - 8;
            (*out)[base + i] = scale * static_cast<float>(low);
            (*out)[base + i + packed.size()] = scale * static_cast<float>(high);
        }
    }

    return true;
}
} // namespace

bool load_gguf_file_view(const char *path,
                         GgufFileView *out_view,
                         ModelConfig *out_config,
                         TensorIndex *out_tensor_index) {
    if (!path || !out_view || !out_config || !out_tensor_index || !host_is_little_endian()) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    std::array<unsigned char, kMinimalHeaderSize> bytes = {};
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        return false;
    }

    for (std::size_t i = 0; i < kGgufMagic.size(); ++i) {
        if (bytes[i] != kGgufMagic[i]) {
            return false;
        }
    }

    GgufFileView view;
    view.header.version = read_u32_le(bytes, 4);
    if (view.header.version != kSupportedGgufVersion) {
        return false;
    }
    view.header.n_tensors = read_u64_le(bytes, 8);
    view.header.n_kv = read_u64_le(bytes, 16);

    ModelConfig config;
    MetadataSeen seen;
    for (std::uint64_t i = 0; i < view.header.n_kv; ++i) {
        std::string key;
        std::uint32_t type_raw = 0;
        if (!read_gguf_string(file, &key) || !read_u32_le(file, &type_raw)) {
            return false;
        }
        if (!parse_metadata_value(file, key, static_cast<GgufValueType>(type_raw), &config, &seen)) {
            return false;
        }
    }

    if (!has_required_metadata(seen)) {
        return false;
    }

    TensorIndex tensor_index;
    if (!parse_tensor_infos(file, view.header.n_tensors, &tensor_index)) {
        return false;
    }

    // This stage builds only a minimal tensor-data boundary view. The tensor
    // bytes remain in the file and are not read, copied, or decoded here.
    const std::streampos tensor_infos_end = file.tellg();
    if (tensor_infos_end < 0) {
        return false;
    }
    view.data_offset = static_cast<std::uint64_t>(tensor_infos_end);

    std::uint64_t file_bytes = 0;
    if (!file_size(file, &file_bytes) || !build_tensor_views(view.data_offset, file_bytes, &tensor_index)) {
        return false;
    }

    *out_view = view;
    *out_config = config;
    *out_tensor_index = std::move(tensor_index);
    return true;
}

bool load_tensor_f32(const char *path,
                     const TensorIndex &tensor_index,
                     const std::string &name,
                     std::vector<float> *out) {
    if (!path || !out) {
        return false;
    }

    const TensorInfo *info = nullptr;
    const TensorView *view = nullptr;
    std::uint64_t elements = 0;
    if (!find_tensor_for_read(tensor_index, name, &info, &view, &elements) ||
        info->gguf_type != kGgmlTypeF32 ||
        !resize_float_output(elements, out)) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file || !seek_abs(file, view->data_begin)) {
        return false;
    }

    for (float &value : *out) {
        if (!read_f32_le(file, &value)) {
            out->clear();
            return false;
        }
    }

    return true;
}

bool load_tensor_as_f32(const char *path,
                        const TensorIndex &tensor_index,
                        const std::string &name,
                        std::vector<float> *out) {
    if (!path || !out) {
        return false;
    }

    const TensorInfo *info = nullptr;
    const TensorView *view = nullptr;
    std::uint64_t elements = 0;
    if (!find_tensor_for_read(tensor_index, name, &info, &view, &elements)) {
        return false;
    }

    if (info->gguf_type == kGgmlTypeF32) {
        return load_tensor_f32(path, tensor_index, name, out);
    }
    if (info->gguf_type == kGgmlTypeQ4_0) {
        return load_tensor_q4_0_as_f32(path, tensor_index, name, out);
    }
    if (info->gguf_type != kGgmlTypeF16 || !resize_float_output(elements, out)) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file || !seek_abs(file, view->data_begin)) {
        return false;
    }

    for (float &value : *out) {
        std::uint16_t bits = 0;
        if (!read_u16_le(file, &bits)) {
            out->clear();
            return false;
        }
        value = ml_fp16_to_fp32(bits);
    }

    return true;
}

}
