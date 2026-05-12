#include "minllama_internal.h"

#include <cstring>
#include <fstream>
#include <limits>

int ml_tokenize(ml_model *model, const char *text, ml_token *out_tokens, size_t capacity) {
    if (!model || !text) {
        return ML_ERROR_INVALID_ARGUMENT;
    }
    if (text[0] == '\0') {
        return 0;
    }
    if (!out_tokens || capacity == 0) {
        return ML_ERROR_BUFFER_TOO_SMALL;
    }

    // Placeholder tokenizer: any non-empty input maps to one stable sentinel.
    out_tokens[0] = ML_SKELETON_TOKEN;
    return 1;
}

int ml_detokenize(ml_model *model, ml_token token, char *out_text, size_t capacity) {
    if (!model || !out_text) {
        return ML_ERROR_INVALID_ARGUMENT;
    }
    if (token != ML_SKELETON_TOKEN) {
        return ML_ERROR_NOT_IMPLEMENTED;
    }

    const size_t required = std::strlen(ML_SKELETON_TEXT) + 1;
    if (capacity < required) {
        return ML_ERROR_BUFFER_TOO_SMALL;
    }

    std::memcpy(out_text, ML_SKELETON_TEXT, required);
    return static_cast<int>(required - 1);
}

namespace minllama {

bool tokenizer_init(SimpleTokenizer &tokenizer,
                    const std::vector<std::string> &vocab,
                    int unk_token_id,
                    int bos_token_id,
                    int eos_token_id) {
    if (vocab.empty()) {
        return false;
    }

    const int n = static_cast<int>(vocab.size());

    // Validate special token IDs.
    auto valid_special = [n](int id) {
        return id == -1 || (id >= 0 && id < n);
    };
    if (!valid_special(unk_token_id) ||
        !valid_special(bos_token_id) ||
        !valid_special(eos_token_id)) {
        return false;
    }

    tokenizer.id_to_token = vocab;

    tokenizer.token_to_id.clear();
    for (int i = 0; i < n; ++i) {
        const auto &token = vocab[i];
        if (tokenizer.token_to_id.find(token) != tokenizer.token_to_id.end()) {
            // Duplicate token.
            return false;
        }
        tokenizer.token_to_id[token] = i;
    }

    tokenizer.unk_token_id = unk_token_id;
    tokenizer.bos_token_id = bos_token_id;
    tokenizer.eos_token_id = eos_token_id;

    return true;
}

namespace {

bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

} // namespace

bool tokenizer_encode_whitespace(const SimpleTokenizer &tokenizer,
                                 const std::string &text,
                                 std::vector<int> &output_ids,
                                 bool add_bos) {
    output_ids.clear();

    // Handle BOS.
    if (add_bos && tokenizer.bos_token_id >= 0) {
        output_ids.push_back(tokenizer.bos_token_id);
    }

    if (text.empty()) {
        return true;
    }

    // Split by whitespace.
    const std::size_t len = text.size();
    std::size_t i = 0;

    while (i < len) {
        // Skip leading whitespace.
        while (i < len && is_whitespace(text[i])) {
            ++i;
        }
        if (i >= len) {
            break;
        }

        // Extract token.
        std::size_t start = i;
        while (i < len && !is_whitespace(text[i])) {
            ++i;
        }

        std::string word(text.data() + start, i - start);

        auto it = tokenizer.token_to_id.find(word);
        if (it != tokenizer.token_to_id.end()) {
            output_ids.push_back(it->second);
        } else {
            if (tokenizer.unk_token_id >= 0) {
                output_ids.push_back(tokenizer.unk_token_id);
            } else {
                return false;
            }
        }
    }

    return true;
}

bool tokenizer_decode_tokens(const SimpleTokenizer &tokenizer,
                             const int *token_ids,
                             int token_count,
                             std::string &output_text) {
    if (token_count < 0) {
        return false;
    }

    output_text.clear();

    if (token_count == 0) {
        return true;
    }

    if (!token_ids) {
        return false;
    }

    const int vocab_size = static_cast<int>(tokenizer.id_to_token.size());
    if (vocab_size == 0) {
        return false;
    }

    for (int i = 0; i < token_count; ++i) {
        const int tid = token_ids[i];
        if (tid < 0 || tid >= vocab_size) {
            return false;
        }

        // Skip special tokens (bos/eos).
        if (tid == tokenizer.bos_token_id || tid == tokenizer.eos_token_id) {
            continue;
        }

        if (!output_text.empty()) {
            output_text += ' ';
        }
        output_text += tokenizer.id_to_token[tid];
    }

    return true;
}

bool minllama_generate_text_greedy_f32(TransformerModelF32 &model,
                                       const SimpleTokenizer &tokenizer,
                                       const std::string &prompt,
                                       int max_new_tokens,
                                       std::string &output_text) {
    if (max_new_tokens < 0) {
        return false;
    }

    output_text.clear();

    // 1. Encode prompt.
    std::vector<int> prompt_ids;
    if (!tokenizer_encode_whitespace(tokenizer, prompt, prompt_ids, true)) {
        return false;
    }

    // 2. Generate.
    std::vector<int> output_ids(max_new_tokens > 0 ? max_new_tokens : 1);
    int output_len = 0;
    if (!transformer_model_generate_greedy_f32(
            model, prompt_ids.data(),
            static_cast<int>(prompt_ids.size()),
            max_new_tokens,
            output_ids.data(),
            static_cast<int>(output_ids.size()),
            &output_len,
            tokenizer.eos_token_id)) {
        return false;
    }

    // 3. Decode.
    if (output_len > 0) {
        if (!tokenizer_decode_tokens(tokenizer, output_ids.data(), output_len,
                                     output_text)) {
            return false;
        }
    }

    return true;
}

bool minllama_generate_text_sample_f32(TransformerModelF32 &model,
                                       const SimpleTokenizer &tokenizer,
                                       const std::string &prompt,
                                       int max_new_tokens,
                                       float temperature,
                                       uint32_t seed,
                                       std::string &output_text,
                                       int top_k,
                                       float top_p) {
    if (max_new_tokens < 0) return false;

    output_text.clear();

    // 1. Encode prompt.
    std::vector<int> prompt_ids;
    if (!tokenizer_encode_whitespace(tokenizer, prompt, prompt_ids, true))
        return false;

    // 2. Generate with sampling.
    uint32_t rng_state = seed;
    std::vector<int> output_ids(max_new_tokens > 0 ? max_new_tokens : 1);
    int output_len = 0;
    if (!transformer_model_generate_sample_f32(
            model, prompt_ids.data(),
            static_cast<int>(prompt_ids.size()),
            max_new_tokens,
            output_ids.data(),
            static_cast<int>(output_ids.size()),
            &output_len,
            tokenizer.eos_token_id,
            temperature,
            &rng_state,
            top_k, top_p))
        return false;

    // 3. Decode.
    if (output_len > 0) {
        if (!tokenizer_decode_tokens(tokenizer, output_ids.data(), output_len,
                                     output_text)) {
            return false;
        }
    }

    return true;
}

namespace {

// Low-level GGUF metadata readers (reused from gguf.cpp pattern).

bool read_gguf_u32(std::ifstream &file, std::uint32_t *out) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char *>(bytes), 4);
    if (file.gcount() != 4) return false;
    *out = static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_gguf_u64(std::ifstream &file, std::uint64_t *out) {
    unsigned char bytes[8];
    file.read(reinterpret_cast<char *>(bytes), 8);
    if (file.gcount() != 8) return false;
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    *out = v;
    return true;
}

bool read_gguf_string(std::ifstream &file, std::string *out) {
    std::uint64_t size = 0;
    if (!read_gguf_u64(file, &size)) return false;
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return false;
    std::string s(static_cast<std::size_t>(size), '\0');
    if (size > 0) {
        file.read(s.data(), static_cast<std::streamsize>(size));
        if (file.gcount() != static_cast<std::streamsize>(size)) return false;
    }
    *out = s;
    return true;
}

bool skip_gguf_value(std::ifstream &file, std::uint32_t type) {
    constexpr std::uint32_t kUint32 = 4;
    constexpr std::uint32_t kFloat32 = 6;
    constexpr std::uint32_t kString = 8;
    constexpr std::uint32_t kArray = 9;
    constexpr std::uint32_t kUint64 = 10;

    switch (type) {
    case kUint32: {
        std::uint32_t ignored;
        return read_gguf_u32(file, &ignored);
    }
    case kUint64: {
        std::uint64_t ignored;
        return read_gguf_u64(file, &ignored);
    }
    case kFloat32: {
        std::uint32_t ignored;
        return read_gguf_u32(file, &ignored);
    }
    case kString: {
        std::string ignored;
        return read_gguf_string(file, &ignored);
    }
    case kArray: {
        std::uint32_t elem_type = 0;
        std::uint64_t count = 0;
        if (!read_gguf_u32(file, &elem_type) || !read_gguf_u64(file, &count))
            return false;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (!skip_gguf_value(file, elem_type)) return false;
        }
        return true;
    }
    default:
        return false;
    }
}

} // namespace

bool load_simple_tokenizer_from_gguf(const ml_model &src,
                                     SimpleTokenizer &tokenizer,
                                     std::string *error) {
    auto set_error = [error](const std::string &msg) {
        if (error) *error = msg;
    };

    std::ifstream file(src.path, std::ios::binary);
    if (!file) {
        set_error("Cannot open GGUF file");
        return false;
    }

    // Read header.
    char magic[4];
    file.read(magic, 4);
    if (file.gcount() != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
        set_error("Invalid GGUF magic");
        return false;
    }

    std::uint32_t version = 0;
    if (!read_gguf_u32(file, &version) || version != 3) {
        set_error("Unsupported GGUF version");
        return false;
    }

    std::uint64_t n_tensors = 0, n_kv = 0;
    if (!read_gguf_u64(file, &n_tensors) || !read_gguf_u64(file, &n_kv)) {
        set_error("Failed to read GGUF header");
        return false;
    }

    // Scan KV pairs for tokenizer info.
    std::vector<std::string> tokens;
    int unk_id = -1, bos_id = -1, eos_id = -1;
    bool has_tokens = false, has_unk = false, has_bos = false, has_eos = false;

    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        std::uint32_t type_raw = 0;
        if (!read_gguf_string(file, &key) || !read_gguf_u32(file, &type_raw)) {
            set_error("Failed to read KV pair");
            return false;
        }

        if (key == "tokenizer.ggml.tokens") {
            if (type_raw != 9) { // Array
                set_error("tokenizer.ggml.tokens is not an array");
                return false;
            }
            std::uint32_t elem_type = 0;
            std::uint64_t count = 0;
            if (!read_gguf_u32(file, &elem_type) || !read_gguf_u64(file, &count)) {
                set_error("Failed to read tokenizer.ggml.tokens header");
                return false;
            }
            if (elem_type != 8) { // String
                set_error("tokenizer.ggml.tokens element type is not string");
                return false;
            }
            tokens.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t j = 0; j < count; ++j) {
                std::string token;
                if (!read_gguf_string(file, &token)) {
                    set_error("Failed to read token at index " + std::to_string(j));
                    return false;
                }
                tokens.push_back(std::move(token));
            }
            has_tokens = true;
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            if (type_raw != 4) { // Uint32
                set_error("tokenizer.ggml.unknown_token_id is not uint32");
                return false;
            }
            std::uint32_t val = 0;
            if (!read_gguf_u32(file, &val)) {
                set_error("Failed to read unknown_token_id");
                return false;
            }
            unk_id = static_cast<int>(val);
            has_unk = true;
        } else if (key == "tokenizer.ggml.bos_token_id") {
            if (type_raw != 4) { // Uint32
                set_error("tokenizer.ggml.bos_token_id is not uint32");
                return false;
            }
            std::uint32_t val = 0;
            if (!read_gguf_u32(file, &val)) {
                set_error("Failed to read bos_token_id");
                return false;
            }
            bos_id = static_cast<int>(val);
            has_bos = true;
        } else if (key == "tokenizer.ggml.eos_token_id") {
            if (type_raw != 4) { // Uint32
                set_error("tokenizer.ggml.eos_token_id is not uint32");
                return false;
            }
            std::uint32_t val = 0;
            if (!read_gguf_u32(file, &val)) {
                set_error("Failed to read eos_token_id");
                return false;
            }
            eos_id = static_cast<int>(val);
            has_eos = true;
        } else {
            // Skip other KV pairs.
            if (!skip_gguf_value(file, type_raw)) {
                set_error("Failed to skip KV: " + key);
                return false;
            }
        }
    }

    if (!has_tokens) {
        set_error("tokenizer.ggml.tokens not found");
        return false;
    }

    if (!tokenizer_init(tokenizer, tokens, unk_id, bos_id, eos_id)) {
        set_error("tokenizer_init failed (empty, duplicate, or bad special ids)");
        return false;
    }

    return true;
}

} // namespace minllama
