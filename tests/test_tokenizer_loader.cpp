#include "minllama.h"
#include "minllama_internal.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Write a minimal GGUF file with custom tokenizer KV pairs only.
// Returns false if the file can't be created.
bool write_tokenizer_gguf(
    const std::string &path,
    const std::vector<std::string> &tokens,
    int unknown_id = -1,
    int bos_id = -1,
    int eos_id = -1,
    bool include_unknown = true,
    bool include_bos = true,
    bool include_eos = true)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    // Count KV pairs.
    int n_kv = 1; // tokenizer.ggml.tokens
    if (include_unknown) ++n_kv;
    if (include_bos) ++n_kv;
    if (include_eos) ++n_kv;

    // Header: magic + version + n_tensors + n_kv.
    out.write("GGUF", 4);
    minllama_test::write_u32_le(out, 3);
    minllama_test::write_u64_le(out, 0); // n_tensors = 0
    minllama_test::write_u64_le(out, static_cast<std::uint64_t>(n_kv));

    // Write tokenizer.ggml.tokens.
    minllama_test::write_metadata_array_string(out, "tokenizer.ggml.tokens", tokens);

    if (include_unknown) {
        minllama_test::write_metadata_u32(out, "tokenizer.ggml.unknown_token_id",
                                          static_cast<std::uint32_t>(unknown_id));
    }
    if (include_bos) {
        minllama_test::write_metadata_u32(out, "tokenizer.ggml.bos_token_id",
                                          static_cast<std::uint32_t>(bos_id));
    }
    if (include_eos) {
        minllama_test::write_metadata_u32(out, "tokenizer.ggml.eos_token_id",
                                          static_cast<std::uint32_t>(eos_id));
    }

    return static_cast<bool>(out);
}

} // namespace

int main() {
    // ----------------------------------------------------------------
    // Test 1: load tokenizer success.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_success.gguf";
        std::remove(path);

        const std::vector<std::string> tokens = {
            "<unk>", "<s>", "</s>", "hello", "world"
        };
        bool ok = write_tokenizer_gguf(path, tokens, 0, 1, 2);
        assert(ok);

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        {
            bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
            assert(loaded);
        }
        assert(error.empty());

        assert(tok.id_to_token.size() == 5);
        assert(tok.id_to_token[0] == "<unk>");
        assert(tok.id_to_token[3] == "hello");
        assert(tok.unk_token_id == 0);
        assert(tok.bos_token_id == 1);
        assert(tok.eos_token_id == 2);

        // Verify encode/decode works.
        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "hello world", ids, false));
        assert(ids.size() == 2);
        assert(ids[0] == 3);
        assert(ids[1] == 4);

        std::string out;
        assert(minllama::tokenizer_decode_tokens(tok, ids.data(), 2, out));
        assert(out == "hello world");

        std::remove(path);
    }

    // ----------------------------------------------------------------
    // Test 2: missing tokenizer.ggml.tokens.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_missing.gguf";
        std::remove(path);

        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write("GGUF", 4);
            minllama_test::write_u32_le(out, 3);
            minllama_test::write_u64_le(out, 0);
            minllama_test::write_u64_le(out, 1);
            // Write a non-tokenizer KV.
            minllama_test::write_metadata_string(out, "general.architecture", "llama");
        }

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
        assert(!loaded);
        assert(!error.empty());

        std::remove(path);
    }

    // ----------------------------------------------------------------
    // Test 3: empty tokens.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_empty.gguf";
        std::remove(path);

        const std::vector<std::string> tokens; // empty
        bool ok = write_tokenizer_gguf(path, tokens);
        assert(ok);

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
        assert(!loaded);
        assert(!error.empty());

        std::remove(path);
    }

    // ----------------------------------------------------------------
    // Test 4: duplicate tokens.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_dup.gguf";
        std::remove(path);

        const std::vector<std::string> tokens = {"a", "a"};
        bool ok = write_tokenizer_gguf(path, tokens);
        assert(ok);

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
        assert(!loaded);
        assert(!error.empty());

        std::remove(path);
    }

    // ----------------------------------------------------------------
    // Test 5: special token id out of range.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_oob.gguf";
        std::remove(path);

        const std::vector<std::string> tokens = {"a", "b", "c"};
        bool ok = write_tokenizer_gguf(path, tokens, -1, 99, -1);
        assert(ok); // bos=99, out of range

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
        assert(!loaded);
        assert(!error.empty());

        std::remove(path);
    }

    // ----------------------------------------------------------------
    // Test 6: optional special ids — all -1 when not provided.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_optional.gguf";
        std::remove(path);

        const std::vector<std::string> tokens = {"hello", "world", "!"};
        // Don't provide unknown/bos/eos at all.
        bool ok = write_tokenizer_gguf(path, tokens, -1, -1, -1,
                                     false, false, false);
        assert(ok);

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        {
            bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
            assert(loaded);
        }
        assert(error.empty());

        assert(tok.unk_token_id == -1);
        assert(tok.bos_token_id == -1);
        assert(tok.eos_token_id == -1);
        assert(tok.id_to_token.size() == 3);

        std::remove(path);
    }

    // ----------------------------------------------------------------
    // Test 7: only some special ids provided.
    // ----------------------------------------------------------------
    {
        const char *path = "test_tok_load_partial.gguf";
        std::remove(path);

        const std::vector<std::string> tokens = {"<unk>", "hello"};
        // Provide unk but not bos/eos.
        bool ok = write_tokenizer_gguf(path, tokens, 0, -1, -1,
                                     true, false, false);
        assert(ok);

        ml_model ml;
        ml.path = path;

        minllama::SimpleTokenizer tok;
        std::string error;
        {
            bool loaded = minllama::load_simple_tokenizer_from_gguf(ml, tok, &error);
            assert(loaded);
        }
        assert(error.empty());

        assert(tok.unk_token_id == 0);
        assert(tok.bos_token_id == -1);
        assert(tok.eos_token_id == -1);

        std::remove(path);
    }

    return 0;
}
