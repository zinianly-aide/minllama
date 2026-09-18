#include "minllama_internal.h"
#include "minicpm5_tokenizer.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void test_minicpm5_synthetic_contract() {
    minllama::BpeTokenizer tok;
    tok.vocab.resize(130560);
    tok.bos_token_id = 0;
    tok.eos_token_id = 1;
    tok.unk_token_id = 130074;
    tok.add_bos_token = false;
    tok.add_eos_token = false;
    minllama::bpe_build_byte_tables(tok);

    auto add = [&](int id, const std::string &token) {
        assert(id >= 0 && id < static_cast<int>(tok.vocab.size()));
        tok.vocab[id] = token;
        tok.token_to_id[token] = id;
    };

    add(0, "<s>");
    add(1, "</s>");
    add(100, "Hello");
    add(101, "\xC4\xA0world"); // GPT-2 byte encoding for " world"
    add(102, "123");
    add(103, "456");
    add(130070, "<|im_start|>");
    add(130073, "<|im_end|>");

    assert(minllama::is_minicpm5_tokenizer(tok));

    // MiniCPM5's first regex splits numeric runs into chunks of 1..3 digits.
    {
        std::vector<int> ids;
        assert(minllama::minicpm5_bpe_encode(tok, "123456", ids));
        assert((ids == std::vector<int>{102, 103}));
    }

    // llama.cpp sets ignore_merges=true for MiniCPM5: exact pre-tokenized
    // pieces win before ordinary BPE merge processing.
    {
        std::vector<int> ids;
        assert(minllama::minicpm5_bpe_encode(tok, "Hello world", ids));
        assert((ids == std::vector<int>{100, 101}));
    }

    // Chat-control tokens must remain atomic instead of being byte-tokenized.
    {
        std::vector<int> ids;
        assert(minllama::minicpm5_bpe_encode(
            tok, "<|im_start|>Hello<|im_end|>", ids));
        assert((ids == std::vector<int>{130070, 100, 130073}));
        assert(minllama::minicpm5_chat_stop_token_id(
                   tok, "<|im_start|>user\nhi<|im_end|>") == 130073);
        assert(minllama::minicpm5_chat_stop_token_id(tok, "plain prompt") == 1);
    }

    std::printf("MiniCPM5 synthetic tokenizer contract passed.\n");
}

} // namespace

int main() {
    // Always run this model-independent compatibility gate in CI, even when the
    // optional SmolLM fixture is not present in the checkout.
    test_minicpm5_synthetic_contract();

    const std::string model_path = "../models/SmolLM-135M.Q4_0.gguf";

    minllama::BpeTokenizer tok;
    std::string error;
    {
        bool ok = minllama::bpe_tokenizer_load(model_path, tok, &error);
        if (!ok) {
            std::fprintf(stderr, "skip bpe model load (synthetic MiniCPM5 gate already passed)\n");
            return 0;
        }
    }

    // Basic metadata checks
    assert(tok.vocab.size() == 49152);
    assert(tok.merges.size() == 48900);
    assert(tok.bos_token_id == 0);
    assert(tok.eos_token_id == 0);
    assert(tok.unk_token_id == 0);
    assert(!tok.add_bos_token);
    assert(!tok.add_eos_token);
    assert(!minllama::is_minicpm5_tokenizer(tok));

    // Byte encoder tables
    assert(tok.byte_encoder.size() == 256);
    assert(tok.byte_decoder.size() == 256);

    std::printf("=== Metadata ===\n");
    std::printf("vocab: %zu tokens, %zu merges\n", tok.vocab.size(), tok.merges.size());
    std::printf("bos=%d eos=%d unk=%d add_bos=%d\n",
                tok.bos_token_id, tok.eos_token_id, tok.unk_token_id,
                (int)tok.add_bos_token);

    // Test 1: "hello world"
    {
        std::vector<int> ids;
        assert(minllama::bpe_encode(tok, "hello world", ids));
        std::printf("\nTest 1: \"hello world\"\n");
        std::printf("  ids:");
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n  tokens:");
        for (int id : ids) std::printf(" '%s'", tok.vocab[id].c_str());
        std::printf("\n");

        // Expected from llama.cpp: [28120, 905] = ["hello"," world"]
        assert(ids.size() == 2);
        assert(ids[0] == 28120);
        assert(ids[1] == 905);

        std::string decoded;
        assert(minllama::bpe_decode(tok, ids.data(), (int)ids.size(), decoded));
        std::printf("  decoded: '%s'\n", decoded.c_str());
        assert(decoded == "hello world");
    }

    // Test 2: leading space " world"
    {
        std::vector<int> ids;
        assert(minllama::bpe_encode(tok, " world", ids));
        std::printf("\nTest 2: \" world\" (leading space)\n");
        std::printf("  ids:");
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n  tokens:");
        for (int id : ids) std::printf(" '%s'", tok.vocab[id].c_str());
        std::printf("\n");

        std::string decoded;
        assert(minllama::bpe_decode(tok, ids.data(), (int)ids.size(), decoded));
        std::printf("  decoded: '%s'\n", decoded.c_str());
        assert(decoded == " world");
    }

    // Test 3: punctuation "hello,"
    {
        std::vector<int> ids;
        assert(minllama::bpe_encode(tok, "hello,", ids));
        std::printf("\nTest 3: \"hello,\"\n");
        std::printf("  ids:");
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n  tokens:");
        for (int id : ids) std::printf(" '%s'", tok.vocab[id].c_str());
        std::printf("\n");

        std::string decoded;
        assert(minllama::bpe_decode(tok, ids.data(), (int)ids.size(), decoded));
        std::printf("  decoded: '%s'\n", decoded.c_str());
        assert(decoded == "hello,");
    }

    // Test 4: multiple spaces
    {
        std::vector<int> ids;
        assert(minllama::bpe_encode(tok, "hello  world", ids));
        std::printf("\nTest 4: \"hello  world\" (double space)\n");
        std::printf("  ids:");
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n  tokens:");
        for (int id : ids) std::printf(" '%s'", tok.vocab[id].c_str());
        std::printf("\n");

        std::string decoded;
        assert(minllama::bpe_decode(tok, ids.data(), (int)ids.size(), decoded));
        std::printf("  decoded: '%s'\n", decoded.c_str());
        assert(decoded == "hello  world");
    }

    // Test 5: unknown byte fallback
    {
        std::string text = "test\xFF";
        std::vector<int> ids;
        assert(minllama::bpe_encode(tok, text, ids));
        std::printf("\nTest 5: unknown byte fallback\n");
        std::printf("  ids:");
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n");
        assert(!ids.empty());
        assert(ids[0] == 2129); // "test"
    }

    // Test 6: empty string
    {
        std::vector<int> ids;
        assert(minllama::bpe_encode(tok, "", ids));
        assert(ids.empty());
        std::string decoded;
        assert(minllama::bpe_decode(tok, ids.data(), 0, decoded));
        assert(decoded.empty());
    }

    // Test 7: decode roundtrip for various texts
    {
        const char *tests[] = {
            "hello world",
            "Hello World",
            "12345",
            "a",
            " the ",
            "hello\nworld",
        };
        for (const char *t : tests) {
            std::vector<int> ids;
            assert(minllama::bpe_encode(tok, t, ids));
            std::string dec;
            assert(minllama::bpe_decode(tok, ids.data(), (int)ids.size(), dec));
            if (dec != t) {
                std::fprintf(stderr, "FAIL roundtrip: '%s' -> '%s'\n", t, dec.c_str());
                assert(false);
            }
        }
        std::printf("\nTest 7: roundtrip passed for %zu strings\n",
                    (size_t)(sizeof(tests) / sizeof(tests[0])));
    }

    std::printf("\nAll BPE tokenizer tests passed.\n");
    return 0;
}
