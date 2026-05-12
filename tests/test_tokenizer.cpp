#include "minllama_internal.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    // ----------------------------------------------------------------
    // Test 1: init success — vocab with special tokens.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "hello", "world"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        assert(tok.id_to_token.size() == 5);
        assert(tok.id_to_token[0] == "<unk>");
        assert(tok.id_to_token[3] == "hello");

        assert(tok.token_to_id.at("<unk>") == 0);
        assert(tok.token_to_id.at("world") == 4);

        assert(tok.unk_token_id == 0);
        assert(tok.bos_token_id == 1);
        assert(tok.eos_token_id == 2);
    }

    // ----------------------------------------------------------------
    // Test 2: init — special token -1 means unused.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a", "b", "c"};

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, -1, 0, -1));

        assert(tok.unk_token_id == -1);
        assert(tok.bos_token_id == 0);
        assert(tok.eos_token_id == -1);
    }

    // ----------------------------------------------------------------
    // Test 3: init — empty vocab invalid.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab;

        minllama::SimpleTokenizer tok;
        assert(!minllama::tokenizer_init(tok, vocab, -1, -1, -1));
    }

    // ----------------------------------------------------------------
    // Test 4: init — duplicate token invalid.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a", "b", "a"};

        minllama::SimpleTokenizer tok;
        assert(!minllama::tokenizer_init(tok, vocab, -1, -1, -1));
    }

    // ----------------------------------------------------------------
    // Test 5: init — special token id out of range.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a", "b"};

        minllama::SimpleTokenizer tok;
        // unk_token_id = 2, but vocab size is 2
        assert(!minllama::tokenizer_init(tok, vocab, 2, -1, -1));
    }

    // ----------------------------------------------------------------
    // Test 6: encode whitespace — "hello world".
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "hello", "world"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "hello world", ids, false));

        assert(ids.size() == 2);
        assert(ids[0] == 3);  // hello
        assert(ids[1] == 4);  // world
    }

    // ----------------------------------------------------------------
    // Test 7: encode with BOS.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "hi"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "hi", ids, true));

        assert(ids.size() == 2);
        assert(ids[0] == 1);  // bos
        assert(ids[1] == 3);  // hi
    }

    // ----------------------------------------------------------------
    // Test 8: encode — unknown with unk_token_id.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "known"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "missing", ids, false));

        assert(ids.size() == 1);
        assert(ids[0] == 0);  // unk
    }

    // ----------------------------------------------------------------
    // Test 9: encode — unknown without unk returns false.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<eos>", "known"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, -1, -1, 0));

        std::vector<int> ids;
        assert(!minllama::tokenizer_encode_whitespace(tok, "missing", ids, false));
    }

    // ----------------------------------------------------------------
    // Test 10: encode — empty text.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "a"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        std::vector<int> ids;

        // Empty, no bos.
        assert(minllama::tokenizer_encode_whitespace(tok, "", ids, false));
        assert(ids.empty());

        // Empty, with bos.
        assert(minllama::tokenizer_encode_whitespace(tok, "", ids, true));
        assert(ids.size() == 1);
        assert(ids[0] == 1);  // bos
    }

    // ----------------------------------------------------------------
    // Test 11: encode — whitespace-only.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "a"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "   ", ids, false));
        assert(ids.empty());
    }

    // ----------------------------------------------------------------
    // Test 12: encode — multiple spaces between words.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "a", "b"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "a   b", ids, false));

        assert(ids.size() == 2);
        assert(ids[0] == 3);  // a
        assert(ids[1] == 4);  // b
    }

    // ----------------------------------------------------------------
    // Test 13: decode — skip bos/eos, join with space.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "<bos>", "<eos>", "hello", "world"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, 1, 2));

        const int token_ids[] = {1, 3, 4, 2};  // bos, hello, world, eos
        std::string out;
        assert(minllama::tokenizer_decode_tokens(tok, token_ids, 4, out));
        assert(out == "hello world");
    }

    // ----------------------------------------------------------------
    // Test 14: decode — empty token list.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a", "b"};

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, -1, -1, -1));

        std::string out;
        assert(minllama::tokenizer_decode_tokens(tok, nullptr, 0, out));
        assert(out.empty());
    }

    // ----------------------------------------------------------------
    // Test 15: decode — invalid id (out of range).
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a", "b"};

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, -1, -1, -1));

        const int token_ids[] = {0, 2, 1};  // 2 is out of range
        std::string out;
        assert(!minllama::tokenizer_decode_tokens(tok, token_ids, 3, out));
    }

    // ----------------------------------------------------------------
    // Test 16: decode — negative id.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a", "b"};

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, -1, -1, -1));

        const int token_ids[] = {-1};
        std::string out;
        assert(!minllama::tokenizer_decode_tokens(tok, token_ids, 1, out));
    }

    // ----------------------------------------------------------------
    // Test 17: decode — nullptr token_ids.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {"a"};

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, -1, -1, -1));

        std::string out;
        assert(!minllama::tokenizer_decode_tokens(tok, nullptr, 1, out));
    }

    // ----------------------------------------------------------------
    // Test 18: encode with add_bos but bos_token_id=-1.
    // ----------------------------------------------------------------
    {
        const std::vector<std::string> vocab = {
            "<unk>", "hi"
        };

        minllama::SimpleTokenizer tok;
        assert(minllama::tokenizer_init(tok, vocab, 0, -1, -1));

        std::vector<int> ids;
        assert(minllama::tokenizer_encode_whitespace(tok, "hi", ids, true));
        // bos is -1, so no bos added
        assert(ids.size() == 1);
        assert(ids[0] == 1);  // hi
    }

    return 0;
}
