#include "minllama_internal.h"
#include "minicpm5_tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    std::string model_path;
    std::string text;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--text" && i + 1 < argc) {
            text = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: minllama_tokenize --model <gguf> --text <text>\n");
            return 0;
        }
    }

    if (model_path.empty() || text.empty()) {
        std::fprintf(stderr, "Usage: minllama_tokenize --model <gguf> --text <text>\n");
        return 1;
    }

    minllama::BpeTokenizer tok;
    std::string error;
    if (!minllama::bpe_tokenizer_load(model_path, tok, &error)) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        return 1;
    }

    const bool minicpm5_mode = minllama::is_minicpm5_tokenizer(tok);
    std::vector<int> ids;
    const bool encoded = minicpm5_mode
        ? minllama::minicpm5_bpe_encode(tok, text, ids)
        : minllama::bpe_encode(tok, text, ids);
    if (!encoded) {
        std::fprintf(stderr, "Error: encode failed\n");
        return 1;
    }

    std::printf("tokenizer_mode: %s\n", minicpm5_mode ? "minicpm5" : "gpt2-compatible");
    std::printf("token_ids:");
    for (int id : ids) std::printf(" %d", id);
    std::printf("\n");

    std::printf("tokens:");
    for (int id : ids) {
        if (id >= 0 && id < static_cast<int>(tok.vocab.size())) {
            std::printf(" %s", tok.vocab[id].c_str());
        } else {
            std::printf(" <OOB:%d>", id);
        }
    }
    std::printf("\n");

    std::string decoded;
    if (minllama::bpe_decode(tok, ids.data(), static_cast<int>(ids.size()), decoded)) {
        std::printf("decoded: %s\n", decoded.c_str());
    }

    return 0;
}
