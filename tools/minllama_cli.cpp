#include "minllama.h"
#include "minllama_internal.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

bool parse_prompt_tokens_csv(const std::string &text, std::vector<int> *out) {
    if (!out || text.empty()) {
        return false;
    }
    out->clear();
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            return false;
        }
        char *end = nullptr;
        long v = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || v < 0 || v > 2147483647L) {
            return false;
        }
        out->push_back(static_cast<int>(v));
    }
    return !out->empty();
}

} // namespace

int main(int argc, const char **argv) {
    minllama::CliOptions opts;
    std::string error;

    if (!minllama::parse_cli_args(argc, argv, opts, &error)) {
        std::cerr << "Error: " << error << "\n";
        std::cerr << "Usage: minllama_cli --model <path> (--prompt <text> | --prompt-tokens <ids>) [--max-new-tokens <n>] [--help]\n";
        return 1;
    }

    if (opts.help) {
        std::cout << "Usage: minllama_cli --model <path> (--prompt <text> | --prompt-tokens <ids>) [--max-new-tokens <n>] [--temperature <t>] [--seed <n>] [--debug-tokens] [--threads <1|2>] [--help]\n";
        std::cout << "\n";
        std::cout << "Options:\n";
        std::cout << "  --model <path>          Path to GGUF model file (required)\n";
        std::cout << "  --prompt <text>         Prompt text (uses BPE tokenizer)\n";
        std::cout << "  --prompt-tokens <ids>   Comma-separated token ids, bypass tokenizer\n";
        std::cout << "  --max-new-tokens <n>    Max tokens to generate (default: 16)\n";
        std::cout << "  --temperature <float>   Sampling temperature (default: 0 = greedy)\n";
        std::cout << "  --seed <uint32>         RNG seed (default: 1)\n";
        std::cout << "  --top-k <int>           Top-K sampling (default: 0 = disabled)\n";
        std::cout << "  --top-p <float>         Top-P / nucleus sampling (default: 1.0 = disabled)\n";
        std::cout << "  --threads <1|2>         Parallel matvec threads (default: 1, only 1 or 2 supported)\n";
        std::cout << "  --debug-tokens          Print tokenizer/token-id debug info\n";
        std::cout << "  --help                  Show this help\n";
        return 0;
    }

    if (opts.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }
    if (opts.prompt.empty() && opts.prompt_tokens_raw.empty()) {
        std::cerr << "Error: one of --prompt or --prompt-tokens is required\n";
        return 1;
    }
    if (!opts.prompt.empty() && !opts.prompt_tokens_raw.empty()) {
        std::cerr << "Error: --prompt and --prompt-tokens are mutually exclusive\n";
        return 1;
    }
    if (opts.max_new_tokens < 0) {
        std::cerr << "Error: --max-new-tokens must be >= 0\n";
        return 1;
    }

    // Load model metadata.
    ml_model *ml = ml_model_load(opts.model_path.c_str());
    if (!ml) {
        std::cerr << "Error: failed to load model from " << opts.model_path << "\n";
        return 1;
    }

    minllama::TransformerModelF32 model;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::cerr << "Error: " << error << "\n";
        ml_model_free(ml);
        return 1;
    }
    model.n_threads = opts.n_threads;

    // Load BPE tokenizer for text prompts.
    minllama::BpeTokenizer bpe_tok;
    if (!minllama::bpe_tokenizer_load(opts.model_path, bpe_tok, &error)) {
        std::cerr << "Error: failed to load BPE tokenizer: " << error << "\n";
        ml_model_free(ml);
        return 1;
    }

    // Also load simple tokenizer for debug token name display.
    minllama::SimpleTokenizer simple_tok;
    minllama::load_simple_tokenizer_from_gguf(*ml, simple_tok, &error);

    std::vector<int> prompt_ids;
    bool add_bos = false;
    if (!opts.prompt_tokens_raw.empty()) {
        if (!parse_prompt_tokens_csv(opts.prompt_tokens_raw, &prompt_ids)) {
            std::cerr << "Error: invalid --prompt-tokens value\n";
            ml_model_free(ml);
            return 1;
        }
        // --prompt-tokens never adds BOS.
    } else {
        // Use BPE tokenizer for text prompt.
        if (!minllama::bpe_encode(bpe_tok, opts.prompt, prompt_ids)) {
            std::cerr << "Error: BPE encode failed\n";
            ml_model_free(ml);
            return 1;
        }
        add_bos = bpe_tok.add_bos_token && !prompt_ids.empty() && prompt_ids[0] != bpe_tok.bos_token_id;
    }

    if (opts.debug_tokens) {
        std::cout << "[debug-tokens]\n";
        std::cout << "tokenizer.ggml.model: gpt2 (BPE)\n";
        std::cout << "BOS id: " << bpe_tok.bos_token_id << "\n";
        std::cout << "EOS id: " << bpe_tok.eos_token_id << "\n";
        std::cout << "UNK id: " << bpe_tok.unk_token_id << "\n";
        std::cout << "add_bos_token (GGUF): " << (bpe_tok.add_bos_token ? "true" : "false") << "\n";
        std::cout << "prompt token ids:";
        for (int id : prompt_ids) std::cout << ' ' << id;
        std::cout << "\n";
        for (std::size_t i = 0; i < prompt_ids.size(); ++i) {
            int id = prompt_ids[i];
            std::string token = (id >= 0 && id < static_cast<int>(bpe_tok.vocab.size()))
                ? bpe_tok.vocab[id] : "<out-of-range>";
            std::cout << "  [" << i << "] id=" << id << " token=" << token << "\n";
        }
    }

    ml_model_free(ml);

    if (prompt_ids.empty()) {
        std::cerr << "Error: prompt produced zero tokens\n";
        return 1;
    }

    std::vector<int> output_ids(opts.max_new_tokens > 0 ? opts.max_new_tokens : 1);
    int output_len = 0;
    bool ok = false;
    int eos_id = bpe_tok.eos_token_id >= 0 ? bpe_tok.eos_token_id : -1;
    if (opts.temperature <= 0.0f) {
        ok = minllama::transformer_model_generate_greedy_f32(
            model, prompt_ids.data(), static_cast<int>(prompt_ids.size()),
            opts.max_new_tokens, output_ids.data(), static_cast<int>(output_ids.size()),
            &output_len, eos_id);
    } else {
        uint32_t rng_state = opts.seed;
        ok = minllama::transformer_model_generate_sample_f32(
            model, prompt_ids.data(), static_cast<int>(prompt_ids.size()),
            opts.max_new_tokens, output_ids.data(), static_cast<int>(output_ids.size()),
            &output_len, eos_id, opts.temperature, &rng_state,
            opts.top_k, opts.top_p);
    }
    if (!ok) {
        std::cerr << "Error: generation failed\n";
        return 1;
    }

    // Decode generated tokens using BPE decoder.
    std::string output;
    if (output_len > 0 && !minllama::bpe_decode(bpe_tok, output_ids.data(), output_len, output)) {
        std::cerr << "Error: decode failed\n";
        return 1;
    }

    if (opts.debug_tokens) {
        std::cout << "generated token ids:";
        for (int i = 0; i < output_len; ++i) std::cout << ' ' << output_ids[i];
        std::cout << "\n";
        for (int i = 0; i < output_len; ++i) {
            int id = output_ids[i];
            std::string token = (id >= 0 && id < static_cast<int>(bpe_tok.vocab.size()))
                ? bpe_tok.vocab[id] : "<out-of-range>";
            std::cout << "  [gen " << i << "] id=" << id << " token=" << token << "\n";
        }
    }

    std::cout << output << std::endl;
    return 0;
}
