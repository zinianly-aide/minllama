#include "minllama.h"
#include "minllama_internal.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, const char **argv) {
    minllama::CliOptions opts;
    std::string error;

    if (!minllama::parse_cli_args(argc, argv, opts, &error)) {
        std::cerr << "Error: " << error << "\n";
        std::cerr << "Usage: minllama_cli --model <path> --prompt <text> [--max-new-tokens <n>] [--help]\n";
        return 1;
    }

    if (opts.help) {
        std::cout << "Usage: minllama_cli --model <path> --prompt <text> [--max-new-tokens <n>] [--temperature <t>] [--seed <n>] [--help]\n";
        std::cout << "\n";
        std::cout << "Options:\n";
        std::cout << "  --model <path>          Path to GGUF model file (required)\n";
        std::cout << "  --prompt <text>         Prompt text (required)\n";
        std::cout << "  --max-new-tokens <n>    Max tokens to generate (default: 16)\n";
        std::cout << "  --temperature <float>   Sampling temperature (default: 0 = greedy)\n";
        std::cout << "  --seed <uint32>         RNG seed (default: 1)\n";
        std::cout << "  --help                  Show this help\n";
        return 0;
    }

    if (opts.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }

    if (opts.prompt.empty()) {
        std::cerr << "Error: --prompt is required\n";
        return 1;
    }

    if (opts.max_new_tokens < 0) {
        std::cerr << "Error: --max-new-tokens must be >= 0\n";
        return 1;
    }

    // Load GGUF.
    ml_model *ml = ml_model_load(opts.model_path.c_str());
    if (!ml) {
        std::cerr << "Error: failed to load model from " << opts.model_path << "\n";
        return 1;
    }

    // Build transformer model.
    minllama::TransformerModelF32 model;
    if (!minllama::load_transformer_model_f32_from_tensors(*ml, model, &error)) {
        std::cerr << "Error: " << error << "\n";
        ml_model_free(ml);
        return 1;
    }

    // Build tokenizer.
    minllama::SimpleTokenizer tokenizer;
    if (!minllama::load_simple_tokenizer_from_gguf(*ml, tokenizer, &error)) {
        std::cerr << "Error: " << error << "\n";
        ml_model_free(ml);
        return 1;
    }

    ml_model_free(ml);

    // Generate.
    std::string output;
    if (!minllama::minllama_generate_text_sample_f32(
            model, tokenizer, opts.prompt, opts.max_new_tokens,
            opts.temperature, opts.seed, output)) {
        std::cerr << "Error: generation failed\n";
        return 1;
    }

    std::cout << output << std::endl;
    return 0;
}
