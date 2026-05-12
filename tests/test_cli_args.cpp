#include "minllama_internal.h"

#include <cassert>
#include <string>

int main() {
    // ----------------------------------------------------------------
    // Test 1: parse success — all args.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {
            "minllama_cli",
            "--model", "a.gguf",
            "--prompt", "hello",
            "--max-new-tokens", "3",
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(error.empty());
        assert(opts.model_path == "a.gguf");
        assert(opts.prompt == "hello");
        assert(opts.max_new_tokens == 3);
        assert(!opts.help);
    }

    // ----------------------------------------------------------------
    // Test 2: default max_new_tokens.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {
            "minllama_cli",
            "--model", "m.gguf",
            "--prompt", "hi",
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.max_new_tokens == 16);  // Default.
    }

    // ----------------------------------------------------------------
    // Test 3: --help flag.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--help"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.help);
    }

    // ----------------------------------------------------------------
    // Test 4: -h short flag.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "-h"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.help);
    }

    // ----------------------------------------------------------------
    // Test 5: --model missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--model"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 6: --prompt missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--prompt"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 7: --max-new-tokens missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--max-new-tokens"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 8: bad --max-new-tokens (negative).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {
            "minllama_cli",
            "--model", "m.gguf",
            "--prompt", "hi",
            "--max-new-tokens", "-1",
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 9: bad --max-new-tokens (non-numeric).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {
            "minllama_cli",
            "--max-new-tokens", "abc",
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 10: unknown argument.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--unknown"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 11: no arguments — all defaults, model and prompt empty.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.model_path.empty());
        assert(opts.prompt.empty());
        assert(opts.max_new_tokens == 16);
    }

    // ----------------------------------------------------------------
    // Test 12: nullptr argv.
    // ----------------------------------------------------------------
    {
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(1, nullptr, opts, &error));
    }

    return 0;
}
