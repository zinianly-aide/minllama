#include "minllama_internal.h"

#include <cassert>
#include <string>

int main() {
    // ----------------------------------------------------------------
    // Test 1: parse --temperature.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {
            "minllama_cli",
            "--model", "m.gguf",
            "--prompt", "hi",
            "--temperature", "0.7",
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.temperature == 0.7f);
    }

    // ----------------------------------------------------------------
    // Test 2: parse --seed.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {
            "minllama_cli",
            "--model", "m.gguf",
            "--prompt", "hi",
            "--seed", "42",
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.seed == 42u);
    }

    // ----------------------------------------------------------------
    // Test 3: default temperature=0, seed=1.
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
        assert(opts.temperature == 0.0f);
        assert(opts.seed == 1u);
    }

    // ----------------------------------------------------------------
    // Test 4: bad temperature (non-numeric).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--temperature", "abc"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 5: bad temperature (negative).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--temperature", "-0.5"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 6: bad seed (non-numeric).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--seed", "xyz"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 7: temperature missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--temperature"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 8: seed missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--seed"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 9: temperature = 0.0 is valid.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--temperature", "0"};
        int argc = sizeof(argv) / sizeof(argv[0]);

        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.temperature == 0.0f);
    }

    return 0;
}
