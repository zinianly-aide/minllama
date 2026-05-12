#include "minllama_internal.h"

#include <cassert>
#include <string>

int main() {
    // ----------------------------------------------------------------
    // Test 1: --top-k 10.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-k", "10"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.top_k == 10);
    }

    // ----------------------------------------------------------------
    // Test 2: --top-p 0.9.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-p", "0.9"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.top_p == 0.9f);
    }

    // ----------------------------------------------------------------
    // Test 3: defaults top_k=0, top_p=1.0.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.top_k == 0);
        assert(opts.top_p == 1.0f);
    }

    // ----------------------------------------------------------------
    // Test 4: bad top-k (negative).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-k", "-1"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 5: bad top-k (non-numeric).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-k", "abc"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 6: bad top-p (non-numeric).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-p", "xyz"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 7: bad top-p (<= 0).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-p", "0"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 8: bad top-p (> 1).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-p", "1.5"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 9: top-k 0 is valid (disabled).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-k", "0"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.top_k == 0);
    }

    // ----------------------------------------------------------------
    // Test 10: top-p 1.0 is valid (disabled).
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-p", "1.0"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(argc, argv, opts, &error));
        assert(opts.top_p == 1.0f);
    }

    // ----------------------------------------------------------------
    // Test 11: top-k missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-k"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    // ----------------------------------------------------------------
    // Test 12: top-p missing value.
    // ----------------------------------------------------------------
    {
        const char *argv[] = {"minllama_cli", "--top-p"};
        int argc = sizeof(argv) / sizeof(argv[0]);
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(argc, argv, opts, &error));
    }

    return 0;
}
