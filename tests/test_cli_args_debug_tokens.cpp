#include "minllama_internal.h"

#include <cassert>
#include <string>

int main() {
    {
        const char *argv[] = {"minllama_cli", "--debug-tokens"};
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(2, argv, opts, &error));
        assert(opts.debug_tokens);
    }
    {
        const char *argv[] = {"minllama_cli", "--prompt-tokens", "1,2,3"};
        minllama::CliOptions opts;
        std::string error;
        assert(minllama::parse_cli_args(3, argv, opts, &error));
        assert(opts.prompt_tokens_raw == "1,2,3");
    }
    {
        const char *argv[] = {"minllama_cli", "--prompt-tokens"};
        minllama::CliOptions opts;
        std::string error;
        assert(!minllama::parse_cli_args(2, argv, opts, &error));
        assert(!error.empty());
    }
    return 0;
}
