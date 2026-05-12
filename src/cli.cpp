#include "minllama_internal.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace minllama {

bool parse_cli_args(int argc, const char **argv, CliOptions &opts, std::string *error) {
    auto set_error = [error](const std::string &msg) {
        if (error) *error = msg;
    };

    if (!argv) {
        set_error("argv is null");
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) {
            set_error("null argument at index " + std::to_string(i));
            return false;
        }

        const std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--model") {
            if (i + 1 >= argc) {
                set_error("--model requires a value");
                return false;
            }
            opts.model_path = argv[++i];
        } else if (arg == "--prompt") {
            if (i + 1 >= argc) {
                set_error("--prompt requires a value");
                return false;
            }
            opts.prompt = argv[++i];
        } else if (arg == "--max-new-tokens") {
            if (i + 1 >= argc) {
                set_error("--max-new-tokens requires a value");
                return false;
            }
            const char *val = argv[++i];
            char *end = nullptr;
            long n = std::strtol(val, &end, 10);
            if (end == val || *end != '\0' || n < 0 || n > 2147483647) {
                set_error("--max-new-tokens must be a non-negative integer: " +
                          std::string(val));
                return false;
            }
            opts.max_new_tokens = static_cast<int>(n);
        } else if (arg == "--temperature") {
            if (i + 1 >= argc) {
                set_error("--temperature requires a value");
                return false;
            }
            const char *val = argv[++i];
            char *end = nullptr;
            float t = std::strtof(val, &end);
            if (end == val || *end != '\0' || t < 0.0f) {
                set_error("--temperature must be a non-negative float: " +
                          std::string(val));
                return false;
            }
            opts.temperature = t;
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                set_error("--seed requires a value");
                return false;
            }
            const char *val = argv[++i];
            char *end = nullptr;
            unsigned long s = std::strtoul(val, &end, 10);
            if (end == val || *end != '\0' ||
                s > static_cast<unsigned long>(UINT32_MAX)) {
                set_error("--seed must be a valid uint32: " +
                          std::string(val));
                return false;
            }
            opts.seed = static_cast<uint32_t>(s);
        } else if (arg == "--top-k") {
            if (i + 1 >= argc) {
                set_error("--top-k requires a value");
                return false;
            }
            const char *val = argv[++i];
            char *end = nullptr;
            long k = std::strtol(val, &end, 10);
            if (end == val || *end != '\0' || k < 0 || k > 2147483647) {
                set_error("--top-k must be a non-negative integer: " +
                          std::string(val));
                return false;
            }
            opts.top_k = static_cast<int>(k);
        } else if (arg == "--top-p") {
            if (i + 1 >= argc) {
                set_error("--top-p requires a value");
                return false;
            }
            const char *val = argv[++i];
            char *end = nullptr;
            float p = std::strtof(val, &end);
            if (end == val || *end != '\0' || p <= 0.0f || p > 1.0f) {
                set_error("--top-p must be in (0, 1]: " +
                          std::string(val));
                return false;
            }
            opts.top_p = p;
        } else {
            set_error("Unknown argument: " + arg);
            return false;
        }
    }

    return true;
}

} // namespace minllama
