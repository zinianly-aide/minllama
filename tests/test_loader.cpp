#include "minllama.h"
#include "test_gguf_util.h"

#include <cassert>
#include <cstdio>
#include <fstream>

namespace {

void expect_loads(const char *path) {
    ml_model *model = ml_model_load(path);
    assert(model != nullptr);
    ml_model_free(model);
}

void expect_fails(const char *path) {
    ml_model *model = ml_model_load(path);
    assert(model == nullptr);
}

} // namespace

int main() {
    const char *valid_path = "test_loader_valid.gguf";
    const char *bad_magic_path = "test_loader_bad_magic.gguf";
    const char *bad_version_path = "test_loader_bad_version.gguf";
    const char *short_header_path = "test_loader_short_header.gguf";
    const char *missing_path = "test_loader_missing.gguf";

    std::remove(valid_path);
    std::remove(bad_magic_path);
    std::remove(bad_version_path);
    std::remove(short_header_path);
    std::remove(missing_path);

    {
        bool ok = minllama_test::write_fake_llama_gguf(valid_path);
        assert(ok);
    }
    expect_loads(valid_path);

    {
        bool ok = minllama_test::write_fake_gguf_header(bad_magic_path, "NOPE");
        assert(ok);
    }
    expect_fails(bad_magic_path);

    {
        bool ok = minllama_test::write_fake_gguf_header(bad_version_path, "GGUF", 2);
        assert(ok);
    }
    expect_fails(bad_version_path);

    {
        std::ofstream out(short_header_path, std::ios::binary | std::ios::trunc);
        out.write("GGUF", 4);
    }
    expect_fails(short_header_path);
    expect_fails(missing_path);

    std::remove(valid_path);
    std::remove(bad_magic_path);
    std::remove(bad_version_path);
    std::remove(short_header_path);
    return 0;
}
