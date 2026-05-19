#include "minllama_internal.h"
#include "architecture.h"

#include <new>

ml_model *ml_model_load(const char *path) {
    if (!path) {
        return nullptr;
    }

    auto *model = new (std::nothrow) ml_model();
    if (!model) {
        return nullptr;
    }

    if (!minllama::load_gguf_file_view(path, &model->gguf, &model->config, &model->tensor_index)) {
        delete model;
        return nullptr;
    }
    model->architecture = model->config.architecture;

    // The runtime is still skeleton-only; GGUF loading builds a minimal
    // metadata + tensor-info index but does not read tensor data.
    model->path = path;
    return model;
}

bool ml_model_is_supported(const ml_model *model) {
    if (!model || !model->config.architecture) {
        return false;
    }
    return minllama::is_supported(static_cast<minllama::Architecture>(model->config.architecture));
}

void ml_model_free(ml_model *model) {
    delete model;
}
