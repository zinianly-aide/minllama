cmake_minimum_required(VERSION 3.16)

project(minllama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(minllama
    src/runtime.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/bpe_tokenizer.cpp
    src/gguf.cpp
    src/quant.cpp
    src/kernels.cpp
    src/platform.cpp
    src/thread_pool.cpp
    src/cli.cpp
)

target_include_directories(minllama
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(minllama_cli tools/minllama_cli.cpp)
target_link_libraries(minllama_cli PRIVATE minllama)
target_include_directories(minllama_cli PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_executable(minllama_scan_tensors tools/minllama_scan_tensors.cpp)
target_link_libraries(minllama_scan_tensors PRIVATE minllama)
target_include_directories(minllama_scan_tensors PRIVATE ${PROJECT_SOURCE_DIR}/src)

enable_testing()

add_executable(test_smoke tests/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE minllama)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE minllama)
target_include_directories(test_quant PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME quant COMMAND test_quant)

add_executable(test_loader tests/test_loader.cpp)
target_link_libraries(test_loader PRIVATE minllama)
add_test(NAME loader COMMAND test_loader)

add_executable(test_metadata tests/test_metadata.cpp)
target_link_libraries(test_metadata PRIVATE minllama)
target_include_directories(test_metadata PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME metadata COMMAND test_metadata)

add_executable(test_tensors tests/test_tensors.cpp)
target_link_libraries(test_tensors PRIVATE minllama)
target_include_directories(test_tensors PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tensors COMMAND test_tensors)

add_executable(test_tensor_data tests/test_tensor_data.cpp)
target_link_libraries(test_tensor_data PRIVATE minllama)
target_include_directories(test_tensor_data PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tensor_data COMMAND test_tensor_data)

add_executable(test_tensor_read tests/test_tensor_read.cpp)
target_link_libraries(test_tensor_read PRIVATE minllama)
target_include_directories(test_tensor_read PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tensor_read COMMAND test_tensor_read)

add_executable(test_q40_decode tests/test_q40_decode.cpp)
target_link_libraries(test_q40_decode PRIVATE minllama)
target_include_directories(test_q40_decode PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME q40_decode COMMAND test_q40_decode)

add_executable(test_matvec tests/test_matvec.cpp)
target_link_libraries(test_matvec PRIVATE minllama)
target_include_directories(test_matvec PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME matvec COMMAND test_matvec)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE minllama)
target_include_directories(test_ops PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME ops COMMAND test_ops)

add_executable(test_quant_blocks tests/test_quant_blocks.cpp)
target_link_libraries(test_quant_blocks PRIVATE minllama)
target_include_directories(test_quant_blocks PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME quant_blocks COMMAND test_quant_blocks)

add_executable(test_rope tests/test_rope.cpp)
target_link_libraries(test_rope PRIVATE minllama)
target_include_directories(test_rope PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME rope COMMAND test_rope)

add_executable(test_attention_math tests/test_attention_math.cpp)
target_link_libraries(test_attention_math PRIVATE minllama)
target_include_directories(test_attention_math PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME attention_math COMMAND test_attention_math)

add_executable(test_attention_single_head tests/test_attention_single_head.cpp)
target_link_libraries(test_attention_single_head PRIVATE minllama)
target_include_directories(test_attention_single_head PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME attention_single_head COMMAND test_attention_single_head)

add_executable(test_kv_cache tests/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE minllama)
target_include_directories(test_kv_cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME kv_cache COMMAND test_kv_cache)

add_executable(test_transformer_layer tests/test_transformer_layer.cpp)
target_link_libraries(test_transformer_layer PRIVATE minllama)
target_include_directories(test_transformer_layer PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME transformer_layer COMMAND test_transformer_layer)

add_executable(test_transformer_layer_ffn tests/test_transformer_layer_ffn.cpp)
target_link_libraries(test_transformer_layer_ffn PRIVATE minllama)
target_include_directories(test_transformer_layer_ffn PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME transformer_layer_ffn COMMAND test_transformer_layer_ffn)

add_executable(test_transformer_model tests/test_transformer_model.cpp)
target_link_libraries(test_transformer_model PRIVATE minllama)
target_include_directories(test_transformer_model PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME transformer_model COMMAND test_transformer_model)

add_executable(test_transformer_logits tests/test_transformer_logits.cpp)
target_link_libraries(test_transformer_logits PRIVATE minllama)
target_include_directories(test_transformer_logits PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME transformer_logits COMMAND test_transformer_logits)

add_executable(test_greedy tests/test_greedy.cpp)
target_link_libraries(test_greedy PRIVATE minllama)
target_include_directories(test_greedy PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME greedy COMMAND test_greedy)

add_executable(test_generate tests/test_generate.cpp)
target_link_libraries(test_generate PRIVATE minllama)
target_include_directories(test_generate PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME generate COMMAND test_generate)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE minllama)
target_include_directories(test_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tokenizer COMMAND test_tokenizer)

add_executable(test_generate_text tests/test_generate_text.cpp)
target_link_libraries(test_generate_text PRIVATE minllama)
target_include_directories(test_generate_text PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME generate_text COMMAND test_generate_text)

add_executable(test_model_loader tests/test_model_loader.cpp)
target_link_libraries(test_model_loader PRIVATE minllama)
target_include_directories(test_model_loader PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME model_loader COMMAND test_model_loader)

add_executable(test_tokenizer_loader tests/test_tokenizer_loader.cpp)
target_link_libraries(test_tokenizer_loader PRIVATE minllama)
target_include_directories(test_tokenizer_loader PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tokenizer_loader COMMAND test_tokenizer_loader)

add_executable(test_cli_args tests/test_cli_args.cpp)
target_link_libraries(test_cli_args PRIVATE minllama)
target_include_directories(test_cli_args PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME cli_args COMMAND test_cli_args)

add_executable(test_generate_eos tests/test_generate_eos.cpp)
target_link_libraries(test_generate_eos PRIVATE minllama)
target_include_directories(test_generate_eos PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME generate_eos COMMAND test_generate_eos)

add_executable(test_sampling tests/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE minllama)
target_include_directories(test_sampling PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME sampling COMMAND test_sampling)

add_executable(test_generate_sampling tests/test_generate_sampling.cpp)
target_link_libraries(test_generate_sampling PRIVATE minllama)
target_include_directories(test_generate_sampling PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME generate_sampling COMMAND test_generate_sampling)

add_executable(test_cli_args_sampling tests/test_cli_args_sampling.cpp)
target_link_libraries(test_cli_args_sampling PRIVATE minllama)
target_include_directories(test_cli_args_sampling PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME cli_args_sampling COMMAND test_cli_args_sampling)

add_executable(test_sampling_topk_topp tests/test_sampling_topk_topp.cpp)
target_link_libraries(test_sampling_topk_topp PRIVATE minllama)
target_include_directories(test_sampling_topk_topp PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME sampling_topk_topp COMMAND test_sampling_topk_topp)

add_executable(test_cli_args_topk_topp tests/test_cli_args_topk_topp.cpp)
target_link_libraries(test_cli_args_topk_topp PRIVATE minllama)
target_include_directories(test_cli_args_topk_topp PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME cli_args_topk_topp COMMAND test_cli_args_topk_topp)

add_executable(test_cli_args_debug_tokens tests/test_cli_args_debug_tokens.cpp)
target_link_libraries(test_cli_args_debug_tokens PRIVATE minllama)
target_include_directories(test_cli_args_debug_tokens PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME cli_args_debug_tokens COMMAND test_cli_args_debug_tokens)

add_executable(test_attention_gqa tests/test_attention_gqa.cpp)
target_link_libraries(test_attention_gqa PRIVATE minllama)
target_include_directories(test_attention_gqa PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME attention_gqa COMMAND test_attention_gqa)

add_executable(test_attention_gqa_reference tests/test_attention_gqa_reference.cpp)
target_link_libraries(test_attention_gqa_reference PRIVATE minllama)
target_include_directories(test_attention_gqa_reference PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME attention_gqa_reference COMMAND test_attention_gqa_reference)

add_executable(test_transformer_layer_gqa tests/test_transformer_layer_gqa.cpp)
target_link_libraries(test_transformer_layer_gqa PRIVATE minllama)
target_include_directories(test_transformer_layer_gqa PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME transformer_layer_gqa COMMAND test_transformer_layer_gqa)

add_executable(test_smollm tools/test_smollm.cpp)
target_link_libraries(test_smollm PRIVATE minllama)
target_include_directories(test_smollm PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_executable(minllama_compare_logits tools/minllama_compare_logits.cpp)
target_link_libraries(minllama_compare_logits PRIVATE minllama)
target_include_directories(minllama_compare_logits PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_executable(test_compare_logits tests/test_compare_logits.cpp)
target_link_libraries(test_compare_logits PRIVATE minllama)
target_include_directories(test_compare_logits PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME compare_logits COMMAND test_compare_logits)

add_executable(minllama_tokenize tools/minllama_tokenize.cpp)
target_link_libraries(minllama_tokenize PRIVATE minllama)
target_include_directories(minllama_tokenize PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_executable(test_bpe_tokenizer tests/test_bpe_tokenizer.cpp)
target_link_libraries(test_bpe_tokenizer PRIVATE minllama)
target_include_directories(test_bpe_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME bpe_tokenizer COMMAND test_bpe_tokenizer)
