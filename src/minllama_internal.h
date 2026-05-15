#ifndef MINLLAMA_INTERNAL_H
#define MINLLAMA_INTERNAL_H

#include "minllama.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct GgufHeader {
    std::uint32_t version = 0;
    std::uint64_t n_tensors = 0;
    std::uint64_t n_kv = 0;
};

struct GgufFileView {
    GgufHeader header;
    std::uint32_t alignment = 32;
    std::uint64_t data_offset = 0;
};

struct ModelConfig {
    std::uint32_t n_vocab = 0;
    std::uint32_t n_layer = 0;
    std::uint32_t n_embd = 0;
    std::uint32_t n_head = 0;
    std::uint32_t n_head_kv = 0;
    std::uint32_t n_ctx_train = 0;
    float rope_theta = 0.0f;
    float rms_norm_eps = 0.0f;
};

constexpr std::uint32_t ML_MAX_TENSOR_DIMS = 4;

struct TensorInfo {
    std::string name;
    std::uint32_t gguf_type = 0;
    std::uint32_t n_dims = 0;
    std::array<std::uint64_t, ML_MAX_TENSOR_DIMS> dims = {};
    std::uint64_t offset = 0;
};

struct TensorView {
    // Minimal readonly boundary view into the GGUF tensor data area. This is
    // only offset arithmetic; tensor bytes are not read or decoded here.
    std::uint64_t data_begin = 0;
    std::uint64_t byte_size = 0;
    std::uint64_t data_end = 0;
};

struct TensorIndex {
    std::vector<TensorInfo> tensors;
    std::vector<TensorView> views;
    std::unordered_map<std::string, std::size_t> by_name;

    const TensorInfo *find(const std::string &name) const {
        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            return nullptr;
        }
        return &tensors[it->second];
    }

    const TensorView *find_view(const std::string &name) const {
        const auto it = by_name.find(name);
        if (it == by_name.end() || it->second >= views.size()) {
            return nullptr;
        }
        return &views[it->second];
    }

    std::size_t size() const {
        return tensors.size();
    }
};

struct ml_model {
    std::string path;
    GgufFileView gguf;
    ModelConfig config;
    TensorIndex tensor_index;
    bool skeleton_only = true;
};

struct ml_context {
    ml_model *model = nullptr;
    std::uint64_t reset_count = 0;
};

constexpr ml_token ML_SKELETON_TOKEN = 1;
constexpr const char *ML_SKELETON_TEXT = "<skeleton>";

float ml_fp16_to_fp32(std::uint16_t value);

namespace minllama {
bool load_gguf_file_view(const char *path,
                         GgufFileView *out_view,
                         ModelConfig *out_config,
                         TensorIndex *out_tensor_index);

// Minimal tensor content access for the future kernel layer. Q4_0 decoding is
// a correctness-first reference path, not an optimized matvec/kernel path.
bool load_tensor_f32(const char *path,
                     const TensorIndex &tensor_index,
                     const std::string &name,
                     std::vector<float> *out);

bool load_tensor_as_f32(const char *path,
                        const TensorIndex &tensor_index,
                        const std::string &name,
                        std::vector<float> *out);

// Load a Q4_0 tensor as raw bytes (no dequant). Used by fused matvec kernels.
bool load_tensor_q4_0_raw(const char *path,
                          const TensorIndex &tensor_index,
                          const std::string &name,
                          std::vector<unsigned char> *out);

// Load a Q8_0 tensor as raw bytes (no dequant). Used by fused matvec kernels.
bool load_tensor_q8_0_raw(const char *path,
                          const TensorIndex &tensor_index,
                          const std::string &name,
                          std::vector<unsigned char> *out);

// Reference quant block decoders used by tests and diagnostics.
bool decode_q4_0_block_f32(std::uint16_t scale_bits,
                           const unsigned char *packed,
                           std::size_t packed_len,
                           float *out,
                           std::size_t out_len);

bool decode_q8_0_block_f32(std::uint16_t scale_bits,
                           const unsigned char *qs,
                           std::size_t qs_len,
                           float *out,
                           std::size_t out_len);

// Correctness-first reference matvec helpers. These are intentionally small
// scalar implementations, not optimized kernels.
bool matvec_f32_f32(const float *matrix,
                    std::size_t rows,
                    std::size_t cols,
                    const float *input,
                    std::size_t input_len,
                    float *out,
                    std::size_t out_len,
                    int n_threads = 1);

bool matvec_f16_f32(const std::uint16_t *matrix,
                    std::size_t rows,
                    std::size_t cols,
                    const float *input,
                    std::size_t input_len,
                    float *out,
                    std::size_t out_len);

bool matvec_tensor_as_f32(const char *path,
                          const TensorIndex &tensor_index,
                          const std::string &name,
                          const std::vector<float> &input,
                          std::vector<float> *out);

bool matvec_q40_f32(const char *path,
                  const TensorIndex &tensor_index,
                  const std::string &name,
                  const std::vector<float> &input,
                  std::vector<float> *out);

// Fused Q4_0 matvec: block-by-block dot accumulation without full dequant.
// q4_data points to raw GGUF Q4_0 tensor data (Q4_0 blocks).
// Uses ARM NEON on aarch64, scalar fallback on other platforms.
bool matvec_q4_0_neon_f32(const unsigned char *q4_data,
                          std::size_t rows,
                          std::size_t cols,
                          const float *input,
                          std::size_t input_len,
                          float *out,
                          std::size_t out_len);

bool matvec_q8_0_fused_f32(const unsigned char *q8_data,
                           std::size_t rows,
                           std::size_t cols,
                           const float *input,
                           std::size_t input_len,
                           float *out,
                           std::size_t out_len,
                           int n_threads = 1);

// Correctness-first reference operator helpers. These are intentionally small
// scalar implementations used to prepare the future transformer layer path.
bool rmsnorm_f32(const float *x,
                 const float *weight,
                 std::size_t len,
                 float eps,
                 float *out,
                 std::size_t out_len);

bool softmax_f32(const float *input,
                 std::size_t len,
                 float *out,
                 std::size_t out_len);

float silu_f32(float x);

bool swiglu_f32(const float *gate,
                const float *up,
                std::size_t len,
                float *out,
                std::size_t out_len);

// Minimal RoPE and attention-math helpers. These are correctness-first
// reference implementations used before wiring full attention blocks.
bool rope_apply_f32(float *vec,
                    std::size_t len,
                    std::size_t position,
                    float rope_theta);

bool dot_f32(const float *a,
             const float *b,
             std::size_t len,
             float *out);

bool scale_f32(float *vec,
               std::size_t len,
               float scale);

bool attention_scores_f32(const float *query,
                          std::size_t dim,
                          const float *keys,
                          std::size_t n_keys,
                          float *out_scores,
                          std::size_t out_len);

bool attention_single_head_f32(const float *query,
                               const float *keys,
                               const float *values,
                               int n_tokens,
                               int dim,
                               float *output);

struct KvCacheF32 {
    std::vector<float> keys;
    std::vector<float> values;
    int max_tokens = 0;
    int dim = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
};

bool kv_cache_init_f32(KvCacheF32 &cache, int max_tokens, int dim);
bool kv_cache_init_gqa_f32(KvCacheF32 &cache, int max_tokens, int n_kv_heads, int head_dim);
bool kv_cache_write_f32(KvCacheF32 &cache, int position, const float *key, const float *value);
bool kv_cache_write_gqa_f32(KvCacheF32 &cache, int position, const float *k, const float *v);
bool attention_decode_single_head_f32(const float *query,
                                      KvCacheF32 &cache,
                                      int position,
                                      float *output);
bool attention_decode_gqa_f32(const float *q,
                              KvCacheF32 &cache,
                              int position,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              float *output);

struct TransformerLayerF32 {
    int dim = 0;
    int n_heads = 1;
    int n_kv_heads = 1;
    int head_dim = 0;
    std::vector<float> rms_att_weight;  // [dim]
    std::vector<float> wq;              // [dim*dim] row-major
    std::vector<float> wk;              // [n_kv_heads*head_dim * dim] row-major
    std::vector<float> wv;              // [n_kv_heads*head_dim * dim] row-major
    std::vector<float> wo;              // [dim*dim] row-major
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-6f;

    // FFN (SwiGLU).  If hidden_dim <= 0 the function degrades to
    // attention-only, keeping backward compatibility with step 13.
    int hidden_dim = 0;
    std::vector<float> rms_ffn_weight;  // [dim]
    std::vector<float> w1;              // [hidden_dim*dim] row-major (f32)
    std::vector<float> w2;              // [dim*hidden_dim] row-major (f32)
    std::vector<float> w3;              // [hidden_dim*dim] row-major (f32)

    // Q4_0 quantized FFN weights (mutually exclusive with f32 counterparts).
    // When non-empty, the NEON or scalar-fallback q4_0 fused matvec is used
    // for faster decode. Only for --model-q4 models at this time.
    std::vector<unsigned char> w1_q4;   // [hidden_dim * blocks_per_row * 18] raw Q4_0
    std::vector<unsigned char> w2_q4;   // [dim * blocks_per_hidden * 18] raw Q4_0
    std::vector<unsigned char> w3_q4;   // [hidden_dim * blocks_per_row * 18] raw Q4_0
};

// Forward trace control: set layer_index to enable tracing for a specific layer only.
// Set to -1 to disable. Only effective in debug builds or when explicitly enabled.
extern int g_forward_trace_layer;
extern bool g_forward_trace_enabled;

bool transformer_layer_decode_f32(const TransformerLayerF32 &layer,
                                  const float *x,
                                  KvCacheF32 &cache,
                                  int position,
                                  float *output,
                                  int n_threads = 1);

struct TransformerModelF32 {
    int dim = 0;
    int n_layers = 0;
    int n_threads = 1;  // thread count for parallel matvec
    std::vector<TransformerLayerF32> layers;
    std::vector<KvCacheF32> kv_caches;

    // Token embedding table: [vocab_size * dim] row-major.
    std::vector<float> token_embedding;

    // Final RMSNorm + LM head for logits.
    std::vector<float> final_norm_weight;  // [dim]
    float rms_norm_eps = 1e-6f;
    int vocab_size = 0;
    std::vector<float> lm_head;             // [vocab_size*dim] row-major

    // Optional Q8_0 lm_head fast path (default off; requires strict guards).
    bool q8_lm_head_enabled = false;        // set by CLI flag --q8-lm-head
    bool lm_head_tied_token_embd = false;   // true when output.weight is absent
    bool token_embd_is_q8_0 = false;        // token_embd.weight gguf type == Q8_0
    std::vector<unsigned char> token_embedding_q8_raw; // raw Q8_0 bytes for token_embd.weight
};

bool transformer_model_decode_f32(TransformerModelF32 &model,
                                  const float *x,
                                  int position,
                                  float *output);

bool transformer_model_logits_f32(TransformerModelF32 &model,
                                  const float *x,
                                  int position,
                                  float *logits);

int argmax_f32(const float *values, int len);

uint32_t rng_next_u32(uint32_t *state);
float rng_uniform01(uint32_t *state);

int sample_temperature_f32(const float *logits,
                           int vocab_size,
                           float temperature,
                           uint32_t *rng_state);

int sample_top_k_top_p_f32(const float *logits,
                           int vocab_size,
                           float temperature,
                           int top_k,
                           float top_p,
                           uint32_t *rng_state);

bool transformer_model_greedy_step_f32(TransformerModelF32 &model,
                                       const float *x,
                                       int position,
                                       int *token_id);

bool transformer_model_sample_step_f32(TransformerModelF32 &model,
                                       const float *x,
                                       int position,
                                       float temperature,
                                       uint32_t *rng_state,
                                       int *token_id,
                                       int top_k = 0,
                                       float top_p = 1.0f);

bool token_embedding_lookup_f32(const TransformerModelF32 &model,
                                int token_id,
                                float *output);

bool transformer_model_greedy_token_step_f32(TransformerModelF32 &model,
                                             int token_id,
                                             int position,
                                             int *next_token_id);

bool transformer_model_generate_greedy_f32(TransformerModelF32 &model,
                                           const int *prompt_tokens,
                                           int prompt_len,
                                           int max_new_tokens,
                                           int *output_tokens,
                                           int output_capacity,
                                           int *output_len,
                                           int eos_token_id = -1);

bool transformer_model_generate_sample_f32(TransformerModelF32 &model,
                                           const int *prompt_tokens,
                                           int prompt_len,
                                           int max_new_tokens,
                                           int *output_tokens,
                                           int output_capacity,
                                           int *output_len,
                                           int eos_token_id,
                                           float temperature,
                                           uint32_t *rng_state,
                                           int top_k = 0,
                                           float top_p = 1.0f);

struct SimpleTokenizer {
    std::string tokenizer_model;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int unk_token_id = -1;
    int bos_token_id = -1;
    int eos_token_id = -1;
    bool add_bos_token = false;  // from GGUF tokenizer.ggml.add_bos_token
    bool add_eos_token = false;  // from GGUF tokenizer.ggml.add_eos_token
};

bool tokenizer_init(SimpleTokenizer &tokenizer,
                    const std::vector<std::string> &vocab,
                    int unk_token_id,
                    int bos_token_id,
                    int eos_token_id);

bool tokenizer_encode_whitespace(const SimpleTokenizer &tokenizer,
                                 const std::string &text,
                                 std::vector<int> &output_ids,
                                 bool add_bos);

bool tokenizer_decode_tokens(const SimpleTokenizer &tokenizer,
                             const int *token_ids,
                             int token_count,
                             std::string &output_text);

bool minllama_generate_text_greedy_f32(TransformerModelF32 &model,
                                       const SimpleTokenizer &tokenizer,
                                       const std::string &prompt,
                                       int max_new_tokens,
                                       std::string &output_text);

bool minllama_generate_text_sample_f32(TransformerModelF32 &model,
                                       const SimpleTokenizer &tokenizer,
                                       const std::string &prompt,
                                       int max_new_tokens,
                                       float temperature,
                                       uint32_t seed,
                                       std::string &output_text,
                                       int top_k = 0,
                                       float top_p = 1.0f);

bool load_transformer_model_f32_from_tensors(const ml_model &src,
                                             TransformerModelF32 &model,
                                             std::string *error);

bool load_simple_tokenizer_from_gguf(const ml_model &src,
                                     SimpleTokenizer &tokenizer,
                                     std::string *error);

struct CliOptions {
    std::string model_path;
    std::string prompt;
    std::string prompt_tokens_raw;
    int max_new_tokens = 16;
    float temperature = 0.0f;
    uint32_t seed = 1;
    int top_k = 0;
    float top_p = 1.0f;
    int n_threads = 1;
    bool help = false;
    bool debug_tokens = false;
    bool debug_load = false;
    bool dump_platform = false;
    bool q8_lm_head = false;
};

extern bool g_debug_load;  // set by CLI --debug-load, read by gguf/model loaders

// BPE tokenizer (GPT-2 / SmolLM style byte-level BPE).
struct BpeTokenizer {
    std::vector<std::string> vocab;               // token id → string
    std::unordered_map<std::string, int> token_to_id; // string → token id
    std::vector<std::string> merges;              // ordered merge strings ("tok1 tok2")
    std::unordered_map<std::string, int> merge_rank; // merge string → rank (lower = earlier)
    std::unordered_map<int, std::string> byte_encoder;   // byte → unicode char(s)
    std::unordered_map<std::string, unsigned char> byte_decoder; // unicode → byte
    int bos_token_id = -1;
    int eos_token_id = -1;
    int unk_token_id = -1;
    bool add_bos_token = false;
    bool add_eos_token = false;
};

// Build the GPT-2 byte encoder/decoder tables.
void bpe_build_byte_tables(BpeTokenizer &tok);

// Load BPE tokenizer directly from GGUF file, skipping the ml_model wrapper.
bool bpe_tokenizer_load(const std::string &gguf_path, BpeTokenizer &tok, std::string *error);

// Encode text to token IDs
bool bpe_encode(const BpeTokenizer &tok, const std::string &text, std::vector<int> &ids);

// Decode token IDs to text
bool bpe_decode(const BpeTokenizer &tok, const int *ids, int count, std::string &text);

bool parse_cli_args(int argc, const char **argv, CliOptions &opts, std::string *error);

// --- Thread pool ---
// Simple thread pool for parallel matvec rows.  n_threads=1 runs inline
// (no thread overhead).  n_threads>1 spawns a pool on first use.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();

    // Block until all tasks finish.
    void wait();

    // Parallel for: calls fn(chunk_start, chunk_end) for each chunk.
    // fn receives [start, end) exclusive range.
    void parallel_for(std::size_t start, std::size_t end,
                      std::function<void(std::size_t, std::size_t)> fn);

    int num_threads() const { return n_threads_; }

private:
    int n_threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_ = false;

    // Pre-assign chunk scheme — no Task needed, no steal.
    std::function<void(std::size_t, std::size_t)> *current_fn_ = nullptr;
    struct Chunk { std::size_t start, end; };
    Chunk *chunks_ = nullptr;
    int n_chunks_ = 0;
    int chunks_done_ = 0;
    int generation_ = 0;  // incremented each parallel_for; workers use this to
                          // detect new tasks and avoid re-processing stale work

    void worker_loop(int worker_id);
};

// Global thread pool, initialized on first call to get_thread_pool().
// n_threads sets thread count on first initialization.
ThreadPool &get_thread_pool(int n_threads = 1);
long get_parallel_for_count();
long get_worker_loops();

// =======================================================================
// Platform / CPU capability detection
// =======================================================================
const char *platform_arch_name();     // e.g. "x86_64", "aarch64"
const char *platform_simd_name();     // active SIMD kernel: "NEON", "AVX2", "SSE2", "scalar"
bool platform_has_neon();
bool platform_has_sse2();
bool platform_has_avx();
bool platform_has_avx2();
void platform_dump_caps();            // print all capabilities to stderr

}  // namespace minllama

#endif
