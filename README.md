# minllama

`minllama` 是一个用于研究和最小实现 `llama.cpp` 核心路径的项目。目标不是复刻完整 `llama.cpp`，而是围绕其官方设计思路，提炼一条可直接实现、可运行、便于验证的最短路径。

## 架构

```
GGUF file
  ├── load_gguf_file_view        → GgufFileView + ModelConfig + TensorIndex
  ├── load_transformer_model_f32_from_tensors  → TransformerModelF32
  └── load_simple_tokenizer_from_gguf          → SimpleTokenizer

TransformerModelF32
  ├── token_embedding            [vocab × dim]
  ├── layers[0..n-1]
  │     ├── RMSNorm (attn)
  │     ├── Wq, Wk, Wv, Wo      [dim × dim]
  │     ├── RoPE
  │     ├── KV cache             [max_tokens × dim]
  │     ├── RMSNorm (ffn)
  │     └── SwiGLU (w1, w2, w3)
  ├── final_norm_weight          [dim]
  └── lm_head                    [vocab × dim]

SimpleTokenizer
  ├── id_to_token / token_to_id  bidirectional vocab map
  ├── tokenizer_encode_whitespace  text → token ids
  └── tokenizer_decode_tokens      token ids → text

Generation
  ├── greedy step: logits → argmax
  ├── sample step: logits → softmax / T → CDF sample (LCG rng)
  ├── generate_greedy: prefill + greedy loop + EOS stop
  └── generate_sample: prefill + sampling loop + EOS stop

CLI: minllama_cli --model <path> --prompt <text> [--temperature <t>] [--seed <n>] [--max-new-tokens <n>]
```

## 第一阶段范围

- Llama 2 风格 decoder-only 架构
- GGUF v3 单文件模型格式
- CPU-only 推理路径
- F32 / F16 / Q4_0 / Q4_1 / Q8_0 权重格式（Q4_1 为 correctness-first 标量支持）
- FP16 KV cache
- Greedy decode + temperature sampling
- EOS 提前停止
- 同步 API
- CLI 推理入口

暂不纳入：多模型架构、多 GPU 后端、continuous batching、OpenAI 兼容 HTTP server、复杂采样器、更多量化格式。

## 文档

- [最小核心研究报告](docs/minimal-core-research-report.md)

## 当前实现状态 (SOP)

**40/40 tests passed (100%)**

### 模块完成度

| 模块 | 状态 | 测试 |
||------|:----:|------|
|| GGUF v3 解析与模型加载 | ✅ | `smoke` `loader` `metadata` `tensors` `tensor_data` `tensor_read` `model_loader` |
|| 权重量化与 matvec (F32/F16/Q4_0/Q4_1/Q8_0) | ✅ | `quant` `q40_decode` `matvec` |
|| 计算内核 (RMSNorm / softmax / SiLU / SwiGLU / RoPE) | ✅ | `ops` `rope` |
|| Attention 前置算子与单头 attention | ✅ | `attention_math` `attention_single_head` |
|| KV cache 读写 + decode attention | ✅ | `kv_cache` |
|| Transformer 层 (Self-Attn + FFN) | ✅ | `transformer_layer` `transformer_layer_ffn` |
|| 完整模型前向 (logits) | ✅ | `transformer_model` `transformer_logits` |
|| Greedy decode 循环 + EOS | ✅ | `greedy` `generate` `generate_eos` |
|| Temperature sampling (LCG RNG) | ✅ | `sampling` `generate_sampling` `sampling_topk_topp` |
|| SimpleTokenizer (编码/解码) | ✅ | `tokenizer` `tokenizer_loader` |
|| CLI 骨架 + 参数解析 | ✅ | `cli_args` `cli_args_sampling` |
|| 端到端文本生成 | ✅ | `generate_text` |
|| 线程池并行 (threads=1/2) | ✅ | 1000 次并行 stress test 通过 |

### 线程池与并行 matvec

`minllama` 使用预分配分块(pre-assign chunk)线程池对 matvec 的行进行并行化。

**实现**: `src/thread_pool.cpp` — 基于 `generation_` 计数器的线程池，每个 `parallel_for` 调用递增 generation，worker 线程通过检测 generation 变化避免处理过期任务。

#### Benchmark (SmolLM-135M Q4_0, M4 CPU, decode 128 tokens)

| --threads | decode tok/s | speedup | deterministic |
|-----------|:-----------:|:-------:|:-------------:|
| 1 | 13.58 tok/s | 1.00x | ✅ |
| 2 | 17.61 tok/s | 1.30x | ✅ |

- **threads=2**: 稳定可用，40/40 tests passed，输出与单线程完全一致 (deterministic)
- **threads=4/8**: ⛔ 不支持 — 已知 hang 问题，留待下一阶段
- **默认值**: `--threads 1` (单线程)，可通过 `--threads 2` 手动开启并行
- **CLI 限制**: `--threads` 只接受 1 或 2，传 4/8 直接报错退出，不做 fallback

### 核心数据结构

- `GgufFileView` / `ModelConfig` / `TensorIndex` — GGUF 解析
- `TransformerModelF32` — 完整模型权重 + KV caches
- `TransformerLayerF32` — 单层 (attn + FFN)
- `KvCacheF32` — KV cache
- `SimpleTokenizer` — 词表双向映射
- `CliOptions` — CLI 参数 (model, prompt, max_new_tokens, temperature, seed, top_k, top_p)

### 核心函数 (公开 API)

```c
// C API
ml_model *ml_model_load(const char *path);
void ml_model_free(ml_model *model);
```

```cpp
// Internal C++ API
// --- Loading ---
bool load_transformer_model_f32_from_tensors(const ml_model&, TransformerModelF32&, string* error);
bool load_simple_tokenizer_from_gguf(const ml_model&, SimpleTokenizer&, string* error);

// --- Embedding ---
bool token_embedding_lookup_f32(const TransformerModelF32&, int token_id, float* output);

// --- Step ---
bool transformer_model_greedy_step_f32(TransformerModelF32&, const float* x, int pos, int* token_id);
bool transformer_model_greedy_token_step_f32(TransformerModelF32&, int token_id, int pos, int* next_token_id);
bool transformer_model_sample_step_f32(TransformerModelF32&, const float* x, int pos, float T, uint32_t* rng, int* token_id);

// --- Generate ---
bool transformer_model_generate_greedy_f32(TransformerModelF32&, const int* prompt, int plen, int max_new, int* out, int cap, int* olen, int eos=-1);
bool transformer_model_generate_sample_f32(TransformerModelF32&, const int* prompt, int plen, int max_new, int* out, int cap, int* olen, int eos, float T, uint32_t* rng);

// --- Text ---
bool minllama_generate_text_greedy_f32(TransformerModelF32&, const SimpleTokenizer&, const string& prompt, int max_new, string& out);
bool minllama_generate_text_sample_f32(TransformerModelF32&, const SimpleTokenizer&, const string& prompt, int max_new, float T, uint32_t seed, string& out);

// --- Sampling ---
int sample_temperature_f32(const float* logits, int vocab, float T, uint32_t* rng);
uint32_t rng_next_u32(uint32_t* state);
float rng_uniform01(uint32_t* state);
```

### 构建与测试

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### CLI 用法

```bash
./build/minllama_cli --help
./build/minllama_cli --model model.gguf --prompt "hello world" --max-new-tokens 8
./build/minllama_cli --model model.gguf --prompt "hello" --temperature 0.7 --seed 42
./build/minllama_cli --model model.gguf --prompt "hello" --threads 2          # 双线程并行
```

### 最近合入（main@4d16fc8）

- 新增 `--q8-lm-head`（默认关闭）可选路径：
  - 仅当以下条件同时满足才启用 Q8 lm_head logits path：
    1. 显式开启 `--q8-lm-head`
    2. `lm_head` 与 `token_embd` 绑定（`output.weight` 缺失）
    3. `token_embd.weight` 类型为 Q8_0
    4. 原始 Q8_0 bytes 可用
  - 否则自动安全回退到原有 f32 matvec logits 路径
- SmolLM-135M.Q4_0 基准（decode 128 tokens）：
  - threads=1: 23.37 → 32.38 tok/s（+38.6%）
  - threads=2: 23.89 → 28.31 tok/s（+18.5%）
- 新增 Q4_1 最小支持（correctness-first）：
  - `dequantize_q4_1(...)`
  - `load_tensor_q4_1(...)`
  - `matvec_q4_1_f32(...)`
  - 目的：兼容“主体 Q4_0、少量 Q4_1 混入”的 GGUF（如 SmolLM2-135M-Instruct-Q4_0.gguf）

### 暂未完成

- threads=4/8 多线程稳定性（已知 hang 问题，下一阶段）
- 更完整的 tokenizer 兼容（SmolLM2/Llama3 tokenizer 行为对齐）
- 与官方 llama.cpp 的 logits 对拍
