# 实现 llama.cpp 核心功能的最小代码研究报告

## 执行摘要

本项目的需求是：围绕 `llama.cpp` 的官方设计思路，提炼一套可直接实现、可运行、便于验证的“最小核心代码”，而不是复刻整个项目。目标不是支持几十种模型、十几种后端和完整服务层的全功能系统，而是一条足以覆盖模型加载、量化权重、KV cache、核心算子、线程策略、API、构建、跨平台、测试与许可的最短实现路径。

基于官方仓库、GGUF 规范、官方最小推理代码和 Llama 2 论文，最合理的最小边界是：

- 只支持 Llama 2 风格 decoder-only 架构。
- 只支持 GGUF v3。
- 第一版只做 CPU 路径。
- 只支持单序列或小 batch。
- 先做 greedy decode。
- 以同步 API 为主，异步用包装层实现。

如果把目标进一步压缩成一句话，结论是：最小实现不应从“通用张量图执行器”起步，而应从“Llama 2 专用 GGUF 解析器 + 两种权重格式 + 一套量化 matvec 核心 + KV cache + 简化 tokenizer + 贪心解码循环”起步。

这样可以在几千行量级内保留 `llama.cpp` 最关键的运行路径；第一版跑通后，再补 SIMD、K-quants、连续批处理、服务接口和 GPU 后端，演进方向也与官方实现兼容。

从工程优先级看，第一版建议如下：

- 文件格式选 GGUF，不兼容旧 GGML/GGMF/GGJT。
- 量化先支持 F16 + Q4_0。
- Q8_0 作为第二步。
- Q4_K/Q5_K/Q6_K 放到第三步。
- KV cache 先固定 FP16。
- tokenizer 先做“可运行的 GGUF 内嵌词表实现”，但预留接入更精确 tokenizer 的接口。
- 线程先按输出行分块并行。
- API 先模仿 `llama_model_load_from_file` / `llama_decode` 这类同步调用，再用队列包装异步。

相较完整 `llama.cpp`，最小实现会放弃多模型、多后端、连续批处理、OpenAI 兼容 server、复杂采样和大量量化格式，但换来显著更低的实现复杂度和更强的可验证性。

## 研究依据与范围

优先参考资料包括：

- `llama.cpp` 官方仓库。
- `ggml` 官方仓库。
- GGUF 规范。
- `llama.cpp` 构建指南。
- `llama-server` README。
- Llama 2 论文。

这些资料共同说明了三件事：

1. `llama.cpp` 的定位本来就是“少依赖、跨硬件、量化友好”的本地推理库。
2. GGUF 已成为官方推荐的模型交换与加载格式。
3. Llama 2 的架构特征，包括 4k 上下文、RMSNorm、SwiGLU、RoPE、70B 上的 GQA，正适合作为“最小实现”的具体目标。

本报告采用的具体范围是：

1. 模型家族限定为 Llama 2 风格 decoder-only transformer。
2. 文件格式限定为 GGUF v3 单文件。
3. 执行路径限定为 CPU-first，避免最小代码被 GPU runtime、后端抽象和设备内存管理吞没。
4. 推理行为限定为单会话或少量会话，batch=1 优先，greedy decode 优先。
5. 量化支持从 F16 + Q4_0 起步，后续再向 Q8_0 和 Q4_K_M 扩展。
6. 服务层不做完整 HTTP server，只在 API 维度说明如何包装异步。

这个边界既足以覆盖 `llama.cpp` 核心机制，又能把实现复杂度压到可控范围。

### 最小系统边界

下面是建议的最小系统边界。它不是完整 `llama.cpp` 的模块图，而是“能跑起来且便于实现”的最短核心链路。

```text
GGUF file
  -> GGUF loader
  -> metadata + tensor index
  -> Model object

Prompt text
  -> Tokenizer
  -> Prompt token ids
  -> Prefill loop
  -> KV cache
  -> Decode loop
  -> Logits
  -> Greedy / sampler
  -> Next token
  -> Detokenize
```

核心思想是：用 GGUF 负责“把模型和 tokenizer 元数据装进一个文件”，用自定义 CPU 核心负责“把量化矩阵乘向量、注意力和 KV cache 跑起来”，其余都尽量推迟。GGUF 适合做边界，因为它从设计上就是 single-file、extensible、mmap-compatible，并且明确描述了 model metadata 与 tensor info。

## 核心实现维度

### 模型文件格式与加载

第一版只支持 GGUF v3、单文件、little-endian、Llama 2 风格标准 tensor names。GGUF 已经取代 GGML/GGMF/GGJT，设计目标就是单文件部署、可扩展元数据、mmap 兼容和“模型加载所需信息完整内嵌”。规范也明确给出了 header、metadata、tensor info 的组织方式，并推荐 `token_embd`、`output_norm`、`blk.N.attn_q`、`blk.N.ffn_up` 等标准命名。

对“最小实现”来说，支持旧格式会把解析器复杂度拉高，但不会显著提升学习价值；因此第一版应显式拒绝旧格式。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 解析 magic/version/tensor_count/kv_count；读取 metadata；建立 `name -> TensorInfo` 索引；校验 alignment、offset、dtype、shape、重复 tensor 名；只接受 `general.architecture=llama` 或同等 Llama 2 子集 |
| 必要数据结构 | `GgufHeader`、`GgufKV`、`TensorInfo`、`ModelConfig`、`TensorView`、`Model` |
| 核心函数原型 | `bool gguf_open(const char *path, GgufFile *out);`、`bool gguf_read_header(...);`、`bool gguf_read_metadata(...);`、`const TensorInfo * gguf_find_tensor(...);` |
| 复杂度与内存估算 | 加载索引约 `O(n_kv + n_tensor)`；若使用 mmap，权重不复制到 heap，常驻额外内存主要是 metadata 与 tensor index，通常为 MB 级以内 |
| 潜在问题 | 大小端、分片 GGUF、未知 metadata key、tensor 命名不规范、offset 溢出、重复 key/tensor |
| 替代方案 | 若追求“最少胶水代码”，可直接链接 GGUF 读取库；若追求“最少外部依赖”，实现一个只覆盖 GGUF v3 子集的私有解析器 |

一个足够小但可工作的加载器结构可以先写成：

```cpp
enum class MlDType : uint32_t { F32, F16, Q4_0, Q8_0 };

struct GgufHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct TensorInfo {
    std::string name;
    MlDType dtype;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t offset;
};

struct ModelConfig {
    int32_t n_vocab = 0;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_ctx_train = 0;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
};

struct Model {
    ModelConfig cfg;
    std::unordered_map<std::string, TensorInfo> tensors;
    void *mapped_base = nullptr;
    size_t mapped_size = 0;
};

bool ml_model_load_gguf(const char *path, Model *out);
const TensorInfo * ml_model_find_tensor(const Model *m, const char *name);
```

这段结构的关键在于第一版就把“配置”和“tensor 索引”分离。后续无论加更多量化类型、LoRA、多后端，还是切换成 GGUF 官方读取器，都不需要推翻 `ModelConfig + TensorIndex + TensorView` 这层组织。

### 权重量化与存储

完整 `llama.cpp` 已支持从 1.5-bit 到 8-bit 的多种量化路线，既有老的 Q4_0/Q5_0/Q8_0，也有更复杂的 Q2_K/Q4_K/Q6_K/IQ*。但对最小实现来说，第一版最值得支持的是 F16 + Q4_0：

- F16 便于做参考真值与数值对拍。
- Q4_0 的 block 结构最简单。
- 第二阶段优先扩展 Q8_0。
- 第三阶段再补 Q4_K_M/Q6_K。

官方 70B 量化对比里，Q4_0 相比 FP16 的 perplexity 增量约为 3.61%，Q4_K_M 约为 1.20%，Q6_K 约为 0.16%。这说明教学最小版可以从 Q4_0 起步，但工程版应尽快补 Q4_K_M。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 第一版支持 F16 和 Q4_0；第二版补 Q8_0；第三版再补 Q4_K_M/Q6_K |
| 必要数据结构 | `BlockQ40`、`BlockQ80`、`TensorView`、`MatVecKernelDesc` |
| 核心函数原型 | `void matvec_q4_0_f32(...);`、`void matvec_q8_0_f32(...);`、`void matvec_f16_f32(...);` |
| 复杂度与内存估算 | decoder 单 token 路径主耗时是量化矩阵乘向量；复杂度随矩阵规模线性增长，通常更受内存带宽限制 |
| 潜在问题 | “先整矩阵解量化、再相乘”会严重放大内存流量；不宜把量化张量整块反量化到临时缓冲区 |
| 替代方案 | 直接复用 ggml/llama.cpp 的现成 quant kernel；若坚持手写，至少做 block-wise on-the-fly dequant |

Q4_0 的最小实现可以非常具体。一个经典 block 是 32 个权重对应一个半精度 scale 和 16 个 packed nibble，重点是在 inner loop 里边解码边累加。

```cpp
struct BlockQ40 {
    uint16_t d;
    uint8_t qs[16];
};

static inline float fp16_to_fp32(uint16_t h);

void matvec_q4_0_f32(
    const BlockQ40 * w,
    const float * x,
    float * y,
    int rows,
    int cols
) {
    const int nb = cols / 32;
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const BlockQ40 & blk = w[r * nb + b];
            const float d = fp16_to_fp32(blk.d);
            const float * xv = x + b * 32;
            for (int i = 0; i < 16; ++i) {
                uint8_t q = blk.qs[i];
                int q0 = (q & 0x0F) - 8;
                int q1 = (q >> 4) - 8;
                acc += d * q0 * xv[i];
                acc += d * q1 * xv[i + 16];
            }
        }
        y[r] = acc;
    }
}
```

这段代码直接定义了“最小实现”的性能上限与代码风格。后续需要 SIMD 时，可把最内层替换成 AVX2/NEON；转向 Q4_K_M 时，核心变化也仍然发生在 block kernel 层，不应波及更外层的模型调度。

### 内存管理

`ggml` 官方说明的一个重要目标是 runtime 零分配，而 `llama.cpp` 的模型参数结构也暴露了 `use_mmap`、`use_direct_io`、`use_mlock`、`check_tensors` 等选项。因此最小实现不应把内存管理当成次要细节，而应把它作为第一层架构：

- 模型权重尽量 mmap。
- KV cache 一次性预分配。
- 临时激活与 scratch 用单调 arena 管理。
- token 循环里不再发生 `malloc/new`。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | `ModelStorage`（mmap/heap 二选一）+ `KVCache` + `ScratchArena` 三段式；加载期可分配，token 热路径零分配 |
| 必要数据结构 | `MMapRegion`、`Arena`、`KVCache`、`RuntimeBuffers` |
| 核心函数原型 | `bool map_file_ro(...);`、`void arena_init(...);`、`void * arena_alloc(...);`、`bool kv_cache_init(...);` |
| 复杂度与内存估算 | 总内存近似为 `weights_resident + kv_cache + scratch + page tables`；mmap 加载时延更低、初始 RSS 更低，但长期仍受页缓存与访问模式影响 |
| 潜在问题 | 首 token 页错误、mlock 权限不足、Windows 文件映射语义差异、NUMA 远程内存、过多小 buffer 导致碎片化 |
| 替代方案 | 若追求实现简单，可放弃 mmap，全部 `read + malloc`；若追求接近官方，可继续加 mlock、huge pages、NUMA pinning |

KV cache 的推荐公式要尽早固定，因为它直接决定 7B/13B/70B 能否跑起来。对于 decoder-only 模型、batch=1、FP16 K/V：

```text
KV_bytes ~= 2 * n_layer * n_ctx * n_head_kv * head_dim * bytes_per_elem
```

其中前面的 `2` 代表 K 和 V 两份缓存。对于 4k 上下文，Llama 2 7B/13B 因为不是 GQA，KV cache 会比很多人想象的大；70B 虽然参数更多，但因为 8 组 KV 投影，KV cache 反而更省。这也是论文里强调 GQA 能改善推理 scalability 的原因。

### 张量与计算内核

Llama 2 的核心结构并不复杂：预归一化 transformer、RMSNorm、RoPE、自注意力、SwiGLU 风格 MLP，以及输出归一化和 lm head。官方最小推理代码直接把这些部件写在紧凑模型定义里；论文也明确指出 Llama 2 延续了标准 transformer 主体，同时采用 RMSNorm、SwiGLU 和 RoPE，并在大模型上使用 GQA。

因此，最小实现无需先做“通用图执行器”，而应先做 Llama 2 专用算子编排器：

```text
attn_norm -> q/k/v/o -> residual -> ffn_norm -> gate/up/down -> residual
```

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 以 matvec 优先，而不是通用 matmul 优先；单 token decode 的主路径全部按“量化矩阵 x FP32/F16 向量”设计 |
| 必要数据结构 | `VecF32`、`TensorView`、`LayerWeights`、`AttentionScratch` |
| 核心函数原型 | `rmsnorm_f32(...)`、`rope_apply_f32(...)`、`attention_decode(...)`、`swiglu_f32(...)` |
| 复杂度与内存估算 | 单 token decode 每层大致为 `O(d^2 + d*ffn + t*d_kv)`；prefill 才真正接近 `GEMM / O(s*d^2 + s^2*d_kv)` |
| 潜在问题 | 若把 decode 也写成普通 GEMM，会增加搬运和 packing 成本；softmax、RoPE 和 cache reorder 的细节容易出错 |
| 替代方案 | 对 F16/F32 路径可接 BLAS；对量化路径若不想自己写 SIMD，可直接借用 ggml kernel |

对最小版来说，应先把输出级并行想清楚，而不是急着上所有 SIMD。单 token 生成时，Q/K/V/O 四个投影和 MLP 的三个矩阵，本质上都是“很多行，共享一个输入向量”的计算。因此最自然的线程划分是按输出行块分工，最自然的 SIMD 方向是在 block 内做 quant unpack + dot product。注意力本身建议保持 FP32 累加，K/V cache 第一版固定 FP16，方便对拍和定位误差。

### 推理流程

在 `llama.cpp` 的 C API 里，输入批次由 `llama_batch` 表示，`llama_decode()` 负责处理一个 batch。Llama 2 论文中，tokenizer 是 32k 词表的 SentencePiece BPE，未知 UTF-8 字符按字节拆解。GGUF 规范进一步指出：模型文件可以嵌入 `tokenizer.ggml.model`、`tokens`、`scores`、`token_type` 等字段，但也提醒内嵌 GGML tokenizer 的实现精度可能低于原始 tokenizer。

因此，最小实现的推理流程最好分成两层：

- 数学核心必须稳定。
- tokenizer 可以先“可运行”，但必须允许以后替换成更精确的实现。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | `tokenize -> prefill -> decode loop -> detokenize` 四段式；第一版先做 greedy；采样器接口预留但不复杂化 |
| 必要数据结构 | `Tokenizer`、`ContextState{pos,last_logits}`、`KVCache`、`GenerateOptions` |
| 核心函数原型 | `int ml_tokenize(...)`、`int ml_prefill(...)`、`int ml_decode_next(...)`、`int ml_detokenize(...)` |
| 复杂度与内存估算 | tokenizer 与 detokenize 属于前后处理；主耗时在 prefill 和 decode；上下文长度对 decode 的增量成本主要通过 attention 读 cache 体现 |
| 潜在问题 | tokenizer 不精确会导致和官方实现输出不一致；chat template、BOS/EOS、digit split、byte fallback 都会影响结果 |
| 替代方案 | “精确模式”接入外部 SentencePiece/HF tokenizer；“内核模式”接受预 tokenized 输入，只验证数学路径 |

用户态流程可以写成非常直接的循环：

```cpp
std::vector<int32_t> ids = tokenizer.encode(prompt, true);

ctx.reset();
ctx.prefill(ids.data(), (int)ids.size());

for (int step = 0; step < max_new_tokens; ++step) {
    int32_t next = ctx.greedy_next_token();
    if (next == tokenizer.eos_id()) {
        break;
    }
    output.push_back(next);
    ctx.prefill(&next, 1);
}

std::string text = tokenizer.decode(output);
```

这个循环剥离了复杂的“批处理 API + 多序列 + 采样器 + server slot”，保留真正不可省略的东西：prompt prefill、位置推进、KV append、基于最后 logits 产出下一个 token。若后续需要多序列或连续批处理，只要把 `ContextState` 扩展成多 slot 即可。

## 工程化维度

### 多线程与并行策略

官方上下文参数把 `n_threads` 和 `n_threads_batch` 分开，说明 `llama.cpp` 已经把“单 token 生成”和“prompt/batch 处理”视为两种不同并行场景。构建文档还明确指出：BLAS 主要改善大 batch 的 prompt processing，对 generation 性能没有帮助。

因此，最小实现不应使用一个并行策略统治所有场景：

- decode 用较保守的线程数按输出行并行。
- prefill 允许更高线程数。
- F16/F32 路径必要时可接 BLAS。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | decode：按输出行块并行；prefill：允许更高并发，必要时可接 BLAS；注意力部分只在 head 或 token 维度轻量并行 |
| 必要数据结构 | `ThreadPool`、`ParallelFor`、`AffinityHints` |
| 核心函数原型 | `parallel_for(begin, end, grain, fn);`、`set_num_threads(decode_n, batch_n);` |
| 复杂度与内存估算 | 多线程提升受内存带宽上限约束，通常不会线性扩展到很高线程数 |
| 潜在问题 | 过度并行导致 cache thrash、false sharing、NUMA 远程访问、线程切换开销 |
| 替代方案 | 第一版可以先不做自定义线程池，只做 OpenMP / `std::thread` 行块划分；后续再换轻量线程池 |

decode 不适合过度并行，因为它本质上是“读大模型、算小向量”的带宽主导工作负载。线程数超过内存控制器的甜点区间后，吞吐通常不再明显上升。官方 `llama-bench` 示例里，7B Q4_0 的 CPU tg16 从 1 线程约 4 tok/s 提升到 8 线程约 16.7 tok/s，但继续加到 16 或 32 线程时增益已经明显变缓。

### API 设计

`llama.cpp` 的核心 C API 是同步式的：模型加载、批次初始化、`llama_decode()`、`llama_synchronize()`、读取 logits，都是显式控制流程。异步、多用户并行和 continuous batching 主要出现在 `llama-server` 一层。

最小实现的 API 设计也应遵循同样分层：

- 内核 API 保持同步和可预期。
- 异步只作为上层包装，不在第一版把状态机复杂化。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 核心库只暴露同步 C API；异步生成通过 worker thread + request queue + per-request context 包装 |
| 必要数据结构 | `ml_model`、`ml_context`、`ml_batch`、`ml_request`、`ml_future` |
| 核心函数原型 | `ml_model_load(...)`、`ml_context_create(...)`、`ml_prefill(...)`、`ml_next_token(...)`、`ml_get_logits(...)`、`ml_generate_async(...)` |
| 复杂度与内存估算 | 同步 API 实现最简单；异步 API 额外成本主要是上下文复制/独占与请求队列 |
| 潜在问题 | 共享一个 context 做多请求会产生竞争；KV cache 是状态性的，不能无锁共享 |
| 替代方案 | 若要服务化，直接做“每个 slot 一个 context”；更复杂的 continuous batching 放到后续版本 |

建议 API 形状模仿下面这个最小集合：

```c
typedef struct ml_model ml_model;
typedef struct ml_context ml_context;

typedef struct {
    int n_ctx;
    int n_threads;
    int n_threads_batch;
    int use_mmap;
    int use_mlock;
} ml_context_params;

ml_model * ml_model_load(const char *path);
void ml_model_free(ml_model *m);

ml_context * ml_context_create(ml_model *m, const ml_context_params *p);
void ml_context_free(ml_context *ctx);

int ml_tokenize(ml_context *ctx, const char *text, int add_bos, int32_t *out_ids, int cap);
int ml_prefill(ml_context *ctx, const int32_t *ids, int n_ids);
int32_t ml_next_token(ml_context *ctx);
const float* ml_get_logits(ml_context *ctx);
```

### 依赖与构建

`llama.cpp` README 明确写了“纯 C/C++ 实现、无依赖”，构建文档则表明 CPU、BLAS、Metal、CUDA、SYCL、HIP、Vulkan 等后端都是逐层加上的。

对最小实现来说，依赖策略应当是：

- 核心库只依赖标准 C/C++ 运行库。
- BLAS、SentencePiece、CUDA 都做成可选编译开关。
- 同时提供 CMakeLists.txt 和 Makefile：前者保证跨平台，后者方便快速复现。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 默认纯 C++17；可选 `MINLLAMA_USE_BLAS`、`MINLLAMA_USE_SENTENCEPIECE`、`MINLLAMA_USE_SIMD`、`MINLLAMA_USE_CUDA` |
| 必要数据结构 | 无特殊结构，关键是 CMake options 与 feature macros |
| 核心函数原型 | 无；构建逻辑由 `CMakeLists.txt` 管理 |
| 复杂度与内存估算 | 纯 CPU 默认构建最容易移植；BLAS 主要利于 prefill；GPU 版本会显著增加代码体量和 CI 复杂度 |
| 潜在问题 | 第三方库版本不齐、Windows/MSVC 的 half 类型与对齐差异、不同平台 SIMD flag 不一致 |
| 替代方案 | 教学版只交付纯 CPU；工程版再分支出 cpu/blas/cuda profile |

这里最值得强调的一点是：官方构建文档说明 BLAS 会提升 batch size 较高时的 prompt processing，但不影响 generation 性能。因此最小内核不应优先把 decode 路径塞进 BLAS；更值得优先优化的是量化 matvec kernel、线程策略和内存布局。

### 跨平台兼容性

官方 README 列出了 Apple Silicon、x86 的 AVX/AMX、RISC-V、CUDA、HIP、MUSA、Vulkan、SYCL 等能力；构建指南也分别说明了 Linux/macOS/Windows 的 CMake 构建方式、macOS 的 Metal 默认开启，以及 Windows 上不同 generator 的差异。

最小实现的合理跨平台策略不是“一次写完所有优化”，而是：

- 保证所有平台有标量 fallback。
- 平台相关只落在 mmap、对齐分配、线程亲和和 SIMD 宏分发层。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 标量路径必须先正确；平台优化做成可选分发：AVX2/AVX512/NEON；文件映射抽象成 `platform_map_file()` |
| 必要数据结构 | `PlatformFileMap`、`AlignedBuffer`、`CpuCaps` |
| 核心函数原型 | `bool platform_map_file(...);`、`void * aligned_alloc64(...);`、`CpuCaps detect_cpu_caps();` |
| 复杂度与内存估算 | 平台抽象增加的代码量很小，却能显著降低后续维护成本 |
| 潜在问题 | Windows 的 `MapViewOfFile`、POSIX `mmap`、macOS `mlock` 权限、MSVC 不支持 GCC-style intrinsics 宏 |
| 替代方案 | 若目标只是“尽快跑通”，第一版可先只支持 Linux/macOS；第二版再补 Windows |

GGUF 文档还提醒了一个常被忽略的点：模型默认 little-endian，但 v3 允许 big-endian；同时“如何识别 big-endian 模型”在文档里仍被标注为待完善。因此第一版应明确只接受 little-endian GGUF，而不是隐式假设所有情况都兼容。

### 测试与验证

官方 `llama-bench` 区分了 prompt processing、text generation、prompt+generation 三类测试，并明确说明吞吐数字不包含 tokenization 与 sampling。官方 perplexity 工具则说明 perplexity 主要用于比较量化模型和 FP16 的质量损失，并提醒不同项目之间的数值不可直接横向比较。

对最小实现而言，验证闭环应当是：

1. 数学一致性。
2. tokenization 一致性。
3. 性能基准。
4. 质量基准。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 先单元测试，再与官方实现对拍 logits/top-1，再做 pp/tg 分离 benchmark，最后做 perplexity 质量验证 |
| 必要数据结构 | `TestCase`、`ReferenceLogits`、`BenchmarkConfig` |
| 核心函数原型 | `run_quant_tests()`、`run_kernel_tests()`、`run_logits_diff()`、`run_pp_bench()`、`run_tg_bench()` |
| 复杂度与内存估算 | 对拍 logits 需要保存参考值；perplexity 需要完整语料遍历，耗时远高于 smoke test |
| 潜在问题 | tokenizer 误差会污染数学对拍；温度、采样和 chat template 会让比较不可复现 |
| 替代方案 | 使用“预 tokenized 提示”先隔离 tokenizer；采样前先只比较 greedy top-1 与 logits 误差 |

建议测试顺序严格执行四层闭环：

1. Parser/quant/kernel 单测：不给模型文件外部变量留空间。
2. 同模型同量化同 prompt 的 logits 对拍：优先看 top-1、cosine similarity、max abs err。
3. pp/tg 分离性能：分别测提示吞吐和生成吞吐。
4. perplexity 或 KL divergence：比较量化质量与 FP16。

### 安全与许可

安全和许可不能放到最后顺带处理。官方安全公告在 2025 年披露过 tokenizer 的 signed/unsigned 溢出问题，影响 `llama_vocab::tokenize` 相关路径，说明输入文本长度、token 个数上界、`size_t/int32_t` 转换都必须严格防御。

许可层面，`llama.cpp` 代码本身是 MIT，但 Llama 2 权重与衍生权重受单独的社区许可证约束。因此“代码 MIT”不等于“模型也 MIT”。如果把 Llama 2 转成 GGUF，这个 GGUF 的权重许可仍然不是 MIT。

| 项目 | 建议 |
| --- | --- |
| 最小实现要点 | 对所有长度、offset、乘法、shape、token 数量做溢出检查；模型文件默认只读打开；分离“代码许可”和“模型许可” |
| 必要数据结构 | `SafeSize` helpers、`LicenseInfo`、`ModelMetadataAudit` |
| 核心函数原型 | `bool checked_mul_u64(...);`、`bool checked_add_u64(...);`、`bool validate_tensor_bounds(...);` |
| 复杂度与内存估算 | 安全检查增加的 CPU 成本很小，但能显著降低解析器与 tokenizer 的攻击面 |
| 潜在问题 | 长文本、恶意 GGUF、重复 key、极端 token 数、越界 offset、错误的 dtype/shape 组合 |
| 替代方案 | 面向不可信输入时，建议把 tokenizer 与 GGUF 解析进一步沙箱化；面向可信内网环境，可保留严格断言但简化隔离机制 |

最小原型里建议默认启用这些安全规则：

1. GGUF header/version/magic 不符立即拒绝。
2. 所有 `offset + size` 都做 64 位有界检查。
3. 所有 `size_t -> int32_t` 转换前必须确认上界。
4. tokenizer 必须先“求长度”，再按长度分配目标缓存，且绝不依赖负值返回做隐式约定。
5. 模型输出到日志时不直接打印全部 metadata。

许可方面，代码仓库内应同时放置 `LICENSE` 和 `NOTICE-MODELS.md`，后者专门记录每个 GGUF 的来源模型及其许可。

## 最小可运行原型

如果把上述边界落成一个教学级、但可跑 7B/13B/70B 的原型，建议文件组织如下。它覆盖 GGUF 解析、F16/Q4_0、CPU decode、同步 API、可选异步包装、基础单测和 benchmark；不覆盖多模型架构、多 GPU、连续批处理、HTTP server、grammar sampling、K/I-quants。

| 文件 | 作用 | 预计代码量 |
| --- | --- | --- |
| `include/minllama.h` | 对外 C API | 150-250 LOC |
| `src/main.cpp` | CLI，负责参数解析与打印结果 | 150-300 LOC |
| `src/gguf.cpp` | GGUF v3 子集解析、tensor index、metadata 读取 | 400-700 LOC |
| `src/tokenizer.cpp` | 内嵌 tokenizer / 可选精确 tokenizer 适配 | 300-700 LOC |
| `src/quant.cpp` | F16/Q4_0/Q8_0 block 结构与 dequant/matvec | 300-600 LOC |
| `src/kernels.cpp` | RMSNorm / RoPE / softmax / SwiGLU / attention | 400-800 LOC |
| `src/model.cpp` | Llama 2 权重绑定与 layer 视图 | 250-500 LOC |
| `src/runtime.cpp` | Context、KV cache、prefill、decode loop | 350-700 LOC |
| `src/platform.cpp` | mmap/Windows file mapping、aligned alloc | 150-300 LOC |
| `src/thread_pool.cpp` | 简单线程池与 `parallel_for` | 150-300 LOC |
| `tests/test_quant.cpp` | 量化与 kernel 单测 | 150-250 LOC |
| `tests/test_logits.cpp` | 与官方实现对拍 logits | 150-300 LOC |
| `tests/bench.cpp` | pp/tg benchmark | 150-250 LOC |
| `CMakeLists.txt` | 主构建脚本 | 60-120 LOC |
| `Makefile` | 快速封装常用命令 | 30-60 LOC |

一个足够小的公共头文件可以长成下面这样：

```c
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ml_model ml_model;
typedef struct ml_context ml_context;

typedef struct {
    int n_ctx;
    int n_threads;
    int n_threads_batch;
    int use_mmap;
    int use_mlock;
} ml_context_params;

ml_model * ml_model_load(const char *path);
void ml_model_free(ml_model *m);

ml_context * ml_context_create(ml_model *m, const ml_context_params *p);
void ml_context_free(ml_context *ctx);
void ml_context_reset(ml_context *ctx);

int ml_tokenize(ml_context *ctx, const char *text, int add_bos, int32_t *out_ids, int cap);
int ml_prefill(ml_context *ctx, const int32_t *ids, int n_ids);
int32_t ml_next_token(ml_context *ctx);
const float* ml_get_logits(ml_context *ctx);
int ml_detokenize(ml_context *ctx, const int32_t *ids, int n_ids, char *out, int cap);

#ifdef __cplusplus
}
#endif
```

示例 `CMakeLists.txt` 策略如下，遵循“默认纯 CPU、可选 BLAS、可选更精确 tokenizer”：

```cmake
cmake_minimum_required(VERSION 3.20)
project(minllama LANGUAGES C CXX)

option(MINLLAMA_USE_BLAS "Use BLAS for dense FP paths" OFF)
option(MINLLAMA_USE_SENTENCEPIECE "Use exact SentencePiece tokenizer" OFF)
option(MINLLAMA_USE_SIMD "Enable SIMD intrinsics when supported" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(minllama
    src/gguf.cpp
    src/tokenizer.cpp
    src/quant.cpp
    src/kernels.cpp
    src/model.cpp
    src/runtime.cpp
    src/platform.cpp
    src/thread_pool.cpp
)

target_include_directories(minllama PUBLIC include)

if (WIN32)
    target_compile_definitions(minllama PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

if (MINLLAMA_USE_SIMD)
    target_compile_definitions(minllama PRIVATE MINLLAMA_USE_SIMD=1)
endif()

if (MINLLAMA_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(minllama PRIVATE MINLLAMA_USE_BLAS=1)
    target_link_libraries(minllama PRIVATE BLAS::BLAS)
endif()

if (MINLLAMA_USE_SENTENCEPIECE)
    find_package(SentencePiece REQUIRED)
    target_compile_definitions(minllama PRIVATE MINLLAMA_USE_SENTENCEPIECE=1)
    target_link_libraries(minllama PRIVATE sentencepiece)
endif()

add_executable(minllama_cli src/main.cpp)
target_link_libraries(minllama_cli PRIVATE minllama)

enable_testing()

add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE minllama)
add_test(NAME test_quant COMMAND test_quant)

add_executable(test_logits tests/test_logits.cpp)
target_link_libraries(test_logits PRIVATE minllama)
add_test(NAME test_logits COMMAND test_logits)

add_executable(minllama_bench tests/bench.cpp)
target_link_libraries(minllama_bench PRIVATE minllama)
```

配套 `Makefile` 可让常用动作一条命令执行：

```makefile
BUILD ?= build
CMAKE ?= cmake

all:
	$(CMAKE) -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(BUILD) -j

blas:
	$(CMAKE) -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DMINLLAMA_USE_BLAS=ON
	$(CMAKE) --build $(BUILD) -j

spm:
	$(CMAKE) -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DMINLLAMA_USE_SENTENCEPIECE=ON
	$(CMAKE) --build $(BUILD) -j

test: all
	ctest --test-dir $(BUILD) --output-on-failure

bench: all
	./$(BUILD)/minllama_bench

run: all
	./$(BUILD)/minllama_cli

clean:
	rm -rf $(BUILD)

.PHONY: all blas spm test bench run clean
```

默认构建命令和官方构建指南保持同一风格，CPU-only 路径最简单：

```bash
# 默认 CPU 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 开启 BLAS，主要提升 prefill 大 batch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMINLLAMA_USE_BLAS=ON
cmake --build build -j

# 运行
./build/minllama_cli -m ./models/llama-2-7b.Q4_0.gguf -p "The capital of France is" -n 8
```

### 示例输入输出

面向用户的 CLI 输出会受具体模型、量化版本、chat template、是否精确 tokenizer 等影响，因此报告里应同时给出一个“机器可验证”的测试输出和一个“用户可感知”的示例输出。前者必须确定，后者允许小幅波动。

机器可验证示例：

```bash
./build/test_logits \
    --model ./models/llama-2-7b.Q4_0.gguf \
    --prompt-ids "1,450,338,263,322" \
    --ref ./refs/llama_cpp_logits.bin
```

期望输出：

```text
PASS top1_match=1
PASS cosine_similarity=0.9991
PASS max_abs_err=0.0217
```

用户可感知示例：

```bash
./build/minllama_cli \
    -m ./models/llama-2-7b-chat.Q4_0.gguf \
    -p "用一句话解释 KV cache。" \
    -n 32
```

示例输出：

```text
KV cache 是把历史 token 的 Key/Value 保存在内存中，从而避免每一步都重算整个前缀。
```

### 测试步骤

最小原型的测试步骤建议固定成下面这组顺序，便于定位问题来源：

```bash
# 1. 编译
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. 单测：量化与 kernel
ctest --test-dir build --output-on-failure -R test_quant

# 3. 对拍：与官方实现比较 logits / top1
ctest --test-dir build --output-on-failure -R test_logits

# 4. 基准：分别测 pp / tg
./build/minllama_bench --model ./models/llama-2-7b.Q4_0.gguf --mode pp --prompt-len 512
./build/minllama_bench --model ./models/llama-2-7b.Q4_0.gguf --mode tg --gen-len 128

# 5. 端到端 smoke test
./build/minllama_cli -m ./models/llama-2-7b.Q4_0.gguf -p "Once upon a time," -n 32
```

## 规模化估算

为了让 7B、13B、70B 的估算有可落地意义，本节统一采用以下假设：

1. 模型家族是 Llama 2 风格。
2. 上下文长度 `n_ctx=4096`。
3. `batch=1`。
4. KV cache 第一版用 FP16。
5. 权重主比较对象是 Q4_0 和 F16。
6. CPU-only 性能估算参考官方 `llama-bench` 中 7B Q4_0 的 CPU tg 数据，再按模型权重规模和 GQA 带来的 KV 差异做一阶外推；这些吞吐只是工程估计，不是官方实测。

官方 benchmark 示例显示，7B Q4_0 在 CPU 上 tg16 可从 1 线程约 4 tok/s 提升到 8 线程约 16.7 tok/s，而官方示例中的 7B/13B Q4_0 CUDA 吞吐分别约在 132/82 tok/s 的量级。这说明完整 `llama.cpp` 的优化空间很大，也说明“最小实现”不应以 GPU 吞吐为第一目标。

| 模型 | 参考结构 | Q4_0 权重大小 | F16 权重大小 | KV@4k FP16 | Q4_0 总内存估算 | F16 总内存估算 | CPU-only 最小实现吞吐估计 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | 32 层，4096 hidden，32 heads | 约 3.56 GiB | 约 13.0 GiB | 约 2.0 GiB | 约 5.8-6.2 GiB | 约 15.3-16 GiB | 约 7-13 tok/s |
| 13B | 40 层，5120 hidden，40 heads | 约 6.86 GiB | 约 24.2 GiB | 约 3.125 GiB | 约 10.4-11.2 GiB | 约 27.7-29 GiB | 约 4-7 tok/s |
| 70B | 80 层，8192 hidden，64 q heads / 8 kv groups | 约 36.20 GiB | 约 128.5 GiB | 约 1.25 GiB | 约 38.5-40.5 GiB | 约 130.8-134 GiB | 约 0.6-1.3 tok/s |

表中 7B 与 13B 的 Q4_0 权重大小取自官方 benchmark 示例，70B 的 Q4_0 / FP16 权重大小取自官方 perplexity 文档中的旧量化记分板；KV cache 按 Llama 2 论文的 GQA 说明和 7B/13B/70B 常见配置计算。可以看到，70B 的真正门槛在权重，不在 KV cache；反过来，7B/13B 因为不是 GQA，4k context 下的 KV cache 更显眼。

70B 的官方旧记分板很有参考价值，因为它直接揭示了“代码复杂度”和“质量损失”之间的交换关系：

| 70B 量化格式 | 官方模型大小 | 相对 FP16 的 PPL 增量 | 对最小实现的启示 |
| --- | --- | --- | --- |
| Q4_0 | 36.20 GiB | 3.61% | 最容易实现，适合教学起步 |
| Q4_K_M | 38.54 GiB | 1.20% | 质量更好，是工程版应尽快补齐的目标 |
| Q6_K | 52.70 GiB | 0.16% | 质量很好，但实现与带宽成本都更高 |
| FP16 | 128.5 GiB | 基线 | 数值参考真值，不适合多数 CPU-only 本地部署 |

这对应建议路线：v1 支持 Q4_0，v2 支持 Q8_0，v3 补 Q4_K_M。如果一开始就追求 Q4_K_M，会更早获得更好的质量，但也会在解析 block packing 和写 SIMD 细节上付出数倍代码复杂度；先用 Q4_0 打通全链路，再引入 Q4_K_M，整体风险更低。

## 与完整 llama.cpp 的对比

完整 `llama.cpp` 今天已经不是一个“只跑 Llama 的最小库”，而是一个支持大量文本/多模态模型、多种量化格式、CPU 与多个 GPU 后端，并带有 OpenAI 兼容 server 和 continuous batching 的大工程。官方 README 与构建文档列出了多后端、多量化和 CPU+GPU hybrid 能力，server README 还明确列出了 parallel decoding、多用户支持和 continuous batching。

因此，最小实现与完整 `llama.cpp` 的差异，本质上不是“对错”，而是工程目标不同。

| 维度 | 完整 llama.cpp | 本项目最小实现 |
| --- | --- | --- |
| 支持模型 | 大量文本与多模态模型，不只 Llama | 只做 Llama 2 风格 decoder-only |
| 文件格式 | GGUF 为主，兼容更广生态 | 只支持 GGUF v3 子集 |
| 量化格式 | 1.5-8 bit 多路线，含 K/I/legacy quants | v1 只做 F16 + Q4_0，v2 补 Q8_0，v3 才补 Q4_K_M |
| 计算后端 | 纯 CPU、BLAS、Metal、CUDA、HIP、MUSA、Vulkan、SYCL 等 | CPU-only；SIMD 可选；GPU 暂不纳入 v1 |
| 内存管理 | 多后端 buffer、offload、hybrid、更多 runtime 选项 | mmap + KV cache + scratch arena 的三段式最小设计 |
| 推理模式 | 多序列、多 slot、parallel decoding、continuous batching | 单序列优先；多请求靠“每请求一 context”包装 |
| API | 完整 C API + 多种工具与绑定 | 同步 C API 最小子集 + 可选异步包装 |
| 服务层 | OpenAI 兼容 HTTP server、监控、schema 约束等 | 不内置 server，只保留库接口 |
| 测试体系 | `llama-bench`、perplexity、更多工具与 CI | 单测 + logits 对拍 + pp/tg benchmark + 可选 perplexity |
| 第三方依赖 | 核心无依赖；server 层会引入 HTTP/JSON 依赖 | 核心无依赖；BLAS/SentencePiece 均可选 |
| 粗略 LOC | 仓库级别、十万行以上数量级 | 核心约 2k-4k LOC；含 tokenizer/CLI/tests 约 4k-8k LOC |
| 性能预期 | 最优 CPU/GPU 路径远强于教学原型 | 单机 CPU 路径可用，但通常低于完整实现；是否接近取决于 SIMD 和 quant kernel 完成度 |

就“功能、性能、代码行数、依赖”这四个维度来说，结论可以归纳为：

> 完整 `llama.cpp` 是“广覆盖、高优化、强工程化”的参考目标；本项目最小实现是“窄边界、可实现、可验证”的教学与原型目标。

如果目标是快速理解和实现核心路径，最小实现更合适；如果目标是上线高并发服务或吃满 GPU，则应回到完整 `llama.cpp` 的设计方向，尤其是 quant kernels、后端抽象和 server slot 体系。

最后，从“实现者今天就能动手”的角度，关键顺序不是把所有 feature 同时做完，而是：

1. 先让 GGUF + F16/Q4_0 + KV cache + greedy decode 跑通。
2. 再做 logits 对拍。
3. 再做 SIMD。
4. 再做更复杂量化。
5. 最后做服务化。

这个顺序既符合官方资料体现的分层，也能最大限度降低第一次实现时的风险。对希望“直接拿去实现”的技术需求来说，这就是实现 `llama.cpp` 核心功能的最小代码路线。

