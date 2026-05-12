# minllama

`minllama` 是一个用于研究和最小实现 `llama.cpp` 核心路径的项目。目标不是复刻完整 `llama.cpp`，而是围绕其官方设计思路，提炼一条可直接实现、可运行、便于验证的最短路径。

第一阶段范围：

- Llama 2 风格 decoder-only 架构
- GGUF v3 单文件模型格式
- CPU-only 推理路径
- F16 + Q4_0 权重格式
- FP16 KV cache
- greedy decode + temperature sampling
- 同步 API

暂不纳入第一阶段的内容包括多模型架构、多 GPU 后端、continuous batching、OpenAI 兼容 HTTP server、复杂采样器和更多量化格式。

## 文档

- [最小核心研究报告](docs/minimal-core-research-report.md)

## 当前实现状态

第一阶段的 correctness-first 参考实现已基本完成，27 组测试全部通过。

### 实现进度（对照研究报告的核心维度）

| 维度 | 状态 | 测试 |
|------|:----:|------|
| GGUF v3 解析与模型加载 | ✅ 完成 | smoke, loader, metadata, tensors, tensor_data, tensor_read, model_loader |
| 权重量化与 matvec（F32/F16/Q4_0） | ✅ 完成 | quant, q40_decode, matvec |
| 计算内核（RMSNorm / softmax / SiLU / SwiGLU / RoPE） | ✅ 完成 | ops, rope |
| Attention 前置算子与单头 attention | ✅ 完成 | attention_math, attention_single_head |
| KV cache（FP16 读写 + decode attention） | ✅ 完成 | kv_cache |
| Transformer 层编排（Self-Attention + FFN） | ✅ 完成 | transformer_layer, transformer_layer_ffn |
| 完整模型前向（logits 输出） | ✅ 完成 | transformer_model, transformer_logits |
| Greedy decode 循环 | ✅ 完成 | greedy, generate, generate_eos |
| Temperature 采样 | ✅ 完成 | sampling |
| Tokenizer（内嵌词表 + 编码/解码） | ✅ 完成 | tokenizer, tokenizer_loader |
| CLI 骨架（参数解析） | ✅ 完成 | cli_args |
| 端到端文本生成 | ✅ 完成 | generate_text |

### 测试总览

```
27/27 tests passed (100%)
```

测试覆盖从 GGUF 文件头解析到端到端文本生成的完整链路。

### API

已实现研究报告建议的最小同步 C API 子集：

- `ml_model_load` / `ml_model_free` — 模型加载与释放
- `ml_context_create` / `ml_context_free` / `ml_context_reset` — 上下文管理
- `ml_tokenize` / `ml_detokenize` — tokenize 与 detokenize
- `ml_prefill` / `ml_next_token` — prefill 与单步 decode
- `ml_get_logits` — 获取当前 logits

### 暂未完成

- 线程池与多线程并行（当前为单线程）
- SIMD 加速（当前为标量 fallback）
- 与官方 llama.cpp 的 logits 对拍（需真实模型文件）
- CLI 端到端真实模型推理（需真实模型文件）
