# minllama

`minllama` 是一个用于研究和最小实现 `llama.cpp` 核心路径的项目。目标不是复刻完整 `llama.cpp`，而是围绕其官方设计思路，提炼一条可直接实现、可运行、便于验证的最短路径。

第一阶段范围：

- Llama 2 风格 decoder-only 架构
- GGUF v3 单文件模型格式
- CPU-only 推理路径
- F16 + Q4_0 权重格式
- FP16 KV cache
- greedy decode
- 同步 API

暂不纳入第一阶段的内容包括多模型架构、多 GPU 后端、continuous batching、OpenAI 兼容 HTTP server、复杂采样器和更多量化格式。

## 文档

- [最小核心研究报告](docs/minimal-core-research-report.md)

## 当前实现状态

当前代码已进入第一阶段第 10 步：最小 RoPE 与 attention 前置算子。

- 提供 C/C++17 项目结构、CMake 构建、Makefile 包装命令和基础测试。
- 对外同步 API 已占位，包括模型加载/释放、context 创建/释放/reset、tokenize、prefill、next token、logits 获取和 detokenize。
- `ml_model_load` 已实际打开文件并校验 GGUF v3 固定 header 子集：magic、version、tensor 数量和 metadata KV 数量。
- 当前已能解析 Llama 2 风格模型所需的最小 metadata 子集和 tensor info，并建立内部 tensor index；同时已能为 F32/F16/Q4_0 tensor 建立 data 区最小边界视图。
- 当前已能读取 F32/F16 tensor 内容并转换为内部 float 数组，也已能参考级解码 Q4_0 tensor。
- 当前已有 correctness-first 参考级 F32/F16/Q4_0 matvec、RMSNorm/softmax/SiLU/SwiGLU，以及最小 RoPE 和 attention 前置数学算子。
- 仍未进入完整 attention、KV cache 和 transformer 层编排。
- CLI 仅打印 skeleton / not implemented 状态信息。

本阶段目标是保证结构干净、可构建、可测试，并开始具备真实文件头校验能力；不包含真实模型推理能力。
