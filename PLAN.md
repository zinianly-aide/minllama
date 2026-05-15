# minllama — 开发计划

## 当前阶段：真实模型 Runtime Correctness 调试

## 优先级

**P0：**
- 真实模型 correctness
- tokenizer/logits/llama.cpp 对拍
- BOS/EOS/position/GQA correctness
- observability/debugability

**P1：**
- Q4_0/Q8_0 fused matvec
- 减少临时 decode

**P2：**
- SIMD
- OpenMP
- Metal

**暂不做：**
- 大规模 tokenizer/BPE 重写
- MoE/VLM
- 多 agent 同时改 kernels.cpp

---

## Agent 分工

- **Hermes/OpenClaw**（本 Agent）：主控 correctness、llama.cpp 对拍、tokenizer/debug、logits/debug、merge decision
- **Codex**：独立工具、独立优化、补测试、diagnostics、benchmark
- **GitHub PR Bot**：code review、regression risk、测试覆盖检查、架构风险提示

## 工作目录规则

- 一个 Codex 一个 worktree
- 一个任务一个 branch
- 不直接改主工作目录
- 不自动 merge main

## 当前主线任务

继续定位 "真实模型生成质量异常"

重点：
1. tokenizer 行为
2. BOS/EOS
3. prompt token ids
4. top logits
5. llama.cpp 对拍
6. tied embedding/lm_head
7. position/RoPE

不要把 tokenizer correctness 和 transformer correctness 混一起。

## Codex 任务板

### Codex A（最高优先级）— logits/tokenizer 对拍工具
新增 `tools/minllama_compare_logits.cpp`

### Codex B — Q4_0/Q8_0 fused matvec 草案

### Codex C — scan_tensors 增强版

## GitHub PR 流程

Codex 完成任务后：
1. `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`
2. `git checkout -b codex/<task-name>`
3. `git add && git commit -m "<type>: <summary>"`
4. push + create PR

PR 描述必须包含：目标、修改文件、新增函数、新增测试、ctest 结果、是否跑真实 SmolLM、已知风险。

## PR Bot 审查重点

1. correctness: GGUF alignment, Q4_0/Q8_0, GQA, KV cache, tokenizer/BOS/EOS
2. 数值安全: NaN/Inf, 越界, dim/head_dim/kv_dim 混淆, 未初始化
3. 测试覆盖: 非法输入, 数学 correctness, regression
4. llama.cpp 行为差异: tokenizer, RoPE, GQA, quant decode

## 当前状态（2026-05-15）

- ✅ 已同步远端 main（merge PR #1: optional raw Q8_0 tied lm_head path）
- ✅ GGUF loader
- ✅ tokenizer loader
- ✅ Q4_0/Q4_1/Q8_0（Q4_1 为最小标量支持）
- ✅ MHA/GQA
- ✅ KV cache
- ✅ transformer
- ✅ sampling
- ✅ CLI
- ✅ tensor alignment 修复
- ✅ 40/40 tests passed
- ✅ SmolLM-135M 可真实运行，无 NaN/Inf
- ✅ SmolLM2-135M-Instruct-Q4_0.gguf 已可加载与生成（不再被少量 Q4_1 阻塞）
- 🔴 SmolLM2 中文/指令质量仍异常（当前焦点转向 tokenizer/模板对齐，而非量化加载）
