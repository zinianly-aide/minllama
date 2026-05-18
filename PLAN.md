# minllama — 开发计划

## 当前阶段：平台化 + 性能稳定性

**重要更新（2026-05-17）**:
- ❌ stop performance micro-optimizations (P2: NO-ROI)
- ✅ transition to platformation phase (P3)
- ✅ P3-1 Benchmark CI infrastructure complete
- ✅ Stability > raw speed

---

## 优先级

**P3（平台化）：**
- ✅ **P3-1**: Benchmark CI infrastructure (COMPLETED)
  - Performance regression detection
  - Stable baselines
  - CI/CD integration
- 🔄 **P3-2**: Model compatibility matrix
- 📋 **P3-3**: Documentation refinement

**P2（性能优化 - 暂停）：**
- ⏸️ SIMD optimization
- ⏸️ OpenMP
- ⏸️ Metal

**P0/P1（已在前阶段完成）：**
- ✅ 真实模型 correctness
- ✅ tokenizer/logits 对拍
- ✅ Q4_0/Q8_0
- ✅ GQA
- ✅ KV cache
- ✅ Transformer forward

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

### P3-1: Benchmark CI（✅ 已完成）
- GitHub Actions workflows (ci.yml + benchmark-ci.yml)
- Local benchmark tools (run_benchmarks.sh + compare_benchmarks.py)
- Performance gate: >5% regression
- 5,745 bytes total documentation

### P3-2: Model Compatibility（🚧 待开始）
目标：扩展 GGUF 支持矩阵
优先模型：
- TinyLlama
- Qwen2
- Gemma
- Phi

输出：
- compatibility matrix
- unsupported feature list
- tokenizer differences

---

## 重要原则（2026-05-17 转折点）

1. **Stability > Speed**
   - 一致性优先于原始速度
   - 可重复性优先于小幅优化

2. **Reality Check First**
   - synthetic benchmark 不可信
   - 真实场景优化才有 ROI

3. **CI as First-Class Citizen**
   - 任何改动必须有 CI 保护
   - 回归检测是必需品

4. **不要过度优化**
   - P2 micro-opt 已 NO-ROI
   - 专注 P3 platformation

---

## 实验记录

### P2 实验失败（2026-05-17）
- **实验 A**: attention loop order (0.76% speedup, NO-ROI)
- **实验 B**: KV read tiling (真实场景 1.00x, NO-ROI)

### 决策
- ❌ stop attention micro-opt
- ❌ stop KV tiling/token blocking/head chunking
- ❌ stop synthetic benchmark optimization
- ✅ transition to P3 platformation

## Codex 任务板（P3阶段）

### Codex A — P3-2 Model Compatibility
新增模型支持：TinyLlama, Qwen2, Gemma, Phi

### Codex B — P3-3 Documentation
完善 API 文档、使用示例、故障排查

### Codex C — P3-1 Integration
确保 CI 流程在生产环境正常运行

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

## 当前状态（2026-05-12）

- ✅ GGUF loader
- ✅ tokenizer loader
- ✅ Q4_0/Q8_0
- ✅ MHA/GQA
- ✅ KV cache
- ✅ transformer
- ✅ sampling
- ✅ CLI
- ✅ tensor alignment 修复
- ✅ 35/35 tests passed
- ✅ SmolLM 可真实运行，无 NaN/Inf
- 🔴 生成质量异常（下一步焦点）
