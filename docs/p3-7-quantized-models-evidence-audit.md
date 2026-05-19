# P3-7 量化模型证据审计报告

**审计时间**: 2026-05-19  
**审计范围**: 本地 Ollama 量化模型 GGUF 支持验证  
**审计目标**: 将 SIMULATED/FAIL 状态转为可验证的 PASS/PARTIAL/FAIL 状态

## 📊 证据审计结果

| Model | Ollama Tag | GGUF Path | File Exists | Inspect GGUF | Validate Model | CLI Load | Generation Output | Final Status | Failure Reason |
|-------|------------|-----------|-------------|--------------|----------------|----------|-------------------|--------------|----------------|
| gemma4 | gemma4:e4b | ~/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a | ✅ | ✅ | ✅ | ✅ (partial) | ❌ | PARTIAL | CLI 无法完成完整生成 |
| qwen3.5 | qwen3.5:latest | ~/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c | ✅ | ✅ | ✅ | ✅ (partial) | ❌ | PARTIAL | CLI 无法完成完整生成 |
| qwen3 | qwen3:0.6b | ~/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa | ✅ | ✅ | ✅ | ✅ (partial) | ❌ | PARTIAL | CLI 无法完成完整生成 |
| deepseek-coder | deepseek-coder:6.7b | ~/.ollama/models/blobs/sha256-59bb50d8116b6a1f9bfbb940d6bb946a05554e591e30c8c2429ed6c854867ecb | ✅ | ✅ | ✅ | ✅ (partial) | ❌ | PARTIAL | CLI 无法完成完整生成 |

## 🔍 详细验证过程

### 1. GGUF 文件定位 ✅
- 所有4个模型均成功定位到 Ollama blob 文件
- GGUF 版本：v3（所有模型）
- 文件大小验证正确

### 2. GGUF 结构验证 ✅
```bash
# 文件存在性检查
✅ 所有 GGUF 文件均存在
✅ 文件大小与 Ollama list 显示一致

# GGUF 版本验证
✅ gemma4: 9.6 GB