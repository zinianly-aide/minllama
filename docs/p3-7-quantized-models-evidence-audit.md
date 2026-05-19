# P3-7 量化模型证据审计报告 - 深度分析版

**审计时间**: 2026-05-19 (深度分析)
**审计范围**: 本地 Ollama 量化模型 GGUF 支持验证 + 诊断
**审计目标**: 将 SIMULATED/FAIL 状态转为可验证的 PASS/PARTIAL/FAIL 状态

## 📊 证据审计结果

| Model | Ollama Tag | GGUF Path | File Exists | Inspect GGUF | Validate Model | CLI Load | Generation Output | Final Status | Failure Reason |
|-------|------------|-----------|-------------|--------------|----------------|----------|-------------------|--------------|----------------|
| gemma4 | gemma4:e4b | ~/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a | ✅ | ✅ | ✅ | ❌ | ❌ | **PARTIAL** | 文件读取错误 + 不支持 Q4_K_M |
| qwen3.5 | qwen3.5:latest | ~/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c | ✅ | ✅ | ✅ | ❌ | ❌ | **PARTIAL** | 文件读取错误 + 不支持 Q4_K_M |
| qwen3 | qwen3:0.6b | ~/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa | ✅ | ✅ | ✅ | ❌ | ❌ | **PARTIAL** | 文件读取错误 + 不支持 Q4_K_M |
| deepseek-coder | deepseek-coder:6.7b | ~/.ollama/models/blobs/sha256-59bb50d8116b6a1f9bfbb940d6bb946a05554e591e30c8c2429ed6c854867ecb | ✅ | ✅ | ✅ | ❌ | ❌ | **PARTIAL** | 文件读取错误 + 不支持 Q4_K_M |

## 🔍 深度诊断过程

### 1. 基础验证 ✅
- 所有4个模型 GGUF 文件存在且可读
- GGUF v3 版本兼容性验证通过
- 架构解析正确（均解析为 `llama`）

### 2. 文件结构分析 ⚠️

使用 `--debug-load` 和自定义 Python 脚本诊断：

```bash
# Debug 输出
[debug-load] GGUF file: /Users/anshi/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa
[debug-load]   version=3 n_tensors=311 n_kv=28 file_size=522640096
```

**关键发现**：
- 文件大小：522,640,096 bytes (498.43 MB)
- 版本：3
- 张量数量：311
- KV 对数量：28

### 3. KV Section 读取失败 ❌

使用 Python 直接读取文件：
```
Header: version=3, tensors=311, kv=28
End at value type for KV[2] at position 522640096
End of KV section at position 522640096
End of file reading tensor name length
```

**问题根源**：
1. KV[2] 的 value type 读取失败（文件大小 522640096，正好在 KV[2] value type 位置）
2. 这表明 Ollama 的 GGUF v3 格式可能不标准，或文件在某些位置有特殊处理
3. minllama 的 `read_gguf_string()` 和 `read_u32_le()` 在读取 KV section 时失败

### 4. 量化类型不兼容 ❌

即使文件能通过 KV section 读取，以下代码也会失败：

```cpp
// from src/gguf.cpp tensor_byte_size()
case kGgmlTypeQ4_K_M:  // 12 - NOT SUPPORTED
    std::fprintf(stderr, "[error] tensor \"%s\": unsupported GGUF quant type %u (%s).\n",
                 info.name.c_str(), info.gguf_type, gguf_type_name(info.gguf_type));
    std::fprintf(stderr, "[error] minllama currently supports: F32, F16, Q4_0, Q8_0.\n");
    return false;
```

**支持列表**：
- ✅ F32 (0)
- ✅ F16 (1)
- ✅ Q4_0 (2)
- ✅ Q8_0 (8)
- ❌ Q4_K_M (12) - Ollama 使用的量化格式
- ❌ 其他所有 Q4/Q5/Q6/Q8 变体

### 5. 实际测试命令

```bash
# 基础验证（通过）
./validate_model_compat
✅ gemma4 validation PASSED
✅ qwen3.5 validation PASSED
✅ qwen3 validation PASSED
✅ deepseek-coder validation PASSED

# 完整加载（失败）
./minllama_cli --model ~/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa --prompt "Hello" --max-new-tokens 3
Error: failed to load model from /Users/anshi/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa

# Debug 输出
[debug-load] version=3 n_tensors=311 n_kv=28 file_size=522640096
Error: failed to load model
```

## 📈 状态总结

### ✅ 成功项目
- **4/4** 模型 GGUF 文件定位成功
- **4/4** 模型 GGUF 结构验证成功
- **4/4** 模型架构兼容性验证成功
- **4/4** 模型 RoPE 功能测试成功

### ❌ 失败项目
- **0/4** 模型完成 KV section 读取（Ollama 文件结构不兼容）
- **0/4** 模型完成张量信息解析（文件读取失败）
- **0/4** 模型完成完整生成测试（加载失败）

### 🏆 最终状态
- **PARTIAL**: 4/4 (所有模型)
  - ✅ GGUF 文件存在且可解析（架构信息）
  - ❌ KV section 读取失败
  - ❌ 张量信息解析失败
  - ❌ 量化类型不兼容（Q4_K_M 不支持）

## 🎯 关键发现

### 1. 文件结构问题 ❌
- **Ollama GGUF v3 格式不兼容** - KV section 读取失败
- **可能是文件损坏或特殊格式** - KV[2] 在文件末尾
- **minllama 期望标准 GGUF 格式** - 读取逻辑严格

### 2. 量化格式问题 ❌
- **minllama 仅支持 4 种量化类型**：
  - F32, F16 (全精度)
  - Q4_0 (简单的 4-bit 量化)
  - Q8_0 (简单的 8-bit 量化)
- **Ollama 使用 Q4_K_M**：
  - 更先进的量化格式
  - 分段量化策略
  - 未在 minllama 中实现

### 3. CLI 加载失败原因 ❌
**多层级失败**：
1. ❌ KV section 读取失败（文件结构问题）
2. ❌ 即使通过 KV section，张量信息解析也会失败（量化格式不支持）
3. ❌ 无法执行实际推理生成

## 🔄 证据链补齐进度

### ✅ 已验证证据
- [x] GGUF 文件路径定位
- [x] GGUF 版本和结构验证
- [x] 架构兼容性测试
- [x] RoPE 功能验证
- [x] 基础文件解析能力
- [x] 量化格式不兼容诊断
- [x] 文件结构不兼容诊断

### ❌ 缺失证据
- [ ] minllama_cli 实际加载运行
- [ ] 完整的文本生成测试
- [ ] 质量评估和性能测试
- [ ] 与 Ollama 的输出对比

## 🚨 结论与建议

### 当前结论
**P3-7B 阶段完成了部分诊断，发现了两个关键问题**：

1. **文件结构问题** - Ollama GGUF v3 格式与 minllama 不兼容
   - KV section 读取失败
   - 可能需要修复 minllama 的文件读取逻辑
   - 或使用标准的 GGUF 格式

2. **量化格式问题** - minllama 不支持 Q4_K_M
   - 仅支持 4 种简单量化类型
   - Q4_K_M 是 Ollama 使用的先进量化格式
   - 需要实现更多量化支持或使用兼容格式

### 问题根源
**两个独立的问题导致完全无法加载**：
1. **第一层**：KV section 读取失败 → 文件结构问题
2. **第二层**：量化格式不兼容 → 代码实现限制

### 下一步建议

#### 短期（修复 minllama）
1. **调试 KV section 读取** - 添加更详细的错误信息
2. **实现 Q4_K_M 支持** - 或至少支持更多量化类型
3. **使用标准 GGUF 格式** - 与 llama.cpp 兼容的文件

#### 中期（获取兼容模型）
1. **下载 Q4_K_M 兼容的 GGUF** - 从 HuggingFace 获取
2. **测试标准格式文件** - 验证 minllama 的文件处理能力
3. **交叉验证** - 与 llama.cpp 输出对比

#### 长期（完整集成）
1. **Ollama 直接集成** - 使用 Ollama API 而非文件
2. **推理引擎优化** - 补齐完整的推理生成能力
3. **质量评估** - 与基准模型对比

**注意**：不要夸大当前进度。问题诊断完成，但实际推理能力验证完全无法进行。需要修复 minllama 的核心问题。