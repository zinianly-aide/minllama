# MiniCPM5-2B compatibility validation

This document defines the compatibility gate for running OpenBMB MiniCPM5-2B with minllama.
The goal is correctness evidence, not a claim that minllama matches llama.cpp performance or supports the model's full 131072-token training context.

## Scope

MiniCPM5-2B is packaged as a Llama-family causal LM. The relevant model contract is compatible with minllama's existing compute path:

- hidden size: 2048
- layers: 42
- attention heads: 16
- KV heads: 2 (GQA)
- head dimension: 128
- SwiGLU + RMSNorm
- RMSNorm epsilon: 1e-6
- RoPE frequency base: 5,000,000
- vocabulary: 130560
- untied output head

minllama already reads the architecture parameters above from standard GGUF metadata, including `llama.rope.freq_base`, `llama.attention.layer_norm_rms_epsilon`, head counts and vocabulary size. Do not hard-code Llama-2 defaults for this model.

## MiniCPM5 tokenizer behavior

llama.cpp added MiniCPM5 tokenizer support in merged PR `ggml-org/llama.cpp#23384`.
MiniCPM5 uses a GPT-2/Llama-3-style byte-level BPE vocabulary with a model-specific pre-tokenizer. Two details matter:

1. numeric runs are split into chunks of 1..3 digits;
2. `ignore_merges=true` means an exact pre-tokenized piece is preferred as a whole token, then ordinary BPE merging is used only when no exact whole token exists.

The compatibility branch implements that behavior only when the vocabulary is positively identified as MiniCPM5 (130560 entries plus its ChatML control tokens). Other GPT-2 BPE models keep the existing tokenizer path.

The runtime also preserves known MiniCPM5 control tokens such as `<|im_start|>` and `<|im_end|>` atomically. For a text prompt that is already in ChatML form, the CLI uses `<|im_end|>` as the single runtime stop token. Plain raw prompts continue to use the GGUF EOS token.

This is deliberately narrow. If the real-model tokenizer parity gate below finds a mismatch, report the exact string and token IDs instead of weakening the test.

## GGUF recommendation

minllama's validated weight formats are F32/F16/Q4_0. Do not use an official Q4_K_M file as evidence that minllama supports Q4_K_M.

For this gate, use either:

- a known-good MiniCPM5-2B F16 GGUF; or
- a Q4_0 GGUF produced with a current llama.cpp conversion + quantization toolchain.

Example conversion flow:

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  /path/to/openbmb/MiniCPM5-2B \
  --outtype f16 \
  --outfile /path/to/MiniCPM5-2B.F16.gguf

/path/to/llama.cpp/build/bin/llama-quantize \
  /path/to/MiniCPM5-2B.F16.gguf \
  /path/to/MiniCPM5-2B.Q4_0.gguf \
  Q4_0
```

Always verify the produced GGUF with llama.cpp before blaming minllama.

## Build gate

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The `bpe_tokenizer` test contains a model-independent MiniCPM5 synthetic contract and therefore must run even when no real model exists under `models/`.

## G0 — metadata

Run minllama with a short token-ID prompt so tokenizer behavior is not involved:

```bash
./build/minllama_cli \
  --model "$MODEL" \
  --prompt-tokens 0 \
  --max-new-tokens 0 \
  --debug-load
```

Required evidence from the loaded model:

```text
n_layers=42
vocab=130560
n_head=16
n_kv_head=2
head_dim=128
rope_theta=5000000
rms_norm_eps=1e-6
```

If these values differ, stop at G0 and inspect the GGUF conversion/metadata.

## G1 — tokenizer parity against llama.cpp

Do not judge tokenizer correctness from readable output alone. Compare token IDs.

Use a corpus that exercises the MiniCPM5 pre-tokenizer:

```text
Hello world
1234567890
I'm testing MiniCPM5.
你好，世界！
hello  world
line1\nline2
<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
```

For each exact string, obtain llama.cpp token IDs with its tokenizer tool/API, then run:

```bash
./build/minllama_tokenize --model "$MODEL" --text '<exact string>'
```

PASS requires exact token-ID equality. A decoded roundtrip is useful evidence but is not sufficient by itself.

## G2 — raw-token forward path

Take one known prompt token sequence from llama.cpp and bypass both tokenizers:

```bash
./build/minllama_verify_greedy \
  --model "$MODEL" \
  --prompt-tokens <comma-separated-token-ids>
```

Then run llama.cpp on the same GGUF and same token IDs. Compare at minimum:

- next-token argmax;
- top-5 token IDs;
- ordering of the top logits.

If raw-token parity fails, the problem is in model loading/forward math, not text tokenization.

## G3 — logits evidence

Use the existing diagnostic tool to capture minllama's top logits:

```bash
./build/minllama_compare_logits \
  --model "$MODEL" \
  --mode logits \
  --top-k 20
```

For strict numeric comparison, use the same prompt token IDs and position in llama.cpp. Record:

- top-1 ID;
- top-5 intersection/order;
- max absolute difference when comparable full logits are available;
- mean absolute difference;
- cosine similarity.

Quantized Q4_0 is allowed to differ numerically; large ranking divergence should be investigated before text-generation claims are made.

## G4 — short text generation

First test a raw prompt:

```bash
./build/minllama_cli \
  --model "$MODEL" \
  --prompt 'Hello, my name is' \
  --max-new-tokens 16 \
  --temperature 0 \
  --debug-tokens
```

Then test explicit ChatML. The CLI intentionally does not invent or auto-apply a chat template:

```bash
./build/minllama_cli \
  --model "$MODEL" \
  --prompt $'<s><|im_start|>user\n你好，请用一句话介绍你自己。<|im_end|>\n<|im_start|>assistant\n' \
  --max-new-tokens 32 \
  --temperature 0 \
  --debug-tokens
```

For ChatML prompts, `--debug-tokens` should show MiniCPM5 tokenizer mode and select the `<|im_end|>` token as the runtime stop marker. The stop marker may remain visible in debug token IDs, but it must not be printed as answer text.

## G5 — context boundary

Do not use 128K context as the first acceptance test. Validate progressively with short contexts (for example 16, 64, 256 and 1024 tokens). minllama is a correctness-first CPU implementation and has not established an optimized 131072-token KV/prefill path.

## Acceptance result format

Report every gate separately:

```text
Build/tests: PASS | FAIL
G0 metadata: PASS | FAIL | BLOCKED
G1 tokenizer parity: PASS | FAIL | BLOCKED
G2 raw-token forward: PASS | FAIL | BLOCKED
G3 logits comparison: PASS | FAIL | BLOCKED
G4 raw generation: PASS | FAIL | BLOCKED
G4 ChatML generation/stop: PASS | FAIL | BLOCKED
G5 short-context progression: PASS | FAIL | BLOCKED
```

Do not convert BLOCKED into PASS. Include exact commands, model SHA/path, llama.cpp commit, minllama commit, first mismatch, and relevant token/logit evidence.
