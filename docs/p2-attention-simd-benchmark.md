# P2 Attention SIMD Benchmark Baseline (P2-G + P2-F)

This document freezes the baseline for future P2-H comparison.
Scope in this update: documentation only.

## 1) P2-G long-context profiling summary

Profiling settings:
- MINLLAMA_PROFILE_RUNTIME=ON
- --q8-lm-head enabled
- max-new-tokens=16
- prompt tokens: 32 / 128 / 512 / 1024
- threads: 1 / 2

### Decode phase (%) baseline

| threads | prompt_tokens | attention % | FFN % | lm_head % | qk % (of attn) | softmax % (of attn) | av % (of attn) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32   | 19.23 | 42.07 | 8.65 | 3.39 | 0.35 | 1.63 |
| 1 | 128  | 21.25 | 40.69 | 8.82 | 10.29 | 0.98 | 5.32 |
| 1 | 512  | 29.37 | 36.83 | 7.55 | 27.74 | 2.40 | 15.64 |
| 1 | 1024 | 38.17 | 32.06 | 6.68 | 37.44 | 3.09 | 23.07 |
| 2 | 32   | 15.12 | 53.79 | 7.50 | 8.34 | 0.70 | 3.15 |
| 2 | 128  | 18.45 | 52.81 | 7.02 | 20.76 | 1.76 | 8.32 |
| 2 | 512  | 28.88 | 46.05 | 6.35 | 41.55 | 3.59 | 16.29 |
| 2 | 1024 | 39.59 | 38.61 | 5.21 | 50.31 | 4.17 | 20.41 |

Observations:
- Long-context decode shows attention becomes a primary hotspot.
- Growth inside attention is mainly qk and av; softmax is secondary.
- rope / kv_rw remain low-priority contributors.

## 2) P2-F isolated SIMD benchmark

Benchmark scope:
- Isolated kernels only (not wired into forward)
- head_dim=64
- seq_len: 1 / 8 / 32 / 128 / 256 / 1024

### qk scalar vs NEON

| seq_len | speedup | max_diff |
|---:|---:|---:|
| 1    | 1.564x | 0.00000095 |
| 8    | 2.000x | 0.00000095 |
| 32   | 1.132x | 0.00000095 |
| 128  | 1.144x | 0.00000191 |
| 256  | 1.124x | 0.00000191 |
| 1024 | 1.153x | 0.00000191 |

### av scalar vs NEON

| seq_len | speedup | max_diff |
|---:|---:|---:|
| 1    | 2.566x | 0.00000000 |
| 8    | 2.914x | 0.00000000 |
| 32   | 4.129x | 0.00000000 |
| 128  | 3.456x | 0.00000000 |
| 256  | 3.584x | 0.00000000 |
| 1024 | 3.437x | 0.00000000 |

## 3) Decision

Attention SIMD forward integration is postponed.

Reasons:
- qk long-seq speedup is only ~1.12x-1.15x (below >=1.5x gate).
- av reaches strong isolated speedup, but av-only integration has unstable ROI.
- Forward integration complexity is relatively high at current gains.

## 4) Future P2-H criteria (gate to consider forward integration)

Proceed to forward integration only when all conditions are met:
1. qk at seq_len >= 128 reaches >=1.5x speedup.
2. Correctness fully passes.
3. End-to-end decode gain is >=10%.
