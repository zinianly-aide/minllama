# P2-G Profiling Report (MINLLAMA_PROFILE_RUNTIME=ON, --q8-lm-head, max-new-tokens=16)

## Conclusion
Long-context decode shows attention becoming a primary hotspot.
- P1024 decode:
  - threads=1: attention 38.17%
  - threads=2: attention 39.59%
- attention growth is dominated by qk and av; softmax secondary; rope/kv_rw low priority.

This supports opening P2-F (isolated attention SIMD kernels).

## Matrix snapshot (key points)
- Short context (32/128): FFN dominates; attention ~15-21%.
- Long context (512/1024): attention rises to ~29-40% in decode.

## Gating notes
- Keep runtime profiling instrumentation behind compile-time flag and default OFF.
- If kept, use a profiling-only commit; do not merge as always-on behavior.
