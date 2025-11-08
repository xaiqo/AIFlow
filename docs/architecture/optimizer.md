# Graph Optimizer

Pass Types
- Fusion: vertical (e.g., Conv+BN+ReLU), horizontal (parallel ops).
- Quantization: PTQ, QAT, mixed precision; per-tensor/per-channel.
- Pruning: structured/unstructured; sparsity-aware rewrites.
- Layout: NCHW ↔ NHWC; cache-aware blocking; memory planning.
- DCE/CF/ConstFold: standard compiler transformations.

Pass Framework
- Pass interface with `match(graph) -> candidates` and `apply(graph, candidate)`.
- Pass pipelines: ordered lists with pre/post-conditions.
- Validation hooks: verify invariants after each apply.

Configuration
- Pass parameters and target hardware constraints are provided via config objects.
- Determinism flags for reproducible experiments.


