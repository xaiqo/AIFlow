# Architecture Overview

This framework provides a modular pipeline:
1. Model Input → 2. Graph IR → 3. Graph Optimization → 4. Kernel Optimization → 5. Hardware Profiling → 6. Auto-Tuning → 7. Visualization → 8. Export

Design Principles
- Separation of concerns: parsing, IR, passes, scheduling, profiling, and tuning are separate packages.
- Extensibility via plugin interfaces and registries.
- Reproducibility: configurations and experiment metadata are first-class.
- Hardware abstraction with targeted backends for portability and performance.

Core Packages
- `aiflow.parsers`: Normalizes external models into the Graph IR.
- `aiflow.ir`: Multi-level IR (HIR/MIR/LIR) with metadata and analysis utilities.
- `aiflow.optimizer`: Graph-pass framework (fusion, quantization, pruning, layout).
- `aiflow.kernel`: Kernel scheduling, tiling, vectorization, and backend integration.
- `aiflow.profiler`: Static and dynamic profiling, roofline and simulation hooks.
- `aiflow.autotuner`: ML-guided search strategies and cost models.
- `aiflow.flows`: Prefect-based orchestration of the end-to-end pipeline.
- `aiflow.plugins`: Extension discovery and registration.

Data Contracts
- Parsers must emit a consistent IR (nodes, tensors, attributes).
- Passes declare pre/post-conditions and preserve graph invariants.
- Kernel backends expose capability descriptors (e.g., vector width, memory hierarchy).


