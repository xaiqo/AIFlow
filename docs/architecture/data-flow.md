# Data Flow

```
User input (S3 URI)
    ↓ CLI / Prefect
Prefect Flow
    ↓
Download from S3
    ↓
Model Parser → Graph IR
    ↓
Graph Optimizer → Optimized Graph IR
    ↓
Kernel Optimizer → Optimized Kernels
    ↓
Hardware Profiler → Performance Metrics
    ↓
Auto-Tuner (feedback loop)
    ↓
Artifacts & Results Export
```


