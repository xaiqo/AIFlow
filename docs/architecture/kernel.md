# Kernel Optimization

Goals
- Transform optimized graphs into efficient kernels for target hardware.
- Apply loop transformations (tiling, unrolling, interchange, fusion/fission).
- Vectorization: AVX/AVX-512/NEON; GPU warp-level ops; tensor cores.
- Memory: coalescing, prefetching, shared/LDS usage, buffer packing.

Backends
- CPU (C++): OpenMP/TBB threading; cache-aware blocking; NUMA hints.
- CUDA: grid/block sizing, occupancy, shared memory utilization.
- OpenCL/ROCm/others: portable kernels with device-specific tuning.

Interfaces
- Backend capability descriptors.
- Schedule representation for transformations and codegen hooks.
- Artifact outputs: source, metadata, and performance counters.




