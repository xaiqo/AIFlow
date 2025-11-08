# Graph IR

Goals
- Represent models uniformly across frameworks.
- Support multi-level IR: HIR (operators) → MIR (loops) → LIR (instructions).
- Enable analysis (shape/dtype inference), transformation, and verification.

Key Entities
- Node: operation with attributes and input/output edges.
- Tensor: typed, shaped data with layout/stride metadata.
- Graph: directed acyclic (with allowances for control flow), annotations for passes.

IR Invariants
- All node inputs/outputs are typed and shaped.
- Graph remains valid after each pass (verified by IR checks).
- Metadata includes layout, quantization scales/zero-points, sparsity, and placement.

Evolution
- Start with HIR + shape inference utilities.
- Introduce MIR for loop-level transformations.
- Add LIR and codegen interfaces for specific hardware targets.




