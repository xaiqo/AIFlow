AI Inference Optimization Framework
===================================

A framework for end-to-end AI inference optimization: from model parsing and graph IR, through graph and kernel optimizations, to hardware profiling, auto-tuning, and visualization.

Status: Initial scaffolding for community contributions. See docs and tasks to get started.

Repository: https://github.com/xaiqo/AIFlow

Key Capabilities (planned)
- Model parsing: ONNX, PyTorch, TensorFlow
- Multi-level Graph IR: HIR → MIR → LIR
- Graph optimizations: fusion, quantization, pruning, layout transforms
- Kernel optimizations: tiling, vectorization, memory scheduling, codegen
- Hardware profiling: static + dynamic profiling, roofline, simulation
- Auto-tuning: ML-guided search (Bayesian/RL/genetic)
- Orchestration: Prefect flows with S3 ingestion; artifacts for a Web UI
 - Parsers: ONNX MVP (Conv/BN/Relu/Add/MatMul/Pool/Reshape/Transpose)

Repository Structure

```
.
├─ src/aiflow/                 # Python package
│  ├─ flows/                   # Prefect orchestration
│  ├─ autotuner/               # Auto-tuner interfaces & strategies
│  ├─ ir/                      # Graph IR data structures
│  ├─ kernel/                  # Kernel scheduling & backend hooks
│  ├─ optimizer/               # Graph optimization passes
│  ├─ parsers/                 # Model parsers (ONNX/PyTorch/TF)
│  ├─ plugins/                 # Plugin registration & discovery
│  ├─ profiler/                # Static/dynamic profiling
│  ├─ cli/                     # CLI entrypoint
│  └─ utils/                   # Shared utilities
├─ tests/                      # Unit/integration tests
├─ examples/                   # Usage examples
├─ docs/                       # Documentation (your originals are preserved)
│  ├─ architecture/            # Advanced architecture docs
│  ├─ core.md                  # (existing)
│  └─ topic.md                 # (existing)
├─ tasks/                      # Per-feature implementation checklists
├─ .github/
│  ├─ ISSUE_TEMPLATE/          # Issue templates
│  ├─ PULL_REQUEST_TEMPLATE.md
│  └─ workflows/ci.yml         # CI (lint + tests)
├─ pyproject.toml              # Build, dependencies, tooling
├─ ruff.toml                   # Ruff configuration
├─ mypy.ini                    # MyPy configuration
└─ .gitignore
```

Quickstart (development)
1) Create a virtual environment and install dependencies
```
python -m venv .venv  # Python 3.13
. .venv/Scripts/Activate.ps1   # PowerShell (Windows)
# or: source .venv/bin/activate
pip install -e .[dev]
```

2) Run linters and tests
```
ruff check .
pytest -q
```

3) Run the Prefect flow from CLI
```
aiflow run s3://your-bucket/your-model.onnx --output-dir ./outputs
```
The flow will automatically detect `.onnx` and parse it via the ONNX parser, validate, and infer shapes.

Contributing
- Read CONTRIBUTING.md for environment setup, style, commit conventions, and PR process.
- Pick a task from tasks/ and open an issue or draft PR.
- Use the plugin system for new backends, passes, and profilers when possible.

License
Apache License 2.0


