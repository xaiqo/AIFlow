# Orchestration (Prefect)

This project uses Prefect to orchestrate the end-to-end pipeline without an HTTP API.

Flow
- Input: user-provided S3 URI (`s3://bucket/key`) for the model artifact.
- Steps: download → parse → graph optimize → kernel optimize → profile → autotune → export.
- Output: artifacts and results written to a local directory.

Running
```
prefect version
aiflow run s3://your-bucket/your-model.onnx --output-dir ./outputs
```

Credentials
- Provide AWS credentials via environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) or an instance profile.

Extensibility
- Add/replace tasks and subflows per component (parsers, passes, backends, profilers, autotuner).
- Use task retries, caching, and concurrency limits as needed for large workloads.


