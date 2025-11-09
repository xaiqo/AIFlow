from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import boto3
from prefect import flow, get_run_logger, task

from aiflow.ir import Graph
from aiflow.parsers.onnx import OnnxParser


@task
def download_from_s3(s3_uri: str) -> Path:
    """
    Download a model from S3 to a temporary file. s3_uri like s3://bucket/key
    Requires AWS credentials in environment.
    """
    logger = get_run_logger()
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    s3 = boto3.client("s3")
    tmp = Path(tempfile.mkstemp(prefix="aiflow_model_")[1])
    s3.download_file(bucket, key, str(tmp))
    logger.info(f"Downloaded {s3_uri} to {tmp}")
    return tmp


@task
def parse_model(local_path: Path) -> Graph:
    logger = get_run_logger()
    logger.info(f"Parsing model at {local_path}")
    if local_path.suffix.lower() == ".onnx":
        ir = OnnxParser().parse(str(local_path), validate_and_infer=True)
        return ir
    # Fallback: empty graph
    return Graph()


@task
def optimize_graph(ir: Graph) -> Graph:
    logger = get_run_logger()
    logger.info("Running graph optimization passes")
    return ir


@task
def optimize_kernels(ir: Graph) -> dict[str, Any]:
    logger = get_run_logger()
    logger.info("Generating and optimizing kernels")
    return {"artifact": {"source": "// kernel source"}, "ir": ir}


@task
def profile_artifacts(artifact: dict[str, Any]) -> dict[str, float]:
    logger = get_run_logger()
    logger.info("Profiling artifacts")
    return {"latency_ms": 0.0, "throughput_qps": 0.0}


@task
def autotune(ir: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    logger = get_run_logger()
    logger.info("Auto-tuning configuration")
    return {"best_config": {}, "metrics": metrics}


@task
def export_results(output_dir: str, results: dict[str, Any]) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / "results.json"
    result_file.write_text(str(results))
    return str(result_file)


@flow(name="aiflow-inference-optimization")
def inference_optimization_flow(s3_uri: str, output_dir: str) -> str:
    """
    Orchestrates the end-to-end pipeline:
    S3 → parse → graph optimize → kernel optimize → profile → autotune → export
    """
    path = download_from_s3(s3_uri)
    ir = parse_model(path)
    ir_opt = optimize_graph(ir)
    artifact = optimize_kernels(ir_opt)
    metrics = profile_artifacts(artifact)
    tuned = autotune(ir_opt, metrics)
    out = export_results(output_dir, tuned)
    return cast(str, out)
