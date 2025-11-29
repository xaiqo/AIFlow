"""Graph optimization passes and pipelines."""

from .passes import Pass, Pipeline
from .passes_impl import ConstantFoldingPass, DeadCodeEliminationPass, FusionCBRPass


def build_default_pipeline(*, enable_fusion: bool = True) -> Pipeline:
    passes: list[Pass] = [ConstantFoldingPass()]
    if enable_fusion:
        passes.append(FusionCBRPass())
    passes.append(DeadCodeEliminationPass())
    return Pipeline(passes)

__all__ = [
    "Pipeline",
    "Pass",
    "ConstantFoldingPass",
    "DeadCodeEliminationPass",
    "FusionCBRPass",
    "build_default_pipeline",
]
