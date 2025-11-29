"""Graph optimization passes and pipelines."""

from .passes import Pass, Pipeline
from .passes_impl import ConstantFoldingPass, DeadCodeEliminationPass, FusionCBRPass

__all__ = ["Pipeline", "Pass", "ConstantFoldingPass", "DeadCodeEliminationPass", "FusionCBRPass"]
