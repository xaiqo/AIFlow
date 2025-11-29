"""Graph optimization passes and pipelines."""

from .passes import Pass, Pipeline
from .passes_impl import ConstantFoldingPass, DeadCodeEliminationPass

__all__ = ["Pipeline", "Pass", "ConstantFoldingPass", "DeadCodeEliminationPass"]
