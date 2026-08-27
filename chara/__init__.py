"""Chara: Thermodynamic Graph Laplacian Survival Inference for Transcriptomic Oncology."""

from .model import CharaModel, load_sample_cohort
from .preprocessing import scale_external_expression, align_expression
from .graph import laplacian_from_edges, heat_kernel, exponential_chara_laplacian
from .metrics import concordance_index

def load_model(path_or_repo=None):
    """Convenience helper to load CharaModel from file path or Hugging Face Hub."""
    return CharaModel.load(path_or_repo)

__all__ = [
    "CharaModel",
    "load_model",
    "load_sample_cohort",
    "concordance_index",
    "scale_external_expression",
    "align_expression",
    "laplacian_from_edges",
    "heat_kernel",
    "exponential_chara_laplacian",
    "__version__",
]

__version__ = "0.2.1"
