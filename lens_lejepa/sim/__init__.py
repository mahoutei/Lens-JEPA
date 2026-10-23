"""Synthetic bring-up data generation.

Requires the optional ``sim`` extra (``pip install -e ".[sim]"``). The data
written by this package is for CI, smoke runs and agent-loop demos only; see
``docs/superpowers/specs/2026-10-23-synthetic-data-design.md``.
"""

from .generate import (
    SimulationConfig,
    generate,
    generate_from_config,
    verify_dataset,
    write_image,
)
from .lens import (
    LensParameters,
    axion_mass_to_length,
    build_lens,
    einstein_radius,
    simulate,
)

__all__ = [
    "LensParameters",
    "SimulationConfig",
    "axion_mass_to_length",
    "build_lens",
    "einstein_radius",
    "generate",
    "generate_from_config",
    "simulate",
    "verify_dataset",
    "write_image",
]
