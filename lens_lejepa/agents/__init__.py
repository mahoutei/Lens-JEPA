"""Agent-driven downstream campaigns, built on the DLens framework.

Importing this package requires ``dlens``; the rest of ``lens_lejepa`` does
not. See ``scripts/install_agent_deps.sh``.
"""

from ._backends import LensJepaInferBackend, LensJepaTrainBackend, build_finetune_config
from ._spec import CampaignSpec, DownstreamSpec, EncoderSpec, validate_delta

__all__ = [
    "CampaignSpec",
    "DownstreamSpec",
    "EncoderSpec",
    "LensJepaInferBackend",
    "LensJepaTrainBackend",
    "build_finetune_config",
    "validate_delta",
]
