"""Agent-driven downstream campaigns, built on the DLens framework.

Importing this package requires ``dlens``; the rest of ``lens_lejepa`` does
not. See ``scripts/install_agent_deps.sh``.
"""

from ._backends import LensJepaInferBackend, LensJepaTrainBackend, build_finetune_config
from ._loop import (
    CampaignResult,
    LensJepaRun,
    LensJepaState,
    build_planner,
    build_planner_prompt,
    run_campaign,
    select_best_run,
)
from ._spec import CampaignSpec, DownstreamSpec, EncoderSpec, validate_delta

__all__ = [
    "CampaignResult",
    "CampaignSpec",
    "DownstreamSpec",
    "EncoderSpec",
    "LensJepaInferBackend",
    "LensJepaRun",
    "LensJepaState",
    "LensJepaTrainBackend",
    "build_finetune_config",
    "build_planner",
    "build_planner_prompt",
    "run_campaign",
    "select_best_run",
    "validate_delta",
]
