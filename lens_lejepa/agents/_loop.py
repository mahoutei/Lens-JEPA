"""The campaign loop: train a configuration, score it, propose the next one.

DLens' own ``ExperimentLoop`` is not reused: it filters the planner's updates
against its own hyper-parameter whitelist and silently discards anything else,
which would drop every knob this campaign searches over.

The loop never reads the test dataset. Only the configuration that validation
selected is scored on it, once, after the loop has finished.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from dlens.agents._base import BaseAgentConfig
from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
from dlens.agents._scripted_planner import make_scripted_planner_model
from dlens.schemas._experiment import ExperimentRun, ExperimentState, PlannerAction, PlannerDecision
from dlens.tools._analysis import compute_analysis

from ..data import CLASSES
from ..utils import logger
from ._backends import LensJepaInferBackend, LensJepaTrainBackend
from ._spec import DOMAINS, CampaignSpec, DownstreamSpec, validate_delta


class LensJepaRun(ExperimentRun):
    """One iteration of the campaign, with the configuration that produced it."""

    spec: DownstreamSpec
    collapsed: bool = False
    rejection: str | None = None


class LensJepaState(ExperimentState):
    """The campaign history, with our per-iteration configuration attached."""

    runs: list[LensJepaRun] = []


class PlannerStrategy(Protocol):
    """Proposes the next configuration.

    ``dlens.agents._experiment_planner.ReActPlannerStrategy`` satisfies it.
    """

    name: str

    async def decide(self, state: LensJepaState) -> PlannerDecision: ...


@dataclass
class CampaignResult:
    """What a campaign produced."""

    state: LensJepaState
    best_run: LensJepaRun | None
    best_checkpoint: str | None
    output_dir: Path


def select_best_run(runs: Sequence[LensJepaRun]) -> LensJepaRun | None:
    """The run with the best validation accuracy; collapsed runs never win."""
    best: LensJepaRun | None = None
    for run in runs:
        if run.collapsed or run.analysis_result is None:
            continue
        if best is None or run.analysis_result.accuracy > best.analysis_result.accuracy:
            best = run
    return best


def build_planner_prompt(campaign: CampaignSpec) -> str:
    """The system prompt: our knob vocabulary, our frozen fields."""
    encoders = ", ".join(item.name for item in campaign.encoders)
    domains = "\n".join(f"  {key:>15s}  number in [{low}, {high}]" for key, (low, high) in DOMAINS.items())
    return f"""\
You are an experiment planner for a self-supervised representation study on
gravitational-lensing images. You are given the campaign history as JSON: for each
iteration, the configuration that was trained (under "spec"), the training metrics,
and the held-out validation analysis (accuracy, macro_auc, confusion_matrix,
per-class precision/recall/F1).

Reason ONLY from these observed validation metrics. Diagnose what limits validation
accuracy and decide exactly ONE concrete change for the next iteration.

You may change only these fields:
  encoder        one of: {encoders}
  adaptation     one of: lora, rslora, dora, loraplus, gated, loha, layerwise, full, linear_probe
{domains}

These are FIXED for the whole campaign and must never appear in updated_params:
  task, data_root, train_dataset, test_dataset, shots, train_fraction,
  val_fraction, cache, image_size, patch_size, seed, val_interval, device, bf16,
  workers, grad_accum, loraplus_ratio, train_jitter, rank_schedule, output_dir,
  max_iterations, patience, min_val_delta, min_val_above_chance, encoders,
  initial, name, hypothesis.

Populate updated_params with the changed field(s) only, and rationale with your
diagnosis citing the observed numbers. Do not repeat a change that already failed
to improve validation accuracy. The test set is never shown to you and must not be
reasoned about.
"""


def build_planner(campaign: CampaignSpec, live: bool = False) -> PlannerStrategy:
    """The offline scripted planner by default; the hosted LLM when ``live``."""
    from dlens.config import build_llm

    if live:
        model: Any = build_llm()
    else:
        model = make_scripted_planner_model(
            [{"action": "train", "rationale": "offline scripted planner", "updated_params": {}}]
        )
    config = BaseAgentConfig(
        name="LensJepaPlanner",
        description="Chooses the next downstream configuration for a Lens-JEPA campaign.",
        custom_system_prompt=build_planner_prompt(campaign),
        model=model,
    )
    return ReActPlannerStrategy(planner=ExperimentPlanner(config=config))


async def run_campaign(
    campaign: CampaignSpec,
    planner: PlannerStrategy,
    *,
    train_backend: LensJepaTrainBackend,
    infer_backend: LensJepaInferBackend,
) -> CampaignResult:
    """Run the campaign and return its history and the selected checkpoint."""
    state = LensJepaState(hypothesis=campaign.hypothesis, max_iterations=campaign.max_iterations)
    spec = campaign.initial
    best_value: float | None = None
    no_improve = 0
    chance = 1.0 / max(len(CLASSES), 2)

    for iteration in range(campaign.max_iterations):
        state.current_iteration = iteration
        logger.info(
            "iteration %d | encoder=%s adaptation=%s rank=%d lr=%g",
            iteration, spec.encoder, spec.adaptation, spec.rank, spec.lr,
        )
        train_result = train_backend.train(spec, campaign.run_name_for(iteration, spec))
        infer_result = infer_backend.infer(train_result.weights_path)
        analysis = compute_analysis(infer_result, list(CLASSES))
        run = LensJepaRun(
            iteration=iteration,
            spec=spec,
            train_result=train_result,
            infer_result=infer_result,
            analysis_result=analysis,
        )
        state.runs.append(run)

        value = float(analysis.accuracy)
        run.collapsed = value <= chance + campaign.min_val_above_chance
        if run.collapsed:
            run.rejection = f"validation accuracy {value:.4f} is at or below chance ({chance:.4f})"
            logger.warning("iteration %d collapsed: %s", iteration, run.rejection)
        improved = not run.collapsed and (best_value is None or value > best_value + campaign.min_val_delta)
        if improved:
            best_value, no_improve = value, 0
        else:
            no_improve += 1
        logger.info("iteration %d | validation accuracy %.4f", iteration, value)

        if no_improve >= campaign.patience:
            logger.info("stopping: validation accuracy did not improve for %d iterations", campaign.patience)
            break

        decision = await planner.decide(state)
        run.planner_decision = decision
        if decision.action in (PlannerAction.REPORT, PlannerAction.STOP):
            logger.info("stopping: the planner chose %s", decision.action.value)
            break
        try:
            spec = validate_delta(spec, decision.updated_params, campaign)
        except ValueError as error:
            # Keep the collapse reason too: both are why this iteration did not count.
            run.rejection = f"{run.rejection}; {error}" if run.rejection else str(error)
            logger.warning("iteration %d: rejected the planner's change: %s", iteration, error)

    best_run = select_best_run(state.runs)
    return CampaignResult(
        state=state,
        best_run=best_run,
        best_checkpoint=best_run.train_result.weights_path if best_run is not None else None,
        output_dir=campaign.output_path,
    )
