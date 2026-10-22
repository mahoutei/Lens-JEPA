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
from ..engine import evaluate
from ..utils import logger, write_json
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
        elif not run.collapsed:
            # A collapsed run is a failed intervention the planner must react to,
            # not a plateau: it does not consume the patience budget.
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


def _iterations(state: LensJepaState) -> list[dict[str, Any]]:
    """One row per iteration: the configuration, the metrics, the outcome."""
    rows = []
    for run in state.runs:
        analysis = run.analysis_result
        rows.append({
            "iteration": run.iteration,
            "spec": run.spec.model_dump(mode="json"),
            "val_accuracy": float(analysis.accuracy) if analysis is not None else None,
            "macro_auc": (
                float(analysis.macro_auc)
                if analysis is not None and analysis.macro_auc is not None
                else None
            ),
            "collapsed": run.collapsed,
            "rejection": run.rejection,
            "run_dir": run.train_result.run_id if run.train_result is not None else None,
            "rationale": run.planner_decision.rationale if run.planner_decision is not None else None,
        })
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    """The campaign table a human reads."""
    campaign = payload["campaign"]
    lines = [
        f"# {campaign['name']}",
        "",
        campaign["hypothesis"] or "_No hypothesis recorded._",
        "",
        "| # | encoder | adaptation | rank | lr | val accuracy | macro AUROC | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["iterations"]:
        accuracy = "n/a" if row["val_accuracy"] is None else f"{row['val_accuracy']:.4f}"
        auc = "n/a" if row["macro_auc"] is None else f"{row['macro_auc']:.4f}"
        spec = row["spec"]
        note = (row["rejection"] or "").replace("|", "\\|")  # a pipe would break the table
        lines.append(
            f"| {row['iteration']} | {spec['encoder']} | {spec['adaptation']} | {spec['rank']} | "
            f"{spec['lr']:g} | {accuracy} | {auc} | {note} |"
        )
    best = payload["best_iteration"]
    lines += [
        "",
        f"Selected iteration: **{best}**" if best is not None else "No iteration was selected.",
        "",
    ]
    if payload["test"]:
        lines += ["## Held-out test metrics", ""]
        for key, value in payload["test"].items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
    return "\n".join(lines) + "\n"


def finalize(result: CampaignResult, campaign: CampaignSpec) -> dict[str, Any]:
    """Score the selected checkpoint on the test set once, then write the artifacts."""
    output = campaign.output_path
    marker = output / "campaign.json"
    if marker.exists():
        raise RuntimeError(
            f"Campaign {campaign.name!r} has already been finalized ({marker}); "
            "run it under a new campaign name instead of scoring the test set twice."
        )
    output.mkdir(parents=True, exist_ok=True)

    test_metrics: dict[str, Any] | None = None
    if result.best_checkpoint is not None and campaign.test_dataset:
        logger.info("evaluating the selected checkpoint on %s", campaign.test_dataset)
        batch_size = result.best_run.spec.batch_size if result.best_run is not None else campaign.initial.batch_size
        test_metrics = evaluate(
            result.best_checkpoint,
            str(Path(campaign.data_root) / campaign.test_dataset),
            campaign.device,
            batch_size=batch_size,
            workers=campaign.workers,
        )

    payload = {
        "campaign": campaign.model_dump(mode="json"),
        "iterations": _iterations(result.state),
        "best_iteration": result.best_run.iteration if result.best_run is not None else None,
        "best_checkpoint": result.best_checkpoint,
        "test": test_metrics,
    }
    write_json(output / "campaign.json", payload)
    (output / "campaign.md").write_text(_markdown(payload))
    return payload


async def write_report(state: LensJepaState, campaign: CampaignSpec, live: bool = False) -> dict[str, Any]:
    """The DLens report agent's summary of the campaign.

    The report agent reads the state, which already carries the campaign's
    history and hypothesis; ``campaign`` names the campaign reported on.
    """
    from dlens.agents._report import ReportAgent
    from dlens.agents._scripted_planner import make_scripted_report_model
    from dlens.config import build_llm

    agent = ReportAgent(model=build_llm() if live else make_scripted_report_model())
    result = await agent.write(state)
    return result.output.model_dump(mode="json")


async def run_from_config(path: str | None = None, overrides: list[str] | None = None) -> Path:
    """Run a campaign described by a YAML file, and return the artifact directory."""
    import os

    from ..config import load_yaml_with_overrides

    campaign = CampaignSpec(**load_yaml_with_overrides(path, overrides))
    live = bool(os.getenv("DLENS_LLM") or os.getenv("OPENAI_API_KEY"))
    logger.info(
        "campaign %s | %d iterations | %s | encoders: %s",
        campaign.name, campaign.max_iterations,
        "live planner" if live else "offline scripted planner",
        ", ".join(item.name for item in campaign.encoders),
    )
    result = await run_campaign(
        campaign,
        build_planner(campaign, live=live),
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    )
    finalize(result, campaign)
    report = await write_report(result.state, campaign, live=live)
    write_json(result.output_dir / "report.json", report)
    logger.info("campaign artifacts: %s", result.output_dir)
    return result.output_dir
