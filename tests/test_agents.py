"""The agent campaign: specs, validation, loop. Offline, deterministic, CPU."""

import json
from pathlib import Path

import pytest

from lens_lejepa.agents._spec import CampaignSpec, DownstreamSpec, validate_delta
from lens_lejepa.data import CLASSES


def test_the_planner_may_change_the_downstream_fields():
    spec = DownstreamSpec()
    updated = validate_delta(spec, {"adaptation": "dora", "rank": 64})
    assert updated.adaptation == "dora"
    assert updated.rank == 64
    assert updated.lr == spec.lr  # untouched fields keep their value


def test_an_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown parameter"):
        validate_delta(DownstreamSpec(), {"learning_rate": 1e-3})


def test_a_frozen_key_is_rejected():
    campaign = CampaignSpec()
    with pytest.raises(ValueError, match="fixed for the whole campaign"):
        validate_delta(DownstreamSpec(), {"shots": 5}, campaign)
    with pytest.raises(ValueError, match="fixed for the whole campaign"):
        validate_delta(DownstreamSpec(), {"seed": 7}, campaign)


def test_an_out_of_domain_value_is_rejected():
    with pytest.raises(ValueError, match="outside the allowed range"):
        validate_delta(DownstreamSpec(), {"rank": 0})
    with pytest.raises(ValueError, match="outside the allowed range"):
        validate_delta(DownstreamSpec(), {"lr": 10.0})
    with pytest.raises(ValueError, match="Unknown adapter"):
        validate_delta(DownstreamSpec(), {"adaptation": "typo"})


def test_a_fractional_count_is_rejected():
    with pytest.raises(ValueError, match="must be a whole number"):
        validate_delta(DownstreamSpec(), {"rank": 32.5})


def test_an_encoder_outside_the_pool_is_rejected():
    campaign = CampaignSpec(encoders=[{"name": "vit_small", "backbone": "vit_small"}])
    with pytest.raises(ValueError, match="not in the pool"):
        validate_delta(DownstreamSpec(), {"encoder": "nope"}, campaign)
    with pytest.raises(KeyError):
        campaign.encoder("nope")


def test_the_default_campaign_starts_inside_its_own_pool():
    campaign = CampaignSpec()
    assert campaign.initial.encoder in [item.name for item in campaign.encoders]
    assert campaign.encoder(campaign.initial.encoder).name == campaign.initial.encoder


def test_the_campaign_exposes_its_pool_and_run_names():
    campaign = CampaignSpec(
        name="demo",
        output_dir="runs/agent",
        encoders=[
            {"name": "lens_lejepa", "backbone": "vit_small"},
            {"name": "lejepa", "backbone": "vit_small"},
        ],
    )
    assert campaign.encoder("lejepa").backbone == "vit_small"
    assert campaign.run_name_for(0, DownstreamSpec(encoder="lejepa", adaptation="lora")) == "iter00_lejepa_lora"
    assert campaign.output_path.as_posix() == "runs/agent/demo"


def test_build_finetune_config_maps_the_campaign_onto_a_run(fake_root, tmp_path):
    from lens_lejepa.agents._backends import build_finetune_config

    campaign = CampaignSpec(
        task="classification",
        data_root=str(fake_root.parent),
        train_dataset=fake_root.name,
        test_dataset=fake_root.name,
        shots=10,
        image_size=32,
        seed=7,
        device="cpu",
        encoders=[{"name": "lens_lejepa", "backbone": "vit_tiny", "pretrained": None}],
    )
    spec = DownstreamSpec(encoder="lens_lejepa", adaptation="lora", rank=8, lr=1e-3, epochs=3)
    config = build_finetune_config(campaign, spec, "iter00")
    assert config.task == "classification"
    assert config.shots == 10
    assert config.seed == 7
    assert config.device == "cpu"
    assert config.backbone == "vit_tiny"
    assert config.adaptation == "lora"
    assert config.rank == 8
    assert config.lr == 1e-3
    assert config.epochs == 3
    assert config.run_name == "iter00"
    # The test set must never be reachable from inside the loop.
    assert config.test_dataset is None


def test_the_backends_train_and_score_the_validation_split(fake_root, tmp_path):
    """The train backend runs our engine; the infer backend scores its checkpoint."""
    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend

    campaign = CampaignSpec(
        task="classification",
        data_root=str(fake_root.parent),
        train_dataset=fake_root.name,
        test_dataset=fake_root.name,
        shots=10,
        image_size=32,
        patch_size=16,
        seed=7,
        device="cpu",
        workers=0,
        val_interval=1,
        output_dir=str(tmp_path),
        encoders=[{"name": "lens_lejepa", "backbone": "vit_tiny"}],
    )
    spec = DownstreamSpec(encoder="lens_lejepa", adaptation="lora", rank=4, alpha=8, lr=1e-3, epochs=2, batch_size=8)

    trained = LensJepaTrainBackend(campaign).train(spec, "iter00")
    assert trained.run_id == "iter00"
    assert trained.num_classes == 3
    assert 0.0 <= trained.metrics["val_accuracy"] <= 1.0
    assert Path(trained.weights_path).exists()

    run_dir = Path(trained.weights_path).parent
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "test" not in summary  # the held-out set was never evaluated

    scored = LensJepaInferBackend(campaign).infer(trained.weights_path)
    assert scored.run_id == "iter00"
    assert scored.num_samples == 6  # two validation images per class
    assert scored.num_classes == 3
    assert len(scored.predictions) == scored.num_samples
    assert len(scored.true_labels) == scored.num_samples
    assert scored.accuracy == pytest.approx(
        sum(p == t for p, t in zip(scored.predictions, scored.true_labels)) / scored.num_samples
    )


class ScriptedPlanner:
    """Emits a fixed decision per iteration; deterministic and offline."""

    name = "scripted"

    def __init__(self, decisions):
        self.decisions = decisions
        self.seen = []  # a snapshot of the state handed to each turn

    async def decide(self, state):
        from dlens.schemas._experiment import PlannerDecision

        index = state.current_iteration
        payload = self.decisions[index] if index < len(self.decisions) else {
            "action": "report", "rationale": "scripted budget exhausted", "updated_params": {}
        }
        self.seen.append(state.model_dump(mode="json"))
        return PlannerDecision(**payload)


class ScriptedTrainBackend:
    """Names the runs it is asked for; no model is trained."""

    name = "scripted"

    def __init__(self, root):
        self.root = Path(root)

    def train(self, spec, run_name):
        from dlens.schemas._downstream import TrainResult

        return TrainResult(
            run_id=run_name,
            weights_path=str(self.root / run_name / "best.pt"),
            backend=self.name,
            num_classes=len(CLASSES),
            metrics={"val_accuracy": 0.0},
            epochs_run=1,
        )


class ScriptedInferBackend:
    """Scores the validation split with a scripted accuracy per iteration."""

    name = "scripted"

    def __init__(self, accuracies):
        self.accuracies = list(accuracies)
        self.index = 0

    def infer(self, weights_path):
        from dlens.schemas._downstream import InferResult

        accuracy = self.accuracies[min(self.index, len(self.accuracies) - 1)]
        self.index += 1
        labels = [index % len(CLASSES) for index in range(12)]
        correct = round(accuracy * len(labels))
        predictions = [
            label if position < correct else (label + 1) % len(CLASSES)
            for position, label in enumerate(labels)
        ]
        probabilities = [
            [0.9 if class_index == prediction else 0.05 for class_index in range(len(CLASSES))]
            for prediction in predictions
        ]
        return InferResult(
            run_id=Path(weights_path).parent.name,
            num_samples=len(labels),
            num_classes=len(CLASSES),
            predictions=predictions,
            probabilities=probabilities,
            true_labels=labels,
            accuracy=correct / len(labels),
            backend=self.name,
        )


def _fake_run(accuracy: float, collapsed: bool = False):
    from dlens.schemas._downstream import AnalysisResult

    from lens_lejepa.agents._loop import LensJepaRun

    return LensJepaRun(
        iteration=0,
        spec=DownstreamSpec(),
        analysis_result=AnalysisResult(
            accuracy=accuracy,
            confusion_matrix=[[1, 0], [0, 1]],
            per_class={"a": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1.0}},
            summary="synthetic",
        ),
        collapsed=collapsed,
    )


def _tiny_campaign(fake_root, tmp_path, **kwargs):
    defaults = dict(
        name="demo",
        data_root=str(fake_root.parent),
        train_dataset=fake_root.name,
        test_dataset=fake_root.name,
        image_size=32,
        device="cpu",
        bf16=False,
        workers=0,
        output_dir=str(tmp_path / "runs"),
        encoders=[{"name": "vit_small", "backbone": "vit_tiny"}],
        initial=DownstreamSpec(encoder="vit_small", adaptation="linear_probe", epochs=10, batch_size=8),
    )
    defaults.update(kwargs)
    return CampaignSpec(**defaults)


def test_the_best_run_is_the_best_validation_accuracy():
    from lens_lejepa.agents._loop import select_best_run

    runs = [_fake_run(0.5), _fake_run(0.9), _fake_run(0.7)]
    assert select_best_run(runs) is runs[1]


def test_a_collapsed_run_is_never_the_best():
    from lens_lejepa.agents._loop import select_best_run

    runs = [_fake_run(0.5), _fake_run(0.99, collapsed=True)]
    assert select_best_run(runs) is runs[0]


def test_the_offline_planner_decides_without_a_network():
    import asyncio

    from dlens.schemas._experiment import PlannerAction

    from lens_lejepa.agents._loop import LensJepaState, build_planner, build_planner_prompt

    campaign = CampaignSpec(encoders=[{"name": "vit_small", "backbone": "vit_tiny"}])
    prompt = build_planner_prompt(campaign)
    assert "vit_small" in prompt  # the encoder pool the planner may choose from
    assert "shots" in prompt  # the fields it may not change
    decision = asyncio.run(build_planner(campaign).decide(LensJepaState(hypothesis="a campaign")))
    assert decision.action == PlannerAction.TRAIN


def test_the_loop_runs_offline_and_records_every_iteration(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "try a different adapter", "updated_params": {"adaptation": "lora"}},
        {"action": "report", "rationale": "done", "updated_params": {}},
    ])
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert len(result.state.runs) == 2
    assert result.state.runs[0].spec.adaptation == "linear_probe"
    assert result.state.runs[1].spec.adaptation == "lora"
    assert result.best_run is not None
    assert result.best_checkpoint.endswith("best.pt")


def test_a_rejected_change_is_recorded_and_the_campaign_continues(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "typo", "updated_params": {"learning_rate": 1e-3}},
        {"action": "report", "rationale": "done", "updated_params": {}},
    ])
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert any("learning_rate" in (run.rejection or "") for run in result.state.runs)
    # The rejected change must not have been applied.
    assert result.state.runs[1].spec.lr == campaign.initial.lr


def test_a_collapsed_run_does_not_consume_the_patience_budget(fake_root, tmp_path):
    """A collapse is a failed intervention the planner must react to, not a plateau."""
    import asyncio

    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2, patience=1)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "react to the collapse", "updated_params": {}},
        {"action": "train", "rationale": "react to the collapse", "updated_params": {}},
    ])
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=ScriptedTrainBackend(tmp_path),
        infer_backend=ScriptedInferBackend([0.0, 1.0]),  # the first run collapses
    ))
    assert result.state.runs[0].collapsed is True
    assert len(result.state.runs) == 2  # the collapse did not end the campaign
    assert result.best_run is result.state.runs[1]


def test_a_run_that_does_not_improve_consumes_the_patience_budget(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=3, patience=1)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "keep going", "updated_params": {}},
        {"action": "train", "rationale": "keep going", "updated_params": {}},
        {"action": "train", "rationale": "keep going", "updated_params": {}},
    ])
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=ScriptedTrainBackend(tmp_path),
        infer_backend=ScriptedInferBackend([1.0, 0.5, 1.0]),  # the second run does not improve
    ))
    assert result.state.runs[0].collapsed is False
    assert len(result.state.runs) == 2  # the flat run ended the campaign, not the budget


def test_the_planner_sees_the_rejection_on_its_next_turn(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "typo", "updated_params": {"learning_rate": 1e-3}},
        {"action": "train", "rationale": "corrected", "updated_params": {}},
    ])
    asyncio.run(run_campaign(
        campaign, planner,
        train_backend=ScriptedTrainBackend(tmp_path),
        infer_backend=ScriptedInferBackend([1.0, 1.0]),
    ))
    first, second = planner.seen
    assert first["runs"][0]["rejection"] is None
    assert "learning_rate" in second["runs"][0]["rejection"]


def test_the_test_set_is_evaluated_once_and_only_after_the_loop(fake_root, tmp_path, monkeypatch):
    import asyncio

    import lens_lejepa.agents._loop as loop_module
    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import finalize, run_campaign

    calls = []
    monkeypatch.setattr(
        loop_module, "evaluate",
        lambda *args, **kwargs: calls.append(args) or {"accuracy": 0.5},
    )

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=1)
    result = asyncio.run(run_campaign(
        campaign, ScriptedPlanner([]),
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert calls == []  # the loop alone never evaluates the test set

    payload = finalize(result, campaign)
    assert len(calls) == 1
    assert payload["test"]["accuracy"] == 0.5
    assert (campaign.output_path / "campaign.json").exists()
    assert (campaign.output_path / "campaign.md").exists()


def test_the_report_agent_summarises_the_campaign(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._loop import run_campaign, write_report

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=1, hypothesis="a hypothesis")
    result = asyncio.run(run_campaign(
        campaign, ScriptedPlanner([]),
        train_backend=ScriptedTrainBackend(tmp_path),
        infer_backend=ScriptedInferBackend([1.0]),
    ))
    report = asyncio.run(write_report(result.state, campaign))
    assert report["title"]
    assert report["hypothesis"] == "a hypothesis"
    assert report["best_accuracy"] == 1.0
