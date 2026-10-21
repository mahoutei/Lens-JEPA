"""The agent campaign: specs, validation, loop. Offline, deterministic, CPU."""

import json
from pathlib import Path

import pytest

from lens_lejepa.agents._spec import CampaignSpec, DownstreamSpec, validate_delta


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
