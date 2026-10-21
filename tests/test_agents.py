"""The agent campaign: specs, validation, loop. Offline, deterministic, CPU."""

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
