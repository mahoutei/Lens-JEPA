"""The two synthetic configs must agree on geometry and checkpoint path.

``load_pretrained_encoder`` strict-loads the state dict, so a mismatch between
the pretrain and campaign ``image_size``/``patch_size`` would only fail after a
full pretraining run.
"""

import pytest

from lens_lejepa.config import PretrainConfig, load_config


def test_pretrain_config_is_cpu_sized_and_matches_the_generator():
    config = load_config(PretrainConfig, "configs/pretrain/sim_smoke.yaml")
    assert (config.image_size, config.patch_size) == (64, 8)
    assert config.encoder == "vit_tiny"
    assert (config.device, config.bf16) == ("cpu", False)
    assert config.data_root == "../datasets_sim"
    assert config.dataset == "Model_I"
    assert config.run_name == "sim_smoke_seed42"


def test_campaign_and_pretrain_geometries_agree():
    pytest.importorskip("dlens")
    from lens_lejepa.agents import CampaignSpec
    from lens_lejepa.config import load_yaml_with_overrides

    campaign = CampaignSpec(**load_yaml_with_overrides("configs/agent/campaign_sim.yaml"))
    pretrain = load_config(PretrainConfig, "configs/pretrain/sim_smoke.yaml")

    assert (campaign.image_size, campaign.patch_size) == (pretrain.image_size, pretrain.patch_size)
    assert len(campaign.encoders) == 1
    encoder = campaign.encoders[0]
    assert encoder.backbone == pretrain.encoder
    assert encoder.pretrained == f"runs/pretrain/lens_lejepa/{pretrain.run_name}/last_pretrain.pt"


def test_campaign_config_points_at_the_synthetic_data():
    pytest.importorskip("dlens")
    from lens_lejepa.agents import CampaignSpec
    from lens_lejepa.config import load_yaml_with_overrides

    campaign = CampaignSpec(**load_yaml_with_overrides("configs/agent/campaign_sim.yaml"))
    assert campaign.data_root == "../datasets_sim"
    assert (campaign.train_dataset, campaign.test_dataset) == ("Model_II", "Model_II_test")
    assert campaign.shots is None
    assert (campaign.device, campaign.bf16) == ("cpu", False)
    assert campaign.name.endswith("_sim")
