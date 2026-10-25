"""The real-data configs must agree on geometry and checkpoint path.

``load_pretrained_encoder`` strict-loads the state dict, so a mismatch between
the pretrain and campaign ``image_size``/``patch_size`` would only fail after a
full pretraining run.
"""

import pytest

from lens_lejepa.config import PretrainConfig, load_config


def test_real_pretrain_is_cpu_sized():
    config = load_config(PretrainConfig, "configs/pretrain/real_model_ii_64.yaml")
    assert (config.image_size, config.patch_size) == (64, 8)
    assert config.encoder == "vit_tiny"
    assert (config.device, config.bf16) == ("cpu", False)
    assert (config.data_root, config.dataset) == ("../datasets", "Model_II")
    assert config.run_name == "real_model_ii_64_seed42"


def test_real_campaign_matches_the_pretrain():
    pytest.importorskip("dlens")
    from lens_lejepa.agents import CampaignSpec
    from lens_lejepa.config import load_yaml_with_overrides

    campaign = CampaignSpec(**load_yaml_with_overrides("configs/agent/campaign_real.yaml"))
    pretrain = load_config(PretrainConfig, "configs/pretrain/real_model_ii_64.yaml")

    assert (campaign.image_size, campaign.patch_size) == (pretrain.image_size, pretrain.patch_size)
    assert len(campaign.encoders) == 1
    encoder = campaign.encoders[0]
    assert encoder.backbone == pretrain.encoder
    assert encoder.pretrained == f"runs/pretrain/lens_lejepa/{pretrain.run_name}/last_pretrain.pt"
    assert campaign.data_root == pretrain.data_root
    assert campaign.train_dataset == pretrain.dataset
