"""End-to-end CPU runs of every command on a tiny synthetic dataset."""

import json

import pytest
import torch

from lens_lejepa.config import FinetuneConfig, PretrainConfig, load_config
from lens_lejepa.engine import evaluate, finetune, pretrain
from lens_lejepa.ssl import LensLeJEPAObjective, ObjectiveConfig, SIGReg
from lens_lejepa.models import LeJEPAProjector, build_vit


def test_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("method: lejepa\nobjective:\n  d4_weight: 0.3\n")
    config = load_config(PretrainConfig, path, ["epochs=3", "objective.rings=4"])
    assert config.epochs == 3 and config.objective.rings == 4 and config.objective.d4_weight == 0.3
    assert config.objective.use_lens_priors is False  # the control never uses the priors
    assert load_config(FinetuneConfig, None, ["shots=null", "rank_schedule=[8,16]"]).rank_schedule == [8, 16]
    with pytest.raises(ValueError):
        load_config(FinetuneConfig, None, ["not_a_field=1"])


def test_sigreg_is_reproducible_and_detects_non_gaussian():
    torch.manual_seed(0)
    gaussian = torch.randn(512, 32)
    a, b = SIGReg(64), SIGReg(64)
    assert torch.equal(a(gaussian), b(gaussian))
    assert SIGReg(64)(torch.zeros(512, 32) + 3.0) > 10 * SIGReg(64)(gaussian)


@pytest.mark.parametrize("use_priors", [True, False])
def test_objective_backpropagates(use_priors):
    vit = build_vit("vit_tiny", 32, 16)
    projector = LeJEPAProjector(vit.embed_dim, 16)
    objective = LensLeJEPAObjective(ObjectiveConfig(use_lens_priors=use_priors, sigreg_slices=16), patch_size=16)
    loss, parts = objective(vit, projector, torch.rand(8, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    assert (parts["d4"] > 0) == use_priors
    assert vit.patch_embed.proj.weight.grad is not None


def _pretrain_config(root, out, method="lens_lejepa", encoder="vit_tiny"):
    return PretrainConfig(
        method=method, data_root=str(root.parent), dataset=root.name, image_size=32, encoder=encoder,
        projector_dim=16, epochs=2, batch_size=12, warmup_epochs=1, workers=0, device="cpu",
        output_dir=str(out), run_name="t", collapse_patience=100,
        objective={"sigreg_slices": 16}, masked={"predictor_dim": 48, "predictor_depth": 1, "npred": 1, "pred_scale": [0.25, 0.25]},
    )


@pytest.mark.parametrize("method,encoder", [("ijepa", "vit_tiny"), ("lens_jepa", "lens_jepa"), ("lens_jepa_sym", "lens_jepa"), ("lens_jepa_focus", "lens_jepa")])
def test_masked_reference_pretraining(fake_root, tmp_path, method, encoder):
    run = pretrain(_pretrain_config(fake_root, tmp_path, method, encoder))
    state = torch.load(run / "last_pretrain.pt", weights_only=False)
    assert state["epoch"] == 2 and "teacher" in state
    config = _pretrain_config(fake_root, tmp_path, method, encoder)
    config.epochs = 3
    assert torch.load(pretrain(config, resume=run / "last_pretrain.pt") / "last_pretrain.pt", weights_only=False)["epoch"] == 3


def test_pretrain_and_resume(fake_root, tmp_path):
    run = pretrain(_pretrain_config(fake_root, tmp_path))
    state = torch.load(run / "last_pretrain.pt", weights_only=False)
    assert state["epoch"] == 2
    config = _pretrain_config(fake_root, tmp_path)
    config.epochs = 3
    run = pretrain(config, resume=run / "last_pretrain.pt")
    assert torch.load(run / "last_pretrain.pt", weights_only=False)["epoch"] == 3
    assert (run / "pretrain_metrics.csv").read_text().count("\n") == 4


@pytest.mark.parametrize(
    "task,backbone,adaptation",
    [
        ("classification", "vit_tiny", "rslora"),
        ("classification", "vit_tiny", "full"),
        ("classification", "resnet18", "rslora"),
        ("classification", "vit", "rslora"),
        ("classification", "vitsd", "rslora"),
        ("classification", "lensiformer", "rslora"),
        ("classification", "lenspinn", "rslora"),
        ("regression", "lenspinn", "rslora"),
        ("classification", "lens_jepa", "lora"),
        ("regression", "vit_tiny", "loraplus"),
        ("super_resolution", "vit_tiny", "lora"),
        ("super_resolution", "rcan", "lora"),
        ("super_resolution", "edsr", "lora"),
        ("super_resolution", "srresnet", "lora"),
        ("super_resolution", "fsrcnn", "lora"),
        ("super_resolution", "rdn", "lora"),
        ("super_resolution", "bicubic", "lora"),
    ],
)
def test_finetune_and_evaluate(fake_root, tmp_path, task, backbone, adaptation):
    from lens_lejepa.models import is_encoder

    method = "lens_jepa" if backbone == "lens_jepa" else "lens_lejepa"
    ssl_run = pretrain(_pretrain_config(fake_root, tmp_path / "ssl", method, backbone if backbone == "lens_jepa" else "vit_tiny"))
    config = FinetuneConfig(
        task=task, data_root=str(fake_root.parent), train_dataset=fake_root.name, test_dataset=fake_root.name,
        image_size=32, backbone=backbone, adaptation=adaptation, rank=4, alpha=8,
        pretrained=str(ssl_run / "last_pretrain.pt") if is_encoder(backbone) else None,
        epochs=2, batch_size=8, val_interval=1, workers=0, device="cpu", output_dir=str(tmp_path), run_name="t",
    )
    run = finetune(config)
    summary = json.loads((run / "summary.json").read_text())
    reloaded = evaluate(run / "best.pt", str(fake_root), "cpu", workers=0)
    for key, value in summary["test"].items():
        if isinstance(value, float):
            assert reloaded[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key


def test_sigreg_scales_with_the_sample_count_like_the_reference():
    """The reference LeJEPA statistic is T = N * sum_t w(t) * (dcos^2 + dsin^2)."""
    from lens_lejepa.ssl.sigreg import EppsPulley

    test = EppsPulley(t_max=3.0, n_points=17)
    x = torch.randn(64, 4)
    doubled = test(x.repeat(2, 1))
    single = test(x)
    # The mean over samples is unchanged by duplicating them, so the statistic
    # must scale exactly with N. A missing N factor would leave it unchanged.
    assert doubled == pytest.approx(2.0 * single, rel=1e-5)


def test_sigreg_weights_include_the_standard_normal_characteristic_function():
    """Upstream folds phi(t) into the trapezoid weights; keep it that way."""
    from lens_lejepa.ssl.sigreg import EppsPulley

    test = EppsPulley(t_max=3.0, n_points=17)
    dt = 3.0 / (17 - 1)
    trapezoid = torch.full((17,), 2 * dt)
    trapezoid[[0, -1]] = dt
    assert torch.allclose(test.weights, trapezoid * test.phi)
    assert torch.allclose(test.phi, torch.exp(-0.5 * test.t.square()))


def test_collapsed_pretraining_leaves_no_loadable_checkpoint(fake_root, tmp_path):
    from pathlib import Path

    config = _pretrain_config(fake_root, tmp_path)
    config.collapse_std_threshold = 10.0  # every epoch counts as collapsed
    config.collapse_patience = 2          # so epoch 1 still writes a checkpoint
    with pytest.raises(RuntimeError, match="collapsed"):
        pretrain(config)
    run_dir = Path(config.output_dir) / "pretrain" / config.method / config.run_name
    assert not (run_dir / "last_pretrain.pt").exists()


def test_grad_accum_scales_the_window_by_its_actual_size(fake_root, tmp_path, monkeypatch):
    """A window holding one gradient must not be scaled as if it held two."""
    recorded = []

    class RecordingAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            recorded.append(
                sum(
                    float(p.grad.abs().sum())
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                )
            )
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)

    def run(grad_accum, name):
        config = FinetuneConfig(
            task="classification", data_root=str(fake_root.parent), train_dataset=fake_root.name,
            test_dataset=None, image_size=32, backbone="vit_tiny", adaptation="linear_probe",
            epochs=1, batch_size=4, grad_accum=grad_accum, max_steps_per_epoch=1,
            val_interval=1, workers=0, device="cpu", output_dir=str(tmp_path), run_name=name,
        )
        finetune(config)

    run(1, "single")
    run(2, "accumulated")
    assert len(recorded) == 2
    # Both runs see the same first batch, so the gradient must be identical:
    # the one-batch window of the second run holds one gradient, not two.
    assert recorded[1] == pytest.approx(recorded[0], rel=1e-5)


def test_pretrain_does_not_fetch_a_batch_past_the_epoch_limit(fake_root, tmp_path, monkeypatch):
    import importlib

    # The engine shadows the submodule name with the function it exports, so the
    # module has to be fetched from the import system, not from the package.
    pretrain_module = importlib.import_module("lens_lejepa.engine.pretrain")

    calls = []
    original = pretrain_module.UnlabelledLensDataset

    class Counting(original):
        def __getitem__(self, index):
            calls.append(index)
            return super().__getitem__(index)

    monkeypatch.setattr(pretrain_module, "UnlabelledLensDataset", Counting)
    config = _pretrain_config(fake_root, tmp_path)
    config.epochs = 1
    config.max_steps_per_epoch = 1
    config.batch_size = 4
    pretrain(config)
    assert len(calls) == 4  # exactly one batch of four, not two


def test_pretrain_rejects_an_epoch_without_steps(fake_root, tmp_path):
    config = _pretrain_config(fake_root, tmp_path)
    config.max_steps_per_epoch = 0
    with pytest.raises(RuntimeError, match="no training steps"):
        pretrain(config)


def test_finetune_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="val_interval"):
        FinetuneConfig(val_interval=0)
    with pytest.raises(ValueError, match="rank_schedule"):
        FinetuneConfig(adaptation="layerwise")
    with pytest.raises(ValueError, match="Unknown adapter"):
        FinetuneConfig(adaptation="typo")
    with pytest.raises(ValueError, match="shots"):
        FinetuneConfig(shots=0)


def test_load_yaml_with_overrides_returns_a_plain_dict(tmp_path):
    from lens_lejepa.config import load_yaml_with_overrides

    path = tmp_path / "campaign.yaml"
    path.write_text("name: demo\nnested:\n  value: 1\n")
    data = load_yaml_with_overrides(path, ["nested.value=2", "extra=3"])
    assert data == {"name": "demo", "nested": {"value": 2}, "extra": 3}


def test_build_trained_model_is_public():
    from lens_lejepa.engine import build_trained_model

    assert callable(build_trained_model)
