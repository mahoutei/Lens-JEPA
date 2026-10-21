import math

import pytest
import torch

from lens_lejepa.models import (
    ADAPTERS,
    EncoderClassifier,
    PatchSuperResolution,
    build_vit,
    count_parameters,
    inject_adapters,
)
from lens_lejepa.models.adapters import LoRALinear


def test_vit_small_shapes_and_size():
    vit = build_vit("vit_small", 160, 16)
    tokens = vit(torch.randn(2, 1, 160, 160))
    assert tokens.shape == (2, 100, 384)
    assert count_parameters(vit) == 21_431_424  # ViT-S/16 as reported in the paper
    assert not vit.pos_embed.requires_grad


def test_rslora_on_vit_small_matches_paper_count():
    vit = build_vit("vit_small", 160, 16)
    targets = inject_adapters(vit, "rslora", rank=32, alpha=64)
    model = EncoderClassifier(vit)
    assert len(targets) == 48  # qkv, proj, fc1, fc2 in each of 12 blocks
    assert count_parameters(model, trainable_only=True) == 2_360_451


@pytest.mark.parametrize("method", [m for m in ADAPTERS if m != "layerwise"] + ["layerwise"])
def test_adapters_start_as_the_frozen_model(method):
    torch.manual_seed(0)
    vit = build_vit("vit_tiny", 32, 16).eval()
    x = torch.randn(2, 1, 32, 32)
    before = vit(x)
    inject_adapters(vit, method, rank=4, alpha=8, rank_schedule=[2, 4] if method == "layerwise" else None)
    assert torch.allclose(vit(x), before, atol=1e-5)
    trainable = [name for name, p in vit.named_parameters() if p.requires_grad]
    assert trainable and all(any(k in name for k in ("lora_", "magnitude", "rank_logits")) for name in trainable)


def test_rslora_scaling():
    layer = torch.nn.Linear(8, 8)
    assert LoRALinear(layer, rank=16, alpha=64).scaling == pytest.approx(4.0)
    assert LoRALinear(layer, rank=16, alpha=64, rank_stabilised=True).scaling == pytest.approx(64 / math.sqrt(16))


def test_patch_super_resolution_output():
    vit = build_vit("vit_tiny", 32, 16)
    out = PatchSuperResolution(vit, 32, 16, output_size=64)(torch.randn(3, 1, 32, 32))
    assert out.shape == (3, 1, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("name", ["vit", "vitsd"])
def test_supervised_vits(name):
    from lens_lejepa.models import SupervisedViT

    model = SupervisedViT(32, 16, 3, dim=64, heads=2, mlp_dim=64, shifted_patches=name == "vitsd")
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 3)


def test_lensiformer_and_lenspinn():
    from lens_lejepa.models import Lensiformer, LensPINN, LensPINNRegressor

    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    probabilities, source = Lensiformer(32, 16, embed_dim=32, num_heads=2).forward_with_source(images, distortion)
    assert probabilities.shape == (2, 3) and torch.allclose(probabilities.sum(1), torch.ones(2), atol=1e-5)
    assert source.shape == (2, 1, 32, 32)
    assert LensPINN()(images, distortion).shape == (2, 3)
    assert LensPINNRegressor()(images, distortion).shape == (2,)


def test_lens_jepa_encoder_and_masks():
    from lens_lejepa.models import BlockMaskCollator, JEPAPredictor, LensJEPAEncoder, MaskConfig

    encoder = LensJEPAEncoder(32, 16)
    assert count_parameters(LensJEPAEncoder(160, 16)) == 2_454_018  # the 2.45M reference backbone
    collator = BlockMaskCollator(MaskConfig(grid_size=2, npred=1, pred_scale=(0.25, 0.25)))
    images, masks_enc, masks_pred = collator([{"image": torch.randn(1, 32, 32)} for _ in range(3)])
    assert not set(masks_enc[0][0].tolist()) & set(masks_pred[0][0].tolist())  # no target leaks into context
    context = encoder(images, masks_enc)
    predicted = JEPAPredictor(4, encoder.embed_dim, 32, depth=1, num_heads=2)(context, masks_enc, masks_pred)
    assert predicted.shape == (3, masks_pred[0].shape[1], encoder.embed_dim)


def test_scatter_average_averages_collisions():
    from lens_lejepa.utils import scatter_average

    values = torch.tensor([[1.0, 3.0, 5.0]])
    index = torch.tensor([[0, 0, 2]])
    out = scatter_average(values, index, size=3)
    assert torch.allclose(out, torch.tensor([[2.0, 0.0, 5.0]]))


def test_scatter_average_is_order_independent_and_drops_no_source():
    from lens_lejepa.utils import scatter_average

    values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    index = torch.tensor([[0, 0, 2, 2]])
    forward = scatter_average(values, index, size=3)
    backward = scatter_average(values.flip(1), index.flip(1), size=3)
    assert torch.allclose(forward, backward)
    # every source still contributes: count-weighted bins sum to the sources' total
    count = torch.tensor([[2.0, 0.0, 2.0]])
    assert (forward * count).sum() == pytest.approx(values.sum())


def test_sis_inverter_averages_colliding_pixels():
    from lens_lejepa.models.lens_jepa import _SISInverter

    inverter = _SISInverter(32, 16, dim=8, heads=2, hidden=8, activation=torch.nn.ELU, blocks=1)
    values = torch.tensor([[1.0, 3.0, 5.0]])
    index = torch.tensor([[0, 0, 7]])
    out = inverter._scatter_source(values, index)
    assert out[0, 0] == pytest.approx(2.0)  # the mean of the two colliding pixels
    assert out[0, 7] == pytest.approx(5.0)
    assert out[0, 1] == 0.0  # nothing landed here


def test_sis_inverter_forward_averages_colliding_pixels():
    """forward() must average the source pixels that land on the same target."""
    from lens_lejepa.models.lens_jepa import _PatchDecoder, _SISInverter

    inverter = _SISInverter(8, 4, dim=8, heads=2, hidden=8, activation=torch.nn.ELU, blocks=1,
                            pixel_scale=0.1, k_min=0.0, k_max=0.2).eval()
    decoder = next(m for m in inverter.modules() if isinstance(m, _PatchDecoder)).patch_decoder
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.zero_()  # sigmoid(0) = 0.5, so k = 0.0 + 0.2 * 0.5 = 0.1
    # k = 0.1 moves every pixel one source pixel towards the centre: the eight
    # pixels around the centre all land on the centre pixel (4, 4), and only
    # (0, 4) lands on (1, 4).
    images = torch.zeros(1, 8, 8)
    images[0, 0, 4] = 4.0
    ring = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)]
    for value, (row, col) in enumerate(ring, start=1):
        images[0, row, col] = float(value)
    with torch.no_grad():
        out = inverter(images, torch.zeros(1, 4, 8))  # a 2x2 grid of tokens; the decoder ignores them
    # the mean of the eight contributors is 4.5, while winner-take-all would keep
    # one of them, so the ratio to the single contributor must be the ratio of 4.5 to 4.0
    assert out[0, 4, 4].item() / out[0, 1, 4].item() == pytest.approx(4.5 / 4.0)


def test_lensiformer_stack_trains_every_block():
    """A stack of blocks must give every block a gradient, not only the last."""
    from lens_lejepa.models import Lensiformer

    model = Lensiformer(32, 16, embed_dim=32, num_heads=2, blocks=2)
    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    probabilities, _ = model.forward_with_source(images, distortion)
    # the softmax output sums to one, so its *sum* carries no gradient at all
    probabilities[:, 0].sum().backward()
    assert model.transformer_blocks[0].mlsa.mha.in_proj_weight.grad is not None
    assert model.transformer_blocks[1].mlsa.mha.in_proj_weight.grad is not None
    assert model.transformer_blocks[0].mlsa.mha.in_proj_weight.grad.abs().max() > 0
    assert model.transformer_blocks[1].mlsa.mha.in_proj_weight.grad.abs().max() > 0


def test_lensiformer_blocks_are_chained():
    """Block n+1 must read block n's output as its key."""
    from lens_lejepa.models import Lensiformer

    torch.manual_seed(0)
    model = Lensiformer(32, 16, embed_dim=32, num_heads=2, blocks=2).eval()
    seen = {}

    def capture(name):
        def hook(module, args, kwargs):
            seen.setdefault(name, {})["key"] = kwargs.get("key")
            seen[name]["output"] = None

        return hook

    outputs = {}

    def forward_hook(name):
        def hook(module, args, output):
            outputs[name] = output

        return hook

    model.transformer_blocks[0].register_forward_pre_hook(capture("b0"), with_kwargs=True)
    model.transformer_blocks[1].register_forward_pre_hook(capture("b1"), with_kwargs=True)
    model.transformer_blocks[0].register_forward_hook(forward_hook("b0"))
    model.transformer_blocks[1].register_forward_hook(forward_hook("b1"))

    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        model.forward_with_source(images, distortion)

    assert seen["b1"]["key"] is outputs["b0"]


def test_lensiformer_first_block_still_reads_the_distortion_tokens():
    """The blocks=1 path must be byte-for-byte the pre-fix behaviour."""
    from lens_lejepa.models import Lensiformer

    torch.manual_seed(0)
    model = Lensiformer(32, 16, embed_dim=32, num_heads=2, blocks=1).eval()
    captured = {}
    model.transformer_blocks[0].register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(key=kwargs["key"]), with_kwargs=True
    )
    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        model.forward_with_source(images, distortion)
    expected = model.tertiary_tokenizer(distortion.reshape(2, 1, 32, 32))
    assert torch.equal(captured["key"], expected)
