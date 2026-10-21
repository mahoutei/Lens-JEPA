"""Train and infer backends that run the real Lens-JEPA pipeline.

The train backend calls ``engine.finetune`` — our adapters, our bf16 training,
our best-validation checkpoint selection — and the infer backend scores the
checkpoint it produced on the validation split. Neither ever touches the test
dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dlens.schemas._downstream import InferResult, TrainResult

from ..config import FinetuneConfig
from ..data import CLASSES, make_loader
from ..engine import build_trained_model, finetune, run_inference
from ..tasks import build_task
from ..utils import resolve_device
from ._spec import CampaignSpec, DownstreamSpec


def build_finetune_config(campaign: CampaignSpec, spec: DownstreamSpec, run_name: str) -> FinetuneConfig:
    """Map a campaign and a downstream specification onto one training run."""
    encoder = campaign.encoder(spec.encoder)
    return FinetuneConfig(
        task=campaign.task,
        data_root=campaign.data_root,
        train_dataset=campaign.train_dataset,
        test_dataset=None,  # the test set is scored once, after the loop
        shots=campaign.shots,
        train_fraction=campaign.train_fraction,
        val_fraction=campaign.val_fraction,
        cache=campaign.cache,
        image_size=campaign.image_size,
        patch_size=campaign.patch_size,
        backbone=encoder.backbone,
        pretrained=encoder.pretrained,
        adaptation=spec.adaptation,
        rank=spec.rank,
        alpha=spec.alpha,
        dropout=spec.dropout,
        rank_schedule=campaign.rank_schedule,
        loraplus_ratio=campaign.loraplus_ratio,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        grad_accum=campaign.grad_accum,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        label_smoothing=spec.label_smoothing,
        train_jitter=campaign.train_jitter,
        val_interval=campaign.val_interval,
        bf16=campaign.bf16,
        seed=campaign.seed,
        workers=campaign.workers,
        device=campaign.device,
        output_dir=str(campaign.output_dir),
        run_name=run_name,
    )


class LensJepaTrainBackend:
    """Runs one downstream training run of the Lens-JEPA pipeline."""

    name = "lens_lejepa"

    def __init__(self, campaign: CampaignSpec) -> None:
        self.campaign = campaign

    def train(self, spec: DownstreamSpec, run_name: str) -> TrainResult:
        config = build_finetune_config(self.campaign, spec, run_name)
        run_dir = finetune(config)
        summary = json.loads((run_dir / "summary.json").read_text())
        monitor = build_task(config).monitor
        return TrainResult(
            run_id=run_dir.name,
            weights_path=str(run_dir / "best.pt"),
            backend=self.name,
            num_classes=len(CLASSES),
            metrics={
                f"val_{monitor}": float(summary[f"val_{monitor}"]),
                "best_epoch": float(summary["best_epoch"]),
            },
            epochs_run=int(summary["best_epoch"]),
        )


class LensJepaInferBackend:
    """Scores a saved downstream checkpoint on the validation split."""

    name = "lens_lejepa"

    def __init__(self, campaign: CampaignSpec) -> None:
        self.campaign = campaign

    def infer(self, weights_path: str) -> InferResult:
        device = resolve_device(self.campaign.device)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        task, model, config = build_trained_model(checkpoint, device)
        _, validation = task.build_datasets()
        loader = make_loader(validation, config.batch_size, shuffle=False, workers=self.campaign.workers)
        outputs = run_inference(task, model, loader, device)["outputs"]
        probabilities = outputs["probabilities"]
        predictions = probabilities.argmax(axis=1)
        labels = outputs["labels"]
        return InferResult(
            run_id=Path(weights_path).parent.name,
            num_samples=int(len(labels)),
            num_classes=int(probabilities.shape[1]),
            predictions=predictions.astype(int).tolist(),
            probabilities=probabilities.tolist(),
            true_labels=labels.astype(int).tolist(),
            accuracy=float((predictions == labels).mean()),
            backend=self.name,
        )
