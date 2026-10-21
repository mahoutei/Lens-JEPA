import numpy as np

from lens_lejepa.data import (
    AxionMassDataset,
    LensClassificationDataset,
    SyntheticSuperResolutionDataset,
    UnlabelledLensDataset,
    few_shot_split,
    quantile_split,
    quantile_subsample,
    stratified_split,
)
from lens_lejepa.data.io import load_lens_image


def test_loader_handles_every_storage_format(tmp_path):
    image = np.random.rand(20, 20).astype(np.float32)
    np.save(tmp_path / "plain.npy", image)
    np.save(tmp_path / "channel.npy", image[None])
    wrapped = np.empty(2, dtype=object)
    wrapped[0], wrapped[1] = image, 1e-23
    np.save(tmp_path / "axion.npy", wrapped, allow_pickle=True)
    for name in ("plain", "channel", "axion"):
        assert load_lens_image(tmp_path / f"{name}.npy").shape == (1, 20, 20)


def test_datasets(fake_root):
    cls = LensClassificationDataset(fake_root, image_size=32)
    item = cls[0]
    assert item["image"].shape == (1, 32, 32)
    assert abs(float(item["image"].mean())) < 1e-4  # standardised
    assert sorted(set(cls.labels.tolist())) == [0, 1, 2]

    reg = AxionMassDataset(fake_root, image_size=32)
    assert len(reg) == 12 and np.all((reg.targets >= -24) & (reg.targets <= -22))

    sr = SyntheticSuperResolutionDataset(fake_root, image_size=32)
    pair = sr[0]
    assert pair["high_res"].shape == (1, 64, 64) and pair["low_res"].shape == (1, 32, 32)
    assert pair["high_res"].min() >= 0 and pair["high_res"].max() <= 1

    ssl = UnlabelledLensDataset(fake_root, image_size=32, max_samples=10)
    assert len(ssl) == 10 and set(ssl[0]) == {"image"}


def test_stratified_split_is_disjoint_and_deterministic():
    labels = np.repeat([0, 1, 2], 100)
    train, val = stratified_split(labels, 0.8, seed=42)
    assert not set(train) & set(val) and len(train) + len(val) == 300
    assert np.bincount(labels[train]).tolist() == [80, 80, 80]
    assert np.array_equal(train, stratified_split(labels, 0.8, seed=42)[0])


def test_few_shot_split():
    labels = np.repeat([0, 1, 2], 100)
    train, val = few_shot_split(labels, shots_per_class=20, val_fraction=0.25, seed=0)
    assert np.bincount(labels[train]).tolist() == [20, 20, 20]
    assert np.bincount(labels[val]).tolist() == [5, 5, 5]
    assert not set(train) & set(val)


def test_quantile_split_covers_range():
    values = np.random.RandomState(0).uniform(-24, -22, 1000)
    train, val = quantile_split(values, 0.8, seed=0)
    assert not set(train) & set(val) and len(train) + len(val) == 1000
    assert values[val].min() < -23.7 and values[val].max() > -22.3
    subset = quantile_subsample(values, 100, seed=0)
    assert len(subset) == len(set(subset)) == 100


def test_shift_image_and_training_jitter(fake_root):
    from lens_lejepa.data.io import shift_image

    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    shifted = shift_image(image, 1, -1)
    assert shifted[0, 1, 0] == image[0, 0, 1] and shifted[0, 0, 0] == image.min()
    jittered = LensClassificationDataset(fake_root, 32, jitter=2)
    assert jittered[0]["image"].shape == (1, 32, 32)


def test_append_csv_does_not_mix_two_runs(tmp_path):
    from lens_lejepa.utils import append_csv

    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 1, "loss": 1.0})
    append_csv(path, {"epoch": 2, "loss": 0.5})
    append_csv(path, {"epoch": 1, "loss": 9.0})  # a new run starts at epoch 1
    text = path.read_text()
    assert text.count("\n") == 2  # header + one row
    assert "9.0" in text
    assert "0.5" not in text


def test_append_csv_appends_a_resumed_run(tmp_path):
    from lens_lejepa.utils import append_csv

    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 1, "loss": 1.0})
    append_csv(path, {"epoch": 2, "loss": 0.5})
    append_csv(path, {"epoch": 3, "loss": 0.25})  # a resumed run continues
    assert path.read_text().count("\n") == 4


def _load_collect_results():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "collect_results.py"
    spec = importlib.util.spec_from_file_location("collect_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(root, task, backbone, adaptation, metrics):
    import json

    run = root / task / f"{backbone}_{adaptation}" / "run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "task": task, "backbone": backbone, "adaptation": adaptation,
        "train_dataset": "Model_II", "shots": None,
    }))
    (run / "summary.json").write_text(json.dumps({
        "test": {"test_root": "/data/Model_II_test", **metrics},
    }))
    return run


def test_collect_results_reports_an_unknown_task(tmp_path):
    collect_results = _load_collect_results()
    _write_summary(tmp_path, "einstein_radius", "vit_small", "rslora", {"accuracy": 0.5, "mae": 0.1})
    rows = collect_results.collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["task"] == "einstein_radius"
    assert rows[0]["adaptation"] == "rslora"


def test_collect_results_labels_the_lens_jepa_encoder(tmp_path):
    collect_results = _load_collect_results()
    _write_summary(tmp_path, "classification", "lens_jepa", "lora", {"accuracy": 0.83})
    rows = collect_results.collect(tmp_path)
    assert rows[0]["adaptation"] == "lora"


def test_collect_results_prints_an_unknown_task_table(tmp_path, capsys, monkeypatch):
    import sys

    collect_results = _load_collect_results()
    _write_summary(tmp_path, "einstein_radius", "vit_small", "rslora", {"accuracy": 0.5, "mae": 0.1})
    monkeypatch.setattr(sys, "argv", ["collect_results.py", str(tmp_path)])
    collect_results.main()
    printed = capsys.readouterr().out
    assert "### einstein_radius" in printed
    assert "| accuracy | mae |" in printed


def test_collect_results_keeps_the_known_task_columns(tmp_path, capsys, monkeypatch):
    import sys

    collect_results = _load_collect_results()
    _write_summary(tmp_path, "classification", "vit_small", "rslora", {"accuracy": 0.83, "mean_auroc": 0.9})
    monkeypatch.setattr(sys, "argv", ["collect_results.py", str(tmp_path)])
    collect_results.main()
    printed = capsys.readouterr().out
    assert "| run | backbone | adaptation | train | shots | test | accuracy | mean_auroc | macro_f1 |" in printed
