"""Writing a synthetic dataset in the DeepLense class-folder layout.

The physics lives in :mod:`lens_lejepa.sim.lens`; this module owns the config,
the per-image seeds, the ``.npy`` writer, the worker pool and verification.

Generated data is for bring-up only — see the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.io import load_axion_image_and_log_mass, load_lens_image
from .lens import EUCLID_PIXEL_SCALE, LensParameters, axion_mass_to_length

CLASSES = ("axion", "cdm", "no_sub")  # this order defines the label indices 0/1/2
_MODEL_IDS = {"Model_I": 0, "Model_II": 1, "Model_II_test": 2}


@dataclass
class SimulationConfig:
    """Everything the generator needs. Defaults are the shipped demo dataset."""

    output_root: str = "../datasets_sim"
    seed: int = 42
    workers: int = 8
    image_size: int = 64
    noise: bool = True
    model_i_per_class: int = 667
    model_ii_per_class: int = 1000
    model_ii_test_per_class: int = 200
    axion_mass_min: float = 1e-24
    axion_mass_max: float = 2e-23
    physics: LensParameters = field(default_factory=LensParameters)

    def __post_init__(self) -> None:
        if isinstance(self.physics, dict):
            self.physics = LensParameters(**self.physics)
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8.")
        for name in ("model_i_per_class", "model_ii_per_class", "model_ii_test_per_class"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if not 0.0 < self.axion_mass_min < self.axion_mass_max:
            raise ValueError("axion_mass_min must be positive and below axion_mass_max.")
        widest = 2.0 * axion_mass_to_length(self.axion_mass_min)
        field_width = self.image_size * EUCLID_PIXEL_SCALE
        if widest > field_width:
            raise ValueError(
                f"The axion vortex at {self.axion_mass_min:g} eV is {widest:.2f}\" wide; "
                f"the {field_width:.2f}\" field is too small."
            )


def write_image(path: str | Path, image: np.ndarray, axion_mass: float | None = None) -> None:
    """Write one image, using the object-array ``[image, mass]`` form for axions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=np.float32)
    if axion_mass is None:
        np.save(path, image)
        return
    payload = np.empty(2, dtype=object)
    payload[0] = image
    payload[1] = float(axion_mass)
    np.save(path, payload, allow_pickle=True)


def _model_counts(config: SimulationConfig) -> list[tuple[str, int]]:
    """The model -> images-per-class plan, shared by generation and verification."""
    return [
        ("Model_I", config.model_i_per_class),
        ("Model_II", config.model_ii_per_class),
        ("Model_II_test", config.model_ii_test_per_class),
    ]


def verify_dataset(root: str | Path, config: SimulationConfig) -> dict:
    """Check counts, shapes, finiteness, class contrast and axion masses.

    Returns a report instead of raising: ``generate`` decides what is fatal, and
    tests can inspect partial failures.
    """
    root = Path(root)
    report: dict = {"ok": True, "models": {}}
    for model, expected in _model_counts(config):
        per_class: dict = {}
        for name in CLASSES:
            files = sorted((root / model / name).glob("*.npy"))
            errors: list[str] = []
            if len(files) != expected:
                errors.append(f"expected {expected} files, found {len(files)}")
            for path in files:
                if len(errors) >= 20:
                    break
                try:
                    if name == "axion":
                        image, log_mass = load_axion_image_and_log_mass(path)
                        lo = np.log10(config.axion_mass_min)
                        hi = np.log10(config.axion_mass_max)
                        if not lo - 1e-12 <= log_mass <= hi + 1e-12:
                            errors.append(f"{path.name}: mass {10.0 ** log_mass:g} outside the configured range")
                    else:
                        image = load_lens_image(path)
                except Exception as exc:  # a malformed file must not abort the report
                    errors.append(f"{path.name}: {exc}")
                    continue
                if image.shape != (1, config.image_size, config.image_size):
                    errors.append(f"{path.name}: shape {image.shape}")
                elif not np.isfinite(image).all():
                    errors.append(f"{path.name}: non-finite values")
                elif float(image.std()) == 0.0:
                    errors.append(f"{path.name}: constant image")
            per_class[name] = {"count": len(files), "errors": errors}
            if errors:
                report["ok"] = False
        report["models"][model] = per_class
    return report
