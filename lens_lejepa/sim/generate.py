"""Writing a synthetic dataset in the DeepLense class-folder layout.

The physics lives in :mod:`lens_lejepa.sim.lens`; this module owns the config,
the per-image seeds, the ``.npy`` writer, the worker pool and verification.

Generated data is for bring-up only — see the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
