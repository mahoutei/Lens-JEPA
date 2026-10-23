# Synthetic Bring-Up Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a small, deterministic, lenstronomy-based dataset in the DeepLense `.npy` layout, plus a tiny pretrain config and a synthetic agent campaign, so every workflow runs on a CPU-only machine with no download.

**Architecture:** A new `lens_lejepa/sim/` package. `lens.py` ports DeepLense's Model II recipe into pure functions (`build_lens`, `simulate`) with no I/O; `generate.py` owns the config, seeding, the `.npy` writer, the worker pool and verification. A lazily imported `generate-data` CLI subcommand wires it into the existing config machinery. Nothing in `data/`, `tasks/`, `engine/` or `agents/` changes.

**Tech Stack:** Python 3.12, `lenstronomy>=1.14,<2` (optional `sim` extra), `astropy` (transitive), NumPy, existing `load_config`/`load_yaml_with_overrides`, `ProcessPoolExecutor`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-synthetic-data-design.md`

## Global Constraints

- `sim` extra is `lenstronomy>=1.14,<2`; the core install must not require it.
- lenstronomy 1.14.2 API: `SimAPI(num_pix=..., kwargs_single_band=..., kwargs_model=...)`. The published DeepLense script's `numpix=` does not work on this version.
- Instrument is fixed: Euclid VIS, `psf_type="GAUSSIAN"`, `coadd_years=6`; pixel scale `0.101` arcsec/px.
- `CLASSES = ("axion", "cdm", "no_sub")` — that order defines label indices 0/1/2. Do not reorder.
- Axion masses are drawn log-uniformly from `[1e-24, 2e-23]` eV. Reject a config whose widest vortex (`2 * axion_mass_to_length(axion_mass_min)`) exceeds `image_size * 0.101`.
- The conversion is `axion_mass_to_length(m) = 0.03e-22 / m`. It is its own inverse; this is deliberate (§4 of the spec). Do not "fix" it.
- Einstein radius uses the published convention: luminosity distances with `D_LS = D_S − D_L`. Expected `1.28″` for the 1e12 M☉ default.
- No module-level `np.random` calls in `lens_lejepa/sim/`. Every random draw comes from the per-image `RandomState`, including the noise seed passed to `noise_for_model`.
- File names are `sim_{index:05d}.npy`. Axion files are object arrays `[image, mass_eV]`; all others are plain `float32` arrays.
- Downstream modules (`data/`, `tasks/`, `engine/`, `agents/`) are not modified.
- Generated data is bring-up only. No metric computed on it may be presented as the paper's; every artefact name says `sim`.
- Run full-suite commands as `rm -rf /tmp/pytest-of-deansama && .venv/bin/python -m pytest -q` (`/tmp` is a small tmpfs here).

## Review Focus

1. **A blank or degenerate class reaches the agent loop.** It would look like a collapsed run and poison best-run selection; `verify_dataset` must reject a constant or non-finite image and a short class (Task 6).
2. **An axion mass below the field-fitting window** produces a 30″ vortex on a 6.46″ field — a featureless image with a meaningless label. `SimulationConfig.__post_init__` must reject it (Task 5).
3. **Nondeterminism** from lenstronomy's noise RNG or from parallel workers. The same seed must give byte-identical files serially and with workers; `noise_for_model(seed=...)` and per-image `RandomState` must be used (Tasks 4 and 7).
4. **A pretrain checkpoint whose `image_size`/`patch_size` differ from the campaign's.** `load_pretrained_encoder` strict-loads, so this fails late; the two shipped configs must agree (Task 9).
5. **An interrupted generation leaves a partial dataset.** Resumption must skip complete files, and the final verification must compare per-class counts against the config (Tasks 6 and 7).

---

### Task 1: The `sim` extra and the install gate

The spec makes this a hard gate: if installing `lenstronomy` damages the core environment, stop rather than proceed.

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: an environment in which `import lenstronomy` works and the core suite is green.

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, after the `agent` extra:

```toml
# Synthetic bring-up data generation (lenstronomy). Not needed for training
# or evaluation; see configs/data/sim.yaml and README "Synthetic dataset".
sim = ["lenstronomy>=1.14,<2"]
```

- [ ] **Step 2: Install it**

```bash
.venv/bin/python -m pip install -e ".[sim]"
```

Expected: `lenstronomy 1.14.x` installs; `scipy` is downgraded from 1.18.1 to a 1.17.x (lenstronomy 1.14.2 requires `scipy<1.18.0`). `numba`, `astropy`, `matplotlib`, `h5py` and friends are pulled in.

If the resolver cannot produce a consistent set, do not improvise a partial install silently: record the failure, stop, and report it. A `--no-deps` install of lenstronomy is acceptable **only** as a documented last resort, with the import smoke test and every sim test run afterwards.

- [ ] **Step 3: Verify the API and the core suite**

```bash
.venv/bin/python -c "import lenstronomy, inspect; from lenstronomy.SimulationAPI.sim_api import SimAPI; print(lenstronomy.__version__); print(inspect.signature(SimAPI.__init__))"
```

Expected: `1.14.x` and `(self, num_pix, kwargs_single_band, kwargs_model)`.

```bash
rm -rf /tmp/pytest-of-deansama && .venv/bin/python -m pytest -q
```

Expected: `144 passed` (plus the pre-existing warning) in roughly 3 minutes. **If anything fails, stop** — restore the environment before continuing (`.venv/bin/python -m pip install "scipy==1.18.1" "numpy==2.5.2"`, then `.venv/bin/python -m pip uninstall -y lenstronomy`) and report.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add the optional sim extra for lenstronomy"
```

---

### Task 2: Lens conversions and Einstein radius

**Files:**
- Create: `lens_lejepa/sim/__init__.py`
- Create: `lens_lejepa/sim/lens.py`
- Create: `tests/test_sim.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `EUCLID_PIXEL_SCALE: float = 0.101`
  - `LensParameters` frozen dataclass with the fields listed below
  - `axion_mass_to_length(mass_eV: float) -> float`
  - `einstein_radius(mass: float, z_halo: float, z_source: float) -> float`

- [ ] **Step 1: Create the package shell**

`lens_lejepa/sim/__init__.py`:

```python
"""Synthetic bring-up data generation.

Requires the optional ``sim`` extra (``pip install -e ".[sim]"``). The data
written by this package is for CI, smoke runs and agent-loop demos only; see
``docs/superpowers/specs/2026-10-23-synthetic-data-design.md``.
"""
```

- [ ] **Step 2: Write the failing tests**

`tests/test_sim.py`:

```python
"""Physics and generation tests for the synthetic bring-up data.

The whole module skips when the optional ``sim`` extra is absent, so CI and a
core install stay green without lenstronomy.
"""

import numpy as np
import pytest

pytest.importorskip("lenstronomy")

from lens_lejepa.sim.lens import LensParameters, axion_mass_to_length, einstein_radius


def test_axion_conversion_matches_the_published_values():
    # 0.03e-22 / 1e-24 = 3.0 arcsec; 0.03e-22 / 2e-23 = 0.15 arcsec.
    assert axion_mass_to_length(1e-24) == pytest.approx(3.0)
    assert axion_mass_to_length(2e-23) == pytest.approx(0.15)


def test_axion_conversion_is_its_own_inverse():
    # DeepLense applies this function in both directions; f(f(x)) == x, so that
    # is correct and must stay that way (spec section 4).
    for mass in (1.0e-24, 3.0e-24, 2.0e-23):
        assert axion_mass_to_length(axion_mass_to_length(mass)) == pytest.approx(mass)


def test_axion_conversion_rejects_non_positive_masses():
    with pytest.raises(ValueError):
        axion_mass_to_length(0.0)


def test_einstein_radius_uses_the_published_distance_convention():
    # 1e12 Msun SIS at z=0.5 -> 1.0, luminosity distances with D_LS = D_S - D_L.
    assert einstein_radius(1e12, 0.5, 1.0) == pytest.approx(1.28, abs=0.01)


def test_einstein_radius_rejects_a_source_in_front_of_the_lens():
    with pytest.raises(ValueError):
        einstein_radius(1e12, 1.0, 0.5)


def test_lens_parameters_carry_the_published_defaults():
    params = LensParameters()
    assert (params.main_halo_mass, params.n_sub, params.vortex_res) == (1e12, 25, 100)
    assert (params.sub_mass_min, params.sub_mass_max, params.sub_beta) == (1e6, 1e10, -0.9)
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'lens_lejepa.sim.lens'`.

- [ ] **Step 4: Implement the module**

`lens_lejepa/sim/lens.py`:

```python
"""Lens simulation ported from DeepLense's Model II recipe.

Reference: ML4SCI/DeepLense,
``DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/Simulations/lens.py``.

The recipe is reproduced, not vendored. Deviations are listed in
``docs/superpowers/specs/2026-10-23-synthetic-data-design.md`` section 4. Images
produced here are for bring-up only and are not a substitute for DeepLense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.constants import G, M_sun, c
from astropy.cosmology import FlatLambdaCDM

EUCLID_PIXEL_SCALE = 0.101  # arcsec / pixel, lenstronomy Euclid VIS
_RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0
_COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
_G_SI = G.value
_M_SUN_KG = M_sun.value
_C_SI = c.value


@dataclass(frozen=True)
class LensParameters:
    """The published Model II recipe; defaults are DeepLense's values."""

    main_halo_mass: float = 1e12
    z_halo: float = 0.5
    z_source: float = 1.0
    n_sub: int = 25
    sub_mass_min: float = 1e6
    sub_mass_max: float = 1e10
    sub_beta: float = -0.9
    sub_radius_min: float = 0.25
    sub_radius_max: float = 2.0
    vortex_mass: float = 1e10
    vortex_res: int = 100


def axion_mass_to_length(mass_eV: float) -> float:
    """de Broglie vortex half-length in arcsec.

    ``0.03e-22 / m``. DeepLense's ``axion_length_to_mass`` evaluates the same
    expression and the conversion is its own inverse (``f(f(x)) = x``), so the
    published mass -> length call is correct. Do not "fix" it.
    """
    if mass_eV <= 0:
        raise ValueError(f"Axion mass must be positive, got {mass_eV!r}.")
    return 0.03e-22 / mass_eV


def einstein_radius(mass: float, z_halo: float, z_source: float) -> float:
    """SIS Einstein radius in arcsec.

    Uses the published distance convention: luminosity distances with
    ``D_LS = D_S - D_L``. The angular-diameter form would give a larger radius
    (1.66" instead of 1.28"); we keep the published one so the images match
    DeepLense's.
    """
    if z_source <= z_halo:
        raise ValueError("The source must lie behind the lens.")
    d_l = _COSMO.luminosity_distance(z_halo).to_value(u.m)
    d_s = _COSMO.luminosity_distance(z_source).to_value(u.m)
    d_ls = d_s - d_l
    theta = np.sqrt(4.0 * _G_SI * mass * _M_SUN_KG / _C_SI**2 * d_ls / (d_l * d_s))
    return float(theta * _RAD_TO_ARCSEC)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add lens_lejepa/sim/__init__.py lens_lejepa/sim/lens.py tests/test_sim.py
git commit -m "feat(sim): port the DeepLense lens conversions"
```

---

### Task 3: The three class builders

**Files:**
- Modify: `lens_lejepa/sim/lens.py`
- Modify: `tests/test_sim.py`

**Interfaces:**
- Consumes: `LensParameters`, `axion_mass_to_length`, `einstein_radius` from Task 2.
- Produces: `build_lens(kind: str, params: LensParameters, rng: np.random.RandomState, axion_mass: float | None = None) -> tuple[list[str], list[dict]]`, where `kind` is one of `"no_sub"`, `"cdm"`, `"axion"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`:

```python
from lens_lejepa.sim.lens import build_lens


def test_no_sub_has_only_the_sis():
    models, kwargs = build_lens("no_sub", LensParameters(), np.random.RandomState(0))
    assert models == ["SIS"]
    assert kwargs[0]["center_x"] == 0.0 and kwargs[0]["center_y"] == 0.0
    assert kwargs[0]["theta_E"] == pytest.approx(einstein_radius(1e12, 0.5, 1.0))


def test_cdm_has_n_sub_point_masses_inside_the_published_annulus():
    params = LensParameters()
    models, kwargs = build_lens("cdm", params, np.random.RandomState(0))
    assert models == ["SIS"] + ["POINT_MASS"] * params.n_sub
    radii = np.hypot(
        [item["center_x"] for item in kwargs[1:]],
        [item["center_y"] for item in kwargs[1:]],
    )
    assert np.all(radii >= params.sub_radius_min - 1e-9)
    assert np.all(radii <= params.sub_radius_max + 1e-9)


def test_axion_vortex_is_colinear_and_sized_by_the_mass():
    params = LensParameters()
    mass = 1e-23
    models, kwargs = build_lens("axion", params, np.random.RandomState(0), axion_mass=mass)
    assert models == ["SIS"] + ["POINT_MASS"] * params.vortex_res

    xs = np.array([item["center_x"] for item in kwargs[1:]])
    ys = np.array([item["center_y"] for item in kwargs[1:]])

    # All points lie on one line through the origin.
    assert np.max(np.abs(xs * ys[0] - ys * xs[0])) < 1e-9

    # The extreme points sit at the de Broglie half-length.
    assert np.hypot(xs[0], ys[0]) == pytest.approx(axion_mass_to_length(mass))

    # Every vortex element carries the same Einstein radius.
    element = einstein_radius(params.vortex_mass / params.vortex_res, 0.5, 1.0)
    assert all(item["theta_E"] == pytest.approx(element) for item in kwargs[1:])


def test_build_lens_rejects_an_axion_without_a_mass():
    with pytest.raises(ValueError, match="axion"):
        build_lens("axion", LensParameters(), np.random.RandomState(0))


def test_build_lens_rejects_a_mass_for_a_non_axion_class():
    with pytest.raises(ValueError, match="axion"):
        build_lens("cdm", LensParameters(), np.random.RandomState(0), axion_mass=1e-23)


def test_build_lens_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="Unknown lens class"):
        build_lens("blobs", LensParameters(), np.random.RandomState(0))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `ImportError: cannot import name 'build_lens'`.

- [ ] **Step 3: Implement the builders**

Append to `lens_lejepa/sim/lens.py`:

```python
_CLASSES = ("no_sub", "cdm", "axion")


def _draw_power_law(rng: np.random.RandomState, params: LensParameters) -> np.ndarray:
    """Truncated power-law masses, as published in ``draw_old_cdm_sub_masses``."""
    u = rng.uniform(0.0, 1.0, params.n_sub)
    low = params.sub_mass_min ** (params.sub_beta + 1.0)
    high = params.sub_mass_max ** (params.sub_beta + 1.0)
    return (low + (high - low) * u) ** (1.0 / (params.sub_beta + 1.0))


def build_lens(
    kind: str,
    params: LensParameters,
    rng: np.random.RandomState,
    axion_mass: float | None = None,
) -> tuple[list[str], list[dict]]:
    """Lens model list and kwargs for ``no_sub``, ``cdm`` or ``axion``."""
    if kind not in _CLASSES:
        raise ValueError(f"Unknown lens class {kind!r}; choose from {_CLASSES}.")
    if (kind == "axion") != (axion_mass is not None):
        raise ValueError("kind='axion' requires axion_mass, and only it accepts one.")

    models = ["SIS"]
    kwargs = [{
        "theta_E": einstein_radius(params.main_halo_mass, params.z_halo, params.z_source),
        "center_x": 0.0,
        "center_y": 0.0,
    }]

    if kind == "cdm":
        masses = _draw_power_law(rng, params)
        radii = rng.uniform(params.sub_radius_min, params.sub_radius_max, params.n_sub)
        angles = rng.uniform(0.0, 2.0 * np.pi, params.n_sub)
        for mass, radius, angle in zip(masses, radii, angles):
            models.append("POINT_MASS")
            kwargs.append({
                "theta_E": einstein_radius(mass, params.z_halo, params.z_source),
                "center_x": float(radius * np.sin(angle)),
                "center_y": float(radius * np.cos(angle)),
            })
    elif kind == "axion":
        length = axion_mass_to_length(axion_mass)
        element_mass = params.vortex_mass / params.vortex_res
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        for coord in np.linspace(-length, length, params.vortex_res):
            models.append("POINT_MASS")
            kwargs.append({
                "theta_E": einstein_radius(element_mass, params.z_halo, params.z_source),
                "center_x": float(coord * np.cos(angle)),
                "center_y": float(coord * np.sin(angle)),
            })
    return models, kwargs
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/sim/lens.py tests/test_sim.py
git commit -m "feat(sim): build the no-sub, CDM and axion lens models"
```

---

### Task 4: Rendering one image

**Files:**
- Modify: `lens_lejepa/sim/lens.py`
- Modify: `tests/test_sim.py`

**Interfaces:**
- Consumes: `build_lens`, `LensParameters`, `_COSMO` from Tasks 2–3.
- Produces: `simulate(kind: str, params: LensParameters, rng: np.random.RandomState, *, num_pix: int, axion_mass: float | None = None, noise: bool = True, noise_seed: int | None = None) -> np.ndarray` returning `float32 [num_pix, num_pix]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`:

```python
from lens_lejepa.sim.lens import simulate


@pytest.mark.parametrize(
    ("kind", "mass"),
    [("no_sub", None), ("cdm", None), ("axion", 1e-23)],
)
def test_simulate_renders_a_finite_non_constant_image(kind, mass):
    image = simulate(kind, LensParameters(), np.random.RandomState(3), num_pix=64, axion_mass=mass)
    assert image.shape == (64, 64)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert image.std() > 0


def test_simulate_is_deterministic_for_a_seeded_rng():
    params = LensParameters()
    first = simulate("cdm", params, np.random.RandomState(11), num_pix=64)
    second = simulate("cdm", params, np.random.RandomState(11), num_pix=64)
    assert np.array_equal(first, second)


def test_simulate_without_noise_skips_the_noise_draw():
    params = LensParameters()
    clean = simulate("no_sub", params, np.random.RandomState(5), num_pix=64, noise=False)
    noisy = simulate("no_sub", params, np.random.RandomState(5), num_pix=64, noise=True)
    assert not np.array_equal(clean, noisy)


def test_simulate_rejects_an_axion_without_a_mass():
    with pytest.raises(ValueError, match="axion"):
        simulate("axion", LensParameters(), np.random.RandomState(0), num_pix=64)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `ImportError: cannot import name 'simulate'`.

- [ ] **Step 3: Implement the renderer**

Append to `lens_lejepa/sim/lens.py`:

```python
def simulate(
    kind: str,
    params: LensParameters,
    rng: np.random.RandomState,
    *,
    num_pix: int,
    axion_mass: float | None = None,
    noise: bool = True,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Render one image with the published instrument and source model.

    ``rng`` drives the source position, the substructure and (when ``noise_seed``
    is not given) the noise realisation, so a fresh ``RandomState`` makes the
    whole render deterministic.
    """
    from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    models, kwargs_lens = build_lens(kind, params, rng, axion_mass)
    instrument = Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6).kwargs_single_band()
    kwargs_model = {
        "lens_model_list": models,
        "lens_redshift_list": [params.z_halo] * len(models),
        "source_light_model_list": ["SERSIC_ELLIPSE"],
        "source_redshift_list": [params.z_source],
        "cosmo": _COSMO,
        "z_source_convention": 2.5,
        "z_source": 1.0,
    }
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=instrument, kwargs_model=kwargs_model)
    image_model = sim.image_model_class({"point_source_supersampling_factor": 1})

    kwargs_source = [{
        "magnitude": 20.0,
        "R_sersic": 0.25,
        "n_sersic": 1.0,
        "e1": -0.1,
        "e2": 0.1,
        "center_x": float(rng.uniform(-0.35, 0.35)),
        "center_y": float(rng.uniform(-0.35, 0.35)),
    }]
    _, kwargs_source, _ = sim.magnitude2amplitude(None, kwargs_source)
    image = image_model.image(kwargs_lens, kwargs_source, None)

    if noise:
        seed = noise_seed if noise_seed is not None else int(rng.randint(0, 2**31 - 1))
        image = image + sim.noise_for_model(model=image, seed=seed)
    return np.asarray(image, dtype=np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `16 passed`, well under a minute.

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/sim/lens.py tests/test_sim.py
git commit -m "feat(sim): render images with lenstronomy's Euclid config"
```

---

### Task 5: `SimulationConfig` and its validation

**Files:**
- Create: `lens_lejepa/sim/generate.py`
- Modify: `tests/test_sim.py`

**Interfaces:**
- Consumes: `LensParameters`, `axion_mass_to_length`, `EUCLID_PIXEL_SCALE` from Task 2; `lens_lejepa.config.load_config`.
- Produces: `SimulationConfig` dataclass with fields `output_root`, `seed`, `workers`, `image_size`, `noise`, `model_i_per_class`, `model_ii_per_class`, `model_ii_test_per_class`, `axion_mass_min`, `axion_mass_max`, `physics`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`:

```python
from lens_lejepa.config import load_config
from lens_lejepa.sim.generate import SimulationConfig


def test_config_defaults_match_the_design():
    config = SimulationConfig()
    assert config.image_size == 64
    assert (config.axion_mass_min, config.axion_mass_max) == (1e-24, 2e-23)
    assert (config.model_i_per_class, config.model_ii_per_class, config.model_ii_test_per_class) == (667, 1000, 200)
    assert isinstance(config.physics, LensParameters)


def test_config_rejects_a_vortex_wider_than_the_field():
    # 1e-25 eV gives a 30" vortex on a 6.46" field.
    with pytest.raises(ValueError, match="vortex"):
        SimulationConfig(axion_mass_min=1e-25)


def test_config_rejects_an_inverted_mass_range():
    with pytest.raises(ValueError, match="axion_mass"):
        SimulationConfig(axion_mass_min=2e-23, axion_mass_max=1e-24)


def test_config_rejects_a_zero_count():
    with pytest.raises(ValueError, match="model_ii_per_class"):
        SimulationConfig(model_ii_per_class=0)


def test_config_rejects_unknown_yaml_keys(tmp_path):
    path = tmp_path / "sim.yaml"
    path.write_text("nope: 1\n")
    with pytest.raises(ValueError, match="Unknown SimulationConfig"):
        load_config(SimulationConfig, path)


def test_config_builds_physics_from_a_nested_dict():
    config = SimulationConfig(physics={"n_sub": 10, "vortex_res": 50})
    assert config.physics.n_sub == 10
    assert config.physics.vortex_res == 50
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `ModuleNotFoundError: No module named 'lens_lejepa.sim.generate'`.

- [ ] **Step 3: Implement the config**

`lens_lejepa/sim/generate.py`:

```python
"""Writing a synthetic dataset in the DeepLense class-folder layout.

The physics lives in :mod:`lens_lejepa.sim.lens`; this module owns the config,
the per-image seeds, the ``.npy`` writer, the worker pool and verification.

Generated data is for bring-up only — see the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .lens import EUCLID_PIXEL_SCALE, LensParameters, axion_mass_to_length, simulate

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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `22 passed`.

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/sim/generate.py tests/test_sim.py
git commit -m "feat(sim): add the simulation config and its field-fit validation"
```

---

### Task 6: The writer and dataset verification

**Files:**
- Modify: `lens_lejepa/sim/generate.py`
- Modify: `tests/test_sim.py`

**Interfaces:**
- Consumes: `lens_lejepa.data.io.load_lens_image`, `lens_lejepa.data.io.load_axion_image_and_log_mass`, `SimulationConfig` from Task 5.
- Produces:
  - `write_image(path: str | Path, image: np.ndarray, axion_mass: float | None = None) -> None`
  - `verify_dataset(root: str | Path, config: SimulationConfig) -> dict` returning `{"ok": bool, "models": {model: {class: {"count": int, "errors": list[str]}}}}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`:

```python
import shutil

from lens_lejepa.data.io import load_axion_image_and_log_mass, load_lens_image
from lens_lejepa.sim.generate import verify_dataset, write_image


def _write_mini_dataset(root):
    """One valid image per model/class, written directly, without rendering."""
    for model in ("Model_I", "Model_II", "Model_II_test"):
        for name in ("axion", "cdm", "no_sub"):
            path = root / model / name / "sim_00000.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            image = np.zeros((64, 64), dtype=np.float32)
            image[32, 32] = 1.0
            write_image(path, image, 1e-23 if name == "axion" else None)


def _mini_config(root):
    return SimulationConfig(
        output_root=str(root),
        workers=1,
        model_i_per_class=1,
        model_ii_per_class=1,
        model_ii_test_per_class=1,
    )


def test_write_image_round_trips_an_axion_file(tmp_path):
    path = tmp_path / "axion.npy"
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 1.0
    write_image(path, image, 3e-24)

    loaded, log_mass = load_axion_image_and_log_mass(path)
    assert loaded.shape == (1, 64, 64)
    assert log_mass == pytest.approx(np.log10(3e-24))


def test_write_image_saves_plain_arrays_without_mass(tmp_path):
    path = tmp_path / "image.npy"
    write_image(path, np.ones((64, 64), dtype=np.float32))
    assert load_lens_image(path).shape == (1, 64, 64)
    assert np.load(path).dtype == np.float32


def test_verify_accepts_a_complete_dataset(tmp_path):
    _write_mini_dataset(tmp_path)
    report = verify_dataset(tmp_path, _mini_config(tmp_path))
    assert report["ok"]
    assert report["models"]["Model_II"]["axion"]["count"] == 1


def test_verify_flags_a_missing_class_folder(tmp_path):
    _write_mini_dataset(tmp_path)
    shutil.rmtree(tmp_path / "Model_II" / "cdm")
    report = verify_dataset(tmp_path, _mini_config(tmp_path))
    assert not report["ok"]
    assert report["models"]["Model_II"]["cdm"]["count"] == 0
    assert any("expected 1 files" in error for error in report["models"]["Model_II"]["cdm"]["errors"])


def test_verify_flags_a_constant_image(tmp_path):
    _write_mini_dataset(tmp_path)
    np.save(tmp_path / "Model_II" / "no_sub" / "sim_00000.npy", np.zeros((64, 64), dtype=np.float32))
    report = verify_dataset(tmp_path, _mini_config(tmp_path))
    assert not report["ok"]
    assert any("constant" in error for error in report["models"]["Model_II"]["no_sub"]["errors"])


def test_verify_flags_an_axion_mass_outside_the_configured_range(tmp_path):
    _write_mini_dataset(tmp_path)
    path = tmp_path / "Model_II" / "axion" / "sim_00000.npy"
    write_image(path, np.ones((64, 64), dtype=np.float32), 1e-22)
    report = verify_dataset(tmp_path, _mini_config(tmp_path))
    assert not report["ok"]
    assert any("outside the configured range" in error for error in report["models"]["Model_II"]["axion"]["errors"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `ImportError: cannot import name 'verify_dataset'`.

- [ ] **Step 3: Implement the writer and verifier**

Add to the imports of `lens_lejepa/sim/generate.py`:

```python
from ..data.io import load_axion_image_and_log_mass, load_lens_image
```

Append to `lens_lejepa/sim/generate.py`:

```python
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


def verify_dataset(root: str | Path, config: SimulationConfig) -> dict:
    """Check counts, shapes, finiteness, class contrast and axion masses.

    Returns a report instead of raising: ``generate`` decides what is fatal, and
    tests can inspect partial failures.
    """
    root = Path(root)
    report: dict = {"ok": True, "models": {}}
    plan = (
        ("Model_I", config.model_i_per_class),
        ("Model_II", config.model_ii_per_class),
        ("Model_II_test", config.model_ii_test_per_class),
    )
    for model, expected in plan:
        per_class: dict = {}
        for name in CLASSES:
            files = sorted((root / model / name).glob("*.npy"))
            errors: list[str] = []
            if len(files) != expected:
                errors.append(f"expected {expected} files, found {len(files)}")
            for path in files:
                try:
                    if name == "axion":
                        image, log_mass = load_axion_image_and_log_mass(path)
                        mass = 10.0 ** log_mass
                        if not config.axion_mass_min <= mass <= config.axion_mass_max:
                            errors.append(f"{path.name}: mass {mass:g} outside the configured range")
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
                if len(errors) >= 20:
                    break
            per_class[name] = {"count": len(files), "errors": errors}
            if errors:
                report["ok"] = False
        report["models"][model] = per_class
    return report
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `29 passed`.

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/sim/generate.py tests/test_sim.py
git commit -m "feat(sim): write the DeepLense npy format and verify a dataset"
```

---

### Task 7: Seeded generation, workers and resumption

**Files:**
- Modify: `lens_lejepa/sim/generate.py`
- Modify: `tests/test_sim.py`

**Interfaces:**
- Consumes: `write_image`, `verify_dataset`, `SimulationConfig`, `simulate` from Tasks 4–6.
- Produces:
  - `generate(config: SimulationConfig) -> dict` (the verification report; raises on failure)
  - `generate_from_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`:

```python
from lens_lejepa.data.datasets import list_class_files
from lens_lejepa.sim.generate import generate


def _tiny_config(root, workers=1):
    return SimulationConfig(
        output_root=str(root),
        workers=workers,
        model_i_per_class=1,
        model_ii_per_class=1,
        model_ii_test_per_class=1,
    )


def test_generate_writes_the_expected_tree(tmp_path):
    report = generate(_tiny_config(tmp_path))
    assert report["ok"]
    assert (tmp_path / "Model_II_test" / "no_sub" / "sim_00000.npy").exists()

    paths, labels = list_class_files(tmp_path / "Model_II")
    assert len(paths) == 3
    assert sorted(labels.tolist()) == [0, 1, 2]


def test_generated_images_load_through_the_real_dataset(tmp_path):
    from lens_lejepa.data.datasets import LensClassificationDataset

    generate(_tiny_config(tmp_path))
    dataset = LensClassificationDataset(tmp_path / "Model_II", 64)
    item = dataset[0]
    assert item["image"].shape == (1, 64, 64)
    assert item["label"].dtype.is_floating_point is False


def test_parallel_generation_is_byte_identical_to_serial(tmp_path):
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    generate(_tiny_config(serial, workers=1))
    generate(_tiny_config(parallel, workers=2))

    for model in ("Model_I", "Model_II", "Model_II_test"):
        for name in ("axion", "cdm", "no_sub"):
            path = serial / model / name / "sim_00000.npy"
            assert path.read_bytes() == (parallel / model / name / path.name).read_bytes()


def test_generate_skips_files_that_already_exist(tmp_path):
    config = _tiny_config(tmp_path)
    generate(config)
    kept = tmp_path / "Model_II" / "cdm" / "sim_00000.npy"
    stamp = kept.stat().st_mtime_ns

    (tmp_path / "Model_II" / "no_sub" / "sim_00000.npy").unlink()
    generate(config)

    assert kept.stat().st_mtime_ns == stamp
    assert (tmp_path / "Model_II" / "no_sub" / "sim_00000.npy").exists()


def test_generate_raises_when_verification_fails(tmp_path, monkeypatch):
    import lens_lejepa.sim.generate as module

    monkeypatch.setattr(
        module,
        "verify_dataset",
        lambda root, config: {"ok": False, "models": {"Model_II": {"cdm": {"count": 0, "errors": ["boom"]}}}},
    )
    with pytest.raises(RuntimeError, match="verification"):
        generate(_tiny_config(tmp_path))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `ImportError: cannot import name 'generate'`.

- [ ] **Step 3: Implement generation**

Add to the imports of `lens_lejepa/sim/generate.py`:

```python
import functools
from concurrent.futures import ProcessPoolExecutor

from ..config import load_config
```

Append to `lens_lejepa/sim/generate.py`:

```python
def _jobs(config: SimulationConfig) -> list[dict]:
    jobs: list[dict] = []
    plan = (
        ("Model_I", config.model_i_per_class),
        ("Model_II", config.model_ii_per_class),
        ("Model_II_test", config.model_ii_test_per_class),
    )
    for model, count in plan:
        for class_index, kind in enumerate(CLASSES):
            for index in range(count):
                jobs.append({
                    "model_id": _MODEL_IDS[model],
                    "kind": kind,
                    "class_index": class_index,
                    "index": index,
                    "path": str(Path(config.output_root) / model / kind / f"sim_{index:05d}.npy"),
                })
    return jobs


def _render_job(config: SimulationConfig, job: dict) -> str:
    path = Path(job["path"])
    if path.exists():  # resumable: deterministic seeds make this safe
        return str(path)

    rng = np.random.RandomState([config.seed, job["model_id"], job["class_index"], job["index"]])
    axion_mass = None
    if job["kind"] == "axion":
        axion_mass = float(
            np.exp(rng.uniform(np.log(config.axion_mass_min), np.log(config.axion_mass_max)))
        )
    image = simulate(
        job["kind"],
        config.physics,
        rng,
        num_pix=config.image_size,
        axion_mass=axion_mass,
        noise=config.noise,
    )
    write_image(path, image, axion_mass)
    return str(path)


def _summarise(report: dict) -> str:
    problems = [
        f"{model}/{name}: {'; '.join(entry['errors'][:3])}"
        for model, classes in report["models"].items()
        for name, entry in classes.items()
        if entry["errors"]
    ]
    return " | ".join(problems[:5])


def generate(config: SimulationConfig) -> dict:
    """Write every image the config describes, then verify the result.

    Existing files are skipped, so an interrupted run resumes. Raises
    ``RuntimeError`` if the resulting dataset is short or degenerate.
    """
    jobs = _jobs(config)
    render = functools.partial(_render_job, config)

    if config.workers == 1:
        for job in tqdm(jobs, desc="simulating", leave=False):
            render(job)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for _ in tqdm(pool.map(render, jobs), total=len(jobs), desc="simulating", leave=False):
                pass

    report = verify_dataset(config.output_root, config)
    if not report["ok"]:
        raise RuntimeError(f"generated dataset failed verification: {_summarise(report)}")
    return report


def generate_from_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Build a :class:`SimulationConfig` from YAML + ``--set`` overrides and run it."""
    return generate(load_config(SimulationConfig, path, overrides))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `34 passed`, roughly 10–20 seconds (18 renders in the serial/parallel test).

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/sim/generate.py tests/test_sim.py
git commit -m "feat(sim): seeded parallel generation with resumption"
```

---

### Task 8: CLI, shipped config, exports and gitignore

**Files:**
- Modify: `lens_lejepa/cli.py`
- Modify: `lens_lejepa/sim/__init__.py`
- Create: `configs/data/sim.yaml`
- Modify: `.gitignore`
- Modify: `tests/test_sim.py`

**Interfaces:**
- Consumes: `SimulationConfig`, `generate`, `generate_from_config`, `verify_dataset`, `write_image`, `simulate`, `build_lens`, `LensParameters`, `axion_mass_to_length`, `einstein_radius`.
- Produces: the `generate-data` subcommand and `configs/data/sim.yaml`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`:

```python
from lens_lejepa.cli import main


def test_shipped_sim_config_matches_the_design():
    config = load_config(SimulationConfig, "configs/data/sim.yaml")
    assert (config.image_size, config.noise) == (64, True)
    assert (config.model_i_per_class, config.model_ii_per_class, config.model_ii_test_per_class) == (667, 1000, 200)
    assert (config.axion_mass_min, config.axion_mass_max) == (1e-24, 2e-23)
    assert config.physics.n_sub == 25


def test_cli_generate_data(tmp_path):
    main([
        "generate-data",
        "--set",
        f"output_root={tmp_path}",
        "workers=1",
        "model_i_per_class=1",
        "model_ii_per_class=1",
        "model_ii_test_per_class=1",
    ])
    assert (tmp_path / "Model_II" / "axion" / "sim_00000.npy").exists()
    assert (tmp_path / "Model_I" / "no_sub" / "sim_00000.npy").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `FileNotFoundError` for `configs/data/sim.yaml`, then `SystemExit` because `generate-data` is not a known subcommand.

- [ ] **Step 3: Add the config and the CLI branch**

`configs/data/sim.yaml`:

```yaml
# Synthetic bring-up dataset: DeepLense's Model II recipe at demo scale.
# NOT paper-grade — see docs/superpowers/specs/2026-10-23-synthetic-data-design.md.
#   python -m lens_lejepa generate-data --config configs/data/sim.yaml
output_root: ../datasets_sim
seed: 42
workers: 8
image_size: 64
noise: true
model_i_per_class: 667
model_ii_per_class: 1000
model_ii_test_per_class: 200
axion_mass_min: 1.0e-24
axion_mass_max: 2.0e-23
physics:
  n_sub: 25
  vortex_mass: 1.0e10
  vortex_res: 100
```

In `lens_lejepa/cli.py`, add to the module docstring:

```
    python -m lens_lejepa generate-data --config configs/data/sim.yaml
```

Next to the `agent-parser` block:

```python
    generate_parser = commands.add_parser("generate-data", help="write a synthetic DeepLense-format dataset")
    _add_config_args(generate_parser)
```

And next to the `agent-loop` branch:

```python
    elif args.command == "generate-data":
        from .sim import generate_from_config

        generate_from_config(args.config, args.set)
```

`lens_lejepa/sim/__init__.py` — replace with:

```python
"""Synthetic bring-up data generation.

Requires the optional ``sim`` extra (``pip install -e ".[sim]"``). The data
written by this package is for CI, smoke runs and agent-loop demos only; see
``docs/superpowers/specs/2026-10-23-synthetic-data-design.md``.
"""

from .generate import (
    SimulationConfig,
    generate,
    generate_from_config,
    verify_dataset,
    write_image,
)
from .lens import (
    LensParameters,
    axion_mass_to_length,
    build_lens,
    einstein_radius,
    simulate,
)

__all__ = [
    "LensParameters",
    "SimulationConfig",
    "axion_mass_to_length",
    "build_lens",
    "einstein_radius",
    "generate",
    "generate_from_config",
    "simulate",
    "verify_dataset",
    "write_image",
]
```

`.gitignore` — append:

```
# generated synthetic dataset (never commit images)
datasets_sim/
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim.py -q
```

Expected: `36 passed`.

Also check the lazy import:

```bash
.venv/bin/python -m lens_lejepa --help
```

Expected: `generate-data` is listed, and the command returns without importing lenstronomy (it only imports the `sim` package inside the branch).

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/cli.py lens_lejepa/sim/__init__.py configs/data/sim.yaml .gitignore tests/test_sim.py
git commit -m "feat(cli): add generate-data and the shipped simulation config"
```

---

### Task 9: Pretrain and campaign configs

**Files:**
- Create: `configs/pretrain/sim_smoke.yaml`
- Create: `configs/agent/campaign_sim.yaml`
- Create: `tests/test_sim_configs.py`

**Interfaces:**
- Consumes: `PretrainConfig` and `load_config` from `lens_lejepa.config`; `CampaignSpec` from `lens_lejepa.agents` (needs the `agent` extra and `dlens`).
- Produces: two configs whose geometry and checkpoint path agree.

- [ ] **Step 1: Write the failing tests**

`tests/test_sim_configs.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sim_configs.py -q
```

Expected: `FileNotFoundError` for `configs/pretrain/sim_smoke.yaml`.

- [ ] **Step 3: Write the configs**

`configs/pretrain/sim_smoke.yaml`:

```yaml
# CPU smoke pretraining on the synthetic bring-up dataset.
#   python -m lens_lejepa pretrain --config configs/pretrain/sim_smoke.yaml
#
# The geometry MUST match configs/agent/campaign_sim.yaml: the encoder is
# strict-loaded downstream, and image_size/patch_size change the state-dict
# shapes. NOT paper-grade.
method: lens_lejepa
data_root: ../datasets_sim
dataset: Model_I
image_size: 64
patch_size: 8
encoder: vit_tiny
projector_dim: 256
epochs: 20
batch_size: 64
lr: 5.0e-4
start_lr: 5.0e-5
final_lr: 5.0e-7
warmup_epochs: 2
weight_decay: 0.05
final_weight_decay: 0.05
objective:
  sigreg_weight: 0.02
  sigreg_slices: 128
  blur_probability: 0.5
  blur_sigma_max: 1.0
  noise_std_max: 0.03
seed: 42
workers: 2
device: cpu
bf16: false
output_dir: runs
run_name: sim_smoke_seed42
```

`configs/agent/campaign_sim.yaml`:

```yaml
# Agent campaign on the synthetic bring-up dataset.
#   python -m lens_lejepa agent-loop --config configs/agent/campaign_sim.yaml
#
# NOT paper-grade: every metric here is measured on generated images and is not
# comparable with the paper's DeepLense numbers.
name: lens_lejepa_adapters_sim
hypothesis: >-
  Which adapter, rank and learning rate give the best validation accuracy on the
  synthetic Model II with the 64 px / patch 8 geometry?

task: classification
data_root: ../datasets_sim
train_dataset: Model_II
test_dataset: Model_II_test
shots: null
seed: 42
device: cpu
bf16: false
workers: 2

image_size: 64
patch_size: 8
val_interval: 5

max_iterations: 4
patience: 2
output_dir: runs/agent

# One entry: encoder switching needs a second pretrain (configs/pretrain/lejepa.yaml
# has the control). Adapters, ranks and learning rates are still searched.
encoders:
  - name: lens_lejepa
    backbone: vit_tiny
    method: lens_lejepa
    pretrained: runs/pretrain/lens_lejepa/sim_smoke_seed42/last_pretrain.pt

initial:
  encoder: lens_lejepa
  adaptation: rslora
  rank: 32
  alpha: 64.0
  lr: 3.0e-4
  weight_decay: 0.01
  dropout: 0.0
  epochs: 20
  batch_size: 64
  label_smoothing: 0.1
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sim_configs.py -q
```

Expected: `3 passed` (three if `dlens` is installed; otherwise one passes and two skip).

- [ ] **Step 5: Commit**

```bash
git add configs/pretrain/sim_smoke.yaml configs/agent/campaign_sim.yaml tests/test_sim_configs.py
git commit -m "feat(sim): add the CPU pretrain and synthetic campaign configs"
```

---

### Task 10: Generate the full demo dataset

This is the first real run. It is not part of CI.

**Files:**
- Modify: `configs/data/sim.yaml` (record the measured runtime)

**Interfaces:**
- Consumes: the `generate-data` command and `configs/data/sim.yaml`.
- Produces: `../datasets_sim` with 5,601 verified images.

- [ ] **Step 1: Generate**

```bash
rm -rf ../datasets_sim
time .venv/bin/python -m lens_lejepa generate-data --config configs/data/sim.yaml
```

Expected: progress bars, then no error. Note the wall-clock time.

- [ ] **Step 2: Verify the size and count**

```bash
find ../datasets_sim -name '*.npy' | wc -l
du -sh ../datasets_sim
```

Expected: `5601` and roughly 100 MB.

- [ ] **Step 3: Spot-check learnability plumbing (not accuracy yet)**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from lens_lejepa.data.datasets import LensClassificationDataset
from lens_lejepa.data.io import load_axion_image_and_log_mass, load_lens_image

root = Path("../datasets_sim/Model_II")
ds = LensClassificationDataset(root, 64)
print("images:", len(ds), "labels:", np.bincount(ds.labels))
for name in ("axion", "cdm", "no_sub"):
    path = sorted((root / name).glob("*.npy"))[0]
    image = load_axion_image_and_log_mass(path)[0] if name == "axion" else load_lens_image(path)
    print(name, "first image mean/std:", float(image.mean()), float(image.std()))
masses = [load_axion_image_and_log_mass(p)[1] for p in sorted((root / "axion").glob("*.npy"))[:3]]
print("log10 masses:", [round(m, 2) for m in masses])
PY
```

Expected: `3000` images, labels `[1000 1000 1000]`, non-zero means, log-masses inside `[-24, -22.7]`.

- [ ] **Step 4: Record the runtime and commit**

Add a measured-runtime comment to `configs/data/sim.yaml`, for example:

```yaml
# Measured on this machine: 3m 42s with 8 workers (5,601 images, ~100 MB).
```

```bash
git add configs/data/sim.yaml
git commit -m "docs(sim): record the measured generation time"
```

---

### Task 11: Pretrain, campaign smoke run and README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the generated dataset (Task 10), `configs/pretrain/sim_smoke.yaml`, `configs/agent/campaign_sim.yaml`.
- Produces: a checkpoint, a campaign artifact with a test accuracy, and the README section.

- [ ] **Step 1: Check the agent dependencies are installed**

```bash
.venv/bin/python -c "import dlens, pydantic_ai; print('ok')"
```

If this fails, run `bash scripts/install_agent_deps.sh` first. Stop and report if that also fails.

- [ ] **Step 2: Pretrain**

```bash
time .venv/bin/python -m lens_lejepa pretrain --config configs/pretrain/sim_smoke.yaml
ls -lh runs/pretrain/lens_lejepa/sim_smoke_seed42/last_pretrain.pt
```

Expected: the run completes without triggering the collapse guard, writes `last_pretrain.pt`, and finishes within the ~10 minute target. If the collapse guard fires, increase `epochs` or lower `lr` in the config, re-run, and record what changed.

- [ ] **Step 3: Run one campaign iteration end to end**

```bash
time .venv/bin/python -m lens_lejepa agent-loop --config configs/agent/campaign_sim.yaml --set max_iterations=1 patience=1
cat runs/agent/lens_lejepa_adapters_sim/campaign.md
```

Expected: one iteration, a validation accuracy above 3-class chance (0.33), a single test evaluation, and no `FileNotFoundError`. If accuracy is at chance, adjust the physics knobs in `configs/data/sim.yaml` (`n_sub`, `sub_mass_min`) — regenerate only Model II with `--set model_i_per_class=...` — and document the change. Do not touch the loader or the task.

- [ ] **Step 4: Document it**

In `README.md`, insert after the DeepLense dataset table (before `## Quick start`):

````markdown
### Synthetic bring-up dataset (not paper-grade)

The generator writes the same `.npy` layout from DeepLense's Model II recipe
(lenstronomy SIS + CDM point masses / axion vortex, Sérsic source, Euclid VIS).
It exists for CI, smoke runs and agent-loop demos.

```bash
pip install -e ".[sim]"
python -m lens_lejepa generate-data --config configs/data/sim.yaml     # ../datasets_sim, 5,601 images, ~3-4 min on 8 workers
python -m lens_lejepa pretrain --config configs/pretrain/sim_smoke.yaml
python -m lens_lejepa agent-loop --config configs/agent/campaign_sim.yaml
```

**These are not the DeepLense images.** Nothing measured on them is comparable
with the paper's numbers, and every run using them records `sim` in its name.
The recipe and the deliberate deviations are documented in
`docs/superpowers/specs/2026-10-23-synthetic-data-design.md`.
````

Then update the repository-layout block:

```
├── cli.py               python -m lens_lejepa {pretrain, finetune, evaluate, tasks, agent-loop, generate-data}
├── sim/                 synthetic bring-up data generation (lenstronomy; optional `sim` extra)
```

and the `configs/` line:

```
configs/                 pretrain/*.yaml, downstream/*.yaml, agent/*.yaml, data/sim.yaml
```

- [ ] **Step 5: Run the full suite one last time**

```bash
rm -rf /tmp/pytest-of-deansama && .venv/bin/python -m pytest -q
```

Expected: all tests pass; `test_sim.py` runs (lenstronomy is installed) and the core tests are unaffected.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document the synthetic bring-up workflow"
```

---

## Self-Review Notes

- **Spec coverage:** §3 recipe (Tasks 2–4), §4 deviations (Tasks 2–4 docstrings and tests), §5 modules/CLI (Tasks 2, 5–8), §6 API (Tasks 2–4), §7 determinism (Tasks 4, 7), §8 layout/config (Tasks 5, 7, 8), §9 configs (Task 9), §10 verification (Task 6), §11 tests (all sim tasks), §12 packaging/docs (Tasks 1, 8, 11), §13 risks (Task 1 gate, Task 11 measurement), §14 decisions (implemented as written).
- **Type consistency:** `SimulationConfig`, `LensParameters`, `build_lens`, `simulate`, `write_image`, `verify_dataset`, `generate`, `generate_from_config` are introduced once and reused with the same names and signatures throughout.
- **Known plan-time correction:** the spec now records `theta_E = 1.28″` under the published luminosity-distance convention; Task 2's test constant matches. Everything else follows the spec text.
