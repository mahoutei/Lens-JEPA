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
