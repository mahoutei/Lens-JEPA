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
