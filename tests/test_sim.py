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
