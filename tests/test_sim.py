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
