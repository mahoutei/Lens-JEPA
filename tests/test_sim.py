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


@pytest.mark.parametrize("mass", [1e-24, 2e-23])
def test_verify_accepts_axion_masses_at_the_configured_bounds(tmp_path, mass):
    _write_mini_dataset(tmp_path)
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 1.0
    write_image(tmp_path / "Model_II" / "axion" / "sim_00000.npy", image, mass)
    report = verify_dataset(tmp_path, _mini_config(tmp_path))
    assert report["ok"], report["models"]["Model_II"]["axion"]["errors"]


def test_verify_caps_the_error_list_for_malformed_files(tmp_path):
    config = SimulationConfig(
        output_root=str(tmp_path),
        workers=1,
        model_i_per_class=25,
        model_ii_per_class=1,
        model_ii_test_per_class=1,
    )
    _write_mini_dataset(tmp_path)
    axion_dir = tmp_path / "Model_I" / "axion"
    for path in axion_dir.glob("*.npy"):
        path.unlink()
    for index in range(25):
        (axion_dir / f"sim_{index:05d}.npy").write_bytes(b"not a npy file")

    report = verify_dataset(tmp_path, config)
    assert not report["ok"]
    assert len(report["models"]["Model_I"]["axion"]["errors"]) <= 20


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
