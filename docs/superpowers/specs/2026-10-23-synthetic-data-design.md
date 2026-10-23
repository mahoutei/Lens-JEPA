# Synthetic bring-up dataset for Lens-LeJEPA — design

**Date:** 2026-10-23
**Status:** approved in chat, pending review of this document
**Scope:** `lens_lejepa/sim/`, `lens_lejepa/cli.py`, `configs/data/`, `configs/pretrain/`, `configs/agent/`, `tests/test_sim.py`, `pyproject.toml`, `.gitignore`, `README.md`

---

## 1. Context

Every workflow in this repository reads the public ML4SCI DeepLense releases from a fixed layout:

```
../datasets/
├── Model_I/{axion,cdm,no_sub}/*.npy          # 150x150, pretraining only
├── Model_II/{axion,cdm,no_sub}/*.npy         # 64x64, Euclid-like
├── Model_II_test/{axion,cdm,no_sub}/*.npy
```

That works, but it makes the repository unusable without a ~200 GB download and a
machine-specific directory. On a clean checkout `pretrain`, `finetune` and `evaluate`
fail in the loader, and the agent loop (landed at `f9a97ae`) reaches
`list_class_files` and dies on `Missing class folder: ../datasets/Model_II/axion`.
No campaign has ever run end to end here.

This design adds a generator that writes that same layout at demo scale. It is built
on `lenstronomy` and ports the published DeepLense Model II recipe, so the images are
physically meaningful rather than blobs. The loaders, tasks, engine and agent code are
**not changed**: the generator is a writer of the format they already read.

**What this data is not.** It is a bring-up artefact for CI, smoke runs and agent-loop
demos. Every artefact and every doc line says so. Metrics measured on it are not
comparable with the paper's DeepLense numbers and must not be reported as such.

---

## 2. Goals and non-goals

**G1.** One command writes a complete dataset in the existing `.npy` layout, with no
changes to `data/`, `tasks/`, `engine/` or `agents/`.

**G2.** The physics comes from `lenstronomy`, following DeepLense's published
generation script (§3), with every deviation listed and tested (§4).

**G3.** Generation is deterministic: the same seed produces byte-identical files,
whether run serially or with workers, and an interrupted run resumes without changing
what is written.

**G4.** The full path runs on a CPU-only laptop with no download:
`generate-data → pretrain → finetune → agent-loop`.

**G5.** The core install stays light and the core test suite stays green: `lenstronomy`
is an optional extra and its tests skip when it is absent.

**Non-goals (v1).** Model III / HST; super-resolution and regression campaigns;
DeepLense's 89k-images-per-class scale; paper comparability; `pyHalo`; any change to
the agent loop or its shipped campaign default.

---

## 3. The recipe we port

Source: `ML4SCI/DeepLense`, `DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/Simulations/lens.py`
and `sim_cdm_sis_2.py` (main branch, read 2026-10-23).

| Component | Value |
|---|---|
| Cosmology | `FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)` |
| Redshifts | lens halo `z = 0.5`, source `z = 1.0` |
| Main lens | `SIS`, `M = 1e12 M☉`, centre `(0, 0)`; `theta_E` from `sqrt(4GM/c² · D_LS/(D_L D_S))`, converted to arcsec |
| Source | `SERSIC_ELLIPSE`, **magnitude** 20, `R_sersic = 0.25″`, `n = 1`, `e1 = -0.1`, `e2 = 0.1`, centre `U(-0.35, 0.35)²` arcsec |
| Instrument | Euclid VIS, `psf_type="GAUSSIAN"`, `coadd_years=6`; pixel scale `0.101″/px` |
| Render | `SimAPI(num_pix, kwargs_single_band, kwargs_model)` → `image_model_class({'point_source_supersampling_factor': 1})` → `magnitude2amplitude(None, kwargs_source)` → `image(...)` |
| Noise | `sim.noise_for_model(model=image, seed=...)` (background + Poisson), added to the model image |
| Passed through | `z_source_convention: 2.5` and `z_source: 1.0`, exactly as published |

The three classes differ only in the substructure appended to the lens model:

| Class | Lens model list | Substructure |
|---|---|---|
| `no_sub` | `["SIS"]` | none |
| `cdm` | `["SIS"] + ["POINT_MASS"] * n_sub` | `n_sub = 25` masses from a truncated power law, `1e6 – 1e10 M☉`, `β = -0.9`; positions `r ~ U(0.25, 2.0)″`, `φ ~ U(0, 2π)`; each `theta_E` from its own mass |
| `axion` | `["SIS"] + ["POINT_MASS"] * vortex_res` | `vortex_res = 100` equal masses of `vortex_mass / 100` (`vortex_mass = 1e10 M☉`) along a line through the centre, angle `U(0, 2π)`; the line's half-length is `L = axion_mass_to_length(m)` |

`axion_mass_to_length(m) = 0.03 · 1e-22 / m` arcsec, so a **larger** mass gives a
**shorter** vortex. The line spans `2L`, and the generator rejects any configuration
whose vortex does not fit inside the field.

Axion files are object arrays `[image, mass_eV]`, matching DeepLense and
`load_axion_image_and_log_mass`. Other classes are plain `float32` arrays.

---

## 4. Deviations from the published script — and one correction

| # | DeepLense | Ours | Why |
|---|---|---|---|
| D1 | `SimAPI(numpix=...)` | `SimAPI(num_pix=...)` | 1.14.2's parameter is `num_pix`; the published call no longer matches the API |
| D2 | `kwargs_single_band['pixel_scale'] /= coeff`, mutating the instrument dict | pixel scale read from a fresh `kwargs_single_band()` on every render | a second call on the same object halves the scale again; we keep no mutable render state |
| D3 | `pyHalo.preset_models.CDM(...)` in `__init__`, used only to obtain an `astropy` cosmology | `FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)` | the CDM and axion paths in that script are NumPy + lenstronomy; `pyHalo` contributes only the cosmology there |
| D4 | `noise_for_model(model)` draws from the global NumPy RNG | `noise_for_model(model, seed=derived_seed)` | determinism (G3); the `seed` parameter is present in lenstronomy 1.14.2 |
| D5 | `axion_mass_to_length` calls `axion_length_to_mass` | the same conversion, defined once and documented as an involution | **verified not a bug** — see below |
| D6 | `draw_old_cdm_sub_masses` draws `poisson(n_sub)` and then ignores it, returning exactly `n_sub` masses | preserved: exactly `n_sub` masses | port fidelity: the released images have a fixed count; "fixing" it would change the distribution |

**Correction: the axion conversion is self-inverse, not a bug.**
DeepLense's `axion_length_to_mass(x) = 0.06 · (2x)⁻¹ · 1e-22 = 0.03e-22 / x`. A function
of the form `C/x` satisfies `f(f(x)) = x`, so applying it in the "mass → length"
direction returns the correct length. `f(1e-24) = 3.0″`, `f(1e-25) = 30″`. The port
defines the conversion once, documents this, and tests both `f(f(m)) ≈ m` and
`f(1e-24) = 3.0`, so a future reader does not "fix" it.

The mass **range** claim stands: `1e-25` is a 30″ vortex, five times the 6.46″ field.

We do not "correct" parameters we cannot validate. `z_source_convention: 2.5` next to
`z_source: 1.0` is passed through verbatim.

---

## 5. Modules

```
lens_lejepa/sim/
  __init__.py     public exports
  lens.py         LensParameters, axion_mass_to_length, einstein_radius, build_lens, simulate
  generate.py     SimulationConfig, generate, verify_dataset, the writer, the worker pool
configs/data/sim.yaml
```

`lens.py` performs no file I/O; `generate.py` contains no lens maths. The split keeps
the physics testable without writing a dataset.

The CLI gains one subcommand, with the same `--config` / `--set` handling as the others:

```
python -m lens_lejepa generate-data --config configs/data/sim.yaml
```

It is imported lazily (like `engine` and `agents`), so `--help` and every non-sim
command work without `lenstronomy` installed.

---

## 6. Physics API

```python
EUCLID_PIXEL_SCALE = 0.101  # arcsec per pixel, lenstronomy Euclid VIS


@dataclass(frozen=True)
class LensParameters:
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
    """de Broglie vortex half-length in arcsec (self-inverse conversion)."""


def einstein_radius(mass: float, z_halo: float, z_source: float) -> float:
    """SIS Einstein radius in arcsec."""


def build_lens(kind: str, params: LensParameters, rng: np.random.RandomState,
               axion_mass: float | None = None) -> tuple[list[str], list[dict]]:
    """Lens model list and kwargs for ``no_sub`` / ``cdm`` / ``axion``."""


def simulate(kind: str, params: LensParameters, rng: np.random.RandomState,
             *, num_pix: int, axion_mass: float | None = None,
             noise: bool = True, noise_seed: int | None = None) -> np.ndarray:
    """Render one image."""
```

`build_lens` is separate from `simulate` so tests can count point masses and inspect
positions without paying for a render. `kind="axion"` requires an `axion_mass`; any
other class rejects one, and an unknown `kind` raises.

---

## 7. Determinism and parallelism

- Each image has one `RandomState`, seeded from
  `[config.seed, model_id, class_index, index]`, where the model ids are
  `0 = Model_I`, `1 = Model_II`, `2 = Model_II_test`. The test split is therefore
  **independent draws**, not a slice of training data — pretraining on Model I cannot
  touch the downstream validation or test images.
- Every random quantity comes from that RNG: source centre, substructure masses and
  positions, vortex angle, and the noise seed (derived from it and passed to
  `noise_for_model`). There are no module-level `np.random` calls.
- The file name encodes the index (`sim_00042.npy`), so resumption — skip if the file
  exists — does not depend on completion order. Delete the directory to regenerate.
- `ProcessPoolExecutor` over `(model, class, index)` tasks; each worker builds its own
  lenstronomy objects and writes its own file. The parent collects paths and stats.
  Serial and parallel runs are byte-identical (tested).

---

## 8. Datasets, layout and config

```
datasets_sim/
├── Model_I/{axion,cdm,no_sub}/sim_00000.npy       667/class, 64 px, independent draws
├── Model_II/{axion,cdm,no_sub}/sim_00000.npy      1000/class, 64 px
└── Model_II_test/{axion,cdm,no_sub}/sim_00000.npy 200/class, 64 px
```

Total 5,601 images. At 64 px × 0.101″/px the field is 6.46″, and a 1e12 M☉ SIS at
z = 0.5 → 1.0 has `theta_E ≈ 1.5″` (≈ 15 px), so the ring is well inside the frame.

Model I is generated at **64 px, not DeepLense's 150 px**. Our smoke pretrain runs at
64 and `load_pretrained_encoder` strict-loads the encoder, so the arc scale must match
downstream; generating at 150 only to resize would waste time and pixels. The classes
of Model I are irrelevant to SSL — the three folders exist because the loader requires
them.

`configs/data/sim.yaml`:

```yaml
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

`SimulationConfig` is a dataclass loaded with the existing `load_config` (unknown keys
rejected), with `physics` constructed in `__post_init__` when given as a dict — the same
pattern `PretrainConfig` uses for `objective`. Validation:

- counts ≥ 1, `image_size ≥ 8`;
- `0 < axion_mass_min < axion_mass_max`;
- the widest vortex fits: `2 · axion_mass_to_length(axion_mass_min) ≤ image_size · 0.101`.
  For the defaults: `2 × 3.0″ = 6.0″ ≤ 6.46″`.

The range `1e-24 … 2e-23` gives vortex lengths 3.0″ down to 0.15″, a 20× spread — enough
for the mass-regression task to have signal. Masses outside it are rejected rather than
silently rendered as a featureless field.

---

## 9. Pretrain and campaign configs

`configs/pretrain/sim_smoke.yaml` — a copy of `lejepa.yaml`'s structure with:

| Field | Value | Why |
|---|---|---|
| `data_root` / `dataset` | `../datasets_sim` / `Model_I` | generated data |
| `image_size` / `patch_size` | 64 / 8 | 8×8 patch grid, D4 acts exactly; matches downstream |
| `encoder` | `vit_tiny` | CPU minutes, not hours |
| `epochs` / `batch_size` | 20 / 64 | target ≤ 10 min CPU, measured and recorded |
| `device` / `bf16` | `cpu` / `false` | no GPU in the demo path |
| `workers` | 2 | CPU contention |
| `output_dir` / `run_name` | `runs` / `sim_smoke_seed42` | checkpoint at `runs/pretrain/lens_lejepa/sim_smoke_seed42/last_pretrain.pt` |

`configs/agent/campaign_sim.yaml` — the shipped campaign with `data_root:
../datasets_sim`, `image_size: 64`, `patch_size: 8`, `device: cpu`, `bf16: false`,
`workers: 2`, `shots: null`, `val_interval: 5`, `max_iterations: 4`, `patience: 2`,
`output_dir: runs/agent`, `initial.epochs: 20`, `initial.batch_size: 64`, and a
**one-entry** encoder pool pointing at the smoke checkpoint. The shipped
`configs/agent/campaign.yaml` is not modified.

A one-entry pool exercises adapter, rank and learning-rate search but not encoder
switching. The loop already tests switching; adding a second pretrain and a second pool
entry is a config-only change when wanted.

`shots: null` (not 100) because 1,000/class exists here: the demo is about the loop
running, not about few-shot stress. With `shots: null`, `stratified_split` uses
`train_fraction: 0.8`, giving 800 train / 200 validation per class; `val_fraction`
applies only to the few-shot path.

Pretrain and downstream sizes must agree: `load_pretrained_encoder` calls
`load_state_dict(..., strict=True)`, and both `image_size` and `patch_size` change the
state-dict shapes. This is why the synthetic campaign is a separate config rather than a
`--set` on the shipped one.

---

## 10. Verification and learnability

`verify_dataset(root, config)` reads every generated file and checks, per model and
class: the expected count, shape `(image_size, image_size)`, all-finite values, and a
non-zero standard deviation. For axion files it also parses the mass, checks it is
finite and inside `[axion_mass_min, axion_mass_max]`. `generate` logs the report and
raises if any class is short or degenerate — an all-zero class must not reach the agent
loop, where it would look like a collapsed run.

Learnability is **measured, not asserted**. After the first full generation, a
one-epoch `finetune` run on Model II establishes accuracy above 3-class chance; the
measured value is recorded in `configs/data/sim.yaml`'s comments and in the README. If
classes prove inseparable, the physics knobs (`n_sub`, `sub_mass_min/max`, `noise`) are
the adjustment — not the loader and not the task.

---

## 11. Testing

`tests/test_sim.py`, with `pytest.importorskip("lenstronomy")` at module level, so the
core suite stays green without the extra. Tiny counts, no network, seconds.

1. **Conversion.** `axion_mass_to_length(1e-24) ≈ 3.0`; `f(f(m)) ≈ m` (D5).
2. **Builder structure.** `no_sub` has no point masses; `cdm` has `n_sub`; `axion` has
   `vortex_res`, all at `vortex_mass / vortex_res`, colinear within tolerance.
3. **Range validation.** `axion_mass_min = 1e-25` raises when the config is built.
4. **Determinism.** Same seed twice → identical arrays; a different seed → different.
5. **Parallel equals serial.** One image per class with `workers=1` and with `workers=2`
   produces byte-identical files.
6. **Layout.** A one-per-class dataset is read by `list_class_files` (three paths,
   labels 0/1/2), `LensClassificationDataset` returns `{"image", "label"}`, and
   `load_axion_image_and_log_mass` returns a finite log mass.
7. **Verification.** `verify_dataset` reports the expected counts and flags a truncated
   folder.
8. **CLI.** `generate-data --set image_size=64 workers=1 ...` writes the expected tree.

Generation is not run in CI; the `sim` extra is not installed there.

---

## 12. Packaging, CI and docs

- `pyproject.toml`: `sim = ["lenstronomy>=1.14,<2"]`, installed with
  `pip install -e ".[sim]"`. Everything else (`astropy`, `matplotlib`, `numba`, `h5py`,
  `scikit-image`, `jampy`, `mgefit`, `schwimmbad`, `multiprocess`, `corner`) arrives
  transitively.
- CI is unchanged: no `sim` extra, `test_sim.py` skips.
- `.gitignore`: `datasets_sim/`.
- `README.md`, Data section: how to generate, the resulting layout, the expected
  runtime, and an explicit statement that this dataset is for bring-up only and its
  metrics are not comparable to the paper's.

---

## 13. Risks

| Risk | Check | Mitigation |
|---|---|---|
| `lenstronomy` install on Python 3.12 / numpy 2.5 | **implementation step 1** | It is pure Python, but 1.14.2 pins `scipy<1.18` and pulls `numba`. The venv has scipy 1.18.1, so pip will downgrade it. No repo code imports scipy directly; after installing, run the full suite. If it breaks, restore the environment and stop — the design does not proceed on a damaged core env. |
| `noise_for_model` cannot be seeded | resolved | `seed=` is present in 1.14.2 (`SimulationAPI/observation_api.py:315`) |
| Wall-clock | measure 100 images, extrapolate | `workers` and the counts are config; expected 2–5 min at 8 workers |
| Classes too easy or too hard | §10 probe | physics knobs |
| Axion regression degenerate | vortex length spans 20× across the range | config validation rejects out-of-field masses |
| Pretrain does not converge at 64 px in 20 epochs | smoke run | `epochs` and `encoder` are config |
| Generated data mistaken for the real release | — | separate `datasets_sim/` root, docs in §12, `synthetic` in the campaign name |

---

## 14. Resolved decisions

| Question | Decision |
|---|---|
| Role of the data | Bring-up, CI and agent-loop demo only; explicitly not paper-grade. |
| Scope | Data generator + a tiny pretrain config + a synthetic campaign config. The shipped campaign default is not rewired. |
| Physics | `lenstronomy`, porting DeepLense's Model II recipe. No `pyHalo`. |
| Scale | Model II 1,000/class train + 200/class test; Model I 667/class. 5,601 images total. |
| Where the code lives | `lens_lejepa/sim/` package + a `generate-data` CLI subcommand (approach A). |
| Model I resolution | 64 px (same as downstream), not DeepLense's 150 px. |
| Axion masses | `1e-24 … 2e-23` eV, config-validated against the field. |
| Encoder pool in the synthetic campaign | One entry; encoder switching is left to the shipped campaign and the tests. |
