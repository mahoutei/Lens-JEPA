# DLens Agent Loop for Lens-JEPA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An agent drives the Lens-JEPA downstream pipeline: a planner proposes the next adapter/hyper-parameter configuration from observed validation metrics, our engine trains it, and the campaign ends with a report on the configuration that validation selected.

**Architecture:** A new `lens_lejepa/agents/` package with three modules — typed specs, two backends over our existing engine, and the loop (which also writes the campaign artifacts). DLens supplies the planner, the report agent, the analysis tool and the experiment schemas; we supply the loop, because DLens' own `ExperimentLoop` silently discards every parameter this campaign searches over. The loop runs offline by default (DLens' scripted planner), so the whole thing is a deterministic CI test with no API key.

**Tech Stack:** Python 3.12, PyTorch 2.x, pydantic (for our specs), pydantic-ai 2.x + DLens (planner/report only), pytest

**Spec:** `docs/superpowers/specs/2026-10-21-dlens-agent-loop-design.md`

**Prerequisite:** `docs/superpowers/plans/2026-10-21-lens-jepa-fix-pass.md` must be completed first. It removes the defects that can move published numbers and the ones that sit on this loop's critical path.

## Global Constraints

- Interpreter: `.venv/bin/python` (Python 3.12), created by Task 0 of the fix-pass plan. Run everything as `.venv/bin/python -m pytest`.
- `lens_lejepa/agents/` is the **only** part of the package that may import `dlens`. `lens_lejepa.config`, `.data`, `.models`, `.ssl`, `.tasks`, `.engine`, `.metrics`, `.utils` and `.cli` must keep working with `dlens` absent. `cli.py` imports `lens_lejepa.agents` lazily, inside the command branch, exactly as it already imports `.engine`.
- The core package keeps `requires-python = ">=3.10"`. The `agent` extra requires Python ≥ 3.12 and is documented as such.
- DLens is pinned to commit `d0d6e4937932e29a1df208178e3d386cdc84ebb2` (DLens main @ 2026-10-18) and installed with `--no-deps`; see Task 1.
- **The loop never reads the test dataset.** Only the finally selected configuration is scored on it, once, after the loop has finished. A test enforces this.
- An unknown or frozen planner key, and a value outside its domain, are **never silently dropped**: the validation raises, the loop records the reason, and the planner sees it on the next turn.
- Every behaviour ships with a test that fails before it.
- Commit after every task.

## Review Focus

The five input classes most likely to bite a person using this software, and the task that owns the test for each:

1. **A planner that emits a key we do not have** (`learning_rate` instead of `lr`) — must be rejected loudly, and the campaign must continue rather than die or silently ignore it (Task 2 and Task 4).
2. **A planner that tries to change the label budget or the dataset** — the frozen fields are the budget-fairness guarantee and must be rejected (Task 2).
3. **A run that collapsed to chance level** — must be flagged, excluded from "best", and shown to the planner rather than reported as the outcome (Task 4).
4. **A campaign re-run on the same output directory** — the per-iteration run directories must not collide, and the artifacts must be rewritten rather than appended to (Task 4 and Task 5).
5. **A campaign run without `dlens` installed, or without an API key** — the core package must still import, and the loop must still run offline (Task 1 and Task 7).

---

### Task 1: Dependency plumbing and the public API the agent needs

**Files:**
- Create: `scripts/install_agent_deps.sh`
- Modify: `pyproject.toml:20-21` (optional-dependencies)
- Modify: `lens_lejepa/config.py:126-138` (`load_config` → `load_yaml_with_overrides`)
- Modify: `lens_lejepa/engine/finetune.py:61-68` (`_build_trained_model` → `build_trained_model`), `:75` (its call site)
- Modify: `lens_lejepa/engine/__init__.py:3-6`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `lens_lejepa.config.load_yaml_with_overrides(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]`
  - `lens_lejepa.engine.build_trained_model(checkpoint: dict, device: torch.device) -> tuple[Task, nn.Module, FinetuneConfig]`
  - `scripts/install_agent_deps.sh` — installs DLens and the three direct dependencies.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_load_yaml_with_overrides_returns_a_plain_dict(tmp_path):
    from lens_lejepa.config import load_yaml_with_overrides

    path = tmp_path / "campaign.yaml"
    path.write_text("name: demo\nnested:\n  value: 1\n")
    data = load_yaml_with_overrides(path, ["nested.value=2", "extra=3"])
    assert data == {"name": "demo", "nested": {"value": 2}, "extra": 3}


def test_build_trained_model_is_public():
    from lens_lejepa.engine import build_trained_model

    assert callable(build_trained_model)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k "yaml_with_overrides or trained_model_is_public" -v`
Expected: FAIL — `ImportError: cannot import name 'load_yaml_with_overrides'`, then `ImportError: cannot import name 'build_trained_model'`.

- [ ] **Step 3: Add `load_yaml_with_overrides` and refactor `load_config`**

In `lens_lejepa/config.py`, replace `load_config` with:

```python
def load_yaml_with_overrides(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read a YAML file into a dict and apply ``key=value`` overrides to it.

    Values are parsed as YAML, so ``shots=100``, ``shots=null`` and
    ``rank_schedule=[8,16]`` all work. Overrides may address nested keys with
    a dot, e.g. ``objective.rings=4``.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        keys, value = _parse_override(item)
        _set_nested(data, keys, value)
    return data


def load_config(cls: type[C], path: str | Path | None = None, overrides: list[str] | None = None) -> C:
    """Build ``cls`` from an optional YAML file plus ``key=value`` overrides."""
    data = load_yaml_with_overrides(path, overrides)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)
```

- [ ] **Step 4: Promote `build_trained_model`**

In `lens_lejepa/engine/finetune.py`, rename the function at line 61 and update its one call site at line 75:

```python
def build_trained_model(checkpoint: dict, device: torch.device) -> tuple[Task, nn.Module, FinetuneConfig]:
    """Rebuild the task and model of a saved downstream checkpoint."""
    config = FinetuneConfig(**checkpoint["config"])
    config.pretrained = None  # the adapted weights are already inside the checkpoint
    task = build_task(config)
    task.load_state_dict(checkpoint.get("task_state", {}))
    model = task.build_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    return task, model.to(device), config


# The previous private name, kept so no existing caller breaks.
_build_trained_model = build_trained_model
```

and in `evaluate` (line 75) change `task, model, config = _build_trained_model(checkpoint, device)` to `task, model, config = build_trained_model(checkpoint, device)`.

In `lens_lejepa/engine/__init__.py`, replace the file with:

```python
"""Training loops: self-supervised pretraining and downstream fine-tuning."""

from .finetune import build_trained_model, evaluate, finetune, run_inference
from .pretrain import pretrain

__all__ = ["build_trained_model", "evaluate", "finetune", "pretrain", "run_inference"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 6: Add the `agent` extra and the install script**

In `pyproject.toml`, replace the optional-dependencies block with:

```toml
[project.optional-dependencies]
dev = ["pytest>=7"]
# The DLens agent framework itself is NOT listed here: it is not published on
# PyPI and its own dependency set (llama-cpp-python, numba, mcp, pandas) is not
# needed for the planner/report path. Run scripts/install_agent_deps.sh, which
# installs it from a pinned commit with --no-deps.
agent = [
    "pydantic-ai>=2.0",
    "loguru>=0.7",
    "python-dotenv>=1.0",
]
```

Create `scripts/install_agent_deps.sh`:

```bash
#!/usr/bin/env bash
# Install the DLens agent dependencies.
#
# DLens declares llama-cpp-python, numba, outlines, mcp, pandas and xmltodict,
# none of which the planner/report path uses: dlens/agents/_models.py imports
# llama_cpp lazily. llama-cpp-python additionally needs cmake and a C++
# compiler at install time. So DLens is installed from a pinned commit with
# --no-deps, and only the three direct dependencies are added.
#
#   bash scripts/install_agent_deps.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DLENS_SHA="${DLENS_SHA:-d0d6e4937932e29a1df208178e3d386cdc84ebb2}"
PY="${PYTHON:-.venv/bin/python}"
UV="${UV:-uv}"

"$UV" pip install --python "$PY" --no-deps \
    "git+https://github.com/ML4SCI/DeepLense-AI-Scientist.git@${DLENS_SHA}"
"$UV" pip install --python "$PY" "pydantic-ai>=2.0" "loguru>=0.7" "python-dotenv>=1.0"

"$PY" -c "import dlens; print('dlens ok')"
"$PY" -c "from dlens.agents._experiment_planner import ExperimentPlanner; print('planner ok')"
```

- [ ] **Step 7: Run the install script and check the import**

Run: `bash scripts/install_agent_deps.sh`
Expected: `dlens ok` and `planner ok`.

If the install fails because `pydantic-ai` pulls a different `pydantic` major version than DLens expects, record the resolved versions in the commit message and continue — do not downgrade DLens.

- [ ] **Step 8: Confirm the core package still imports**

Run: `.venv/bin/python -c "import lens_lejepa.cli, lens_lejepa.engine, lens_lejepa.tasks; print('core ok')"`
Expected: `core ok`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml scripts/install_agent_deps.sh lens_lejepa/config.py lens_lejepa/engine/__init__.py lens_lejepa/engine/finetune.py tests/test_pipeline.py
git commit -m "feat(agent): dependency plumbing and the public API the agent needs

Add the 'agent' extra, a script that installs DLens from a pinned commit
with --no-deps (its llama-cpp-python dependency needs a C++ compiler and
is imported lazily by DLens anyway), a load_yaml_with_overrides helper,
and promote _build_trained_model to build_trained_model."
```

---

### Task 2: The campaign specs and the strict update validation

**Note:** the default encoder pool is the spec's resolved `{lens_lejepa, lejepa}`
(`docs/superpowers/specs/2026-10-21-dlens-agent-loop-design.md`, "May the planner switch
encoders?"). The first draft of this task's snippet said `vit_small` only, which contradicted
`DownstreamSpec.encoder = "lens_lejepa"` and left the default campaign's own initial spec naming
an encoder outside its pool. The snippet below is corrected to the spec's pool.

**Files:**
- Create: `lens_lejepa/agents/__init__.py`
- Create: `lens_lejepa/agents/_spec.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `lens_lejepa.models.adapters.canonical_adapter_name` (raises `ValueError` for an unknown adapter).
- Produces:
  - `EncoderSpec(name: str, backbone: str, pretrained: str | None = None, method: str = "")`
  - `DownstreamSpec(encoder: str = "lens_lejepa", adaptation: str = "rslora", rank: int = 32, alpha: float = 64.0, lr: float = 3e-4, weight_decay: float = 0.01, dropout: float = 0.0, epochs: int = 50, batch_size: int = 128, label_smoothing: float = 0.1)`
  - `CampaignSpec` — every field the planner may not change, plus `encoders: list[EncoderSpec]` and `initial: DownstreamSpec`.
  - `CampaignSpec.encoder(name: str) -> EncoderSpec` — raises `KeyError` for an unknown name.
  - `CampaignSpec.run_name_for(iteration: int, spec: DownstreamSpec) -> str`
  - `CampaignSpec.output_path -> Path`
  - `DOMAINS: dict[str, tuple[float, float]]` — the allowed range of every numeric field.
  - `validate_delta(spec: DownstreamSpec, updates: dict[str, Any], campaign: CampaignSpec | None = None) -> DownstreamSpec` — raises `ValueError` for an unknown key, a frozen key, or a value outside its domain.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agents.py`:

```python
"""The agent campaign: specs, validation, loop. Offline, deterministic, CPU."""

from pathlib import Path

import pytest

from lens_lejepa.agents._spec import CampaignSpec, DownstreamSpec, validate_delta


def test_the_planner_may_change_the_downstream_fields():
    spec = DownstreamSpec()
    updated = validate_delta(spec, {"adaptation": "dora", "rank": 64})
    assert updated.adaptation == "dora"
    assert updated.rank == 64
    assert updated.lr == spec.lr  # untouched fields keep their value


def test_an_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown parameter"):
        validate_delta(DownstreamSpec(), {"learning_rate": 1e-3})


def test_a_frozen_key_is_rejected():
    campaign = CampaignSpec()
    with pytest.raises(ValueError, match="fixed for the whole campaign"):
        validate_delta(DownstreamSpec(), {"shots": 5}, campaign)
    with pytest.raises(ValueError, match="fixed for the whole campaign"):
        validate_delta(DownstreamSpec(), {"seed": 7}, campaign)


def test_an_out_of_domain_value_is_rejected():
    with pytest.raises(ValueError, match="outside the allowed range"):
        validate_delta(DownstreamSpec(), {"rank": 0})
    with pytest.raises(ValueError, match="outside the allowed range"):
        validate_delta(DownstreamSpec(), {"lr": 10.0})
    with pytest.raises(ValueError, match="Unknown adapter"):
        validate_delta(DownstreamSpec(), {"adaptation": "typo"})


def test_an_encoder_outside_the_pool_is_rejected():
    campaign = CampaignSpec(encoders=[{"name": "vit_small", "backbone": "vit_small"}])
    with pytest.raises(ValueError, match="not in the pool"):
        validate_delta(DownstreamSpec(), {"encoder": "nope"}, campaign)


def test_the_campaign_exposes_its_pool_and_run_names():
    campaign = CampaignSpec(
        name="demo",
        output_dir="runs/agent",
        encoders=[
            {"name": "lens_lejepa", "backbone": "vit_small"},
            {"name": "lejepa", "backbone": "vit_small"},
        ],
    )
    assert campaign.encoder("lejepa").backbone == "vit_small"
    assert campaign.run_name_for(0, DownstreamSpec(encoder="lejepa", adaptation="lora")) == "iter00_lejepa_lora"
    assert campaign.output_path.as_posix() == "runs/agent/demo"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens_lejepa.agents'`.

- [ ] **Step 3: Write the specs**

Create `lens_lejepa/agents/_spec.py`:

```python
"""Typed specifications for an agent campaign.

``DownstreamSpec`` holds exactly the fields the planner may change.
``CampaignSpec`` holds everything else — the task, the data, the label budget,
the seed — plus the pool of pretrained encoders the planner may choose from.
The separation is the budget-fairness guarantee: the planner can change how a
model is trained, never what it is trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..models.adapters import canonical_adapter_name


class EncoderSpec(BaseModel):
    """One pretrained encoder in the pool."""

    name: str = Field(description="Identifier the planner uses to select it.")
    backbone: str = Field(description="Encoder name, e.g. vit_small.")
    pretrained: str | None = Field(default=None, description="Path to last_pretrain.pt.")
    method: str = Field(default="", description="Pretraining method that produced it.")


class DownstreamSpec(BaseModel):
    """The fields the planner is allowed to change."""

    encoder: str = "lens_lejepa"
    adaptation: str = "rslora"
    rank: int = 32
    alpha: float = 64.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.0
    epochs: int = 50
    batch_size: int = 128
    label_smoothing: float = 0.1


class CampaignSpec(BaseModel):
    """Everything the planner may not change, plus the pool it may choose from."""

    name: str = "campaign"
    hypothesis: str = ""
    task: str = "classification"
    data_root: str = "../datasets"
    train_dataset: str = "Model_II"
    test_dataset: str | None = "Model_II_test"
    shots: int | None = None
    train_fraction: float = 0.8
    val_fraction: float = 0.25
    cache: bool = False
    image_size: int = 160
    patch_size: int = 16
    seed: int = 42
    val_interval: int = 10
    device: str = "cuda"
    bf16: bool = True
    workers: int = 8
    grad_accum: int = 1
    loraplus_ratio: float = 2.0
    train_jitter: int = 0
    rank_schedule: list[int] = [8, 16, 32, 64]
    output_dir: str = "runs/agent"
    max_iterations: int = 8
    patience: int = 3
    min_val_delta: float = 0.005
    min_val_above_chance: float = 0.05
    encoders: list[EncoderSpec] = [EncoderSpec(name="lens_lejepa", backbone="vit_small"), EncoderSpec(name="lejepa", backbone="vit_small")]
    initial: DownstreamSpec = DownstreamSpec()

    def encoder(self, name: str) -> EncoderSpec:
        """The pooled encoder called ``name``."""
        for item in self.encoders:
            if item.name == name:
                return item
        raise KeyError(f"No encoder named {name!r} in the pool {[e.name for e in self.encoders]}.")

    def run_name_for(self, iteration: int, spec: DownstreamSpec) -> str:
        """A unique run-directory name for one iteration."""
        return f"iter{iteration:02d}_{spec.encoder}_{spec.adaptation}"

    @property
    def output_path(self) -> Path:
        """Where the campaign artifacts are written."""
        return Path(self.output_dir) / self.name


#: Allowed range for every numeric field the planner may set.
DOMAINS: dict[str, tuple[float, float]] = {
    "rank": (1, 256),
    "alpha": (0.0, 1024.0),
    "lr": (1e-6, 1e-1),
    "weight_decay": (0.0, 1.0),
    "dropout": (0.0, 0.999),
    "epochs": (1, 200),
    "batch_size": (1, 4096),
    "label_smoothing": (0.0, 1.0),
}

#: Fields that must be whole numbers.
INTEGER_FIELDS = ("rank", "epochs", "batch_size")


def validate_delta(
    spec: DownstreamSpec,
    updates: dict[str, Any],
    campaign: CampaignSpec | None = None,
) -> DownstreamSpec:
    """Apply the planner's updates to ``spec``.

    Raises ``ValueError`` for a key we do not have, a key that is fixed for the
    whole campaign, or a value outside its domain. The loop catches the error,
    records the reason and shows it to the planner on the next turn, so a
    change is never silently dropped.
    """
    allowed = set(DownstreamSpec.model_fields)
    frozen = set(CampaignSpec.model_fields) - allowed
    for key, value in (updates or {}).items():
        if key in frozen:
            raise ValueError(f"{key!r} is fixed for the whole campaign and cannot be changed.")
        if key not in allowed:
            raise ValueError(f"Unknown parameter {key!r}; allowed: {sorted(allowed)}.")
        if key == "encoder":
            if campaign is not None:
                try:
                    campaign.encoder(str(value))
                except KeyError as error:
                    raise ValueError(f"encoder {value!r} is not in the pool.") from error
            continue
        if key == "adaptation":
            canonical_adapter_name(str(value))
            continue
        low, high = DOMAINS[key]
        number = float(value)
        if not low <= number <= high:
            raise ValueError(f"{key}={value!r} is outside the allowed range [{low}, {high}].")
        if key in INTEGER_FIELDS and int(number) != number:
            raise ValueError(f"{key}={value!r} must be a whole number.")
    data = spec.model_dump()
    data.update(updates)
    return DownstreamSpec(**data)
```

- [ ] **Step 4: Create the package init**

Create `lens_lejepa/agents/__init__.py`:

```python
"""Agent-driven downstream campaigns, built on the DLens framework.

Importing this package requires ``dlens``; the rest of ``lens_lejepa`` does
not. See ``scripts/install_agent_deps.sh``.
"""

from ._spec import CampaignSpec, DownstreamSpec, EncoderSpec, validate_delta

__all__ = ["CampaignSpec", "DownstreamSpec", "EncoderSpec", "validate_delta"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: PASS — all six tests.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lens_lejepa/agents/ tests/test_agents.py
git commit -m "feat(agents): campaign specs and strict planner-update validation

DownstreamSpec holds exactly what the planner may change; CampaignSpec
holds the rest. An unknown key, a frozen key or an out-of-domain value
raises instead of being silently dropped."
```

---

### Task 3: The train and infer backends over our engine

**Files:**
- Create: `lens_lejepa/agents/_backends.py`
- Modify: `lens_lejepa/agents/__init__.py`
- Modify: `.github/workflows/tests.yml` (see the CI note below)
- Test: `tests/test_agents.py`

**CI note (added after Task 2).** This is the first task whose production code imports `dlens`
(`_backends.py` imports `dlens.schemas._downstream`), so from here on the test suite cannot be
collected in CI without DLens installed. Add one step to `.github/workflows/tests.yml`, after the
install step:

```yaml
      - name: Install the agent dependencies
        run: bash scripts/install_agent_deps.sh
```

`scripts/install_agent_deps.sh` pins DLens to a commit and installs it with `--no-deps`, so this
adds the DLens checkout and three small pure-Python dependencies and nothing else. Do not change
anything else in the workflow.

**Interfaces:**
- Consumes: `CampaignSpec`, `DownstreamSpec` (Task 2); `lens_lejepa.engine.finetune`, `run_inference`, `build_trained_model`; `lens_lejepa.tasks.build_task`; `lens_lejepa.data.CLASSES`, `make_loader`.
- Produces:
  - `build_finetune_config(campaign: CampaignSpec, spec: DownstreamSpec, run_name: str) -> FinetuneConfig`
  - `LensJepaTrainBackend(campaign).train(spec: DownstreamSpec, run_name: str) -> dlens.schemas._downstream.TrainResult`
  - `LensJepaInferBackend(campaign).infer(weights_path: str) -> dlens.schemas._downstream.InferResult`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agents.py`:

```python
def test_build_finetune_config_maps_the_campaign_onto_a_run(fake_root, tmp_path):
    from lens_lejepa.agents._backends import build_finetune_config

    campaign = CampaignSpec(
        task="classification",
        data_root=str(fake_root.parent),
        train_dataset=fake_root.name,
        test_dataset=fake_root.name,
        shots=10,
        image_size=32,
        seed=7,
        device="cpu",
        encoders=[{"name": "lens_lejepa", "backbone": "vit_tiny", "pretrained": None}],
    )
    spec = DownstreamSpec(encoder="lens_lejepa", adaptation="lora", rank=8, lr=1e-3, epochs=3)
    config = build_finetune_config(campaign, spec, "iter00")
    assert config.task == "classification"
    assert config.shots == 10
    assert config.seed == 7
    assert config.device == "cpu"
    assert config.backbone == "vit_tiny"
    assert config.adaptation == "lora"
    assert config.rank == 8
    assert config.lr == 1e-3
    assert config.epochs == 3
    assert config.run_name == "iter00"
    # The test set must never be reachable from inside the loop.
    assert config.test_dataset is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agents.py -k finetune_config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens_lejepa.agents._backends'`.

- [ ] **Step 3: Write the backends**

Create `lens_lejepa/agents/_backends.py`:

```python
"""Train and infer backends that run the real Lens-JEPA pipeline.

The train backend calls ``engine.finetune`` — our adapters, our bf16 training,
our best-validation checkpoint selection — and the infer backend scores the
checkpoint it produced on the validation split. Neither ever touches the test
dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dlens.schemas._downstream import InferResult, TrainResult

from ..config import FinetuneConfig
from ..data import CLASSES, make_loader
from ..engine import build_trained_model, finetune, run_inference
from ..tasks import build_task
from ..utils import resolve_device
from ._spec import CampaignSpec, DownstreamSpec


def build_finetune_config(campaign: CampaignSpec, spec: DownstreamSpec, run_name: str) -> FinetuneConfig:
    """Map a campaign and a downstream specification onto one training run."""
    encoder = campaign.encoder(spec.encoder)
    return FinetuneConfig(
        task=campaign.task,
        data_root=campaign.data_root,
        train_dataset=campaign.train_dataset,
        test_dataset=None,  # the test set is scored once, after the loop
        shots=campaign.shots,
        train_fraction=campaign.train_fraction,
        val_fraction=campaign.val_fraction,
        cache=campaign.cache,
        image_size=campaign.image_size,
        patch_size=campaign.patch_size,
        backbone=encoder.backbone,
        pretrained=encoder.pretrained,
        adaptation=spec.adaptation,
        rank=spec.rank,
        alpha=spec.alpha,
        dropout=spec.dropout,
        rank_schedule=campaign.rank_schedule,
        loraplus_ratio=campaign.loraplus_ratio,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        grad_accum=campaign.grad_accum,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        label_smoothing=spec.label_smoothing,
        train_jitter=campaign.train_jitter,
        val_interval=campaign.val_interval,
        bf16=campaign.bf16,
        seed=campaign.seed,
        workers=campaign.workers,
        device=campaign.device,
        output_dir=str(campaign.output_dir),
        run_name=run_name,
    )


class LensJepaTrainBackend:
    """Runs one downstream training run of the Lens-JEPA pipeline."""

    name = "lens_lejepa"

    def __init__(self, campaign: CampaignSpec) -> None:
        self.campaign = campaign

    def train(self, spec: DownstreamSpec, run_name: str) -> TrainResult:
        config = build_finetune_config(self.campaign, spec, run_name)
        run_dir = finetune(config)
        summary = json.loads((run_dir / "summary.json").read_text())
        monitor = build_task(config).monitor
        return TrainResult(
            run_id=run_dir.name,
            weights_path=str(run_dir / "best.pt"),
            backend=self.name,
            num_classes=len(CLASSES),
            metrics={
                f"val_{monitor}": float(summary[f"val_{monitor}"]),
                "best_epoch": float(summary["best_epoch"]),
            },
            epochs_run=int(summary["best_epoch"]),
        )


class LensJepaInferBackend:
    """Scores a saved downstream checkpoint on the validation split."""

    name = "lens_lejepa"

    def __init__(self, campaign: CampaignSpec) -> None:
        self.campaign = campaign

    def infer(self, weights_path: str) -> InferResult:
        device = resolve_device(self.campaign.device)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        task, model, config = build_trained_model(checkpoint, device)
        _, validation = task.build_datasets()
        loader = make_loader(validation, config.batch_size, shuffle=False, workers=self.campaign.workers)
        outputs = run_inference(task, model, loader, device)["outputs"]
        probabilities = outputs["probabilities"]
        predictions = probabilities.argmax(axis=1)
        labels = outputs["labels"]
        return InferResult(
            run_id=Path(weights_path).parent.name,
            num_samples=int(len(labels)),
            num_classes=int(probabilities.shape[1]),
            predictions=predictions.astype(int).tolist(),
            probabilities=probabilities.tolist(),
            true_labels=labels.astype(int).tolist(),
            accuracy=float((predictions == labels).mean()),
            backend=self.name,
        )
```

- [ ] **Step 4: Export the backends**

In `lens_lejepa/agents/__init__.py`, replace the file with:

```python
"""Agent-driven downstream campaigns, built on the DLens framework.

Importing this package requires ``dlens``; the rest of ``lens_lejepa`` does
not. See ``scripts/install_agent_deps.sh``.
"""

from ._backends import LensJepaInferBackend, LensJepaTrainBackend, build_finetune_config
from ._spec import CampaignSpec, DownstreamSpec, EncoderSpec, validate_delta

__all__ = [
    "CampaignSpec",
    "DownstreamSpec",
    "EncoderSpec",
    "LensJepaInferBackend",
    "LensJepaTrainBackend",
    "build_finetune_config",
    "validate_delta",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lens_lejepa/agents/ tests/test_agents.py
git commit -m "feat(agents): train and infer backends over the Lens-JEPA engine

The train backend calls engine.finetune; the infer backend scores the
checkpoint it produced on the deterministic validation split. The
FinetuneConfig they build always has test_dataset=None."
```

---

### Task 4: The loop, its guards, and the selection of the best run

**Files:**
- Create: `lens_lejepa/agents/_loop.py`
- Modify: `lens_lejepa/agents/__init__.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `CampaignSpec`, `DownstreamSpec`, `DOMAINS`, `validate_delta` (Task 2); `LensJepaTrainBackend`, `LensJepaInferBackend` (Task 3); `dlens.schemas._experiment.{ExperimentRun, ExperimentState, PlannerAction, PlannerDecision}`; `dlens.tools._analysis.compute_analysis`.
- Produces:
  - `LensJepaRun(ExperimentRun)` — adds `spec: DownstreamSpec`, `collapsed: bool = False`, `rejection: str | None = None`.
  - `LensJepaState(ExperimentState)` — `runs: list[LensJepaRun]`.
  - `select_best_run(runs: Sequence[LensJepaRun]) -> LensJepaRun | None` — the run with the best validation accuracy, ignoring collapsed runs.
  - `CampaignResult` — dataclass with `state`, `best_run`, `best_checkpoint`, `output_dir`.
  - `build_planner_prompt(campaign: CampaignSpec) -> str`
  - `build_planner(campaign: CampaignSpec, live: bool = False) -> PlannerStrategy`
  - `run_campaign(campaign: CampaignSpec, planner, *, train_backend, infer_backend) -> CampaignResult` — async; never touches the test set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agents.py`:

```python
class ScriptedPlanner:
    """Emits a fixed decision per iteration; deterministic and offline."""

    name = "scripted"

    def __init__(self, decisions):
        self.decisions = decisions

    async def decide(self, state):
        from dlens.schemas._experiment import PlannerDecision

        index = state.current_iteration
        payload = self.decisions[index] if index < len(self.decisions) else {
            "action": "report", "rationale": "scripted budget exhausted", "updated_params": {}
        }
        return PlannerDecision(**payload)


def _fake_run(accuracy: float, collapsed: bool = False):
    from dlens.schemas._downstream import AnalysisResult

    from lens_lejepa.agents._loop import LensJepaRun

    return LensJepaRun(
        iteration=0,
        spec=DownstreamSpec(),
        analysis_result=AnalysisResult(
            accuracy=accuracy,
            confusion_matrix=[[1, 0], [0, 1]],
            per_class={"a": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1.0}},
            summary="synthetic",
        ),
        collapsed=collapsed,
    )


def _tiny_campaign(fake_root, tmp_path, **kwargs):
    defaults = dict(
        name="demo",
        data_root=str(fake_root.parent),
        train_dataset=fake_root.name,
        test_dataset=fake_root.name,
        image_size=32,
        device="cpu",
        bf16=False,
        workers=0,
        output_dir=str(tmp_path / "runs"),
        encoders=[{"name": "vit_small", "backbone": "vit_tiny"}],
        initial=DownstreamSpec(encoder="vit_small", adaptation="linear_probe", epochs=1, batch_size=8),
    )
    defaults.update(kwargs)
    return CampaignSpec(**defaults)


def test_the_best_run_is_the_best_validation_accuracy():
    from lens_lejepa.agents._loop import select_best_run

    runs = [_fake_run(0.5), _fake_run(0.9), _fake_run(0.7)]
    assert select_best_run(runs) is runs[1]


def test_a_collapsed_run_is_never_the_best():
    from lens_lejepa.agents._loop import select_best_run

    runs = [_fake_run(0.5), _fake_run(0.99, collapsed=True)]
    assert select_best_run(runs) is runs[0]


def test_the_loop_runs_offline_and_records_every_iteration(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "try a different adapter", "updated_params": {"adaptation": "lora"}},
        {"action": "report", "rationale": "done", "updated_params": {}},
    ])
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert len(result.state.runs) == 2
    assert result.state.runs[0].spec.adaptation == "linear_probe"
    assert result.state.runs[1].spec.adaptation == "lora"
    assert result.best_run is not None
    assert result.best_checkpoint.endswith("best.pt")


def test_a_rejected_change_is_recorded_and_the_campaign_continues(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2)
    planner = ScriptedPlanner([
        {"action": "train", "rationale": "typo", "updated_params": {"learning_rate": 1e-3}},
        {"action": "report", "rationale": "done", "updated_params": {}},
    ])
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert any("learning_rate" in (run.rejection or "") for run in result.state.runs)
    # The rejected change must not have been applied.
    assert result.state.runs[1].spec.lr == campaign.initial.lr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens_lejepa.agents._loop'`.

- [ ] **Step 3: Write the loop**

Create `lens_lejepa/agents/_loop.py`:

```python
"""The campaign loop: train a configuration, score it, propose the next one.

DLens' own ``ExperimentLoop`` is not reused: it filters the planner's updates
against its own hyper-parameter whitelist and silently discards anything else,
which would drop every knob this campaign searches over.

The loop never reads the test dataset. Only the configuration that validation
selected is scored on it, once, after the loop has finished.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from dlens.agents._base import BaseAgentConfig
from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
from dlens.agents._scripted_planner import make_scripted_planner_model
from dlens.schemas._experiment import ExperimentRun, ExperimentState, PlannerAction, PlannerDecision
from dlens.tools._analysis import compute_analysis

from ..data import CLASSES
from ..utils import logger
from ._backends import LensJepaInferBackend, LensJepaTrainBackend
from ._spec import DOMAINS, CampaignSpec, DownstreamSpec, validate_delta


class LensJepaRun(ExperimentRun):
    """One iteration of the campaign, with the configuration that produced it."""

    spec: DownstreamSpec
    collapsed: bool = False
    rejection: str | None = None


class LensJepaState(ExperimentState):
    """The campaign history, with our per-iteration configuration attached."""

    runs: list[LensJepaRun] = []


class PlannerStrategy(Protocol):
    """Proposes the next configuration.

    ``dlens.agents._experiment_planner.ReActPlannerStrategy`` satisfies it.
    """

    name: str

    async def decide(self, state: LensJepaState) -> PlannerDecision: ...


@dataclass
class CampaignResult:
    """What a campaign produced."""

    state: LensJepaState
    best_run: LensJepaRun | None
    best_checkpoint: str | None
    output_dir: Path


def select_best_run(runs: Sequence[LensJepaRun]) -> LensJepaRun | None:
    """The run with the best validation accuracy; collapsed runs never win."""
    best: LensJepaRun | None = None
    for run in runs:
        if run.collapsed or run.analysis_result is None:
            continue
        if best is None or run.analysis_result.accuracy > best.analysis_result.accuracy:
            best = run
    return best


def build_planner_prompt(campaign: CampaignSpec) -> str:
    """The system prompt: our knob vocabulary, our frozen fields."""
    encoders = ", ".join(item.name for item in campaign.encoders)
    domains = "\n".join(f"  {key:>15s}  number in [{low}, {high}]" for key, (low, high) in DOMAINS.items())
    return f"""\
You are an experiment planner for a self-supervised representation study on
gravitational-lensing images. You are given the campaign history as JSON: for each
iteration, the configuration that was trained (under "spec"), the training metrics,
and the held-out validation analysis (accuracy, macro_auc, confusion_matrix,
per-class precision/recall/F1).

Reason ONLY from these observed validation metrics. Diagnose what limits validation
accuracy and decide exactly ONE concrete change for the next iteration.

You may change only these fields:
  encoder        one of: {encoders}
  adaptation     one of: lora, rslora, dora, loraplus, gated, loha, layerwise, full, linear_probe
{domains}

These are FIXED for the whole campaign and must never appear in updated_params:
  task, data_root, train_dataset, test_dataset, shots, train_fraction,
  val_fraction, cache, image_size, patch_size, seed, val_interval, device, bf16,
  workers, grad_accum, loraplus_ratio, train_jitter, rank_schedule, output_dir,
  max_iterations, patience, min_val_delta, min_val_above_chance, encoders,
  initial, name, hypothesis.

Populate updated_params with the changed field(s) only, and rationale with your
diagnosis citing the observed numbers. Do not repeat a change that already failed
to improve validation accuracy. The test set is never shown to you and must not be
reasoned about.
"""


def build_planner(campaign: CampaignSpec, live: bool = False) -> PlannerStrategy:
    """The offline scripted planner by default; the hosted LLM when ``live``."""
    from dlens.config import build_llm

    if live:
        model: Any = build_llm()
    else:
        model = make_scripted_planner_model(
            [{"action": "train", "rationale": "offline scripted planner", "updated_params": {}}]
        )
    config = BaseAgentConfig(
        name="LensJepaPlanner",
        description="Chooses the next downstream configuration for a Lens-JEPA campaign.",
        custom_system_prompt=build_planner_prompt(campaign),
        model=model,
    )
    return ReActPlannerStrategy(planner=ExperimentPlanner(config=config))


async def run_campaign(
    campaign: CampaignSpec,
    planner: PlannerStrategy,
    *,
    train_backend: LensJepaTrainBackend,
    infer_backend: LensJepaInferBackend,
) -> CampaignResult:
    """Run the campaign and return its history and the selected checkpoint."""
    state = LensJepaState(hypothesis=campaign.hypothesis, max_iterations=campaign.max_iterations)
    spec = campaign.initial
    best_value: float | None = None
    no_improve = 0
    chance = 1.0 / max(len(CLASSES), 2)

    for iteration in range(campaign.max_iterations):
        state.current_iteration = iteration
        logger.info(
            "iteration %d | encoder=%s adaptation=%s rank=%d lr=%g",
            iteration, spec.encoder, spec.adaptation, spec.rank, spec.lr,
        )
        train_result = train_backend.train(spec, campaign.run_name_for(iteration, spec))
        infer_result = infer_backend.infer(train_result.weights_path)
        analysis = compute_analysis(infer_result, list(CLASSES))
        run = LensJepaRun(
            iteration=iteration,
            spec=spec,
            train_result=train_result,
            infer_result=infer_result,
            analysis_result=analysis,
        )
        state.runs.append(run)

        value = float(analysis.accuracy)
        run.collapsed = value <= chance + campaign.min_val_above_chance
        if run.collapsed:
            run.rejection = f"validation accuracy {value:.4f} is at or below chance ({chance:.4f})"
            logger.warning("iteration %d collapsed: %s", iteration, run.rejection)
        improved = not run.collapsed and (best_value is None or value > best_value + campaign.min_val_delta)
        if improved:
            best_value, no_improve = value, 0
        else:
            no_improve += 1
        logger.info("iteration %d | validation accuracy %.4f", iteration, value)

        if no_improve >= campaign.patience:
            logger.info("stopping: validation accuracy did not improve for %d iterations", campaign.patience)
            break

        decision = await planner.decide(state)
        run.planner_decision = decision
        if decision.action in (PlannerAction.REPORT, PlannerAction.STOP):
            logger.info("stopping: the planner chose %s", decision.action.value)
            break
        try:
            spec = validate_delta(spec, decision.updated_params, campaign)
        except ValueError as error:
            run.rejection = str(error)
            logger.warning("iteration %d: rejected the planner's change: %s", iteration, error)

    best_run = select_best_run(state.runs)
    return CampaignResult(
        state=state,
        best_run=best_run,
        best_checkpoint=best_run.train_result.weights_path if best_run is not None else None,
        output_dir=campaign.output_path,
    )
```

- [ ] **Step 4: Export the loop**

In `lens_lejepa/agents/__init__.py`, add to the imports:

```python
from ._loop import (
    CampaignResult,
    LensJepaRun,
    LensJepaState,
    build_planner,
    build_planner_prompt,
    run_campaign,
    select_best_run,
)
```

and add those names to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: PASS — all tests in the file. The two loop tests train a `vit_tiny` with a linear probe on the synthetic fixture, on CPU, in a few seconds each.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lens_lejepa/agents/ tests/test_agents.py
git commit -m "feat(agents): the campaign loop, its guards and the best-run selection

The loop trains a configuration, scores it on the validation split, and
applies the planner's next change under strict validation: an unknown
key, a frozen key or an out-of-domain value is recorded as a rejection
and shown to the planner instead of being dropped. A collapsed run never
becomes the selected configuration, and the test set is never touched."
```

---

### Task 5: The campaign artifacts and the single test evaluation

**Files:**
- Modify: `lens_lejepa/agents/_loop.py` (add `finalize`, `write_report` and their helpers)
- Modify: `lens_lejepa/agents/__init__.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `CampaignResult`, `LensJepaState` (Task 4); `lens_lejepa.engine.evaluate`; `lens_lejepa.utils.write_json`.
- Produces:
  - `finalize(result: CampaignResult, campaign: CampaignSpec) -> dict[str, Any]` — evaluates the selected checkpoint on the test set **once**, then writes `campaign.json` and `campaign.md`. Returns the payload.
  - `write_report(state: LensJepaState, campaign: CampaignSpec, live: bool = False) -> dict[str, Any]` — async; returns the DLens `ReportDocument` as a dict.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agents.py`:

```python
def test_the_test_set_is_evaluated_once_and_only_after_the_loop(fake_root, tmp_path, monkeypatch):
    import asyncio

    import lens_lejepa.agents._loop as loop_module
    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import finalize, run_campaign

    calls = []
    monkeypatch.setattr(
        loop_module, "evaluate",
        lambda *args, **kwargs: calls.append(args) or {"accuracy": 0.5},
    )

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=1)
    result = asyncio.run(run_campaign(
        campaign, ScriptedPlanner([]),
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert calls == []  # the loop alone never evaluates the test set

    payload = finalize(result, campaign)
    assert len(calls) == 1
    assert payload["test"]["accuracy"] == 0.5
    assert (campaign.output_path / "campaign.json").exists()
    assert (campaign.output_path / "campaign.md").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agents.py -k evaluated_once -v`
Expected: FAIL — `ImportError: cannot import name 'finalize' from 'lens_lejepa.agents._loop'`.

- [ ] **Step 3: Write the artifact functions**

In `lens_lejepa/agents/_loop.py`, add `from ..engine import evaluate` and `from ..utils import write_json` to the imports, and append at the end of the file:

```python
def _iterations(state: LensJepaState) -> list[dict[str, Any]]:
    """One row per iteration: the configuration, the metrics, the outcome."""
    rows = []
    for run in state.runs:
        analysis = run.analysis_result
        rows.append({
            "iteration": run.iteration,
            "spec": run.spec.model_dump(mode="json"),
            "val_accuracy": float(analysis.accuracy) if analysis is not None else None,
            "macro_auc": (
                float(analysis.macro_auc)
                if analysis is not None and analysis.macro_auc is not None
                else None
            ),
            "collapsed": run.collapsed,
            "rejection": run.rejection,
            "run_dir": run.train_result.run_id if run.train_result is not None else None,
            "rationale": run.planner_decision.rationale if run.planner_decision is not None else None,
        })
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    """The campaign table a human reads."""
    campaign = payload["campaign"]
    lines = [
        f"# {campaign['name']}",
        "",
        campaign["hypothesis"] or "_No hypothesis recorded._",
        "",
        "| # | encoder | adaptation | rank | lr | val accuracy | macro AUROC | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["iterations"]:
        accuracy = "n/a" if row["val_accuracy"] is None else f"{row['val_accuracy']:.4f}"
        auc = "n/a" if row["macro_auc"] is None else f"{row['macro_auc']:.4f}"
        spec = row["spec"]
        note = row["rejection"] or ""
        lines.append(
            f"| {row['iteration']} | {spec['encoder']} | {spec['adaptation']} | {spec['rank']} | "
            f"{spec['lr']:g} | {accuracy} | {auc} | {note} |"
        )
    best = payload["best_iteration"]
    lines += [
        "",
        f"Selected iteration: **{best}**" if best is not None else "No iteration was selected.",
        "",
    ]
    if payload["test"]:
        lines += ["## Held-out test metrics", ""]
        for key, value in payload["test"].items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
    return "\n".join(lines) + "\n"


def finalize(result: CampaignResult, campaign: CampaignSpec) -> dict[str, Any]:
    """Score the selected checkpoint on the test set once, then write the artifacts."""
    output = campaign.output_path
    output.mkdir(parents=True, exist_ok=True)

    test_metrics: dict[str, Any] | None = None
    if result.best_checkpoint is not None and campaign.test_dataset:
        logger.info("evaluating the selected checkpoint on %s", campaign.test_dataset)
        batch_size = result.best_run.spec.batch_size if result.best_run is not None else campaign.initial.batch_size
        test_metrics = evaluate(
            result.best_checkpoint,
            f"{campaign.data_root}/{campaign.test_dataset}",
            campaign.device,
            batch_size=batch_size,
            workers=campaign.workers,
        )

    payload = {
        "campaign": campaign.model_dump(mode="json"),
        "iterations": _iterations(result.state),
        "best_iteration": result.best_run.iteration if result.best_run is not None else None,
        "best_checkpoint": result.best_checkpoint,
        "test": test_metrics,
    }
    write_json(output / "campaign.json", payload)
    (output / "campaign.md").write_text(_markdown(payload))
    return payload


async def write_report(state: LensJepaState, campaign: CampaignSpec, live: bool = False) -> dict[str, Any]:
    """The DLens report agent's summary of the campaign."""
    from dlens.agents._report import ReportAgent
    from dlens.agents._scripted_planner import make_scripted_report_model
    from dlens.config import build_llm

    agent = ReportAgent(model=build_llm() if live else make_scripted_report_model())
    result = await agent.write(state)
    return result.output.model_dump(mode="json")
```

- [ ] **Step 4: Export the artifact helpers**

In `lens_lejepa/agents/__init__.py`, add `finalize` and `write_report` to the `._loop` import list and to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lens_lejepa/agents/ tests/test_agents.py
git commit -m "feat(agents): campaign artifacts and the single test evaluation

finalize is the only place in the agent layer that scores the test
dataset, and it scores it once, for the configuration validation
selected. The history, the selected configuration and the held-out
metrics are written to campaign.json and campaign.md."
```

---

### Task 6: The CLI subcommand and a campaign config

**Files:**
- Modify: `lens_lejepa/cli.py:40` (a new subparser), `:46-67` (the dispatch)
- Modify: `lens_lejepa/agents/_loop.py` (add `run_from_config`)
- Modify: `lens_lejepa/agents/__init__.py`
- Create: `configs/agent/campaign.yaml`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: everything from Tasks 2-5.
- Produces: `python -m lens_lejepa agent-loop --config configs/agent/campaign.yaml [--set key=value]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agents.py`:

```python
def test_the_campaign_config_loads_and_validates():
    from lens_lejepa.config import load_yaml_with_overrides

    path = Path(__file__).resolve().parent.parent / "configs" / "agent" / "campaign.yaml"
    campaign = CampaignSpec(**load_yaml_with_overrides(path))
    assert campaign.task == "classification"
    assert campaign.encoders
    assert campaign.initial.adaptation in (
        "lora", "rslora", "dora", "loraplus", "gated", "loha", "layerwise", "full", "linear_probe",
    )


def test_the_cli_exposes_the_agent_loop_subcommand():
    from lens_lejepa.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["agent-loop", "--help"])
    assert exit_info.value.code == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agents.py -k "campaign_config or agent_loop_subcommand" -v`
Expected: FAIL — `FileNotFoundError: .../configs/agent/campaign.yaml`, and the CLI test fails because `agent-loop` is not a known subcommand (argparse exits with code 2, not 0).

- [ ] **Step 3: Write the campaign config**

Create `configs/agent/campaign.yaml`:

```yaml
# An agent campaign over the Lens-JEPA downstream pipeline.
#
#   python -m lens_lejepa agent-loop --config configs/agent/campaign.yaml
#
# The planner chooses the adapter and the hyper-parameters; "initial" is
# the starting configuration. The task, the data, the label budget and the
# seed are fixed for the whole campaign.
name: lens_lejepa_adapters
hypothesis: >-
  Which parameter-efficient adapter, at which rank and learning rate, gives
  the best validation accuracy on Model II with 100 labelled images per class?

task: classification
data_root: ../datasets
train_dataset: Model_II
test_dataset: Model_II_test
shots: 100
seed: 42
device: cuda
bf16: true
workers: 8

image_size: 160
patch_size: 16
val_interval: 10

max_iterations: 8
patience: 3
output_dir: runs/agent

# The pool the planner may choose an encoder from.
encoders:
  - name: lens_lejepa
    backbone: vit_small
    method: lens_lejepa
    pretrained: runs/pretrain/lens_lejepa/lens_lejepa_model_i_seed42/last_pretrain.pt
  - name: lejepa
    backbone: vit_small
    method: lejepa
    pretrained: runs/pretrain/lejepa/lejepa_model_i_seed42/last_pretrain.pt

# The first configuration of the campaign.
initial:
  encoder: lens_lejepa
  adaptation: rslora
  rank: 32
  alpha: 64.0
  lr: 3.0e-4
  weight_decay: 0.01
  dropout: 0.0
  epochs: 50
  batch_size: 128
  label_smoothing: 0.1
```

- [ ] **Step 4: Add `run_from_config`**

In `lens_lejepa/agents/_loop.py`, append at the end of the file:

```python
async def run_from_config(path: str | None = None, overrides: list[str] | None = None) -> Path:
    """Run a campaign described by a YAML file, and return the artifact directory."""
    import os

    from ..config import load_yaml_with_overrides

    campaign = CampaignSpec(**load_yaml_with_overrides(path, overrides))
    live = bool(os.getenv("DLENS_LLM") or os.getenv("OPENAI_API_KEY"))
    logger.info(
        "campaign %s | %d iterations | %s | encoders: %s",
        campaign.name, campaign.max_iterations,
        "live planner" if live else "offline scripted planner",
        ", ".join(item.name for item in campaign.encoders),
    )
    result = await run_campaign(
        campaign,
        build_planner(campaign, live=live),
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    )
    payload = finalize(result, campaign)
    report = await write_report(result.state, campaign, live=live)
    write_json(result.output_dir / "report.json", report)
    logger.info("campaign artifacts: %s", result.output_dir)
    return result.output_dir
```

`write_json` must be imported at the top of `_loop.py` (Task 5 adds it).

- [ ] **Step 5: Add the CLI subcommand**

In `lens_lejepa/cli.py`, add the subparser after the `tasks` subparser:

```python
    agent_parser = commands.add_parser("agent-loop", help="run an agent-driven downstream campaign")
    _add_config_args(agent_parser)
```

and add the dispatch branch, after the `tasks` branch:

```python
    elif args.command == "agent-loop":
        import asyncio

        from .agents import run_from_config

        asyncio.run(run_from_config(args.config, args.set))
```

- [ ] **Step 6: Export `run_from_config`**

In `lens_lejepa/agents/__init__.py`, add `run_from_config` to the `._loop` import list and to `__all__`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 8: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 9: Smoke-test the CLI offline**

Run: `.venv/bin/python -m lens_lejepa agent-loop --config configs/agent/campaign.yaml --set max_iterations=1`
Expected: it fails with a clear error that the dataset `../datasets/Model_II` is missing (the real data is not on this machine). That is the expected outcome here — the point of the smoke test is that the CLI parses the config, resolves the encoders and starts the loop.

- [ ] **Step 10: Commit**

```bash
git add lens_lejepa/cli.py lens_lejepa/agents/ configs/agent/campaign.yaml tests/test_agents.py
git commit -m "feat(cli): add the agent-loop subcommand and a campaign config

python -m lens_lejepa agent-loop --config configs/agent/campaign.yaml
runs the campaign offline with DLens' scripted planner, or live when
DLENS_LLM or OPENAI_API_KEY is set."
```

---

### Task 7: The end-to-end agent tests

**Files:**
- Modify: `tests/test_agents.py`

**Interfaces:**
- Consumes: everything from Tasks 2-6.
- Produces: no new production code.

- [ ] **Step 1: Add the DLens-integration test**

Append to `tests/test_agents.py`:

```python
def test_the_dlens_planner_drives_the_loop_offline(fake_root, tmp_path):
    import asyncio

    from dlens.agents._base import BaseAgentConfig
    from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
    from dlens.agents._scripted_planner import make_scripted_planner_model

    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import run_campaign

    campaign = _tiny_campaign(fake_root, tmp_path, max_iterations=2)
    agent_config = BaseAgentConfig(
        name="test",
        description="test",
        custom_system_prompt="choose",
        model=make_scripted_planner_model([
            {"action": "train", "rationale": "switch adapter", "updated_params": {"adaptation": "lora"}},
        ]),
    )
    planner = ReActPlannerStrategy(planner=ExperimentPlanner(config=agent_config))
    result = asyncio.run(run_campaign(
        campaign, planner,
        train_backend=LensJepaTrainBackend(campaign),
        infer_backend=LensJepaInferBackend(campaign),
    ))
    assert result.state.runs[0].spec.adaptation == "linear_probe"
    assert result.state.runs[1].spec.adaptation == "lora"


def test_two_campaigns_with_the_same_seed_agree(fake_root, tmp_path):
    import asyncio

    from lens_lejepa.agents._backends import LensJepaInferBackend, LensJepaTrainBackend
    from lens_lejepa.agents._loop import run_campaign

    accuracies = []
    for name in ("a", "b"):
        campaign = _tiny_campaign(fake_root, tmp_path / name, max_iterations=1)
        result = asyncio.run(run_campaign(
            campaign, ScriptedPlanner([]),
            train_backend=LensJepaTrainBackend(campaign),
            infer_backend=LensJepaInferBackend(campaign),
        ))
        accuracies.append([run.analysis_result.accuracy for run in result.state.runs])
    assert accuracies[0] == accuracies[1]


def test_importing_the_core_does_not_import_the_agent_layer():
    """The agent layer is optional; nothing else may need it."""
    import subprocess
    import sys

    code = (
        "import lens_lejepa.cli, lens_lejepa.engine, lens_lejepa.tasks; "
        "import sys; print('lens_lejepa.agents' in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: PASS — all tests in the file.

If `test_two_campaigns_with_the_same_seed_agree` fails, the pipeline is not deterministic on CPU: find the source of the difference before continuing (most likely `set_seed` is not called for the DataLoader worker order, or the run directory name leaks into the result).

- [ ] **Step 3: Run the whole suite and check the timing**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, and the suite finishes in well under two minutes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agents.py
git commit -m "test(agents): end-to-end coverage of the offline campaign

The DLens planner drives our loop offline with the scripted model, two
campaigns with the same seed agree, and importing the core package does
not pull in the agent layer."
```

---

### Task 8: Document the agent workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/ADDING_A_TASK.md`

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: no new production code.

- [ ] **Step 1: Add the README section**

Insert this section in `README.md`, after the "Reproducibility notes" section and before "Acknowledgements":

````markdown
## Agent workflow

The downstream experiments can also be driven by an agent, using the
[DLens](https://github.com/ML4SCI/DeepLense-AI-Scientist) multi-agent framework.
A planner reads the campaign history and proposes the next configuration
(adapter family, rank, learning rate, ...) from the observed validation metrics;
our own engine trains it.

```bash
bash scripts/install_agent_deps.sh          # DLens, pinned, plus its three direct deps
python -m lens_lejepa agent-loop --config configs/agent/campaign.yaml
```

The loop is **offline by default** — DLens' scripted planner, no API key, fully
deterministic — and goes live when `DLENS_LLM` or `OPENAI_API_KEY` is set.

| Rule | Why |
|---|---|
| The planner may change only the adapter and the hyper-parameters | the task, the data, the label budget and the seed are fixed for the campaign, so every iteration is a fair comparison |
| The test set is scored once, after the loop | the planner must never see it |
| An unknown or out-of-range change is rejected and reported back | a silently dropped change makes the loop repeat itself |
| A collapsed run is flagged and never selected | it would otherwise be reported as the outcome |
| The selected configuration is the best validation result, not the last | the last iteration may have been a failed intervention |

Artifacts land in `runs/agent/<campaign>/`: `campaign.json` (the full history),
`campaign.md` (the table) and `report.json` (the agent's narrative). The
per-iteration training runs live beside them under `runs/agent/<campaign>/<task>/`.
````

- [ ] **Step 2: Note what a new task needs**

Append to `docs/ADDING_A_TASK.md`:

```markdown
## 5. Making a task agent-drivable

The agent campaign (`python -m lens_lejepa agent-loop`) drives the
classification task today. To add a task, it needs:

- a `monitor` and `higher_is_better` (already part of the `Task` interface),
- an analysis the agent can read: the campaign uses DLens' `compute_analysis`,
  which expects per-class predictions and labels, so it is classification-shaped,
- a `collect` that returns `probabilities` and `labels` under those names.

Everything else — adapters, bf16 training, validation schedule, best-checkpoint
selection, held-out evaluation — is inherited from the task and the engine.
```

- [ ] **Step 3: Verify the README's install instructions still work**

Run: `bash scripts/install_agent_deps.sh`
Expected: `dlens ok` and `planner ok`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/ADDING_A_TASK.md
git commit -m "docs: document the agent workflow and what a task needs to be agent-drivable"
```

---

## Notes for the implementer

- **Order:** Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. The module graph is acyclic: `_spec` ← `_backends` ← `_loop` ← `__init__`. `_loop` imports `_backends` at module level; nothing imports `_loop` except `__init__`.
- **The `dlens` private imports are the established pattern.** DLens' own scripts import `from dlens.agents._experiment_loop import ExperimentLoop`; its public `__init__.py` files do not re-export the experiment schemas. Import from `dlens.schemas._experiment`, `dlens.schemas._downstream`, `dlens.tools._analysis`, `dlens.agents._base`, `dlens.agents._experiment_planner`, `dlens.agents._report`, `dlens.agents._scripted_planner`, `dlens.config`.
- **DLens' own scripts have bugs; do not copy them.** `scripts/run_pipeline.py` calls `loop.run(dataset=...)` while `ExperimentLoop.run` takes `train_ref`/`val_ref`, and it calls `make_scripted_planner_model(report_after=1)` while the function takes a list of decisions. Our code calls `make_scripted_planner_model([...])`.
- **Do not add `pytest-asyncio`.** The async tests call `asyncio.run(...)` inside synchronous test functions, so the dev extra stays `pytest` only.
- **Scope note:** the campaign covers classification only. `compute_analysis` is classification-shaped, and the regression and super-resolution tasks need their own analysis before they can be agent-driven. The `Task.monitor` / `higher_is_better` interface already supports them.
- **`ReActPlannerStrategy.decide` returns a `PlannerDecision`**, while `ExperimentPlanner.decide` returns the agent's run result. Use `ReActPlannerStrategy`, not `ExperimentPlanner`, as the planner.
