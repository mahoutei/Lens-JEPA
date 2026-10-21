# Lens-JEPA Correctness Fix Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the defects that can move published numbers and the defects that sit on the agent loop's critical path, so the DLens agent work builds on a pipeline whose metrics are trustworthy.

**Architecture:** Nine independent, small changes to existing modules, each with its own test that fails before the change. No new subsystems and no new runtime dependencies. The one shared piece is a new helper `scatter_average` in `lens_lejepa/utils.py`, which replaces two hand-rolled scatter-average blocks that had drifted apart.

**Tech Stack:** Python 3.12, PyTorch 2.x (CPU-only for the test suite), pytest, numpy, pyyaml

**Spec:** `docs/superpowers/specs/2026-10-21-dlens-agent-loop-design.md` — §4 is this plan's contract; §6.7 (guardrails) is why the agent-path tasks are in scope.

## Global Constraints

- Interpreter: `~/.local/bin/python3.12` (3.12.14). Virtualenv at `.venv`, created with `uv venv --python 3.12`. Run everything as `.venv/bin/python -m pytest`.
- The core package must keep working on Python ≥ 3.10 and must **not** gain a dependency on `dlens`, `pydantic-ai`, or any agent library. Those arrive in the second plan (`docs/superpowers/plans/2026-10-21-dlens-agent-loop.md`).
- Every behaviour change ships with a test that fails before the change and passes after it.
- **Two tasks change numerical behaviour:** Task 1 (the SIS scatter) and Task 8 (the pretraining batch loop). Task 1 alters the `lens_jepa` encoder's outputs, so README:153 (`lens_jepa` rows: 0.8298 → 0.8309, Sym 0.8660 → 0.8670) may need re-measuring after this plan. Task 8 stops fetching the batch past the epoch limit, which changes the RNG consumption at the epoch boundary and advances the masked methods' block-mask sampler one step less per epoch. Task 2 explicitly preserves `blocks=1` behaviour; Task 3 changes no behaviour at all.
- Public API changes are additive: existing names keep working. Where a private helper becomes public, keep the private name as an alias.
- Commit after every task. Use Conventional Commit prefixes already in use in this repo: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
- No task needs network access. The suite is CPU-only and takes ~155 s.

## Review Focus

The five input classes most likely to bite a person using this software, and the task that owns the test for each:

1. **Re-running the same config twice** — the second run must not mix its metrics with the first (Task 5).
2. **A collapsed pretraining run** — must leave no loadable checkpoint behind, so no downstream task can pick it up (Task 4).
3. **`adaptation=layerwise` without a `rank_schedule`** — must raise a clear, early error naming the fix, not a `KeyError` deep inside the adapter (Task 9).
4. **A checkpoint from a task that is not one of the three built-ins** — `scripts/collect_results.py` must report it, not crash with a `KeyError` after all the training is done (Task 6).
5. **`grad_accum > 1` with an epoch length that is not a multiple of `grad_accum`** — the short final window must be weighted like every other window (Task 7).

---

### Task 0: Development environment and baseline

**Files:**
- Modify: `.gitignore` (add `.venv/`)
- Create: `.venv/` (not committed)

**Interfaces:**
- Consumes: nothing.
- Produces: `.venv/bin/python` — the interpreter every later task's commands use.

- [ ] **Step 1: Create the virtualenv and install the dependencies**

```bash
cd /home/deansama/projects/Lens-JEPA
uv venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy scikit-learn pyyaml tqdm pytest
uv pip install --python .venv/bin/python --no-deps -e .
```

`--no-deps` on the editable install keeps pip from pulling the ~2 GB CUDA wheel of torch from PyPI; the CPU wheel is already installed by the first command. `torchvision` is imported by `lens_lejepa/data/io.py` and `lens_lejepa/models/supervised.py`, so it must be installed too — add it to the second command:

```bash
uv pip install --python .venv/bin/python torchvision --index-url https://download.pytorch.org/whl/cpu
```

- [ ] **Step 2: Confirm the package imports and the suite is collected**

Run: `.venv/bin/python -c "import lens_lejepa, torch; print(torch.__version__)"`
Expected: prints a version, no traceback.

Run: `.venv/bin/python -m pytest --collect-only -q`
Expected: ~82 tests collected, no collection errors.

- [ ] **Step 3: Record the baseline**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -30`
Expected: a clean run, or a list of failures.

**If any test fails at this point, stop and report it.** A pre-existing failure is not this plan's business and must not be silently "fixed" inside a later task — it changes the baseline every later task is judged against. Record the exact failing test IDs here before continuing.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add .venv to gitignore"
```

---

### Task 1: Average colliding pixels in the SIS lens inversion (spec F1)

**Files:**
- Modify: `lens_lejepa/utils.py` (add `scatter_average` after `check_finite`)
- Modify: `lens_lejepa/models/lens_jepa.py:110-131` (`_SISInverter.forward`, add `_scatter_source`)
- Modify: `lens_lejepa/models/supervised.py:268-287` (`invert_lens`)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `lens_lejepa.utils.scatter_average(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor` — `values` is `[B, K]`, `index` is `[B, K]` int64 with entries in `[0, size)`, result is `[B, size]`, colliding indices averaged, bins with no source are `0`.

**Background.** `lens_lejepa/models/lens_jepa.py:126-129` currently reads:

```python
out = torch.zeros(b, n * n, device=images.device, dtype=flat_in.dtype)
current = out.gather(1, index)
values = flat_in[:, nonzero]
out.scatter_(1, index, torch.where(current == 0, values, (current + values) / 2))
```

`current` is gathered from a **freshly zeroed** `out`, so it is always `0`; the `(current + values) / 2` branch is unreachable and `scatter_` keeps one arbitrary winner among colliding source pixels. The result therefore depends on write order and drops energy. `lens_lejepa/models/supervised.py:285-287` does the same job correctly with `scatter_add_` + a count. This task makes both use one correct helper.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_scatter_average_averages_collisions():
    from lens_lejepa.utils import scatter_average

    values = torch.tensor([[1.0, 3.0, 5.0]])
    index = torch.tensor([[0, 0, 2]])
    out = scatter_average(values, index, size=3)
    assert torch.allclose(out, torch.tensor([[2.0, 0.0, 5.0]]))


def test_scatter_average_is_order_independent_and_conserves_the_sum():
    from lens_lejepa.utils import scatter_average

    values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    index = torch.tensor([[0, 0, 2, 2]])
    forward = scatter_average(values, index, size=3)
    backward = scatter_average(values.flip(1), index.flip(1), size=3)
    assert torch.allclose(forward, backward)
    assert forward.sum() == pytest.approx(values.sum())


def test_sis_inverter_averages_colliding_pixels():
    from lens_lejepa.models.lens_jepa import _SISInverter

    inverter = _SISInverter(32, 16, dim=8, heads=2, hidden=8, activation=torch.nn.ELU, blocks=1)
    values = torch.tensor([[1.0, 3.0, 5.0]])
    index = torch.tensor([[0, 0, 7]])
    out = inverter._scatter_source(values, index)
    assert out[0, 0] == pytest.approx(2.0)  # the mean of the two colliding pixels
    assert out[0, 7] == pytest.approx(5.0)
    assert out[0, 1] == 0.0  # nothing landed here
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_models.py -k scatter_average or sis_inverter -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens_lejepa.utils'` is not raised; instead `ImportError: cannot import name 'scatter_average' from 'lens_lejepa.utils'` and `AttributeError: '_SISInverter' object has no attribute '_scatter_source'`.

- [ ] **Step 3: Add the helper**

In `lens_lejepa/utils.py`, after the `check_finite` function, add:

```python
def scatter_average(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Mean of ``values`` into bins given by ``index``; bins with no source are zero.

    ``values`` is ``[B, K]`` and ``index`` is ``[B, K]`` of dtype ``int64`` with
    entries in ``[0, size)``. The result is ``[B, size]``.

    Unlike ``Tensor.scatter_``, colliding indices are *averaged*, so the result
    does not depend on the order of the contributions and no energy is dropped.
    """
    total = torch.zeros(values.shape[0], size, device=values.device, dtype=values.dtype).scatter_add_(1, index, values)
    count = torch.zeros_like(total).scatter_add_(1, index, torch.ones_like(total))
    return torch.where(count > 0, total / count.clamp_min(1), total)
```

- [ ] **Step 4: Use the helper in the SIS inverter**

In `lens_lejepa/models/lens_jepa.py`, add the import next to `from .masking import apply_masks`:

```python
from ..utils import scatter_average
```

In `_SISInverter`, replace the body of `forward` from the `flat_in = ...` line to the end with:

```python
        flat_in = images.reshape(b, -1)
        out = self._scatter_source(flat_in[:, nonzero], index)
        out = out.view(b, n, n)
        return out / (out.amax(dim=(1, 2), keepdim=True) + self.eps)
```

and add this method to `_SISInverter`, directly above `forward`:

```python
    def _scatter_source(self, values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Map source pixels onto the source plane, averaging colliding pixels."""
        return scatter_average(values, index, self.size * self.size)
```

- [ ] **Step 5: Use the same helper in `invert_lens`**

In `lens_lejepa/models/supervised.py`, add the import after the existing `import torch.nn.functional as F`:

```python
from ..utils import scatter_average
```

Replace the last three lines of `invert_lens`:

```python
    index = (bx * n + by).view(b, -1)
    total = torch.zeros(b, n * n, device=images.device, dtype=images.dtype).scatter_add_(1, index, images.reshape(b, -1))
    count = torch.zeros_like(total).scatter_add_(1, index, torch.ones_like(total))
    return torch.where(count > 0, total / count.clamp_min(1), total).view(b, n, n)
```

with:

```python
    index = (bx * n + by).view(b, -1)
    out = scatter_average(images.reshape(b, -1), index, n * n)
    return out.view(b, n, n)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS — all tests in the file, including the three new ones.

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, and the count matches the baseline plus 3.

- [ ] **Step 8: Commit**

```bash
git add lens_lejepa/utils.py lens_lejepa/models/lens_jepa.py lens_lejepa/models/supervised.py tests/test_models.py
git commit -m "fix(models): average colliding pixels in the lens inversion

The SIS inverter gathered the 'current' value from a freshly zeroed
tensor, so its averaging branch was unreachable and scatter_ kept one
arbitrary winner among colliding source pixels. That made the result
write-order dependent and is the actual cause of the nondeterminism
the README attributes to the GPU. The supervised invert_lens already
averaged correctly; both now share one helper."
```

---

### Task 2: Chain the Lensiformer blocks (spec F2)

**Files:**
- Modify: `lens_lejepa/models/supervised.py:343-354` (`Lensiformer.forward_with_source`)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new. `Lensiformer.forward_with_source(images, distortions)` keeps its signature and its `blocks=1` behaviour.

**Background.** `forward_with_source` currently reads:

```python
x = image_tokens
for block in self.transformer_blocks:
    x = block(key=distortion_tokens, value=image_tokens, query=source_tokens)
```

Every block reads the same three fixed token sets, so block *i*'s output is discarded by block *i+1*. With `blocks > 1` all but the last block receive **no gradient** and stay at their initialisation. `models/lens_jepa.py:153-154` chains correctly. The default is `blocks=1`, where the defect is invisible — so the fix must preserve the `blocks=1` behaviour exactly, because README:146 reports a published `lensiformer` number produced with it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_lensiformer_stack_trains_every_block():
    """A stack of blocks must give every block a gradient, not only the last."""
    from lens_lejepa.models import Lensiformer

    model = Lensiformer(32, 16, embed_dim=32, num_heads=2, blocks=2)
    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    probabilities, _ = model.forward_with_source(images, distortion)
    probabilities.sum().backward()
    assert model.transformer_blocks[0].mlsa.mha.in_proj_weight.grad is not None
    assert model.transformer_blocks[1].mlsa.mha.in_proj_weight.grad is not None


def test_lensiformer_blocks_are_chained():
    """Block n+1 must read block n's output as its key."""
    from lens_lejepa.models import Lensiformer

    torch.manual_seed(0)
    model = Lensiformer(32, 16, embed_dim=32, num_heads=2, blocks=2).eval()
    seen = {}

    def capture(name):
        def hook(module, args, kwargs):
            seen.setdefault(name, {})["key"] = kwargs.get("key")
            seen[name]["output"] = None

        return hook

    outputs = {}

    def forward_hook(name):
        def hook(module, args, output):
            outputs[name] = output

        return hook

    model.transformer_blocks[0].register_forward_pre_hook(capture("b0"), with_kwargs=True)
    model.transformer_blocks[1].register_forward_pre_hook(capture("b1"), with_kwargs=True)
    model.transformer_blocks[0].register_forward_hook(forward_hook("b0"))
    model.transformer_blocks[1].register_forward_hook(forward_hook("b1"))

    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        model.forward_with_source(images, distortion)

    assert seen["b1"]["key"] is outputs["b0"]


def test_lensiformer_first_block_still_reads_the_distortion_tokens():
    """The blocks=1 path must be byte-for-byte the pre-fix behaviour."""
    from lens_lejepa.models import Lensiformer

    torch.manual_seed(0)
    model = Lensiformer(32, 16, embed_dim=32, num_heads=2, blocks=1).eval()
    captured = {}
    model.transformer_blocks[0].register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(key=kwargs["key"]), with_kwargs=True
    )
    images, distortion = torch.randn(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        model.forward_with_source(images, distortion)
    expected = model.tertiary_tokenizer(distortion.reshape(2, 1, 32, 32))
    assert torch.equal(captured["key"], expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_models.py -k lensiformer -v`
Expected: FAIL — `test_lensiformer_stack_trains_every_block` fails on
`assert model.transformer_blocks[0].mlsa.mha.in_proj_weight.grad is not None`
(the gradient is `None`), and `test_lensiformer_blocks_are_chained` fails
because block 1's key is the distortion tokens, not block 0's output.
`test_lensiformer_first_block_still_reads_the_distortion_tokens` PASSES
already — it is the guard that keeps the published number safe.

- [ ] **Step 3: Chain the blocks**

In `lens_lejepa/models/supervised.py`, replace the loop in `forward_with_source`:

```python
        x = image_tokens
        for block in self.transformer_blocks:
            # Every block reads the same three token sets, as in the reference code.
            x = block(key=distortion_tokens, value=image_tokens, query=source_tokens)
```

with:

```python
        x = image_tokens
        for index, block in enumerate(self.transformer_blocks):
            # The first block reads the distortion tokens, exactly as before; every
            # later block reads the previous block's output, so a stack of blocks
            # actually trains instead of discarding all but the last one.
            key = distortion_tokens if index == 0 else x
            x = block(key=key, value=image_tokens, query=source_tokens)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_models.py -k lensiformer -v`
Expected: PASS — all three.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add lens_lejepa/models/supervised.py tests/test_models.py
git commit -m "fix(models): chain the Lensiformer blocks

Every block was handed the same fixed key/value/query, so each block's
output was discarded by the next one and, with blocks>1, all but the
last received no gradient at all. The first block still reads the
distortion tokens, so the default blocks=1 path is unchanged."
```

---

### Task 3: Pin the SIGReg statistic to the reference implementation (spec F3)

**Files:**
- Modify: `lens_lejepa/ssl/sigreg.py:1-11` (module docstring), `:20-40` (`EppsPulley` docstring)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new. **No behaviour change in this task.**

**Background — this task changes no code behaviour, and that is the point.** The reference implementation at `galilai-group/lejepa` (`lejepa/univariate/epps_pulley.py`) computes:

```python
self.register_buffer("weights", weights * self.phi)
...
return (err @ self.weights) * N * self.world_size
```

which is exactly what `lens_lejepa/ssl/sigreg.py` does. The `× N` factor is part of the Epps–Pulley statistic `T = N·∫|φ̂ − φ|² w(t) dt`, not a batch-size artefact, and the `weights * phi` product is upstream's own choice of weight function. An earlier audit flagged both as bugs; **they are not**. This task locks the behaviour down with a test and rewrites the docstrings so nobody "fixes" it later.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_sigreg_scales_with_the_sample_count_like_the_reference():
    """The reference LeJEPA statistic is T = N * sum_t w(t) * (dcos^2 + dsin^2)."""
    from lens_lejepa.ssl.sigreg import EppsPulley

    test = EppsPulley(t_max=3.0, n_points=17)
    x = torch.randn(64, 4)
    doubled = test(x.repeat(2, 1))
    single = test(x)
    # The mean over samples is unchanged by duplicating them, so the statistic
    # must scale exactly with N. A missing N factor would leave it unchanged.
    assert doubled == pytest.approx(2.0 * single, rel=1e-5)


def test_sigreg_weights_include_the_standard_normal_characteristic_function():
    """Upstream folds phi(t) into the trapezoid weights; keep it that way."""
    from lens_lejepa.ssl.sigreg import EppsPulley

    test = EppsPulley(t_max=3.0, n_points=17)
    dt = 3.0 / (17 - 1)
    trapezoid = torch.full((17,), 2 * dt)
    trapezoid[[0, -1]] = dt
    assert torch.allclose(test.weights, trapezoid * test.phi)
    assert torch.allclose(test.phi, torch.exp(-0.5 * test.t.square()))
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k sigreg -v`
Expected: PASS. This task's test pins existing correct behaviour; if it FAILS, the implementation has drifted from upstream and that is a finding, not a test bug.

- [ ] **Step 3: Rewrite the docstrings**

In `lens_lejepa/ssl/sigreg.py`, replace the module docstring:

```python
"""SIGReg: sketched isotropic Gaussian regularisation (Balestriero & LeCun, 2025).

The embeddings are projected onto random unit directions ("slices"). For each
slice an Epps-Pulley statistic compares the empirical characteristic function
with that of a standard normal. Keeping every 1-D projection Gaussian keeps
the whole embedding distribution close to an isotropic Gaussian, which is what
stops the teacher-free objective from collapsing.

Follows the reference implementation at github.com/galilai-group/lejepa
(single-process version).
"""
```

with:

```python
"""SIGReg: sketched isotropic Gaussian regularisation (Balestriero & LeCun, 2025).

The embeddings are projected onto random unit directions ("slices"). For each
slice the Epps-Pulley statistic

    T = N * sum_t w(t) * ((cos_mean(t) - phi(t))**2 + sin_mean(t)**2)

compares the empirical characteristic function with that of a standard normal.
Keeping every 1-D projection Gaussian keeps the whole embedding distribution
close to an isotropic Gaussian, which is what stops the teacher-free objective
from collapsing.

This follows the reference implementation at github.com/galilai-group/lejepa
(``lejepa/univariate/epps_pulley.py``, single-process version) exactly:
the trapezoid weights are multiplied by ``phi(t)`` and the statistic is
multiplied by the sample count N, which is part of the Epps-Pulley statistic
itself and not a batch-size artefact. Do not "normalise" either one away.
"""
```

and in `EppsPulley`, replace the class docstring:

```python
    """Univariate Epps-Pulley test statistic against N(0, 1)."""
```

with:

```python
    """Univariate Epps-Pulley test statistic against N(0, 1).

    ``weights`` already include the standard normal characteristic function
    ``phi(t) = exp(-t^2/2)``; the returned statistic is multiplied by the
    number of samples. Both match the reference LeJEPA implementation.
    """
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add lens_lejepa/ssl/sigreg.py tests/test_pipeline.py
git commit -m "docs(ssl): pin the SIGReg statistic to the reference LeJEPA formula

The Epps-Pulley statistic legitimately scales with the sample count and
folds phi(t) into its weights, exactly as galilai-group/lejepa does.
Document that and add tests, so the next reader does not remove either
factor."
```

---

### Task 4: A collapsed pretraining run must leave no loadable checkpoint (spec F4)

**Files:**
- Modify: `lens_lejepa/engine/pretrain.py:170-171`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new. `pretrain(config, resume=None) -> Path` keeps its signature.

**Background.** The collapse guard raises `RuntimeError` at `pretrain.py:170-171`, but the previous epochs already wrote `last_pretrain.pt` and that file stays on disk, loadable by `load_pretrained_encoder`. The comment at `pretrain.py:174` ("Only healthy checkpoints are written, so a collapsed run can never be used downstream") and README:190-192 are therefore false.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_collapsed_pretraining_leaves_no_loadable_checkpoint(fake_root, tmp_path):
    from pathlib import Path

    config = _pretrain_config(fake_root, tmp_path)
    config.collapse_std_threshold = 10.0  # every epoch counts as collapsed
    config.collapse_patience = 2          # so epoch 1 still writes a checkpoint
    with pytest.raises(RuntimeError, match="collapsed"):
        pretrain(config)
    run_dir = Path(config.output_dir) / "pretrain" / config.method / config.run_name
    assert not (run_dir / "last_pretrain.pt").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k collapsed -v`
Expected: FAIL — `assert not (run_dir / "last_pretrain.pt").exists()` fails,
because epoch 1 wrote the checkpoint and epoch 2's collapse only raised.

- [ ] **Step 3: Delete the checkpoint when the guard trips**

In `lens_lejepa/engine/pretrain.py`, replace:

```python
        if collapse_epochs >= config.collapse_patience:
            raise RuntimeError(f"Representation collapsed for {collapse_epochs} epochs: {diagnostics}. Run rejected.")
```

with:

```python
        if collapse_epochs >= config.collapse_patience:
            # A collapsed run must leave nothing loadable behind, otherwise a
            # downstream task can still pick up the encoder of a rejected run.
            (run_dir / "last_pretrain.pt").unlink(missing_ok=True)
            raise RuntimeError(f"Representation collapsed for {collapse_epochs} epochs: {diagnostics}. Run rejected.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add lens_lejepa/engine/pretrain.py tests/test_pipeline.py
git commit -m "fix(engine): delete the checkpoint when the collapse guard trips

The guard raised but left the previous epoch's last_pretrain.pt on disk,
where it stayed loadable by load_pretrained_encoder - so a collapsed run
could still reach a downstream task, contradicting the README."
```

---

### Task 5: Re-running a config must not mix two runs' metrics (agent path)

**Files:**
- Modify: `lens_lejepa/utils.py:58-69` (`append_csv`)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `append_csv(path: Path, row: dict[str, Any]) -> None` keeps its signature; its behaviour changes only in the case described below.

**Background.** `make_run_dir` (`utils.py:38-40`) creates the run directory with `exist_ok=True` under a deterministic name, and `append_csv` appends. Re-running a config therefore leaves one CSV containing the epochs of both runs, and overwrites `best.pt`. The agent loop in the second plan creates many run directories, so this must not silently corrupt a campaign's metrics.

The fix uses the fact that epochs increase monotonically *within* a run: a new row whose `epoch` is not greater than the highest `epoch` already in the file belongs to a fresh run, so the file is replaced.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data.py`:

```python
def test_append_csv_does_not_mix_two_runs(tmp_path):
    from lens_lejepa.utils import append_csv

    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 1, "loss": 1.0})
    append_csv(path, {"epoch": 2, "loss": 0.5})
    append_csv(path, {"epoch": 1, "loss": 9.0})  # a new run starts at epoch 1
    text = path.read_text()
    assert text.count("\n") == 2  # header + one row
    assert "9.0" in text
    assert "0.5" not in text


def test_append_csv_appends_a_resumed_run(tmp_path):
    from lens_lejepa.utils import append_csv

    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 1, "loss": 1.0})
    append_csv(path, {"epoch": 2, "loss": 0.5})
    append_csv(path, {"epoch": 3, "loss": 0.25})  # a resumed run continues
    assert path.read_text().count("\n") == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_data.py -k append_csv -v`
Expected: FAIL — the first test finds 4 lines instead of 2.

- [ ] **Step 3: Make `append_csv` replace a file from a previous run**

In `lens_lejepa/utils.py`, replace `append_csv` with:

```python
def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one row, rewriting the header if new columns appear.

    Epochs increase monotonically within a run, so a row whose ``epoch`` is not
    greater than the highest epoch already in the file belongs to a *new* run
    with the same name: the file is replaced rather than appended to, so
    re-running a config never mixes two runs' metrics in one file.
    """
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    epoch = row.get("epoch")
    if rows and epoch is not None:
        previous = [float(r["epoch"]) for r in rows if r.get("epoch") not in (None, "")]
        if previous and float(epoch) <= max(previous):
            rows = []
    rows.append({key: value for key, value in row.items() if np.isscalar(value) or value is None})
    fieldnames = list(dict.fromkeys(key for r in rows for key in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass. `tests/test_pipeline.py:72` asserts
`pretrain_metrics.csv` has 4 lines after a 2-epoch run plus a 1-epoch resume — that still holds, because a resumed run continues at epoch 3.

- [ ] **Step 6: Commit**

```bash
git add lens_lejepa/utils.py tests/test_data.py
git commit -m "fix(utils): replace a metrics CSV that belongs to a previous run

Run directories are reused (exist_ok=True) under deterministic names, so
re-running a config appended its epochs to the previous run's CSV. A row
whose epoch does not advance past the file's highest epoch now starts a
fresh file."
```

---

### Task 6: `collect_results.py` must handle any task and label encoders correctly (agent path)

**Files:**
- Modify: `scripts/collect_results.py:21-41` (`collect`)
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `collect(runs: Path) -> list[dict]` keeps its signature. The row dict gains no new keys; the `adaptation` value becomes correct for encoder backbones that are not named `vit*`.

**Background.** `collect_results.py:38` does `COLUMNS[config["task"]]`, which raises `KeyError` for any task added following `docs/ADDING_A_TASK.md` — and it is the **last** command of `scripts/reproduce_paper.sh`, so the crash happens after all the training is done. `collect_results.py:34` labels the adaptation as `-` for every backbone whose name does not start with `vit`, which mislabels the `lens_jepa` encoder rows (README:153).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
def _load_collect_results():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "collect_results.py"
    spec = importlib.util.spec_from_file_location("collect_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(root, task, backbone, adaptation, metrics):
    import json

    run = root / task / f"{backbone}_{adaptation}" / "run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "task": task, "backbone": backbone, "adaptation": adaptation,
        "train_dataset": "Model_II", "shots": None,
    }))
    (run / "summary.json").write_text(json.dumps({
        "test": {"test_root": "/data/Model_II_test", **metrics},
    }))
    return run


def test_collect_results_reports_an_unknown_task(tmp_path):
    collect_results = _load_collect_results()
    _write_summary(tmp_path, "einstein_radius", "vit_small", "rslora", {"accuracy": 0.5, "mae": 0.1})
    rows = collect_results.collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["task"] == "einstein_radius"
    assert rows[0]["adaptation"] == "rslora"


def test_collect_results_labels_the_lens_jepa_encoder(tmp_path):
    collect_results = _load_collect_results()
    _write_summary(tmp_path, "classification", "lens_jepa", "lora", {"accuracy": 0.83})
    rows = collect_results.collect(tmp_path)
    assert rows[0]["adaptation"] == "lora"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_data.py -k collect_results -v`
Expected: FAIL — `KeyError: 'einstein_radius'` in the first test, and
`assert 'lora' == '-'` in the second.

- [ ] **Step 3: Make the column lookup task-agnostic**

In `scripts/collect_results.py`, replace the imports at the top with:

```python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens_lejepa.models import is_encoder
```

and replace `collect` with:

```python
def _columns(task: str, test: dict) -> list[str]:
    """Metric columns for a task; unknown tasks report their numeric metrics."""
    known = COLUMNS.get(task)
    if known is not None:
        return known
    return sorted(
        key for key, value in test.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def collect(runs: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(runs.glob("*/*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if "test" not in summary:
            continue
        config = json.loads((summary_path.parent / "config.json").read_text())
        test = summary["test"]
        rows.append(
            {
                "task": config["task"],
                "run": summary_path.parent.name,
                "backbone": config["backbone"],
                "adaptation": config["adaptation"] if is_encoder(config["backbone"]) else "-",
                "train": config["train_dataset"],
                "shots": "full" if config["shots"] is None else config["shots"],
                "test": Path(test["test_root"]).name,
                **{key: test.get(key) for key in _columns(config["task"], test)},
            }
        )
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Check the script still runs against the repo's own runs directory**

Run: `.venv/bin/python scripts/collect_results.py runs`
Expected: no traceback. If `runs/` does not exist yet, the script prints nothing and exits 0 — that is also fine.

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_results.py tests/test_data.py
git commit -m "fix(scripts): report any task in collect_results and label encoders

COLUMNS[task] raised KeyError for every task added following
docs/ADDING_A_TASK.md, and it ran as the last command of
reproduce_paper.sh, after all the training. The adaptation column also
mislabelled the lens_jepa encoder as having no adaptation."
```

---

### Task 7: Weight the final gradient-accumulation window by its actual size (agent path)

**Files:**
- Modify: `lens_lejepa/engine/finetune.py:110-125` (the training loop)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new. `finetune(config: FinetuneConfig) -> Path` keeps its signature.

**Background.** `finetune.py:120` does `(loss / config.grad_accum).backward()` and flushes when `step % config.grad_accum == 0 or step == steps`. The last window is usually shorter than `grad_accum`, so its gradients are divided by a number they were not accumulated with — the tail step is systematically down-weighted. With the default `grad_accum=1` the behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_grad_accum_scales_the_window_by_its_actual_size(fake_root, tmp_path, monkeypatch):
    """A window holding one gradient must not be scaled as if it held two."""
    recorded = []

    class RecordingAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            recorded.append(
                sum(
                    float(p.grad.abs().sum())
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                )
            )
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)

    def run(grad_accum, name):
        config = FinetuneConfig(
            task="classification", data_root=str(fake_root.parent), train_dataset=fake_root.name,
            test_dataset=None, image_size=32, backbone="vit_tiny", adaptation="linear_probe",
            epochs=1, batch_size=4, grad_accum=grad_accum, max_steps_per_epoch=1,
            val_interval=1, workers=0, device="cpu", output_dir=str(tmp_path), run_name=name,
        )
        finetune(config)

    run(1, "single")
    run(2, "accumulated")
    assert len(recorded) == 2
    # Both runs see the same first batch, so the gradient must be identical:
    # the one-batch window of the second run holds one gradient, not two.
    assert recorded[1] == pytest.approx(recorded[0], rel=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k grad_accum -v`
Expected: FAIL — `recorded[1]` is about half of `recorded[0]`, because the tail
window was divided by `grad_accum=2`.

- [ ] **Step 3: Scale the gradients by the window size at flush time**

In `lens_lejepa/engine/finetune.py`, replace the training loop:

```python
            model.train()
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, total=steps, desc=f"{task.name} {epoch}/{config.epochs}", leave=False), start=1):
                if step > steps:
                    break
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with _autocast(config, device):
                    loss = task.loss(model(*task.model_input(batch)), batch)
                check_finite("training loss", loss)
                (loss / config.grad_accum).backward()
                if step % config.grad_accum == 0 or step == steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
```

with:

```python
            model.train()
            losses = []
            optimizer.zero_grad(set_to_none=True)
            window = 0
            for step, batch in enumerate(tqdm(train_loader, total=steps, desc=f"{task.name} {epoch}/{config.epochs}", leave=False), start=1):
                if step > steps:
                    break
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with _autocast(config, device):
                    loss = task.loss(model(*task.model_input(batch)), batch)
                check_finite("training loss", loss)
                loss.backward()
                window += 1
                if step % config.grad_accum == 0 or step == steps:
                    # Average over the gradients actually accumulated, so a short
                    # final window counts as much as a full one.
                    scale = 1.0 / window
                    for group in optimizer.param_groups:
                        for param in group["params"]:
                            if param.grad is not None:
                                param.grad.mul_(scale)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    window = 0
                losses.append(loss.item())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add lens_lejepa/engine/finetune.py tests/test_pipeline.py
git commit -m "fix(engine): average the gradient-accumulation window by its size

The loss was divided by grad_accum before every backward, so the final
partial window - which usually holds fewer gradients than grad_accum -
was systematically down-weighted. The gradients are now averaged over
the window that was actually accumulated."
```

---

### Task 8: Pretraining must not fetch a batch past the epoch limit, and must reject an empty epoch (agent path)

**Files:**
- Modify: `lens_lejepa/engine/pretrain.py:107` (after `steps_per_epoch` is computed), `:137-139` (the batch loop)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new. `pretrain(config, resume=None) -> Path` keeps its signature.

**Background.** The loop breaks *after* fetching the next batch, so every epoch loads one batch more than it trains on; for the masked methods that also advances the block-mask sampler's iteration counter, so the mask sequence drifts by one per epoch. And when `steps_per_epoch == 0` the loop body never runs, leaving `parts` undefined at `pretrain.py:160` — a `NameError` where a clear message belongs.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_pretrain_does_not_fetch_a_batch_past_the_epoch_limit(fake_root, tmp_path, monkeypatch):
    import lens_lejepa.engine.pretrain as pretrain_module

    calls = []
    original = pretrain_module.UnlabelledLensDataset

    class Counting(original):
        def __getitem__(self, index):
            calls.append(index)
            return super().__getitem__(index)

    monkeypatch.setattr(pretrain_module, "UnlabelledLensDataset", Counting)
    config = _pretrain_config(fake_root, tmp_path)
    config.max_steps_per_epoch = 1
    config.batch_size = 4
    pretrain(config)
    assert len(calls) == 4  # exactly one batch of four, not two


def test_pretrain_rejects_an_epoch_without_steps(fake_root, tmp_path):
    config = _pretrain_config(fake_root, tmp_path)
    config.max_steps_per_epoch = 0
    with pytest.raises(RuntimeError, match="no training steps"):
        pretrain(config)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k "epoch_limit or without_steps" -v`
Expected: FAIL — the first finds 8 calls instead of 4; the second raises
`NameError: name 'parts' is not defined` instead of the expected `RuntimeError`.

- [ ] **Step 3: Guard against an empty epoch**

In `lens_lejepa/engine/pretrain.py`, add `from itertools import islice` to the imports, and after the line that computes `steps_per_epoch`:

```python
    steps_per_epoch = len(loader) if config.max_steps_per_epoch is None else min(len(loader), config.max_steps_per_epoch)
```

add:

```python
    if steps_per_epoch == 0:
        raise RuntimeError("An epoch has no training steps: the dataset is empty or max_steps_per_epoch is 0.")
```

- [ ] **Step 4: Stop fetching after the last training step**

In `lens_lejepa/engine/pretrain.py`, replace:

```python
        progress = tqdm(loader, total=steps_per_epoch, desc=f"pretrain {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if step > steps_per_epoch:
                break
```

with:

```python
        progress = tqdm(loader, total=steps_per_epoch, desc=f"pretrain {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(islice(progress, steps_per_epoch), start=1):
```

`islice` stops the iteration at the limit, so the loader never hands over a batch that is not trained on.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add lens_lejepa/engine/pretrain.py tests/test_pipeline.py
git commit -m "fix(engine): stop the pretraining loop at the epoch limit

The loop broke after fetching the next batch, so every epoch loaded one
batch more than it trained on and the masked methods' mask sampler
advanced one extra time per epoch. An epoch with no steps also left the
epoch's loss parts undefined and crashed with a NameError."
```

---

### Task 9: Validate `FinetuneConfig` before a run starts (agent path)

**Files:**
- Modify: `lens_lejepa/config.py:69-107` (`FinetuneConfig`, add `__post_init__`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FinetuneConfig.__post_init__(self) -> None` — new. Every existing construction site keeps working; `FinetuneConfig(**checkpoint["config"])` in `engine/finetune.py:62` is unaffected because valid configs still validate.

**Background.** `FinetuneConfig` has no validation at all. `val_interval=0` divides by zero at `engine/finetune.py:127`; `adaptation="layerwise"` without a `rank_schedule` raises a bare `ValueError` from deep inside `models/adapters.py:157`; an unknown adaptation name is silently accepted for non-encoder backbones and ignored.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_finetune_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="val_interval"):
        FinetuneConfig(val_interval=0)
    with pytest.raises(ValueError, match="rank_schedule"):
        FinetuneConfig(adaptation="layerwise")
    with pytest.raises(ValueError, match="Unknown adapter"):
        FinetuneConfig(adaptation="typo")
    with pytest.raises(ValueError, match="shots"):
        FinetuneConfig(shots=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k invalid_values -v`
Expected: FAIL — no `ValueError` is raised for any of the four cases.

- [ ] **Step 3: Add validation**

In `lens_lejepa/config.py`, add the import:

```python
from .models.adapters import canonical_adapter_name
```

(`lens_lejepa/models/adapters.py` imports only `math`, `re` and torch, so this
introduces no import cycle: `lens_lejepa.models` does not import `lens_lejepa.config`.)

Then add this method to `FinetuneConfig`, after its field definitions:

```python
    def __post_init__(self) -> None:
        if self.val_interval <= 0:
            raise ValueError(f"val_interval must be positive, got {self.val_interval}.")
        if self.shots is not None and self.shots <= 0:
            raise ValueError(f"shots must be a positive number of images per class, got {self.shots}.")
        if self.adaptation == "layerwise" and not self.rank_schedule:
            raise ValueError("adaptation='layerwise' needs a rank_schedule, e.g. [8, 16, 32, 64].")
        if self.adaptation not in ("full", "linear_probe"):
            canonical_adapter_name(self.adaptation)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass. If any existing test constructs a `FinetuneConfig`
with `val_interval=0` or an unknown adaptation, fix that construction to a
valid value — do not weaken the validation.

- [ ] **Step 6: Commit**

```bash
git add lens_lejepa/config.py tests/test_pipeline.py
git commit -m "feat(config): validate FinetuneConfig before a run starts

val_interval=0 divided by zero mid-run, adaptation=layerwise without a
rank_schedule failed deep inside the adapter, and an unknown adaptation
name was silently ignored for non-encoder backbones."
```

---

### Task 10: Correct the README claims this plan made stale

**Files:**
- Modify: `README.md:59`, `README.md:167-169`, `README.md:190-192`

**Interfaces:**
- Consumes: the behaviour changes landed by Tasks 1, 4 and 9.
- Produces: no new production code.

**Background.** This task was added after Task 4's review flagged it: three statements in the README
are now inaccurate because of the fixes in this plan. Only prose changes — no code, no tests.

1. `README.md:59` claims "77 tests, CPU only, ~15 s". The suite had 82 tests before this plan and
   had **101** after it, and it takes ~155 s on this machine, not ~15 s. (The count moved again
   afterwards, when the final review's fix wave added two more tests — write the number the suite
   actually reports, as Step 2 says.)
2. `README.md:167-169` attributes the 0.1-point discrepancies of three rows to the scatter in the
   lens inversion being non-deterministic on the GPU. That was true before Task 1: the old code
   kept one arbitrary winner among colliding pixels. The inversion now **averages** them, so the
   result no longer depends on which write wins and only the floating-point summation order can
   differ, in the last bits. The three rows were measured before that fix and should be
   re-measured.
3. `README.md:191` says the collapse guard stops the run "before a checkpoint is written". After
   Task 4 the guard stops the run and **deletes** the checkpoint it had written — the operative
   guarantee ("a collapsed encoder can never reach a downstream task") is now true, but the
   wording is not.

- [ ] **Step 1: Make the three edits**

Edit `README.md` line 59:

```bash
pytest                           # 101 tests, CPU only, ~2.5 min
```

Edit `README.md` lines 167-169:

```markdown
Lensiformer and the Lens-JEPA encoder invert the lens with scatter operations. When several pixels land on the
same source pixel the inversion averages them, so the result no longer depends on which write wins; only the
floating-point summation order on the GPU can differ, and only in the last bits. The three rows below were
measured before that fix and should be re-measured.
```

Edit `README.md` lines 190-192:

```markdown
* **Collapse guard.** Every pretraining epoch records pooled feature standard deviation and effective-rank fraction.
  Three consecutive epochs below `0.02` / `0.04` stop the run and delete the checkpoint it had written, so a
  collapsed encoder can never reach a downstream task.
```

- [ ] **Step 2: Verify the claims you just wrote**

Run: `.venv/bin/python -m pytest --collect-only -q | tail -2`
Expected: the reported count is 101.

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -2`
Expected: all tests pass.

If either number differs from what you wrote above, write the number the run actually reports.

- [ ] **Step 3: Check nothing else in the README contradicts the fixes**

Run: `grep -n "77 tests\|before a checkpoint\|not deterministic" README.md`
Expected: no matches (all three statements were corrected).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: correct the README claims the fix pass made stale

The test count and runtime were wrong, the lens-inversion paragraph
described the pre-fix scatter, and the collapse-guard sentence said the
run stops before a checkpoint is written when it now deletes it."
```

---

## Notes for the implementer

- **Order matters only in one place:** Task 1 must land before Task 6, because Task 6 imports `lens_lejepa.models`, which needs torch importable. Otherwise the tasks are independent.
- **After this plan lands**, the `lens_jepa` encoder's outputs have changed (Task 1). Re-measure the two `lens_jepa` rows in README:153 before quoting them again:
  `python -m lens_lejepa pretrain --config configs/pretrain/lens_jepa.yaml` then the matching `finetune` runs from `scripts/reproduce_paper.sh:77-85`.
- **Left for later, deliberately:** `torch.load(..., weights_only=False)` on user-supplied checkpoints (`models/__init__.py:38`, `engine/finetune.py:74`, `engine/pretrain.py:120`); `AxionMassDataset` decoding all 89k images at construction to read the masses (`data/datasets.py:101`); SSIM's zero padding (`metrics.py:56-60`); `mae_by_mass_decile` being dropped from the CSV and checkpoint (`metrics.py:45` → `utils.py:64` → `engine/finetune.py:156`); `config.py:62` silently overwriting `objective.use_lens_priors`; the duplicated `_LSABlock`/`_MLP`/tokenizer families between `models/lens_jepa.py` and `models/supervised.py`; and `scripts/linear_probe.py` duplicating `adaptation=linear_probe` with a different estimator. None of them block the agent loop.
