# DLens agent loop for Lens-LeJEPA — design

**Date:** 2026-10-21
**Status:** approved in chat, pending review of this document
**Scope:** the `lens_lejepa` package, `configs/agent/`, `tests/test_agents.py`, `scripts/`

---

## 1. Context

`lens_lejepa` is a fixed-method research codebase: self-supervised pretraining
(LeJEPA objective + two lensing priors) followed by frozen-encoder adaptation
(rank-stabilised LoRA and five other adapter families) and three downstream
tasks. The experiments are expressed as YAML configs and run through
`python -m lens_lejepa {pretrain,finetune,evaluate}`; `scripts/reproduce_paper.sh`
walks the paper's tables.

That works, but the exploration space is a grid someone has to write by hand.
The paper's Table 5 (adapter families × label budgets) and Table 7 (label
efficiency) are grids, and grids are exactly what a planner agent is for.

[DLens](https://github.com/ML4SCI/DeepLense-AI-Scientist) is a Pydantic-AI
multi-agent framework for autonomous gravitational-lensing workflows. Its
pipeline is:

```
hypothesis → simulate (lenstronomy) → dataset → ExperimentLoop(design → train → infer → analysis → planner) → report
```

Every stage is a `Protocol`, so it is designed for extension:

| Extension point | Contract |
|---|---|
| `SimBackend` | `generate(SimConfig) -> list[np.ndarray]` |
| `TrainBackend` | `train(DatasetRef, ArchitectureSpec, TrainingConfig) -> TrainResult` |
| `InferBackend` | `infer(weights_path, DatasetRef) -> InferResult` |
| `PlannerStrategy` | `decide(state) -> PlannerDecision` |
| `ReportAgent` | `ExperimentState -> ReportDocument` |

This design makes DLens drive **our** pipeline: a planner agent proposes the
next downstream configuration, our engine trains it, the results feed back, and
the campaign ends with a report.

---

## 2. Goals

- **G1.** An agent chooses the next downstream configuration (adaptation
  family, rank, LR, weight decay, dropout, epochs, encoder) from observed
  validation metrics.
- **G2.** The loop runs our real pipeline — `engine.finetune`, our adapters,
  our bf16 training, our best-validation checkpoint selection — not a
  re-implementation of it.
- **G3.** The loop is **offline-capable and deterministic**: with no API key
  it runs end to end on a scripted planner, in CI, in seconds.
- **G4.** The test set is scored exactly once, for the finally selected
  configuration. The planner never sees it.
- **G5.** The campaign is reproducible from its artifacts: same spec + same
  seed ⇒ same result; every decision and every metric is recorded.

## 3. Non-goals (v1)

- **Pretraining search.** The pool of pretrained encoders is given, not
  discovered. Pretraining is ~6 GPU-hours per candidate (README:91); putting it
  inside the loop makes every iteration cost that much.
- **Regression and super-resolution tasks.** DLens' `compute_analysis` is
  classification-shaped (`AnalysisResult` requires a confusion matrix and
  per-class precision/recall/F1). Our `Task.monitor` / `higher_is_better`
  already generalise; the analysis layer does not, yet.
- **Data generation.** DLens' `DeepLensBackend` (lenstronomy 1.9.2) can
  simulate our dataset. Separate concern, separate change.
- **DLens' `ExperimentLoop`, `ArchitectureSpec`, `TrainingConfig`.** See §5.

## 4. Prerequisite: the fix pass

The agent's input is metrics. If the metrics come from a defective pipeline,
the planner optimises noise and the report is confidently wrong. The agent
work is gated on these four fixes, all of which are one-to-three-line changes
and all of which can move published numbers:

| # | Site | Defect | Why it gates the agent |
|---|---|---|---|
| F1 | `lens_lejepa/models/lens_jepa.py:127-129` | `current = out.gather(1, index)` reads a freshly zeroed `out`, so the `(current+values)/2` branch is unreachable; many-to-one SIS pixels are never averaged and `scatter_` picks an arbitrary winner | This is the actual cause of the nondeterminism README:167-169 attributes to the GPU. It changes the `lens_jepa` encoder's outputs. |
| F2 | `lens_lejepa/models/supervised.py:351-353` | Lensiformer blocks do not chain (`x = block(key=distortion_tokens, value=image_tokens, query=source_tokens)`), so block *i*'s output is discarded by block *i+1*; with `blocks>1` all but the last get no gradient. `x = image_tokens` (`:350`) is dead | Latent at the default `blocks=1`, but the `lensiformer` baseline row in README:146 depends on this model. `models/lens_jepa.py:153-154` chains correctly — the two "Lensiformer"s disagree. |
| F3 | `lens_lejepa/ssl/sigreg.py:40` | `* x.shape[0]` makes the SIGReg statistic scale linearly with sample count (2×B), so λ=0.02 is not a fixed-strength regulariser | Changes the pretraining objective's effective strength; also `sigreg.py:34` folds the N(0,1) characteristic function into the trapezoid weights, so the statistic is not the textbook Epps–Pulley one the docstring claims. Both need checking against `galilai-group/lejepa`. |
| F4 | `lens_lejepa/engine/pretrain.py:170-171` | The collapse guard raises, but the previous epochs' `last_pretrain.pt` is still on disk and still loadable, so a collapsed run *can* reach a downstream task — contradicting the comment at `:174` and README:190-192 | The agent would happily start a campaign from a collapsed encoder. |

Also in the prerequisite: `git init` + a CI workflow that runs the test suite.
An agent writes checkpoints; they need to be versioned. *(Done: the repository
is now under git with a baseline commit at `6d473a1`.)*

## 5. Why our own loop, not DLens'

DLens' `ExperimentLoop` (`src/dlens/agents/_experiment_loop.py`) cannot drive
our search space as written:

1. **Its delta filter silently drops our knobs.** `_apply()` (`:200-223`) keeps
   only `_ARCH_KEYS = ("name", "family", "physics_informed")` and `_CFG_KEYS`
   (loss, optimizer, lr, batch_size, epochs, weight_decay, lr_scheduler,
   dropout, augment, early_stop_patience). Our search space — `adaptation`,
   `rank`, `alpha`, `shots`, `encoder`, `label_smoothing` — would be discarded
   **without error**, and the loop would silently re-run the same configuration
   every iteration.
2. **Its stop guard is wrong for an adapter search.** "train/val gap closed ⇒
   REPORT" would end the campaign on the first well-fit configuration. We want
   patience-based stopping.
3. **Its architecture space is a different space.** `ArchitectureSpec` families
   are {CNN, ResNet, ViT, Equivariant, MLPMixer, Hybrid} × depths/widths —
   supervised classifiers from scratch. Our "architecture" is
   *(pretrained encoder × adaptation family × rank)*.

What we *do* reuse from DLens, unchanged:

- `ExperimentPlanner` — coupled to us only through `ExperimentState`, so a
  subclassed state carrying our spec works as-is. We supply our own system
  prompt via `BaseAgentConfig.custom_system_prompt` (theirs names their knobs
  and would steer the planner into a space we do not have).
- `ReportAgent` — `REPORT_SYSTEM_PROMPT` is generic; it consumes
  `ExperimentState` and emits `ReportDocument`.
- `compute_analysis` — pure numpy, gives accuracy, macro one-vs-rest AUC,
  confusion matrix and per-class precision/recall/F1. Strictly more than our
  `classification_metrics`.
- The typed schemas `ExperimentState`, `ExperimentRun`, `PlannerDecision`,
  `PlannerAction`, `TrainResult`, `InferResult`, `AnalysisResult`, and the
  `PlannerStrategy` protocol.

## 6. Architecture

### 6.1 Modules

```
lens_lejepa/agents/
  __init__.py      public API
  _spec.py         our pydantic specs
  _backends.py     train + infer backends over our engine
  _loop.py         the loop, the stop guards, the planner prompt
configs/agent/campaign.yaml
```

New CLI subcommand, in the existing `lens_lejepa/cli.py`:

```
python -m lens_lejepa agent-loop --config configs/agent/campaign.yaml
```

`lens_lejepa/agents/` is imported lazily by the CLI (as `engine` already is),
so the core package keeps working without the agent dependencies installed.

### 6.2 Our specs

```python
class EncoderSpec(BaseModel):
    """One pretrained encoder in the pool."""
    name: str                    # "lens_lejepa" — referenced by DownstreamSpec.encoder
    backbone: str                # "vit_small"
    pretrained: str | None       # path to last_pretrain.pt
    method: str                  # lens_lejepa | lejepa | lens_jepa_sym | lens_jepa_focus | ijepa


class DownstreamSpec(BaseModel):
    """Exactly the fields the planner may change."""
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
    """Everything that is not planner-controlled."""
    task: str = "classification"        # frozen
    data_root: str = "../datasets"      # frozen
    train_dataset: str = "Model_II"    # frozen
    test_dataset: str | None = "Model_II_test"   # frozen
    shots: int | None = None            # frozen
    seed: int = 42                      # frozen
    val_interval: int = 10              # frozen
    device: str = "cuda"                # frozen
    bf16: bool = True                   # frozen
    image_size: int = 160               # frozen
    patch_size: int = 16                # frozen
    encoders: list[EncoderSpec]         # the pool
    hypothesis: str = ""                # shown to the planner and the report agent
    max_iterations: int = 8
    patience: int = 3
    output_dir: str = "runs/agent"
    rank_schedule: list[int] = [8, 16, 32, 64]   # frozen; required by adaptation=layerwise
```

**Frozen fields** are the budget-fairness guarantee: the planner can never
change the task, the data, the label budget, the seed, or the split. This is
the same concern as DLens' `docs/BUDGET_FAIRNESS.md`.

`rank_schedule` is frozen rather than planner-controlled: `adaptation=layerwise`
requires it (`models/adapters.py:157` raises without it), and letting the
planner invent one would make "one change per iteration" ambiguous.

### 6.3 Data flow, one iteration

```
ExperimentPlanner.decide(state)
  → PlannerDecision(action, rationale, updated_params)
      ↓  validate_and_apply(spec, updated_params)
        · unknown key            → raise (never silently dropped)
        · frozen key             → raise
        · value out of domain   → reject, reason returned to the planner next turn
      ↓
DownstreamSpec'
      ↓
LensJepaTrainBackend.train(spec)
  → build a FinetuneConfig from (CampaignSpec, DownstreamSpec)
  → engine.finetune(config with test_dataset=None)
  → run_dir, best.pt
      ↓
LensJepaInferBackend.infer(best.pt)
  → engine.build_trained_model(best.pt) → task, model
  → task.build_datasets()[1]            # the deterministic validation split
  → engine.run_inference(task, model, val_loader, device)
  → predictions, labels, probabilities
      ↓
dlens.tools._analysis.compute_analysis(infer_result, class_names)
  → AnalysisResult(accuracy, macro_auc, confusion_matrix, per_class, summary)
      ↓
ExperimentState.runs.append(run)  →  our stop guards
```

`FinetuneConfig` mapping — every field accounted for, so nothing is silently
left at a default that differs from the campaign:

| `FinetuneConfig` field | Source |
|---|---|
| `task`, `data_root`, `train_dataset`, `shots`, `train_fraction`, `val_fraction`, `seed`, `val_interval`, `device`, `bf16`, `image_size`, `patch_size` | `CampaignSpec` (frozen) |
| `test_dataset` | `None` inside the loop; `CampaignSpec.test_dataset` for the single final evaluation |
| `backbone`, `pretrained` | `CampaignSpec.encoders[DownstreamSpec.encoder]` |
| `adaptation`, `rank`, `alpha`, `lr`, `weight_decay`, `dropout`, `epochs`, `batch_size`, `label_smoothing` | `DownstreamSpec` (planner-controlled) |
| `rank_schedule` | `CampaignSpec` (frozen) |
| `cache`, `grad_accum`, `loraplus_ratio`, `train_jitter`, `workers`, `max_steps_per_epoch` | `CampaignSpec` (frozen; defaults matching `configs/downstream/classification.yaml`) |
| `output_dir`, `run_name` | `CampaignSpec.output_dir` / a per-iteration unique name, e.g. `iter{n}_{encoder}_{adaptation}` |

After the loop ends:

1. Select the best run by the task's validation monitor (`accuracy`, higher is
   better) — **never** the last run; the last run may have been a failed
   intervention.
2. Evaluate that run's `best.pt` on the test set **once**.
3. Write `campaign.json`, `campaign.md`, `report.json`.

### 6.4 Backends

`LensJepaTrainBackend` implements DLens' `TrainBackend` shape:

```python
class LensJepaTrainBackend:
    name = "lens_lejepa"
    def train(self, dataset, architecture, training_config) -> TrainResult: ...
```

It is our own object, not their `ArchitectureSpec`/`TrainingConfig`; the loop
passes it what it needs. It returns their `TrainResult` with:

- `run_id` — the run directory name
- `weights_path` — `<run_dir>/best.pt`
- `backend` — `"lens_lejepa"`
- `num_classes` — `len(CLASSES)`
- `metrics` — `{"val_accuracy": …, "train_loss": …}`
- `epochs_run` — the epoch count that produced the best checkpoint

`LensJepaInferBackend` implements `infer(weights_path, dataset) -> InferResult`
with `predictions`, `true_labels`, `probabilities`, `accuracy`. It reuses
`engine.run_inference`, so inference stays full-precision float32
(`engine/finetune.py:47`).

**One small public-API change is needed.** Rebuilding a task and model from a
saved checkpoint currently lives in the private
`engine/finetune.py:61 _build_trained_model`. The agent backend needs it, so
it is promoted to a public `build_trained_model` and exported from
`lens_lejepa/engine/__init__.py` alongside `evaluate`, `finetune`,
`pretrain`, `run_inference`. The private name stays as an alias so no
existing caller breaks.

### 6.5 Dependency strategy

DLens' own dependency set (`pydantic-ai`, `llama-cpp-python`, `outlines`,
`mcp`, `numba`, `pandas`, `xmltodict`, `loguru`, `dotenv`) is heavier than we
need, and `llama-cpp-python` requires cmake + a C++ compiler at install time.

`src/dlens/agents/_models.py` imports `llama_cpp` **lazily** (only inside
`OutlinesLlamaCppModel.__init__`), so the planner/report paths work without it.

Therefore:

```bash
DLENS_SHA=d0d6e4937932e29a1df208178e3d386cdc84ebb2   # DLens main @ 2026-10-18
pip install --no-deps "git+https://github.com/ML4SCI/DeepLense-AI-Scientist.git@${DLENS_SHA}"
pip install "pydantic-ai>=2.0" "loguru>=0.7" "python-dotenv>=1.0"
```

- The SHA is pinned in `scripts/install_agent_deps.sh` for reproducibility, and
  re-pinning is a one-line change when DLens moves.
- `pyproject.toml` gains an optional extra `agent` listing the three direct
  dependencies; `dlens` itself is installed from the pinned git SHA because
  it is not published on PyPI.
- DLens requires Python ≥ 3.12. Our core package keeps `requires-python =
  ">=3.10"`; the `agent` extra documents the ≥ 3.12 requirement.
- **Fallback (Approach C):** if the `dlens` install proves unworkable on the
  target machine, re-implement `ExperimentPlanner` and `ReportAgent` on
  `pydantic-ai` directly in `lens_lejepa/agents/_agents.py` (~120 lines). The
  rest of the design is unaffected — the loop only needs an object with
  `decide(state) -> PlannerDecision` and one with `write(state)`.

### 6.6 Planner prompt

Our own system prompt, passed as `custom_system_prompt`. It must:

- describe **our** knob vocabulary (`adaptation`, `rank`, `alpha`, `lr`,
  `weight_decay`, `dropout`, `epochs`, `batch_size`, `label_smoothing`,
  `encoder`) and the valid domains;
- state that the label budget, task, dataset and seed are frozen;
- forbid proposing a change already tried that failed to improve validation;
- require the rationale to cite the observed numbers;
- state that the encoder may be switched to another name in the pool.

### 6.7 Guardrails

1. **Test-set discipline.** The loop's backends never receive or open the test
   dataset. Only the finally selected configuration is scored on test, once.
   Asserted by a test that fails if the loop touches the test path.
2. **Unknown planner key → error.** Never a silent drop. (The direct lesson
   from DLens' `_apply`.)
3. **Frozen key → error.** The planner cannot change task, data, shots, seed,
   val_interval, device, bf16.
4. **Out-of-domain value → rejection, not a crash.** `rank ∈ [1, 256]`,
   `alpha ∈ [0, 1024]`, `lr ∈ [1e-6, 1e-1]`, `weight_decay ∈ [0, 1]`,
   `dropout ∈ [0, 1)`, `epochs ∈ [1, 200]`, `batch_size ∈ [1, 4096]`,
   `adaptation ∈ ADAPTERS`, `encoder ∈ pool`. A rejection is recorded and the
   reason is shown to the planner on the next turn, so it can correct itself.
5. **Collapse guard.** A run whose validation accuracy is at or below chance +
   margin is flagged collapsed, excluded from "best", and the loop continues —
   the planner needs to see and react to the collapse, not have the campaign
   end on it.
6. **Stop conditions.** Patience (validation monitor flat for N consecutive
   iterations) | iteration budget | the planner emits `report` or `stop`.
   Deliberately **not** DLens' "train/val gap closed" guard.
7. **Determinism.** Every run is seeded from `CampaignSpec.seed`; the seed is
   recorded in the campaign artifact. Same spec + same seed ⇒ same result.
8. **Best ≠ last.** The selected configuration is the one with the best
   validation monitor, never the final iteration.

### 6.8 Offline-first

- **Default (no API key):** DLens' `make_scripted_planner_model()` supplies a
  deterministic planner. The whole loop runs in CI.
- **Live:** `DLENS_LLM` / `OPENAI_API_KEY` → `ExperimentPlanner` with our
  prompt. DLens' `build_llm()` handles the hosted-model plumbing; local
  ollama/vLLM work through `build_model_from_env()`.

### 6.9 Artifacts

```
runs/agent/<campaign>/
  campaign.json      full history: hypothesis, per-iteration spec, metrics,
                     decision, rationale, guard outcome
  campaign.md        human-readable iteration table + the selected config
  report.json        DLens ReportDocument (title, summary, findings, conclusion)
  <run_dir>/         our engine's own outputs (best.pt, val_metrics.csv, …)
```

`campaign.md` is the artefact a human reads; `report.json` is the agent's
narrative over the same history. The final test metrics appear in
`campaign.json` / `campaign.md` and nowhere in the iteration history.

## 7. Testing

`tests/test_agents.py` — CPU-only, seconds, no network, no API key, in the
style of the existing `tests/test_pipeline.py` and its `fake_root` fixture:

1. **Loop runs offline.** The scripted planner drives N iterations to
   completion on the synthetic dataset; `campaign.json` contains N iterations.
2. **Unknown planner key raises.** A decision with `{"nope": 1}` is rejected.
3. **Frozen key raises.** A decision with `{"shots": 10}` is rejected.
4. **Out-of-domain value is rejected, not fatal.** `{"rank": 0}` is recorded
   as a rejection with a reason; the loop continues.
5. **The loop never touches the test set.** The test dataset root is made
   unreadable / monkeypatched; the loop still completes.
6. **Best-run selection.** The selected configuration is the one with the best
   validation monitor, and it is not necessarily the last iteration.
7. **Collapse is excluded from best.** A run at chance level does not become
   the selected configuration.
8. **One test evaluation.** The campaign artifact contains exactly one test
   evaluation.
9. **Determinism.** Two offline campaigns with the same seed produce the same
   iteration specs and metrics.

The DLens-dependent tests are skipped when `dlens` is not installed
(`pytest.importorskip`), so the core suite stays runnable without it.

## 8. Docs

- `README.md`: a new section documenting the agent workflow, the install
  command, the offline/live modes, and the guardrails.
- `docs/ADDING_A_TASK.md`: a note on what a new task must provide to be
  agent-drivable (a monitor, an analysis that produces an `AnalysisResult`).

## 9. Resolved decisions

| Question | Decision |
|---|---|
| Which outcome? | Agents drive our experiments (approach A). |
| Loop scope | Finetune-only. The encoder pool is given, not searched. |
| Loop wiring | Our loop, DLens' planner/report/analysis/schemas. |
| LLM | Offline scripted planner by default; live via `DLENS_LLM`/`OPENAI_API_KEY`. |
| May the planner switch encoders? | **Yes.** `encoder` is in the planner's vocabulary, drawn from the pool (default `{lens_lejepa, lejepa}`). One change per iteration is enforced by the prompt, not by the schema. |
| Dependency | DLens from a pinned SHA with `--no-deps`; three direct deps in the `agent` extra. Fallback: Approach C. |
| Regression / SR | Out of scope for v1. |
| Pretraining search | Out of scope for v1 (a separate, explicitly-priced phase). |
