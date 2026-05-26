# Architecture: pdomain-ocr-training

**Status:** Current as of 2026-05-22 (IEvalRunner + LocalEvalRunner shipped).

## Purpose

`pdomain-ocr-training` owns all torch/DocTR OCR model-training and evaluation code
for the `pd-*` suite. No other `pd-*` repo imports torch. Consumers (currently
the future `pdomain-ocr-trainer-spa`) depend only on the typed Protocols, never on
concrete training modules.

## Module layout

```text
pdomain_ocr_training/
    __init__.py      Public API re-exports (lazy for LocalTrainingRunner)
    protocols.py     ITrainingRunner + IEvalRunner Protocols; all config + result models
    local.py         LocalTrainingRunner — callback→generator bridge
    local_eval.py    LocalEvalRunner — synchronous eval wrapper (stub entry points)
    detect.py        Verbatim-moved DocTR detection training (legacy; per-file-ignores)
    recog.py         Verbatim-moved DocTR recognition training (legacy; per-file-ignores)
    datasets.py      ExportManager — on-disk dataset layout manager (legacy)
    utils.py         Shared training utilities (legacy)
```

## Install modes

```bash
pip install pdomain-ocr-training          # Torch-free base
pip install 'pdomain-ocr-training[train]' # Adds torch / DocTR / matplotlib
```

The base install exposes all typed models, both Protocols, and `LocalEvalRunner`.
`LocalTrainingRunner` is available lazily — `import pdomain_ocr_training` never
imports torch; accessing `LocalTrainingRunner` without `[train]` raises a clear
`ImportError`.

## Protocols

### ITrainingRunner

Training is long-running and streams progress across many epochs. The Protocol
returns `Iterator[TrainingEvent]`. `LocalTrainingRunner` bridges the legacy
DocTR callback-style training loops into that iterator via a thread + Queue.

```text
pdomain-ocr-trainer-spa                pdomain-ocr-training
─────────────────       inject        ─────────────────────
ITrainingRunner  ◄──────────────── LocalTrainingRunner
                                        │
                                    thread bridge
                                        │
                                   detect.main() / recog.main()
```

Supported methods:

- `train_detection(profile, config) -> Iterator[TrainingEvent]`
- `train_recognition(profile, config) -> Iterator[TrainingEvent]`

### IEvalRunner

Eval is a single synchronous forward pass — no epoch loop, no progress stream.
The Protocol returns result objects directly. `LocalEvalRunner` delegates to
module-level stub entry points that can be monkeypatched in tests.

```text
pdomain-ocr-trainer-spa                pdomain-ocr-training
─────────────────       inject        ─────────────────────
IEvalRunner      ◄──────────────── LocalEvalRunner
                                        │
                                   evaluate_detection_from_config()  [stub]
                                   evaluate_recognition_from_config() [stub]
```

Supported methods:

- `evaluate_detection(profile, config) -> DetectionEvalResult`
- `evaluate_recognition(profile, config) -> RecognitionEvalResult`

The stub entry points raise `NotImplementedError`. The real DocTR eval backend
is tracked by GH issue #3.

## Config and result models

All models live in `protocols.py` and are Pydantic v2 `BaseModel`s.

**Training:**

- `DetectionConfig` — train/val paths + DocTR hyperparameters
- `RecognitionConfig` — train/val paths + DocTR hyperparameters
- `TrainingEvent` — streaming event (`kind`, `message`, optional metrics)

**Eval:**

- `DetectionEvalConfig` — val path, model checkpoint path
- `RecognitionEvalConfig` — val path, model checkpoint path
- `DetectionEvalResult` — precision, recall, f1, IoU metrics, slices
- `RecognitionEvalResult` — CER, WER, exact-match rate, slices
- `EvalSlice` — per-feature breakdown (for M12/M13 slicing; empty list in M7)

## Legacy modules

`detect.py`, `recog.py`, `datasets.py`, and `utils.py` are verbatim moves from
the legacy `pd-ocr-trainer` repo. They carry per-file-ignores in `pyproject.toml`
(`ANN`, `D`, `BLE`, `S`, plus several additional families for `detect.py` /
`recog.py`) pending an annotation follow-up pass. Do not rewrite them as part
of unrelated tasks.

## Error handling

- `ITrainingRunner`: exceptions are wrapped in `kind="error"` `TrainingEvent`s
  so the streaming caller always gets a clean iterator.
- `IEvalRunner`: exceptions propagate directly. The consumer decides how to
  surface them (API error response, log, etc.).

## ADRs

- `docs/decisions/2026-05-21-ieval-runner-protocol.md` — rationale for
  sibling `IEvalRunner` Protocol rather than extending `ITrainingRunner`.
