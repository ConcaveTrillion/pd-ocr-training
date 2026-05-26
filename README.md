# pdomain-ocr-training

DocTR OCR model training and evaluation pipeline for the `pd-*` OCR suite.

This package owns all torch/DocTR training code — detection and recognition model fine-tuning, dataset management, model export, and evaluation. Isolating torch here keeps every other `pd-*` SPA backend (e.g. `pdomain-ocr-labeler-spa`, `pdomain-prep-for-pgdp`) torch-free and deployment-lightweight.

Supersedes the legacy `pd-ocr-trainer` repo.

## Install modes

The heavy training stack (torch / DocTR / matplotlib) is an **optional extra**,
so the base install stays torch-free.

```bash
# Base — torch-free. Exposes the typed config models, ITrainingRunner,
# IEvalRunner, and LocalEvalRunner. Use this in a long-lived web process
# (e.g. pdomain-ocr-trainer-spa) that injects the runners but does not train.
pip install pdomain-ocr-training

# Full training stack — adds torch / DocTR / matplotlib and makes
# LocalTrainingRunner usable. Use this in the training worker process.
pip install 'pdomain-ocr-training[train]'
```

## Protocols and concrete runners

Two sibling Protocols, each with a concrete implementation:

| Protocol | Concrete | Install mode |
|---|---|---|
| `ITrainingRunner` | `LocalTrainingRunner` | `[train]` extra required |
| `IEvalRunner` | `LocalEvalRunner` | base (torch-free); real DocTR eval stubs pending |

`LocalTrainingRunner` is exported lazily — importing `pdomain_ocr_training` does
**not** pull in torch. Accessing it without the `[train]` extra raises a clear
`ImportError`. `LocalEvalRunner` is torch-free and importable in the base
install; its underlying eval entry points currently raise `NotImplementedError`
pending the real DocTR eval backend implementation.

### Torch-free usage (config models + Protocols)

```python
from pdomain_ocr_training import (
    DetectionConfig,
    DetectionEvalConfig,
    IEvalRunner,
    ITrainingRunner,
    LocalEvalRunner,
    RecognitionConfig,
    RecognitionEvalConfig,
    TrainingEvent,
)

# Type dependency-injection seams without importing torch:
def run_training(runner: ITrainingRunner, cfg: DetectionConfig) -> None:
    for event in runner.train_detection("my-run", cfg):
        print(event.kind, event.message)

def run_eval(runner: IEvalRunner, cfg: RecognitionEvalConfig) -> None:
    result = runner.evaluate_recognition("eval-001", cfg)
    print(f"CER: {result.cer:.4f}  WER: {result.wer:.4f}")
```

### Full training usage (`[train]` extra)

```python
from pdomain_ocr_training import DetectionConfig, ITrainingRunner, LocalTrainingRunner

runner: ITrainingRunner = LocalTrainingRunner()
cfg = DetectionConfig(train_path="data/train", val_path="data/val")
for event in runner.train_detection("my-run", cfg):
    print(event.kind, event.message)
```

### Eval usage (torch-free; real impl pending)

```python
from pdomain_ocr_training import IEvalRunner, LocalEvalRunner, RecognitionEvalConfig

runner: IEvalRunner = LocalEvalRunner()
cfg = RecognitionEvalConfig(val_path="data/val", model_path="checkpoints/best.pt")
# NOTE: raises NotImplementedError until the real DocTR eval backend is wired in.
result = runner.evaluate_recognition("eval-001", cfg)
```
