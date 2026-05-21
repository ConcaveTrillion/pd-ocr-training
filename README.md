# pd-ocr-training

DocTR OCR model training pipeline for the `pd-*` OCR suite.

This package owns all torch/DocTR training code — detection and recognition model fine-tuning, dataset management, and model export. Isolating torch here keeps every other `pd-*` SPA backend (e.g. `pd-ocr-labeler-spa`, `pd-prep-for-pgdp`) torch-free and deployment-lightweight.

Supersedes the legacy `pd-ocr-trainer` repo.

## Install modes

The heavy training stack (torch / DocTR / matplotlib) is an **optional extra**,
so the base install stays torch-free.

```bash
# Base — torch-free. Exposes the typed config models and the
# ITrainingRunner Protocol only. Use this in a long-lived web process
# (e.g. pd-ocr-trainer-spa) that injects the runner but does not train.
pip install pd-ocr-training

# Full training stack — adds torch / DocTR / matplotlib and makes
# LocalTrainingRunner usable. Use this in the training worker process.
pip install 'pd-ocr-training[train]'
```

### Torch-free usage (config models + Protocol)

```python
from pd_ocr_training import (
    DetectionConfig,
    ITrainingRunner,
    RecognitionConfig,
    TrainingEvent,
)

# Type a dependency-injection seam without importing torch:
def run(runner: ITrainingRunner, cfg: DetectionConfig) -> None:
    for event in runner.train_detection("my-run", cfg):
        print(event.kind, event.message)
```

`LocalTrainingRunner` is exported lazily — importing `pd_ocr_training` does
**not** pull in torch. Accessing `pd_ocr_training.LocalTrainingRunner` without
the `[train]` extra installed raises a clear `ImportError` pointing at
`pip install 'pd-ocr-training[train]'`.

### Full training usage (`[train]` extra)

```python
from pd_ocr_training import DetectionConfig, ITrainingRunner, LocalTrainingRunner

runner: ITrainingRunner = LocalTrainingRunner()
cfg = DetectionConfig(train_path="data/train", val_path="data/val")
for event in runner.train_detection("my-run", cfg):
    print(event.kind, event.message)
```
