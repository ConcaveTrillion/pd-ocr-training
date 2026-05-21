"""DocTR OCR model training pipeline for the pd-* OCR suite.

Two install modes
-----------------
``pip install pd-ocr-training``
    Torch-free base install. Exposes the typed config models
    (``DetectionConfig``, ``RecognitionConfig``, ``TrainingEvent``) and the
    ``ITrainingRunner`` Protocol. Suitable for a long-lived web process
    (e.g. ``pd-ocr-trainer-spa``) that only needs the interface.

``pip install pd-ocr-training[train]``
    Adds the heavy training stack (torch / DocTR / matplotlib) and makes
    ``LocalTrainingRunner`` usable. The actual training runs in a separate
    worker process.

``LocalTrainingRunner`` is exported lazily: it is only imported on first
attribute access. Accessing it without the ``[train]`` extra installed
raises an ``ImportError`` with install guidance rather than a raw
``ModuleNotFoundError`` at package import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pd_ocr_training.protocols import (
    DetectionConfig,
    ITrainingRunner,
    RecognitionConfig,
    TrainingEvent,
)

if TYPE_CHECKING:
    from pd_ocr_training.local import LocalTrainingRunner

__version__ = "0.1.0"
__all__ = [
    "DetectionConfig",
    "ITrainingRunner",
    "LocalTrainingRunner",
    "RecognitionConfig",
    "TrainingEvent",
]


def __getattr__(name: str) -> object:
    """Lazily resolve ``LocalTrainingRunner`` so the base import stays torch-free.

    ``LocalTrainingRunner`` pulls in ``detect.py`` / ``recog.py`` and therefore
    ``torch`` / ``DocTR``. Importing it eagerly would make ``import
    pd_ocr_training`` fail in a torch-free environment. Resolving it here keeps
    the package importable and turns a missing training stack into a clear,
    actionable error.
    """
    if name == "LocalTrainingRunner":
        try:
            from pd_ocr_training.local import LocalTrainingRunner
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "LocalTrainingRunner requires the optional training stack "
                "(torch / DocTR). Install it with: pip install "
                "'pd-ocr-training[train]'"
            ) from exc
        return LocalTrainingRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
