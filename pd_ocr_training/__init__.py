"""DocTR OCR model training pipeline for the pd-* OCR suite.

Two install modes
-----------------
``pip install pd-ocr-training``
    Torch-free base install. Exposes the typed config models
    (``DetectionConfig``, ``RecognitionConfig``, ``TrainingEvent``,
    ``DetectionEvalConfig``, ``RecognitionEvalConfig``, ``EvalSlice``,
    ``DetectionEvalResult``, ``RecognitionEvalResult``) and the
    ``ITrainingRunner`` / ``IEvalRunner`` Protocols. Suitable for a
    long-lived web process (e.g. ``pd-ocr-trainer-spa``) that only needs
    the interfaces.

``pip install pd-ocr-training[train]``
    Adds the heavy training stack (torch / DocTR / matplotlib) and makes
    ``LocalTrainingRunner`` usable. ``LocalEvalRunner`` is torch-free and
    importable in the base install; the real DocTR eval wrappers are a
    follow-up task.

``LocalTrainingRunner`` is exported lazily: it is only imported on first
attribute access. Accessing it without the ``[train]`` extra installed
raises an ``ImportError`` with install guidance rather than a raw
``ModuleNotFoundError`` at package import time.

``LocalEvalRunner`` is also exported lazily for consistency, though it does
not actually require torch (its stub entry points raise ``NotImplementedError``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pd_ocr_training.protocols import (
    DetectionConfig,
    DetectionEvalConfig,
    DetectionEvalResult,
    EvalSlice,
    GlyphFeatureSet,
    IEvalRunner,
    ITrainingRunner,
    RecognitionConfig,
    RecognitionEvalConfig,
    RecognitionEvalResult,
    TrainingEvent,
)

if TYPE_CHECKING:
    from pd_ocr_training.local import LocalTrainingRunner
    from pd_ocr_training.local_eval import LocalEvalRunner

__version__ = "0.1.0"
__all__ = [
    "DetectionConfig",
    "DetectionEvalConfig",
    "DetectionEvalResult",
    "EvalSlice",
    "GlyphFeatureSet",
    "IEvalRunner",
    "ITrainingRunner",
    "LocalEvalRunner",
    "LocalTrainingRunner",
    "RecognitionConfig",
    "RecognitionEvalConfig",
    "RecognitionEvalResult",
    "TrainingEvent",
]


def __getattr__(name: str) -> object:
    """Lazily resolve ``LocalTrainingRunner`` / ``LocalEvalRunner`` so the base import stays torch-free.

    ``LocalTrainingRunner`` pulls in ``detect.py`` / ``recog.py`` and therefore
    ``torch`` / ``DocTR``. Importing it eagerly would make ``import
    pd_ocr_training`` fail in a torch-free environment. Resolving it here keeps
    the package importable and turns a missing training stack into a clear,
    actionable error.

    ``LocalEvalRunner`` is torch-free (its stub entry points raise
    ``NotImplementedError``) but is also resolved lazily for consistency.
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
    if name == "LocalEvalRunner":
        try:
            from pd_ocr_training.local_eval import LocalEvalRunner
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "LocalEvalRunner requires the optional training stack "
                "(torch / DocTR). Install it with: pip install "
                "'pd-ocr-training[train]'"
            ) from exc
        return LocalEvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
