"""Concrete local implementation of ``IEvalRunner``.

``LocalEvalRunner`` wraps the DocTR eval entry points
(``evaluate_detection_from_config`` / ``evaluate_recognition_from_config``)
behind the ``IEvalRunner`` Protocol.

Call model
----------
Unlike ``LocalTrainingRunner`` (which needs a thread-safe queue to bridge a
callback-style training loop into a generator), eval is a single blocking
forward pass.  ``LocalEvalRunner`` therefore calls the underlying function
synchronously on the calling thread and returns the result directly.

Any exception raised by the underlying eval function propagates to the
caller unchanged.  It is the caller's responsibility to handle errors (e.g.
a ``RuntimeError`` when the checkpoint file is missing); the runner does not
wrap them in result objects.

Eval entry points
-----------------
The module-level eval entry points (``evaluate_detection_from_config`` /
``evaluate_recognition_from_config``) delegate to the real DocTR backend in
``pd_ocr_training._eval_backend``.  That backend module is imported *lazily*
inside each function -- never at module scope -- so that:

1. The package can be imported and the Protocol contract validated in a
   torch-free environment (importing ``local_eval`` pulls in no torch/DocTR).
2. Tests can monkeypatch these module-level names, or the
   ``_eval_backend.evaluate_*_impl`` functions, rather than importing the real
   torch stack.

Running an actual eval requires the ``[train]`` extra (torch + DocTR); calling
either function without it raises ``ImportError`` from the lazy import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from pd_ocr_training.protocols import (
        DetectionEvalConfig,
        DetectionEvalResult,
        RecognitionEvalConfig,
        RecognitionEvalResult,
    )


class _DetectionEvalKwargs(TypedDict):
    val_path: str | Path
    model_path: str | Path
    arch: str
    batch_size: int
    input_size: int
    rotation: bool
    workers: int
    amp: bool
    device: int | None
    profile: str


class _RecognitionEvalKwargs(TypedDict):
    val_path: str | Path
    model_path: str | Path
    arch: str
    batch_size: int
    input_size: int
    vocab: str
    workers: int
    amp: bool
    device: int | None
    profile: str


# ---------------------------------------------------------------------------
# Eval entry points -- delegate to the lazily-imported DocTR backend
# ---------------------------------------------------------------------------


def evaluate_detection_from_config(
    *,
    val_path: str | Path,
    model_path: str | Path,
    arch: str = "db_resnet50",
    batch_size: int = 2,
    input_size: int = 1024,
    rotation: bool = False,
    workers: int = 4,
    amp: bool = False,
    device: int | None = None,
    profile: str = "eval",
) -> DetectionEvalResult:
    """Run a detection evaluation pass via the real DocTR backend.

    Reconstructs a :class:`DetectionEvalConfig` from the flattened kwargs and
    delegates to :func:`pd_ocr_training._eval_backend.evaluate_detection_impl`.
    The ``_eval_backend`` module is imported lazily so importing ``local_eval``
    stays torch-free.

    Args:
        val_path: Validation dataset path.
        model_path: Checkpoint path.
        arch: DocTR detection architecture name.
        batch_size: Eval batch size.
        input_size: Square input image size.
        rotation: Whether to evaluate rotated/polygon detection.
        workers: DataLoader worker count.
        amp: Whether to use automatic mixed precision.
        device: CUDA device index, or ``None`` for auto-selection.
        profile: Logical run identifier.

    Returns:
        A populated ``DetectionEvalResult``.

    Raises:
        ImportError: When the ``[train]`` extra (torch / DocTR) is not installed.
    """
    from pd_ocr_training import _eval_backend
    from pd_ocr_training.protocols import DetectionEvalConfig

    config = DetectionEvalConfig.model_validate(
        {
            "val_path": val_path,
            "model_path": model_path,
            "arch": arch,
            "batch_size": batch_size,
            "input_size": input_size,
            "rotation": rotation,
            "workers": workers,
            "amp": amp,
            "device": device,
        }
    )
    return _eval_backend.evaluate_detection_impl(profile, config)


def evaluate_recognition_from_config(
    *,
    val_path: str | Path,
    model_path: str | Path,
    arch: str = "crnn_vgg16_bn",
    batch_size: int = 64,
    input_size: int = 32,
    vocab: str = "french",
    workers: int = 4,
    amp: bool = False,
    device: int | None = None,
    profile: str = "eval",
) -> RecognitionEvalResult:
    """Run a recognition evaluation pass via the real DocTR backend.

    Reconstructs a :class:`RecognitionEvalConfig` from the flattened kwargs and
    delegates to :func:`pd_ocr_training._eval_backend.evaluate_recognition_impl`.
    The ``_eval_backend`` module is imported lazily so importing ``local_eval``
    stays torch-free.

    Args:
        val_path: Validation dataset path.
        model_path: Checkpoint path.
        arch: DocTR recognition architecture name.
        batch_size: Eval batch size.
        input_size: Recognition input height.
        vocab: DocTR vocab name or custom vocab string.
        workers: DataLoader worker count.
        amp: Whether to use automatic mixed precision.
        device: CUDA device index, or ``None`` for auto-selection.
        profile: Logical run identifier.

    Returns:
        A populated ``RecognitionEvalResult``.

    Raises:
        ImportError: When the ``[train]`` extra (torch / DocTR) is not installed.
    """
    from pd_ocr_training import _eval_backend
    from pd_ocr_training.protocols import RecognitionEvalConfig

    config = RecognitionEvalConfig.model_validate(
        {
            "val_path": val_path,
            "model_path": model_path,
            "arch": arch,
            "batch_size": batch_size,
            "input_size": input_size,
            "vocab": vocab,
            "workers": workers,
            "amp": amp,
            "device": device,
        }
    )
    return _eval_backend.evaluate_recognition_impl(profile, config)


# ---------------------------------------------------------------------------
# LocalEvalRunner
# ---------------------------------------------------------------------------


def _build_detection_eval_kwargs(
    profile: str,
    config: DetectionEvalConfig,
) -> _DetectionEvalKwargs:
    """Build the kwargs dict for ``evaluate_detection_from_config``.

    Args:
        profile: Logical run identifier (currently passed through for
            logging; unused by the stub but reserved for the real impl).
        config: Typed detection eval configuration.

    Returns:
        Keyword-argument dict ready to pass to
        ``evaluate_detection_from_config(**kwargs)``.
    """
    return {
        "val_path": str(config.val_path),
        "model_path": str(config.model_path),
        "arch": config.arch,
        "batch_size": config.batch_size,
        "input_size": config.input_size,
        "rotation": config.rotation,
        "workers": config.workers,
        "amp": config.amp,
        "device": config.device,
        "profile": profile,
    }


def _build_recognition_eval_kwargs(
    profile: str,
    config: RecognitionEvalConfig,
) -> _RecognitionEvalKwargs:
    """Build the kwargs dict for ``evaluate_recognition_from_config``.

    Args:
        profile: Logical run identifier.
        config: Typed recognition eval configuration.

    Returns:
        Keyword-argument dict ready to pass to
        ``evaluate_recognition_from_config(**kwargs)``.
    """
    return {
        "val_path": str(config.val_path),
        "model_path": str(config.model_path),
        "arch": config.arch,
        "batch_size": config.batch_size,
        "input_size": config.input_size,
        "vocab": config.vocab,
        "workers": config.workers,
        "amp": config.amp,
        "device": config.device,
        "profile": profile,
    }


class LocalEvalRunner:
    """Concrete ``IEvalRunner`` that runs evaluation locally.

    Detection evaluation delegates to ``evaluate_detection_from_config``;
    recognition evaluation delegates to ``evaluate_recognition_from_config``.
    Both are called synchronously on the calling thread and return result
    objects directly.

    The underlying eval functions are module-level names in this module so
    they can be monkeypatched in tests without requiring a GPU.

    Example::

        runner = LocalEvalRunner()
        cfg = RecognitionEvalConfig(
            val_path="data/val", model_path="checkpoints/best.pt"
        )
        result = runner.evaluate_recognition("eval-001", cfg)
        print(f"CER: {result.cer:.4f}  WER: {result.wer:.4f}")
    """

    def evaluate_detection(
        self,
        profile: str,
        config: DetectionEvalConfig,
    ) -> DetectionEvalResult:
        """Run a detection evaluation pass locally and return metrics.

        Delegates to ``evaluate_detection_from_config``, called synchronously.
        Any exception raised propagates directly to the caller.

        Args:
            profile: Logical run identifier (used for logging).
            config: Fully-specified detection evaluation configuration.

        Returns:
            ``DetectionEvalResult`` with overall metrics and slices.

        Raises:
            NotImplementedError: When the stub eval function has not been
                replaced with a real implementation.
            Any other exception raised by the underlying eval function.
        """
        kwargs = _build_detection_eval_kwargs(profile, config)
        return evaluate_detection_from_config(**kwargs)

    def evaluate_recognition(
        self,
        profile: str,
        config: RecognitionEvalConfig,
    ) -> RecognitionEvalResult:
        """Run a recognition evaluation pass locally and return metrics.

        Delegates to ``evaluate_recognition_from_config``, called
        synchronously.  Any exception raised propagates directly to the
        caller.

        Args:
            profile: Logical run identifier (used for logging).
            config: Fully-specified recognition evaluation configuration.

        Returns:
            ``RecognitionEvalResult`` with overall metrics and slices.

        Raises:
            NotImplementedError: When the stub eval function has not been
                replaced with a real implementation.
            Any other exception raised by the underlying eval function.
        """
        kwargs = _build_recognition_eval_kwargs(profile, config)
        return evaluate_recognition_from_config(**kwargs)
