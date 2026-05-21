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

Stub functions
--------------
The production eval entry points (``evaluate_detection_from_config`` /
``evaluate_recognition_from_config``) call into DocTR / torch, which are
optional heavy deps.  The stub implementations provided in this module raise
``NotImplementedError`` so that:

1. The package can be imported and the Protocol contract validated in a
   torch-free environment.
2. Tests can monkeypatch the stubs rather than importing the real torch stack.

When the ``[train]`` extra is installed the stubs should be replaced with
real DocTR eval wrappers; that migration is deferred to the follow-up
eval-implementation task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pd_ocr_training.protocols import (
        DetectionEvalConfig,
        DetectionEvalResult,
        RecognitionEvalConfig,
        RecognitionEvalResult,
    )


# ---------------------------------------------------------------------------
# Stub eval entry points (replaceable via monkeypatch in tests)
# ---------------------------------------------------------------------------


def evaluate_detection_from_config(**kwargs: Any) -> DetectionEvalResult:
    """Stub detection eval entry point.

    This stub exists so ``LocalEvalRunner`` can be imported and tested in a
    torch-free environment.  Replace this function (via monkeypatch or direct
    assignment) with a real DocTR detection eval wrapper once the eval
    implementation is available.

    Args:
        **kwargs: Detection eval kwargs forwarded from ``LocalEvalRunner``.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "evaluate_detection_from_config is not yet implemented. "
        "Install the [train] extra and provide a real DocTR eval wrapper."
    )


def evaluate_recognition_from_config(**kwargs: Any) -> RecognitionEvalResult:
    """Stub recognition eval entry point.

    This stub exists so ``LocalEvalRunner`` can be imported and tested in a
    torch-free environment.  Replace this function (via monkeypatch or direct
    assignment) with a real DocTR recognition eval wrapper once the eval
    implementation is available.

    Args:
        **kwargs: Recognition eval kwargs forwarded from ``LocalEvalRunner``.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "evaluate_recognition_from_config is not yet implemented. "
        "Install the [train] extra and provide a real DocTR eval wrapper."
    )


# ---------------------------------------------------------------------------
# LocalEvalRunner
# ---------------------------------------------------------------------------


def _build_detection_eval_kwargs(
    profile: str,
    config: DetectionEvalConfig,
) -> dict[str, Any]:
    """Build the kwargs dict for ``evaluate_detection_from_config``.

    Args:
        profile: Logical run identifier (currently passed through for
            logging; unused by the stub but reserved for the real impl).
        config: Typed detection eval configuration.

    Returns:
        Keyword-argument dict ready to pass to
        ``evaluate_detection_from_config(**kwargs)``.
    """
    kwargs: dict[str, Any] = config.model_dump()
    kwargs["profile"] = profile
    kwargs["val_path"] = str(config.val_path)
    kwargs["model_path"] = str(config.model_path)
    return kwargs


def _build_recognition_eval_kwargs(
    profile: str,
    config: RecognitionEvalConfig,
) -> dict[str, Any]:
    """Build the kwargs dict for ``evaluate_recognition_from_config``.

    Args:
        profile: Logical run identifier.
        config: Typed recognition eval configuration.

    Returns:
        Keyword-argument dict ready to pass to
        ``evaluate_recognition_from_config(**kwargs)``.
    """
    kwargs: dict[str, Any] = config.model_dump()
    kwargs["profile"] = profile
    kwargs["val_path"] = str(config.val_path)
    kwargs["model_path"] = str(config.model_path)
    return kwargs


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
