"""Tests that local_eval stubs delegate to the real _eval_backend.

These verify the wiring added by issue #3: ``evaluate_detection_from_config`` /
``evaluate_recognition_from_config`` are no longer ``NotImplementedError``
stubs -- they delegate to ``_eval_backend.evaluate_*_impl``.  The backend impls
are monkeypatched so no GPU is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pd_ocr_training import local_eval
from pd_ocr_training.local_eval import LocalEvalRunner
from pd_ocr_training.protocols import (
    DetectionEvalConfig,
    DetectionEvalResult,
    RecognitionEvalConfig,
    RecognitionEvalResult,
)

if TYPE_CHECKING:
    import pytest


def test_recognition_from_config_delegates_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_recognition_from_config calls _eval_backend.evaluate_recognition_impl."""
    captured: dict[str, Any] = {}

    def fake_impl(profile: str, config: RecognitionEvalConfig) -> RecognitionEvalResult:
        captured["profile"] = profile
        captured["config"] = config
        return RecognitionEvalResult(
            cer=0.1,
            wer=0.2,
            exact_match_rate=0.7,
            sample_count=10,
            excluded_count=0,
            duration_seconds=1.0,
        )

    monkeypatch.setattr("pd_ocr_training._eval_backend.evaluate_recognition_impl", fake_impl)

    runner = LocalEvalRunner()
    cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
    result = runner.evaluate_recognition("real-run", cfg)

    assert isinstance(result, RecognitionEvalResult)
    assert result.cer == 0.1
    assert captured["profile"] == "real-run"
    assert captured["config"].val_path == "/tmp/val"


def test_detection_from_config_delegates_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_detection_from_config calls _eval_backend.evaluate_detection_impl."""
    captured: dict[str, Any] = {}

    def fake_impl(profile: str, config: DetectionEvalConfig) -> DetectionEvalResult:
        captured["profile"] = profile
        captured["config"] = config
        return DetectionEvalResult(
            precision=0.9,
            recall=0.85,
            f1=0.87,
            iou_50=0.8,
            iou_50_95=0.6,
            sample_count=15,
            excluded_count=1,
            duration_seconds=2.0,
        )

    monkeypatch.setattr("pd_ocr_training._eval_backend.evaluate_detection_impl", fake_impl)

    runner = LocalEvalRunner()
    cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
    result = runner.evaluate_detection("real-det-run", cfg)

    assert isinstance(result, DetectionEvalResult)
    assert result.precision == 0.9
    assert captured["profile"] == "real-det-run"
    assert captured["config"].arch == "db_resnet50"


def test_local_eval_module_is_torch_free() -> None:
    """local_eval.py must not import torch/doctr at module scope."""
    import inspect

    source = inspect.getsource(local_eval)
    # The torch-heavy import must be deferred inside the delegating functions,
    # never at module top level.
    module_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    joined = "\n".join(module_lines)
    assert "torch" not in joined
    assert "doctr" not in joined
    assert "_eval_backend" not in joined
