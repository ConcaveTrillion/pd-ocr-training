"""Tests for pd_ocr_training.local_eval — LocalEvalRunner.

All tests monkeypatch the underlying eval functions so no GPU is required.
Monkeypatch targets:
  - ``pd_ocr_training.local_eval.evaluate_detection_from_config``
  - ``pd_ocr_training.local_eval.evaluate_recognition_from_config``
"""

import time
from typing import Any
from unittest.mock import patch

import pytest

from pd_ocr_training.local_eval import LocalEvalRunner
from pd_ocr_training.protocols import (
    DetectionEvalConfig,
    DetectionEvalResult,
    EvalSlice,
    IEvalRunner,
    RecognitionEvalConfig,
    RecognitionEvalResult,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_eval_runner_satisfies_protocol() -> None:
    """LocalEvalRunner is an instance of IEvalRunner (runtime_checkable)."""
    assert isinstance(LocalEvalRunner(), IEvalRunner)


# ---------------------------------------------------------------------------
# Recognition eval — happy path
# ---------------------------------------------------------------------------


def _make_recog_result(**kwargs: Any) -> RecognitionEvalResult:
    defaults: dict[str, Any] = {
        "cer": 0.04,
        "wer": 0.07,
        "exact_match_rate": 0.83,
        "sample_count": 100,
        "excluded_count": 2,
        "duration_seconds": 5.0,
        "slices": [],
    }
    defaults.update(kwargs)
    return RecognitionEvalResult(**defaults)


def _make_det_result(**kwargs: Any) -> DetectionEvalResult:
    defaults: dict[str, Any] = {
        "precision": 0.92,
        "recall": 0.88,
        "f1": 0.90,
        "iou_50": 0.85,
        "iou_50_95": 0.62,
        "sample_count": 200,
        "excluded_count": 0,
        "duration_seconds": 3.0,
        "slices": [],
    }
    defaults.update(kwargs)
    return DetectionEvalResult(**defaults)


def test_evaluate_recognition_returns_result() -> None:
    """evaluate_recognition returns a RecognitionEvalResult from the stub."""
    expected = _make_recog_result()

    with patch(
        "pd_ocr_training.local_eval.evaluate_recognition_from_config",
        return_value=expected,
    ):
        runner = LocalEvalRunner()
        cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/model.pt")
        result = runner.evaluate_recognition("test-run", cfg)

    assert isinstance(result, RecognitionEvalResult)
    assert result.cer == 0.04
    assert result.sample_count == 100


def test_evaluate_recognition_with_slices() -> None:
    """evaluate_recognition result preserves slices list."""
    slices = [EvalSlice(feature="bold", n_pos=10, n_neg=90, n_excluded=0)]
    expected = _make_recog_result(slices=slices)

    with patch(
        "pd_ocr_training.local_eval.evaluate_recognition_from_config",
        return_value=expected,
    ):
        runner = LocalEvalRunner()
        cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/model.pt")
        result = runner.evaluate_recognition("test-run", cfg)

    assert len(result.slices) == 1
    assert result.slices[0].feature == "bold"


def test_evaluate_recognition_passes_config_fields() -> None:
    """evaluate_recognition forwards config fields to the underlying function."""
    captured: dict[str, Any] = {}

    def fake_eval(**kwargs: Any) -> RecognitionEvalResult:
        captured.update(kwargs)
        return _make_recog_result()

    with patch(
        "pd_ocr_training.local_eval.evaluate_recognition_from_config",
        side_effect=fake_eval,
    ):
        runner = LocalEvalRunner()
        cfg = RecognitionEvalConfig(
            val_path="/tmp/val",
            model_path="/tmp/model.pt",
            arch="crnn_vgg16_bn",
            vocab="english",
            batch_size=32,
        )
        runner.evaluate_recognition("my-run", cfg)

    assert captured["val_path"] == "/tmp/val"
    assert captured["model_path"] == "/tmp/model.pt"
    assert captured["vocab"] == "english"
    assert captured["batch_size"] == 32


def test_evaluate_recognition_raises_on_error() -> None:
    """evaluate_recognition propagates exceptions from the underlying function."""
    with patch(
        "pd_ocr_training.local_eval.evaluate_recognition_from_config",
        side_effect=RuntimeError("model not found"),
    ):
        runner = LocalEvalRunner()
        cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/missing.pt")
        with pytest.raises(RuntimeError, match="model not found"):
            runner.evaluate_recognition("failing-run", cfg)


# ---------------------------------------------------------------------------
# Detection eval — happy path
# ---------------------------------------------------------------------------


def test_evaluate_detection_returns_result() -> None:
    """evaluate_detection returns a DetectionEvalResult from the stub."""
    expected = _make_det_result()

    with patch(
        "pd_ocr_training.local_eval.evaluate_detection_from_config",
        return_value=expected,
    ):
        runner = LocalEvalRunner()
        cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/tmp/model.pt")
        result = runner.evaluate_detection("test-run", cfg)

    assert isinstance(result, DetectionEvalResult)
    assert result.precision == 0.92
    assert result.sample_count == 200


def test_evaluate_detection_with_slices() -> None:
    """evaluate_detection result preserves slices list."""
    slices = [EvalSlice(feature="header", n_pos=5, n_neg=95, n_excluded=1)]
    expected = _make_det_result(slices=slices)

    with patch(
        "pd_ocr_training.local_eval.evaluate_detection_from_config",
        return_value=expected,
    ):
        runner = LocalEvalRunner()
        cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/tmp/model.pt")
        result = runner.evaluate_detection("test-run", cfg)

    assert len(result.slices) == 1
    assert result.slices[0].feature == "header"


def test_evaluate_detection_passes_config_fields() -> None:
    """evaluate_detection forwards config fields to the underlying function."""
    captured: dict[str, Any] = {}

    def fake_eval(**kwargs: Any) -> DetectionEvalResult:
        captured.update(kwargs)
        return _make_det_result()

    with patch(
        "pd_ocr_training.local_eval.evaluate_detection_from_config",
        side_effect=fake_eval,
    ):
        runner = LocalEvalRunner()
        cfg = DetectionEvalConfig(
            val_path="/tmp/val",
            model_path="/tmp/model.pt",
            arch="db_resnet50",
            batch_size=4,
        )
        runner.evaluate_detection("my-run", cfg)

    assert captured["val_path"] == "/tmp/val"
    assert captured["model_path"] == "/tmp/model.pt"
    assert captured["arch"] == "db_resnet50"
    assert captured["batch_size"] == 4


def test_evaluate_detection_raises_on_error() -> None:
    """evaluate_detection propagates exceptions from the underlying function."""
    with patch(
        "pd_ocr_training.local_eval.evaluate_detection_from_config",
        side_effect=ValueError("bad checkpoint"),
    ):
        runner = LocalEvalRunner()
        cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/bad.pt")
        with pytest.raises(ValueError, match="bad checkpoint"):
            runner.evaluate_detection("failing-run", cfg)


# ---------------------------------------------------------------------------
# duration_seconds is populated
# ---------------------------------------------------------------------------


def test_evaluate_recognition_duration_is_positive() -> None:
    """duration_seconds in the result is > 0 when the stub takes time."""

    def fake_eval(**kwargs: Any) -> RecognitionEvalResult:
        time.sleep(0.01)
        return _make_recog_result(duration_seconds=0.01)

    with patch(
        "pd_ocr_training.local_eval.evaluate_recognition_from_config",
        side_effect=fake_eval,
    ):
        runner = LocalEvalRunner()
        cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
        result = runner.evaluate_recognition("timed-run", cfg)

    assert result.duration_seconds > 0


# ---------------------------------------------------------------------------
# Torch-free import contract
# ---------------------------------------------------------------------------


def test_local_eval_runner_class_importable() -> None:
    """LocalEvalRunner is importable from pd_ocr_training (lazy export)."""
    import pd_ocr_training

    runner_cls = pd_ocr_training.LocalEvalRunner
    assert runner_cls is LocalEvalRunner
