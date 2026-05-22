"""Tests for pd_ocr_training._eval_backend — real DocTR eval backend.

The pure metric helpers (``_levenshtein``, ``_cer``, ``_wer``, ``_f1``) are
fully GPU-free and unit-tested directly.  The ``evaluate_*_impl`` entry points
are exercised with monkeypatched DocTR/torch internals so the suite stays
GPU-free per the workspace rule.
"""

from __future__ import annotations

from typing import Any

import pytest

from pd_ocr_training import _eval_backend
from pd_ocr_training.protocols import (
    DetectionEvalConfig,
    DetectionEvalResult,
    RecognitionEvalConfig,
    RecognitionEvalResult,
)

# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "abd", 1),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
    ],
)
def test_levenshtein(a: str, b: str, expected: int) -> None:
    """_levenshtein computes the edit distance between two sequences."""
    assert _eval_backend._levenshtein(a, b) == expected


def test_levenshtein_on_token_lists() -> None:
    """_levenshtein works on token (word) lists, not just strings."""
    assert _eval_backend._levenshtein(["the", "cat"], ["the", "dog"]) == 1
    assert _eval_backend._levenshtein(["a", "b", "c"], ["a", "c"]) == 1


def test_cer_perfect() -> None:
    """CER is 0.0 when every prediction exactly matches its ground truth."""
    assert _eval_backend._cer(["hello", "world"], ["hello", "world"]) == 0.0


def test_cer_partial() -> None:
    """CER is total-char-edits / total-gt-chars."""
    # "hello" vs "hallo" -> 1 edit; gt has 5 chars -> 0.2
    assert _eval_backend._cer(["hallo"], ["hello"]) == pytest.approx(0.2)


def test_cer_empty_ground_truth() -> None:
    """CER is 0.0 when there are no ground-truth characters at all."""
    assert _eval_backend._cer([""], [""]) == 0.0


def test_wer_perfect() -> None:
    """WER is 0.0 when every word matches."""
    assert _eval_backend._wer(["the cat sat"], ["the cat sat"]) == 0.0


def test_wer_partial() -> None:
    """WER is total-word-edits / total-gt-words."""
    # one word wrong out of three -> 1/3
    assert _eval_backend._wer(["the cat sat"], ["the dog sat"]) == pytest.approx(1 / 3)


def test_f1_normal() -> None:
    """_f1 is the harmonic mean of precision and recall."""
    assert _eval_backend._f1(0.5, 0.5) == pytest.approx(0.5)
    assert _eval_backend._f1(1.0, 1.0) == pytest.approx(1.0)


def test_f1_zero_denominator() -> None:
    """_f1 is 0.0 when precision and recall are both 0 (no divide-by-zero)."""
    assert _eval_backend._f1(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# evaluate_recognition_impl — monkeypatched DocTR internals
# ---------------------------------------------------------------------------


def test_evaluate_recognition_impl_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_recognition_impl returns a populated RecognitionEvalResult."""

    def fake_run(profile: str, config: RecognitionEvalConfig) -> dict[str, Any]:
        return {
            "predictions": ["hello", "wolrd"],
            "ground_truths": ["hello", "world"],
            "exact_match_rate": 0.5,
            "sample_count": 2,
            "excluded_count": 0,
        }

    monkeypatch.setattr(_eval_backend, "_run_recognition_inference", fake_run)

    cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
    result = _eval_backend.evaluate_recognition_impl("run-1", cfg)

    assert isinstance(result, RecognitionEvalResult)
    assert result.sample_count == 2
    assert result.excluded_count == 0
    assert result.exact_match_rate == 0.5
    # "wolrd" vs "world": 2 char edits over 10 gt chars -> 0.2
    assert result.cer == pytest.approx(0.2)
    assert result.wer == pytest.approx(0.5)  # one of two single-word strings wrong
    assert result.duration_seconds >= 0.0
    assert result.slices == []


def test_evaluate_recognition_impl_perfect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A perfect prediction set yields cer == wer == 0.0."""

    def fake_run(profile: str, config: RecognitionEvalConfig) -> dict[str, Any]:
        return {
            "predictions": ["alpha", "beta"],
            "ground_truths": ["alpha", "beta"],
            "exact_match_rate": 1.0,
            "sample_count": 2,
            "excluded_count": 1,
        }

    monkeypatch.setattr(_eval_backend, "_run_recognition_inference", fake_run)
    cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
    result = _eval_backend.evaluate_recognition_impl("run-2", cfg)

    assert result.cer == 0.0
    assert result.wer == 0.0
    assert result.excluded_count == 1


# ---------------------------------------------------------------------------
# evaluate_detection_impl — monkeypatched DocTR internals
# ---------------------------------------------------------------------------


def test_evaluate_detection_impl_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_detection_impl maps inference output into a DetectionEvalResult."""

    def fake_run(profile: str, config: DetectionEvalConfig) -> dict[str, Any]:
        return {
            "precision": 0.9,
            "recall": 0.8,
            "iou_50": 0.85,
            "iou_50_95": 0.6,
            "sample_count": 20,
            "excluded_count": 0,
        }

    monkeypatch.setattr(_eval_backend, "_run_detection_inference", fake_run)
    cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
    result = _eval_backend.evaluate_detection_impl("run-3", cfg)

    assert isinstance(result, DetectionEvalResult)
    assert result.precision == 0.9
    assert result.recall == 0.8
    # f1 is the harmonic mean of precision (0.9) and recall (0.8)
    assert result.f1 == pytest.approx(2 * 0.9 * 0.8 / (0.9 + 0.8))
    assert result.iou_50 == 0.85
    assert result.iou_50_95 == 0.6
    assert result.sample_count == 20
    assert result.duration_seconds >= 0.0
    assert result.slices == []


def test_evaluate_detection_impl_zero_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that detects nothing yields f1 == 0.0 without a divide error."""

    def fake_run(profile: str, config: DetectionEvalConfig) -> dict[str, Any]:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "iou_50": 0.0,
            "iou_50_95": 0.0,
            "sample_count": 5,
            "excluded_count": 2,
        }

    monkeypatch.setattr(_eval_backend, "_run_detection_inference", fake_run)
    cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/tmp/m.pt")
    result = _eval_backend.evaluate_detection_impl("run-4", cfg)

    assert result.f1 == 0.0
    assert result.excluded_count == 2
