"""Tests for the eval Protocol and models in pd_ocr_training.protocols.

Covers:
- Eval config models (DetectionEvalConfig, RecognitionEvalConfig)
- Eval result models (EvalSlice, DetectionEvalResult, RecognitionEvalResult)
- IEvalRunner Protocol runtime_checkable behaviour
"""

from pd_ocr_training.protocols import (
    DetectionEvalConfig,
    DetectionEvalResult,
    EvalSlice,
    IEvalRunner,
    RecognitionEvalConfig,
    RecognitionEvalResult,
)

# ---------------------------------------------------------------------------
# EvalSlice model
# ---------------------------------------------------------------------------


def test_eval_slice_minimal() -> None:
    """EvalSlice can be constructed with required fields only."""
    s = EvalSlice(feature="drop_cap", n_pos=10, n_neg=90, n_excluded=0)
    assert s.feature == "drop_cap"
    assert s.n_pos == 10
    assert s.n_neg == 90
    assert s.n_excluded == 0
    assert s.cer_pos is None
    assert s.delta_cer is None
    assert s.low_support is False


def test_eval_slice_full() -> None:
    """EvalSlice accepts all optional metric fields."""
    s = EvalSlice(
        feature="italic",
        n_pos=50,
        n_neg=950,
        n_excluded=5,
        cer_pos=0.05,
        cer_neg=0.02,
        wer_pos=0.08,
        wer_neg=0.03,
        delta_cer=0.03,
        low_support=True,
    )
    assert s.cer_pos == 0.05
    assert s.delta_cer == 0.03
    assert s.low_support is True


# ---------------------------------------------------------------------------
# RecognitionEvalResult model
# ---------------------------------------------------------------------------


def test_recognition_eval_result_minimal() -> None:
    """RecognitionEvalResult can be constructed with required metric fields."""
    r = RecognitionEvalResult(
        cer=0.04,
        wer=0.07,
        exact_match_rate=0.83,
        sample_count=1000,
        excluded_count=5,
        duration_seconds=12.3,
    )
    assert r.cer == 0.04
    assert r.wer == 0.07
    assert r.exact_match_rate == 0.83
    assert r.slices == []


def test_recognition_eval_result_with_slices() -> None:
    """RecognitionEvalResult stores slices list."""
    slices = [EvalSlice(feature="bold", n_pos=20, n_neg=80, n_excluded=0)]
    r = RecognitionEvalResult(
        cer=0.04,
        wer=0.07,
        exact_match_rate=0.83,
        sample_count=100,
        excluded_count=0,
        duration_seconds=5.0,
        slices=slices,
    )
    assert len(r.slices) == 1
    assert r.slices[0].feature == "bold"


# ---------------------------------------------------------------------------
# DetectionEvalResult model
# ---------------------------------------------------------------------------


def test_detection_eval_result_minimal() -> None:
    """DetectionEvalResult can be constructed with required metric fields."""
    r = DetectionEvalResult(
        precision=0.92,
        recall=0.88,
        f1=0.90,
        iou_50=0.85,
        iou_50_95=0.62,
        sample_count=500,
        excluded_count=0,
        duration_seconds=8.7,
    )
    assert r.precision == 0.92
    assert r.f1 == 0.90
    assert r.slices == []


def test_detection_eval_result_with_slices() -> None:
    """DetectionEvalResult stores slices list."""
    slices = [EvalSlice(feature="header", n_pos=5, n_neg=95, n_excluded=1)]
    r = DetectionEvalResult(
        precision=0.9,
        recall=0.85,
        f1=0.875,
        iou_50=0.80,
        iou_50_95=0.55,
        sample_count=200,
        excluded_count=1,
        duration_seconds=3.1,
        slices=slices,
    )
    assert len(r.slices) == 1


# ---------------------------------------------------------------------------
# Eval config models
# ---------------------------------------------------------------------------


def test_recognition_eval_config_defaults() -> None:
    """RecognitionEvalConfig has sensible defaults."""
    cfg = RecognitionEvalConfig(val_path="/tmp/val", model_path="/tmp/model.pt")
    assert cfg.arch == "crnn_vgg16_bn"
    assert cfg.batch_size == 64
    assert cfg.vocab == "french"
    assert cfg.device is None


def test_detection_eval_config_defaults() -> None:
    """DetectionEvalConfig has sensible defaults."""
    cfg = DetectionEvalConfig(val_path="/tmp/val", model_path="/tmp/model.pt")
    assert cfg.arch == "db_resnet50"
    assert cfg.batch_size == 2
    assert cfg.device is None


# ---------------------------------------------------------------------------
# IEvalRunner Protocol — runtime_checkable
# ---------------------------------------------------------------------------


def test_eval_protocol_is_runtime_checkable_with_both_methods() -> None:
    """A class implementing both methods satisfies IEvalRunner."""

    class Stub:
        def evaluate_detection(
            self,
            profile: str,
            config: DetectionEvalConfig,
        ) -> DetectionEvalResult:
            return DetectionEvalResult(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                iou_50=1.0,
                iou_50_95=1.0,
                sample_count=0,
                excluded_count=0,
                duration_seconds=0.0,
            )

        def evaluate_recognition(
            self,
            profile: str,
            config: RecognitionEvalConfig,
        ) -> RecognitionEvalResult:
            return RecognitionEvalResult(
                cer=0.0,
                wer=0.0,
                exact_match_rate=1.0,
                sample_count=0,
                excluded_count=0,
                duration_seconds=0.0,
            )

    assert isinstance(Stub(), IEvalRunner)


def test_eval_protocol_missing_method_not_instance() -> None:
    """A class missing evaluate_recognition is NOT an IEvalRunner instance."""

    class IncompleteStub:
        def evaluate_detection(
            self,
            profile: str,
            config: DetectionEvalConfig,
        ) -> DetectionEvalResult:
            return DetectionEvalResult(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                iou_50=1.0,
                iou_50_95=1.0,
                sample_count=0,
                excluded_count=0,
                duration_seconds=0.0,
            )

    assert not isinstance(IncompleteStub(), IEvalRunner)


def test_eval_protocol_empty_class_not_instance() -> None:
    """An empty class is NOT an IEvalRunner instance."""

    class EmptyStub:
        pass

    assert not isinstance(EmptyStub(), IEvalRunner)
