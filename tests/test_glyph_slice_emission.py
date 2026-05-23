"""Tests for per-feature glyph slice emission in evaluate_recognition_impl.

Issue #9: when slice_glyph_features=True, evaluate_recognition_impl must load
the JSON sidecar and emit one EvalSlice per feature (ligature:<kind>, long_s,
swash) with correct positive/negative/excluded counts and delta fields.

All tests monkeypatch _run_recognition_inference so no GPU is required.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pd_ocr_training import _eval_backend
from pd_ocr_training.protocols import (
    EvalSlice,
    GlyphFeatureSet,
    RecognitionEvalConfig,
    RecognitionEvalResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sidecar(tmp: Path, data: dict[str, dict[str, Any]]) -> Path:
    """Write a glyph-feature sidecar JSON and return the path."""
    p = tmp / "glyph_features.json"
    p.write_text(json.dumps(data))
    return p


def _make_fake_run(
    predictions: list[str],
    ground_truths: list[str],
    crop_ids: list[str],
) -> Any:
    """Return a monkeypatch-ready replacement for _run_recognition_inference."""

    def fake_run(profile: str, config: RecognitionEvalConfig) -> dict[str, Any]:
        return {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "crop_ids": crop_ids,
            "exact_match_rate": 1.0 if predictions == ground_truths else 0.0,
            "sample_count": len(predictions),
            "excluded_count": 0,
        }

    return fake_run


# ---------------------------------------------------------------------------
# No-slicing path (backward compat)
# ---------------------------------------------------------------------------


def test_no_slicing_slices_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With slice_glyph_features=False (default), slices is always []."""
    sidecar = _write_sidecar(
        tmp_path, {"img.png": {"ligatures": ["fi"], "long_s": False, "swash": False}}
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["hello"], ["hello"], ["img.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=False,
        # sidecar path supplied but slicing disabled
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("no-slice", cfg)
    assert result.slices == []


# ---------------------------------------------------------------------------
# Slice emission — long_s and swash
# ---------------------------------------------------------------------------


def test_long_s_slice_emitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """EvalSlice for long_s is emitted when slice_glyph_features=True."""
    # 2 samples: one has long_s, one does not
    sidecar = _write_sidecar(
        tmp_path,
        {
            "img1.png": {"ligatures": [], "long_s": True, "swash": False},
            "img2.png": {"ligatures": [], "long_s": False, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["hello", "world"], ["hello", "world"], ["img1.png", "img2.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("long-s", cfg)

    features = {s.feature: s for s in result.slices}
    assert "long_s" in features
    sl = features["long_s"]
    assert sl.n_pos == 1
    assert sl.n_neg == 1
    assert sl.n_excluded == 0
    assert sl.low_support is True  # n_pos < 30


def test_swash_slice_emitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """EvalSlice for swash is emitted when swash words are present."""
    sidecar = _write_sidecar(
        tmp_path,
        {
            "a.png": {"ligatures": [], "long_s": False, "swash": True},
            "b.png": {"ligatures": [], "long_s": False, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["alpha", "beta"], ["alpha", "beta"], ["a.png", "b.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("swash", cfg)
    features = {s.feature: s for s in result.slices}
    assert "swash" in features
    sl = features["swash"]
    assert sl.n_pos == 1
    assert sl.n_neg == 1


# ---------------------------------------------------------------------------
# Ligature slicing — per-kind, never lumped
# ---------------------------------------------------------------------------


def test_ligature_slices_per_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each distinct ligature kind gets its own EvalSlice (never lumped)."""
    sidecar = _write_sidecar(
        tmp_path,
        {
            "fi_word.png": {"ligatures": ["fi"], "long_s": False, "swash": False},
            "fl_word.png": {"ligatures": ["fl"], "long_s": False, "swash": False},
            "both.png": {"ligatures": ["fi", "fl"], "long_s": False, "swash": False},
            "none.png": {"ligatures": [], "long_s": False, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(
            ["w1", "w2", "w3", "w4"],
            ["w1", "w2", "w3", "w4"],
            ["fi_word.png", "fl_word.png", "both.png", "none.png"],
        ),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("ligatures", cfg)
    features = {s.feature: s for s in result.slices}

    # fi: positive = fi_word + both = 2; negative = fl_word + none = 2
    assert "ligature:fi" in features
    fi = features["ligature:fi"]
    assert fi.n_pos == 2
    assert fi.n_neg == 2
    assert fi.n_excluded == 0

    # fl: positive = fl_word + both = 2; negative = fi_word + none = 2
    assert "ligature:fl" in features
    fl = features["ligature:fl"]
    assert fl.n_pos == 2
    assert fl.n_neg == 2

    # No "ligature" lumped bucket
    assert "ligature" not in features


def test_ligature_kinds_never_lumped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A single 'ligature' bucket must NOT appear — each kind is its own feature."""
    sidecar = _write_sidecar(
        tmp_path,
        {
            "x.png": {"ligatures": ["fi", "ct"], "long_s": False, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["x"], ["x"], ["x.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("no-lump", cfg)
    feature_names = [s.feature for s in result.slices]
    assert "ligature" not in feature_names
    assert "ligature:fi" in feature_names
    assert "ligature:ct" in feature_names


# ---------------------------------------------------------------------------
# Excluded samples (missing crop id in sidecar)
# ---------------------------------------------------------------------------


def test_excluded_samples_not_in_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Samples whose crop id is absent from the sidecar are counted as excluded."""
    # Only img1.png is in sidecar; img2.png is absent → excluded
    sidecar = _write_sidecar(
        tmp_path,
        {
            "img1.png": {"ligatures": [], "long_s": True, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["hello", "world"], ["hello", "world"], ["img1.png", "img2.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("excluded", cfg)
    features = {s.feature: s for s in result.slices}
    sl = features["long_s"]
    # img1 → positive, img2 → excluded (not negative)
    assert sl.n_pos == 1
    assert sl.n_neg == 0
    assert sl.n_excluded == 1


def test_sidecar_crop_id_absent_from_valset_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crop id in the sidecar but absent from val set is silently ignored."""
    sidecar = _write_sidecar(
        tmp_path,
        {
            "img1.png": {"ligatures": [], "long_s": True, "swash": False},
            "ghost.png": {"ligatures": [], "long_s": True, "swash": False},  # not in val set
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["hello"], ["hello"], ["img1.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    # Should not raise; ghost.png simply doesn't contribute
    result = _eval_backend.evaluate_recognition_impl("ghost", cfg)
    features = {s.feature: s for s in result.slices}
    sl = features["long_s"]
    assert sl.n_pos == 1
    assert sl.n_neg == 0
    assert sl.n_excluded == 0


# ---------------------------------------------------------------------------
# CER / WER per slice
# ---------------------------------------------------------------------------


def test_cer_wer_pos_neg_computed_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cer_pos/wer_pos are computed only over the positive set; neg is separate."""
    # positive sample: prediction error ("helo" vs "hello" → 1 char edit)
    # negative sample: perfect ("world" == "world")
    sidecar = _write_sidecar(
        tmp_path,
        {
            "pos.png": {"ligatures": [], "long_s": True, "swash": False},
            "neg.png": {"ligatures": [], "long_s": False, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["helo", "world"], ["hello", "world"], ["pos.png", "neg.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("metrics", cfg)
    features = {s.feature: s for s in result.slices}
    sl = features["long_s"]
    assert sl.cer_pos == pytest.approx(0.2)  # 1 edit / 5 chars
    assert sl.cer_neg == pytest.approx(0.0)  # perfect
    assert sl.delta_cer == pytest.approx(0.2)  # cer_pos - cer_neg


def test_delta_wer_populated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """delta_wer = wer_pos - wer_neg; both present when both sides are non-empty."""
    sidecar = _write_sidecar(
        tmp_path,
        {
            "p.png": {"ligatures": [], "long_s": True, "swash": False},
            "n.png": {"ligatures": [], "long_s": False, "swash": False},
        },
    )
    # pos: "the cat" vs "the bat" → 1 word wrong / 2 → wer=0.5
    # neg: "hello" == "hello" → wer=0.0
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["the cat", "hello"], ["the bat", "hello"], ["p.png", "n.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("wer-delta", cfg)
    features = {s.feature: s for s in result.slices}
    sl = features["long_s"]
    assert sl.wer_pos == pytest.approx(0.5)
    assert sl.wer_neg == pytest.approx(0.0)
    assert sl.delta_wer == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# delta_cer / delta_wer is None when a side is empty
# ---------------------------------------------------------------------------


def test_delta_none_when_neg_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """delta_cer and delta_wer are None when the negative set is empty."""
    # All samples are positive (long_s=True); no negative exists
    sidecar = _write_sidecar(
        tmp_path,
        {
            "a.png": {"ligatures": [], "long_s": True, "swash": False},
            "b.png": {"ligatures": [], "long_s": True, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["x", "y"], ["x", "y"], ["a.png", "b.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("no-neg", cfg)
    features = {s.feature: s for s in result.slices}
    sl = features["long_s"]
    assert sl.n_neg == 0
    assert sl.delta_cer is None
    assert sl.delta_wer is None


def test_delta_none_when_pos_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """delta_cer and delta_wer are None when the positive set is empty."""
    sidecar = _write_sidecar(
        tmp_path,
        {
            "a.png": {"ligatures": [], "long_s": False, "swash": False},
            "b.png": {"ligatures": [], "long_s": False, "swash": False},
        },
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["x", "y"], ["x", "y"], ["a.png", "b.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("no-pos", cfg)
    features = {s.feature: s for s in result.slices}
    sl = features["long_s"]
    assert sl.n_pos == 0
    assert sl.delta_cer is None
    assert sl.delta_wer is None


# ---------------------------------------------------------------------------
# low_support threshold
# ---------------------------------------------------------------------------


def test_low_support_true_when_n_pos_below_30(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """low_support is True exactly when n_pos < 30."""
    # n_pos = 1 → low_support=True
    sidecar = _write_sidecar(
        tmp_path,
        {"only.png": {"ligatures": [], "long_s": True, "swash": False}},
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["x"], ["x"], ["only.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("low-support", cfg)
    features = {s.feature: s for s in result.slices}
    assert features["long_s"].low_support is True


def test_low_support_false_when_n_pos_ge_30(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """low_support is False when n_pos >= 30."""
    # 30 positive samples
    n = 30
    crop_ids = [f"pos_{i}.png" for i in range(n)]
    sidecar_data = {cid: {"ligatures": [], "long_s": True, "swash": False} for cid in crop_ids}
    sidecar = _write_sidecar(tmp_path, sidecar_data)
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["x"] * n, ["x"] * n, crop_ids),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("high-support", cfg)
    features = {s.feature: s for s in result.slices}
    assert features["long_s"].low_support is False


# ---------------------------------------------------------------------------
# Feature universe — long_s and swash always present in universe even if zero
# ---------------------------------------------------------------------------


def test_long_s_and_swash_always_in_universe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """long_s and swash EvalSlices are emitted even when all samples have them absent."""
    sidecar = _write_sidecar(
        tmp_path,
        {"a.png": {"ligatures": [], "long_s": False, "swash": False}},
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["x"], ["x"], ["a.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("always-emit", cfg)
    feature_names = {s.feature for s in result.slices}
    assert "long_s" in feature_names
    assert "swash" in feature_names


# ---------------------------------------------------------------------------
# Return type is still RecognitionEvalResult
# ---------------------------------------------------------------------------


def test_result_is_recognition_eval_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """evaluate_recognition_impl always returns RecognitionEvalResult."""
    sidecar = _write_sidecar(
        tmp_path,
        {"x.png": {"ligatures": ["fi"], "long_s": False, "swash": False}},
    )
    monkeypatch.setattr(
        _eval_backend,
        "_run_recognition_inference",
        _make_fake_run(["fi"], ["fi"], ["x.png"]),
    )
    cfg = RecognitionEvalConfig(
        val_path="/tmp/val",
        model_path="/tmp/m.pt",
        slice_glyph_features=True,
        glyph_annotations_path=sidecar,
    )
    result = _eval_backend.evaluate_recognition_impl("type-check", cfg)
    assert isinstance(result, RecognitionEvalResult)
    assert isinstance(result.slices, list)
    assert all(isinstance(s, EvalSlice) for s in result.slices)


# ---------------------------------------------------------------------------
# GlyphFeatureSet round-trips through JSON sidecar
# ---------------------------------------------------------------------------


def test_glyph_feature_set_round_trips(tmp_path: Path) -> None:
    """GlyphFeatureSet serializes and deserializes correctly via JSON."""
    original = GlyphFeatureSet(ligatures=["fi", "fl"], long_s=True, swash=False)
    as_dict = original.model_dump()
    restored = GlyphFeatureSet.model_validate(as_dict)
    assert restored.ligatures == ["fi", "fl"]
    assert restored.long_s is True
    assert restored.swash is False
