"""Real DocTR forward-pass eval backend.

This module owns the torch/DocTR evaluation code.  It is imported lazily by
``local_eval.py`` only when an eval is actually run, so the base (torch-free)
install never pays the import cost.

Public entry points
-------------------
``evaluate_recognition_impl`` / ``evaluate_detection_impl`` build a DocTR model
and validation DataLoader, run a single forward pass over the val set, and
return a populated :class:`RecognitionEvalResult` / :class:`DetectionEvalResult`.

The two ``_run_*_inference`` functions hold the torch-dependent forward-pass
code; ``test_eval_backend.py`` monkeypatches them so the metric-mapping logic
in the ``*_impl`` functions can be tested without a GPU.

Metric mapping notes
--------------------
* **Recognition.** DocTR's ``TextMatch`` metric reports *exact-match* rates
  (``raw`` / ``caseless`` / ``unicase``), not CER/WER.  ``raw`` is used as
  ``exact_match_rate``.  CER and WER are computed directly here from the
  collected (prediction, ground-truth) string pairs using an exact Levenshtein
  edit distance -- char-level for CER, whitespace-token-level for WER.
* **Detection.** DocTR's ``LocalizationConfusion`` reports
  ``(recall, precision, mean_iou)`` at a single IoU threshold.  ``iou_50`` uses
  a 0.50-threshold instance; ``iou_50_95`` averages ``mean_iou`` over the
  standard COCO sweep (0.50, 0.55, ..., 0.95).  ``f1`` is the harmonic mean of
  precision and recall.

Crop-id threading (issue #8)
----------------------------
``_run_recognition_inference`` now threads each sample's crop id (the DocTR
recognition val-set label key — the per-crop filename / relative path) alongside
its prediction and ground-truth strings. The crop id is the join key into the
glyph-feature sidecar loaded by ``evaluate_recognition_impl`` when
``config.slice_glyph_features`` is ``True``. Keying by crop id (not by
iteration index) is robust to any filtering or reordering of the val set.

Per-feature glyph slicing (issue #9)
-------------------------------------
When ``config.slice_glyph_features`` is ``True`` and
``config.glyph_annotations_path`` is set, ``evaluate_recognition_impl`` loads
the JSON sidecar into a ``dict[str, GlyphFeatureSet]`` and emits one
:class:`EvalSlice` per feature into ``RecognitionEvalResult.slices``.

Feature universe: ``{"long_s", "swash"}`` plus one ``"ligature:<kind>"`` entry
for every distinct ligature kind appearing in the sidecar.  Each feature's
positive / negative / excluded sets are built per the spec (§4):

- **positive** — crop id present in sidecar **and** feature is present.
- **negative** — crop id present in sidecar **and** feature is absent.
- **excluded** — crop id absent from sidecar.

``delta_cer = cer_pos - cer_neg``, ``delta_wer = wer_pos - wer_neg`` — both
``None`` when either side is empty.  ``low_support = (n_pos < 30)``.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, cast

from pdomain_ocr_training.protocols import (
    DetectionEvalResult,
    EvalSlice,
    GlyphFeatureSet,
    RecognitionEvalResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pdomain_ocr_training.protocols import (
        DetectionEvalConfig,
        RecognitionEvalConfig,
    )

# COCO-style IoU sweep used for the iou_50_95 metric.
_IOU_SWEEP: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

_DeviceSpec = str


class _RecognitionInferenceOutput(TypedDict):
    predictions: list[str]
    ground_truths: list[str]
    exact_match_rate: float
    sample_count: int
    excluded_count: int
    crop_ids: NotRequired[list[str]]


class _DetectionInferenceOutput(TypedDict):
    precision: float
    recall: float
    iou_50: float
    iou_50_95: float
    sample_count: int
    excluded_count: int


class _TensorLike(Protocol):
    def to(self, device: _DeviceSpec) -> object: ...


class _RecognitionDatasetLike(Protocol):
    data: list[tuple[object, str]]
    collate_fn: object

    def __len__(self) -> int: ...


class _DetectionDatasetLike(Protocol):
    class_names: list[str]
    collate_fn: object

    def __len__(self) -> int: ...


class _RecognitionModelOutput(TypedDict):
    preds: list[tuple[str, object]]


class _DetectionModelOutput(TypedDict):
    preds: list[Mapping[str, object]]


class _RecognitionModelLike(Protocol):
    def from_pretrained(self, path: str) -> None: ...

    def to(self, device: _DeviceSpec) -> _RecognitionModelLike: ...

    def eval(self) -> None: ...

    def __call__(
        self, batch: object, targets: Sequence[str], *, return_preds: bool
    ) -> _RecognitionModelOutput: ...


class _DetectionModelLike(Protocol):
    def from_pretrained(self, path: str) -> None: ...

    def to(self, device: _DeviceSpec) -> _DetectionModelLike: ...

    def eval(self) -> None: ...

    def __call__(
        self,
        batch: object,
        targets: Sequence[Mapping[str, object]],
        *,
        return_preds: bool,
    ) -> _DetectionModelOutput: ...


class _RecognitionFactory(Protocol):
    def __call__(self, *, pretrained: bool, vocab: str) -> _RecognitionModelLike: ...


class _DetectionFactory(Protocol):
    def __call__(
        self,
        *,
        pretrained: bool,
        assume_straight_pages: bool,
        class_names: list[str],
    ) -> _DetectionModelLike: ...


class _DataLoaderFactory(Protocol):
    def __call__(
        self,
        dataset: object,
        *,
        batch_size: int,
        drop_last: bool,
        num_workers: int,
        sampler: object,
        pin_memory: bool,
        collate_fn: object,
    ) -> Iterable[object]: ...


class _SequentialSamplerFactory(Protocol):
    def __call__(self, data_source: object) -> object: ...


class _LocalizationMetricLike(Protocol):
    def update(self, *, gts: object, preds: object) -> None: ...

    def summary(self) -> tuple[float | None, float | None, float | None]: ...


# ---------------------------------------------------------------------------
# Pure metric helpers (GPU-free, unit-tested directly)
# ---------------------------------------------------------------------------


def _levenshtein(a: str | list[str], b: str | list[str]) -> int:
    """Compute the Levenshtein edit distance between two sequences.

    Works on strings (char-level) or on lists of tokens (word-level).

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        The minimum number of single-element insertions, deletions, or
        substitutions needed to turn ``a`` into ``b``.
    """
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    previous = list(range(len_b + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def _cer(predictions: list[str], ground_truths: list[str]) -> float:
    """Compute the corpus Character Error Rate.

    Args:
        predictions: Predicted strings.
        ground_truths: Ground-truth strings (same length / order as predictions).

    Returns:
        Total character edits divided by total ground-truth characters.
        ``0.0`` when there are no ground-truth characters.
    """
    total_edits = 0
    total_chars = 0
    for pred, gt in zip(predictions, ground_truths, strict=True):
        total_edits += _levenshtein(pred, gt)
        total_chars += len(gt)
    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


def _wer(predictions: list[str], ground_truths: list[str]) -> float:
    """Compute the corpus Word Error Rate.

    Words are whitespace-delimited tokens.

    Args:
        predictions: Predicted strings.
        ground_truths: Ground-truth strings (same length / order as predictions).

    Returns:
        Total word edits divided by total ground-truth words.  ``0.0`` when
        there are no ground-truth words.
    """
    total_edits = 0
    total_words = 0
    for pred, gt in zip(predictions, ground_truths, strict=True):
        gt_words = gt.split()
        total_edits += _levenshtein(pred.split(), gt_words)
        total_words += len(gt_words)
    if total_words == 0:
        return 0.0
    return total_edits / total_words


def _f1(precision: float, recall: float) -> float:
    """Compute the F1 score (harmonic mean of precision and recall).

    Args:
        precision: Precision value in ``[0, 1]``.
        recall: Recall value in ``[0, 1]``.

    Returns:
        The F1 score, or ``0.0`` when precision and recall are both ``0``.
    """
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


# ---------------------------------------------------------------------------
# Torch-dependent inference (monkeypatched in tests)
# ---------------------------------------------------------------------------


def _select_device(device_index: int | None) -> _DeviceSpec:
    """Resolve the torch device for an eval run.

    Args:
        device_index: Explicit CUDA device index, or ``None`` to auto-select
            (CUDA device 0 if available, else CPU).

    Returns:
        A device string accepted by torch APIs (for example ``"cuda:0"``).

    Raises:
        RuntimeError: When an explicit index is given but no GPU is accessible
            or the index is out of range.
    """
    import torch

    if isinstance(device_index, int):
        if not torch.cuda.is_available():
            raise RuntimeError("A device index was given but no GPU is accessible.")
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid CUDA device index: {device_index}")
        return f"cuda:{device_index}"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _run_recognition_inference(
    _profile: str,
    config: RecognitionEvalConfig,
) -> _RecognitionInferenceOutput:
    """Run the recognition forward pass over the validation set.

    Builds the DocTR recognition model and validation DataLoader, restores the
    checkpoint, and collects predicted / ground-truth strings.

    Args:
        profile: Logical run identifier (used for logging only).
        config: Recognition eval configuration.

    Returns:
        A dict with keys ``predictions``, ``ground_truths``,
        ``exact_match_rate``, ``sample_count``, ``excluded_count``.
    """
    import torch
    from doctr import transforms as t
    from doctr.datasets import VOCABS, RecognitionDataset
    from doctr.models import recognition
    from torch.utils.data import DataLoader, SequentialSampler

    device = _select_device(config.device)
    vocab = (
        config.vocab.removeprefix("CUSTOM:")
        if config.vocab.startswith("CUSTOM:")
        else VOCABS.get(config.vocab, config.vocab)
    )

    val_set_raw = RecognitionDataset(
        img_folder=os.path.join(str(config.val_path), "images"),
        labels_path=os.path.join(str(config.val_path), "labels.json"),
        img_transforms=t.Resize(
            (config.input_size, 4 * config.input_size), preserve_aspect_ratio=True
        ),
    )
    val_set = cast(
        "_RecognitionDatasetLike",
        cast(
            "object",
            val_set_raw,
        ),
    )
    data_loader = cast("_DataLoaderFactory", cast("object", DataLoader))
    sequential_sampler = cast("_SequentialSamplerFactory", cast("object", SequentialSampler))
    val_loader = cast(
        "Iterable[tuple[_TensorLike, list[str]]]",
        data_loader(
            val_set_raw,
            batch_size=config.batch_size,
            drop_last=False,
            num_workers=config.workers,
            sampler=sequential_sampler(val_set_raw),
            pin_memory=torch.cuda.is_available(),
            collate_fn=val_set.collate_fn,
        ),
    )
    norm_mean = (0.694, 0.695, 0.693)
    norm_std = (0.299, 0.296, 0.301)

    model_factory = cast("_RecognitionFactory", recognition.__dict__[config.arch])
    model = model_factory(pretrained=False, vocab=vocab)
    model.from_pretrained(str(config.model_path))
    model = model.to(device)
    model.eval()

    # Build a flat list of crop ids (the val-set label key for each sample) in
    # SequentialSampler order.  DocTR's RecognitionDataset stores its entries as
    # a list of (img_path, label) tuples in val_set.data; the crop id is the
    # basename of the image path, which matches the key in ``labels.json``.
    # Using basename (not the full path) makes the id robust to where the
    # images/ folder is mounted, and matches the key format DocTR uses when it
    # builds the dataset from a flat labels.json.
    all_crop_ids: list[str] = [os.path.basename(str(img_path)) for img_path, _label in val_set.data]

    predictions: list[str] = []
    ground_truths: list[str] = []
    crop_ids: list[str] = []
    exact = 0
    sample_idx = 0
    with torch.no_grad():
        for images, targets in val_loader:
            image_tensor = cast("torch.Tensor", images.to(device))
            mean_tensor = image_tensor.new_tensor(norm_mean).view(1, 3, 1, 1)
            std_tensor = image_tensor.new_tensor(norm_std).view(1, 3, 1, 1)
            batch: object = (image_tensor - mean_tensor) / std_tensor
            if config.amp:
                with torch.autocast("cuda"):
                    out = model(batch, targets, return_preds=True)
            else:
                out = model(batch, targets, return_preds=True)
            words = [word for word, _score in out["preds"]]
            for pred, gt in zip(words, targets, strict=False):
                predictions.append(pred)
                ground_truths.append(gt)
                crop_ids.append(all_crop_ids[sample_idx])
                sample_idx += 1
                if pred == gt:
                    exact += 1

    sample_count = len(ground_truths)
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "crop_ids": crop_ids,
        "exact_match_rate": (exact / sample_count) if sample_count else 0.0,
        "sample_count": sample_count,
        "excluded_count": len(val_set) - sample_count,
    }


def _run_detection_inference(
    _profile: str,
    config: DetectionEvalConfig,
) -> _DetectionInferenceOutput:
    """Run the detection forward pass over the validation set.

    Builds the DocTR detection model and validation DataLoader, restores the
    checkpoint, and accumulates a ``LocalizationConfusion`` metric at each IoU
    threshold in the COCO sweep.

    Args:
        profile: Logical run identifier (used for logging only).
        config: Detection eval configuration.

    Returns:
        A dict with keys ``precision``, ``recall``, ``iou_50``, ``iou_50_95``,
        ``sample_count``, ``excluded_count``.
    """
    import numpy as np
    import torch
    from doctr import transforms as t
    from doctr.datasets import DetectionDataset
    from doctr.models import detection
    from doctr.utils.metrics import LocalizationConfusion
    from torch.utils.data import DataLoader, SequentialSampler

    device = _select_device(config.device)

    sample_transforms = (
        [
            t.Resize(
                (config.input_size, config.input_size),
                preserve_aspect_ratio=True,
                symmetric_pad=True,
            )
        ]
        if not config.rotation
        else [
            t.Resize(config.input_size, preserve_aspect_ratio=True),
            t.Resize(
                (config.input_size, config.input_size),
                preserve_aspect_ratio=True,
                symmetric_pad=True,
            ),
        ]
    )
    val_set_raw = DetectionDataset(
        img_folder=os.path.join(str(config.val_path), "images"),
        label_path=os.path.join(str(config.val_path), "labels.json"),
        sample_transforms=t.SampleCompose(sample_transforms),
        use_polygons=config.rotation,
    )
    val_set = cast(
        "_DetectionDatasetLike",
        cast(
            "object",
            val_set_raw,
        ),
    )
    data_loader = cast("_DataLoaderFactory", cast("object", DataLoader))
    sequential_sampler = cast("_SequentialSamplerFactory", cast("object", SequentialSampler))
    val_loader = cast(
        "Iterable[tuple[_TensorLike, list[Mapping[str, object]]]]",
        data_loader(
            val_set_raw,
            batch_size=config.batch_size,
            drop_last=False,
            num_workers=config.workers,
            sampler=sequential_sampler(val_set_raw),
            pin_memory=torch.cuda.is_available(),
            collate_fn=val_set.collate_fn,
        ),
    )
    norm_mean = (0.798, 0.785, 0.772)
    norm_std = (0.264, 0.2749, 0.287)

    model_factory = cast("_DetectionFactory", detection.__dict__[config.arch])
    model = model_factory(
        pretrained=False,
        assume_straight_pages=not config.rotation,
        class_names=val_set.class_names,
    )
    model.from_pretrained(str(config.model_path))
    model = model.to(device)
    model.eval()

    metrics: dict[float, _LocalizationMetricLike] = {
        thresh: cast(
            "_LocalizationMetricLike",
            cast(
                "object",
                LocalizationConfusion(iou_thresh=thresh, use_polygons=config.rotation),
            ),
        )
        for thresh in _IOU_SWEEP
    }
    with torch.no_grad():
        for images, targets in val_loader:
            image_tensor = cast("torch.Tensor", images.to(device))
            mean_tensor = image_tensor.new_tensor(norm_mean).view(1, 3, 1, 1)
            std_tensor = image_tensor.new_tensor(norm_std).view(1, 3, 1, 1)
            batch: object = (image_tensor - mean_tensor) / std_tensor
            if config.amp:
                with torch.autocast("cuda"):
                    out = model(batch, targets, return_preds=True)
            else:
                out = model(batch, targets, return_preds=True)
            for target, loc_pred in zip(targets, out["preds"], strict=False):
                for boxes_gt_obj, boxes_pred_obj in zip(
                    target.values(), loc_pred.values(), strict=False
                ):
                    boxes = cast(
                        "np.ndarray[tuple[int, ...], np.dtype[np.float64]]", boxes_pred_obj
                    )
                    if boxes.ndim == 2 and boxes.shape[1] == 5:
                        boxes = boxes[:, :4]
                    preds = boxes if len(boxes) else np.zeros((0, 4), dtype=np.float64)
                    for metric in metrics.values():
                        metric.update(gts=boxes_gt_obj, preds=preds)

    recall_50, precision_50, iou_50 = metrics[0.50].summary()
    mean_ious = [metrics[thresh].summary()[2] or 0.0 for thresh in _IOU_SWEEP]
    return {
        "precision": precision_50 or 0.0,
        "recall": recall_50 or 0.0,
        "iou_50": iou_50 or 0.0,
        "iou_50_95": sum(mean_ious) / len(mean_ious),
        "sample_count": len(val_set),
        "excluded_count": 0,
    }


# ---------------------------------------------------------------------------
# Per-feature glyph slice emission (issue #9)
# ---------------------------------------------------------------------------

_LOW_SUPPORT_THRESHOLD = 30


def _emit_glyph_slices(
    predictions: list[str],
    ground_truths: list[str],
    crop_ids: list[str],
    sidecar: dict[str, GlyphFeatureSet],
) -> list[EvalSlice]:
    """Build per-feature EvalSlice entries from glyph-feature sidecar data.

    Feature universe: ``{"long_s", "swash"}`` plus one ``"ligature:<kind>"``
    entry for every distinct ligature kind seen across all sidecar entries.
    ``long_s`` and ``swash`` are always emitted; ligature kinds are only emitted
    when at least one sidecar entry has that kind.

    Args:
        predictions: Predicted strings, parallel to ``crop_ids``.
        ground_truths: Ground-truth strings, parallel to ``crop_ids``.
        crop_ids: Recognition crop ids (val-set label keys), parallel to
            ``predictions`` and ``ground_truths``.
        sidecar: Parsed glyph-feature JSON — maps crop id →
            :class:`GlyphFeatureSet`.

    Returns:
        List of :class:`EvalSlice` — one per feature in the universe.
    """
    # Build the feature universe: fixed features + per-kind ligature features.
    ligature_kinds: set[str] = set()
    for feat in sidecar.values():
        ligature_kinds.update(feat.ligatures)
    features: list[str] = ["long_s", "swash", *sorted(f"ligature:{k}" for k in ligature_kinds)]

    # Build per-sample buckets: pos_idx / neg_idx / excl_idx for each feature.
    # A sample is "excluded" from ALL features when its crop id is absent from
    # the sidecar.
    slices: list[EvalSlice] = []
    for feature in features:
        pos_preds: list[str] = []
        pos_gts: list[str] = []
        neg_preds: list[str] = []
        neg_gts: list[str] = []
        n_excluded = 0

        for pred, gt, crop_id in zip(predictions, ground_truths, crop_ids, strict=True):
            if crop_id not in sidecar:
                n_excluded += 1
                continue
            feat_set = sidecar[crop_id]
            if feature == "long_s":
                present = feat_set.long_s
            elif feature == "swash":
                present = feat_set.swash
            else:
                kind = feature.removeprefix("ligature:")
                present = kind in feat_set.ligatures

            if present:
                pos_preds.append(pred)
                pos_gts.append(gt)
            else:
                neg_preds.append(pred)
                neg_gts.append(gt)

        n_pos = len(pos_preds)
        n_neg = len(neg_preds)

        cer_pos: float | None = _cer(pos_preds, pos_gts) if n_pos > 0 else None
        cer_neg: float | None = _cer(neg_preds, neg_gts) if n_neg > 0 else None
        wer_pos: float | None = _wer(pos_preds, pos_gts) if n_pos > 0 else None
        wer_neg: float | None = _wer(neg_preds, neg_gts) if n_neg > 0 else None

        delta_cer: float | None = (
            (cer_pos - cer_neg) if (cer_pos is not None and cer_neg is not None) else None
        )
        delta_wer: float | None = (
            (wer_pos - wer_neg) if (wer_pos is not None and wer_neg is not None) else None
        )

        slices.append(
            EvalSlice(
                feature=feature,
                n_pos=n_pos,
                n_neg=n_neg,
                n_excluded=n_excluded,
                cer_pos=cer_pos,
                cer_neg=cer_neg,
                wer_pos=wer_pos,
                wer_neg=wer_neg,
                delta_cer=delta_cer,
                delta_wer=delta_wer,
                low_support=n_pos < _LOW_SUPPORT_THRESHOLD,
            )
        )
    return slices


# ---------------------------------------------------------------------------
# Public eval entry points
# ---------------------------------------------------------------------------


def evaluate_recognition_impl(
    profile: str,
    config: RecognitionEvalConfig,
) -> RecognitionEvalResult:
    """Evaluate a recognition model and return populated metrics.

    Args:
        profile: Logical run identifier (used for logging).
        config: Fully-specified recognition eval configuration.

    Returns:
        A :class:`RecognitionEvalResult` with overall ``cer`` / ``wer`` /
        ``exact_match_rate``.  When ``config.slice_glyph_features`` is ``True``
        and ``config.glyph_annotations_path`` is set, ``slices`` is populated
        with per-feature :class:`EvalSlice` entries keyed by crop id.
        Otherwise ``slices`` is an empty list (backward-compatible default).
    """
    start = time.monotonic()
    raw = _run_recognition_inference(profile, config)
    predictions = raw["predictions"]
    ground_truths = raw["ground_truths"]
    # crop_ids are threaded through by _run_recognition_inference for glyph
    # slicing (#8).  They are parallel to predictions / ground_truths.
    crop_ids = raw.get("crop_ids", [])

    # Glyph slice emission (#9): load sidecar and emit per-feature EvalSlices.
    slices: list[EvalSlice] = []
    if config.slice_glyph_features and config.glyph_annotations_path is not None:
        # json.loads returns Any; cast to the expected mapping shape before
        # iterating so basedpyright does not emit reportAny / reportExplicitAny.
        sidecar_raw = cast(
            "dict[str, object]", json.loads(config.glyph_annotations_path.read_text())
        )
        sidecar: dict[str, GlyphFeatureSet] = {
            k: GlyphFeatureSet.model_validate(v) for k, v in sidecar_raw.items()
        }
        slices = _emit_glyph_slices(predictions, ground_truths, crop_ids, sidecar)

    return RecognitionEvalResult(
        cer=_cer(predictions, ground_truths),
        wer=_wer(predictions, ground_truths),
        exact_match_rate=raw["exact_match_rate"],
        slices=slices,
        sample_count=raw["sample_count"],
        excluded_count=raw["excluded_count"],
        duration_seconds=time.monotonic() - start,
    )


def evaluate_detection_impl(
    profile: str,
    config: DetectionEvalConfig,
) -> DetectionEvalResult:
    """Evaluate a detection model and return populated metrics.

    Args:
        profile: Logical run identifier (used for logging).
        config: Fully-specified detection eval configuration.

    Returns:
        A :class:`DetectionEvalResult` with ``precision`` / ``recall`` / ``f1``
        / ``iou_50`` / ``iou_50_95`` and an empty ``slices`` list.
    """
    start = time.monotonic()
    raw = _run_detection_inference(profile, config)
    return DetectionEvalResult(
        precision=raw["precision"],
        recall=raw["recall"],
        f1=_f1(raw["precision"], raw["recall"]),
        iou_50=raw["iou_50"],
        iou_50_95=raw["iou_50_95"],
        slices=[],
        sample_count=raw["sample_count"],
        excluded_count=raw["excluded_count"],
        duration_seconds=time.monotonic() - start,
    )
