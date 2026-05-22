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

Out of scope (issue #3 baseline)
--------------------------------
Per-slice ``EvalSlice`` breakdown -- ``slices`` stays ``[]``.  Populating it is
pd-ocr-trainer-spa M12/M13 work and needs a glyph-feature labelling scheme that
is not yet defined.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from pd_ocr_training.protocols import (
    DetectionEvalResult,
    RecognitionEvalResult,
)

if TYPE_CHECKING:
    from pd_ocr_training.protocols import (
        DetectionEvalConfig,
        RecognitionEvalConfig,
    )

# COCO-style IoU sweep used for the iou_50_95 metric.
_IOU_SWEEP: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


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


def _select_device(device_index: int | None) -> Any:
    """Resolve the torch device for an eval run.

    Args:
        device_index: Explicit CUDA device index, or ``None`` to auto-select
            (CUDA device 0 if available, else CPU).

    Returns:
        A ``torch.device``.

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
        return torch.device("cuda", device_index)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _run_recognition_inference(
    profile: str,
    config: RecognitionEvalConfig,
) -> dict[str, Any]:
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
    from torchvision.transforms.v2 import Normalize

    device = _select_device(config.device)
    vocab = (
        config.vocab.removeprefix("CUSTOM:")
        if config.vocab.startswith("CUSTOM:")
        else VOCABS.get(config.vocab, config.vocab)
    )

    val_set = RecognitionDataset(
        img_folder=os.path.join(str(config.val_path), "images"),
        labels_path=os.path.join(str(config.val_path), "labels.json"),
        img_transforms=t.Resize(
            (config.input_size, 4 * config.input_size), preserve_aspect_ratio=True
        ),
    )
    val_loader = DataLoader(
        # DocTR's RecognitionDataset is a torch-compatible Dataset at runtime;
        # its bundled stubs do not declare the torch Dataset base class.
        val_set,  # pyright: ignore[reportArgumentType]
        batch_size=config.batch_size,
        drop_last=False,
        num_workers=config.workers,
        sampler=SequentialSampler(val_set),  # pyright: ignore[reportArgumentType]
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set.collate_fn,
    )
    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    model = recognition.__dict__[config.arch](pretrained=False, vocab=vocab)
    model.from_pretrained(str(config.model_path))
    model = model.to(device)
    model.eval()

    predictions: list[str] = []
    ground_truths: list[str] = []
    exact = 0
    with torch.no_grad():
        for images, targets in val_loader:
            batch = batch_transforms(images.to(device))
            if config.amp:
                with torch.amp.autocast("cuda"):
                    out = model(batch, targets, return_preds=True)
            else:
                out = model(batch, targets, return_preds=True)
            words = [w for w, _ in out["preds"]] if out["preds"] else []
            for pred, gt in zip(words, targets, strict=False):
                predictions.append(pred)
                ground_truths.append(gt)
                if pred == gt:
                    exact += 1

    sample_count = len(ground_truths)
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "exact_match_rate": (exact / sample_count) if sample_count else 0.0,
        "sample_count": sample_count,
        "excluded_count": len(val_set) - sample_count,
    }


def _run_detection_inference(
    profile: str,
    config: DetectionEvalConfig,
) -> dict[str, Any]:
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
    from torchvision.transforms.v2 import Normalize

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
    val_set = DetectionDataset(
        img_folder=os.path.join(str(config.val_path), "images"),
        label_path=os.path.join(str(config.val_path), "labels.json"),
        sample_transforms=t.SampleCompose(sample_transforms),
        use_polygons=config.rotation,
    )
    val_loader = DataLoader(
        # DocTR's DetectionDataset is a torch-compatible Dataset at runtime;
        # its bundled stubs do not declare the torch Dataset base class.
        val_set,  # pyright: ignore[reportArgumentType]
        batch_size=config.batch_size,
        drop_last=False,
        num_workers=config.workers,
        sampler=SequentialSampler(val_set),  # pyright: ignore[reportArgumentType]
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set.collate_fn,
    )
    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))

    model = detection.__dict__[config.arch](
        pretrained=False,
        assume_straight_pages=not config.rotation,
        class_names=val_set.class_names,
    )
    model.from_pretrained(str(config.model_path))
    model = model.to(device)
    model.eval()

    metrics = {
        thresh: LocalizationConfusion(iou_thresh=thresh, use_polygons=config.rotation)
        for thresh in _IOU_SWEEP
    }
    with torch.no_grad():
        for images, targets in val_loader:
            batch = batch_transforms(images.to(device))
            if config.amp:
                with torch.amp.autocast("cuda"):
                    out = model(batch, targets, return_preds=True)
            else:
                out = model(batch, targets, return_preds=True)
            for target, loc_pred in zip(targets, out["preds"], strict=False):
                for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values(), strict=False):
                    boxes = boxes_pred
                    if isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] == 5:
                        boxes = boxes[:, :4]
                    preds = boxes if len(boxes) else np.zeros((0, 4))
                    for metric in metrics.values():
                        metric.update(gts=boxes_gt, preds=preds)

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
        ``exact_match_rate`` and an empty ``slices`` list.
    """
    start = time.monotonic()
    raw = _run_recognition_inference(profile, config)
    predictions: list[str] = raw["predictions"]
    ground_truths: list[str] = raw["ground_truths"]
    return RecognitionEvalResult(
        cer=_cer(predictions, ground_truths),
        wer=_wer(predictions, ground_truths),
        exact_match_rate=raw["exact_match_rate"],
        slices=[],
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
