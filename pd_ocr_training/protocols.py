"""Training and evaluation runner Protocols and supporting data models for pd-ocr-training.

Design rationale
----------------
The ``ITrainingRunner`` and ``IEvalRunner`` Protocols mirror the workspace idiom
established in ``pd-ocr-ops`` (see ``pd_ocr_ops.gpu.protocols``): a
``@runtime_checkable`` ``Protocol`` defines the contract; a ``Local*``
implementation ships separately so consumer apps depend only on the interface.

Two-Protocol design (ITrainingRunner + IEvalRunner)
----------------------------------------------------
Training and evaluation are separate concerns with different call shapes:

- ``ITrainingRunner`` returns ``Iterator[TrainingEvent]`` — training is a
  long-running job that streams progress events over many epochs.
- ``IEvalRunner`` returns a result object synchronously — eval is a single
  forward pass with no epoch loop, so the callback->iterator bridge is
  unnecessary overhead.

Keeping them separate follows the Single-Responsibility Principle and mirrors
the multi-Protocol pattern used in ``pd-ocr-ops``
(``StageDispatcher`` / ``LongJobRunner``).

Config-type decision
--------------------
The existing entry points in ``detect.py`` and ``recog.py`` already expose
stable, well-documented keyword-argument surfaces via ``detect_from_config``
and ``train_from_config``.  We capture these as typed pydantic ``BaseModel``
classes (``DetectionConfig`` / ``RecognitionConfig``) rather than raw
``dict[str, object]`` because:

1. Typed models give IDE completion and inline validation.
2. The parameter shapes are stable (no dynamic keys).
3. Pydantic v2 is already a workspace dependency (used throughout pd-ocr-ops).

The two configs are intentionally separate because the tasks have different
required parameters: detection uses ``rotation`` and ``input_size=1024``;
recognition uses ``vocab`` and ``input_size=32``.

Progress design
---------------
``ITrainingRunner`` methods return ``Iterator[TrainingEvent]`` so that callers
can consume events synchronously via a ``for`` loop.  ``LocalTrainingRunner``
(Task 6) will implement this by bridging the existing callback-style
``progress_hook`` into a thread-safe queue and draining it as a generator.

``from __future__ import annotations`` is present and safe here.  Structural
``runtime_checkable`` Protocol ``isinstance()`` checks inspect method *names*
only — not their annotations — so converting annotations to string-form via
the future import does not affect isinstance() behaviour.
"""

from __future__ import annotations

from pathlib import (
    Path,  # noqa: TC003 — keep Path importable at runtime; pydantic resolves the annotation at model-build time
)
from typing import TYPE_CHECKING, Literal, runtime_checkable

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Progress event
# ---------------------------------------------------------------------------


class TrainingEvent(BaseModel):
    """A single progress event emitted during a training run.

    Attributes:
        kind: Public/normalised event category — one of ``"log"``,
            ``"epoch"``, ``"metric"``, ``"done"``, or ``"error"``.  Raw
            internal events emitted by ``detect.py``/``recog.py`` progress
            hooks (e.g. ``"train_batch"``, ``"val_batch"``, ``"epoch_end"``)
            are translated into these kinds by the runner implementation
            (Task 6).
        message: Human-readable description of the event.
        progress: Optional normalized progress in ``[0.0, 1.0]``; present on
            ``"epoch"`` and ``"done"`` events.
        data: Optional structured payload (e.g. loss, lr, recall values).
    """

    kind: Literal["log", "epoch", "metric", "done", "error"]
    message: str
    progress: float | None = None
    data: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Typed training config models
# ---------------------------------------------------------------------------


class DetectionConfig(BaseModel):
    """Configuration for a detection training run.

    Maps directly onto the parameters of ``detect.detect_from_config``.

    Attributes:
        train_path: Path to the training data folder (must contain
            ``images/`` and ``labels.json``).
        val_path: Path to the validation data folder (same layout).
        arch: DocTR detection architecture name, e.g. ``"db_resnet50"``.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Initial learning rate.
        weight_decay: L2 weight-decay coefficient.
        optimizer: Optimizer name; one of ``"adam"`` or ``"adamw"``.
        scheduler: LR scheduler; one of ``"cosine"``, ``"onecycle"``,
            or ``"poly"``.
        input_size: Square input image size in pixels (height == width).
        rotation: Whether to train with rotated bounding-box polygons.
        workers: Number of DataLoader worker processes.
        amp: Enable PyTorch Automatic Mixed Precision.
        early_stop: Enable early stopping.
        early_stop_epochs: Patience (epochs without improvement) before
            stopping.
        early_stop_delta: Minimum improvement delta for early stopping.
        output_dir: Directory where model checkpoints are written.
        device: GPU device index; ``None`` selects the default device.
        pretrained: Initialise from pretrained weights before fine-tuning.
        name: Experiment name used as the checkpoint filename stem.
    """

    train_path: str | Path
    val_path: str | Path
    arch: str = "db_resnet50"
    epochs: int = 100
    batch_size: int = 2
    lr: float = 0.002
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "poly"
    input_size: int = 1024
    rotation: bool = False
    workers: int = 4
    amp: bool = False
    early_stop: bool = False
    early_stop_epochs: int = 5
    early_stop_delta: float = 0.01
    output_dir: str | Path = Field(default=".")
    device: int | None = None
    pretrained: bool = True
    name: str | None = None


class RecognitionConfig(BaseModel):
    """Configuration for a recognition training run.

    Maps directly onto the parameters of ``recog.train_from_config``.

    Attributes:
        train_path: Path to the training data folder.
        val_path: Path to the validation data folder.
        arch: DocTR recognition architecture name, e.g. ``"crnn_vgg16_bn"``.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Initial learning rate.
        weight_decay: L2 weight-decay coefficient.
        optimizer: Optimizer name; one of ``"adam"`` or ``"adamw"``.
        scheduler: LR scheduler; one of ``"cosine"``, ``"onecycle"``,
            or ``"poly"``.
        input_size: Input image height in pixels (width is ``4 * input_size``).
        vocab: Vocabulary name (e.g. ``"french"``, ``"english"``) or
            ``"CUSTOM:<chars>"`` for a custom character set.
        workers: Number of DataLoader worker processes.
        amp: Enable PyTorch Automatic Mixed Precision.
        early_stop: Enable early stopping.
        early_stop_epochs: Patience (epochs without improvement) before
            stopping.
        early_stop_delta: Minimum improvement delta for early stopping.
        output_dir: Directory where model checkpoints are written.
        device: GPU device index; ``None`` selects the default device.
        pretrained: Initialise from pretrained weights before fine-tuning.
        name: Experiment name used as the checkpoint filename stem.
    """

    train_path: str | Path
    val_path: str | Path
    arch: str = "crnn_vgg16_bn"
    epochs: int = 10
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "cosine"
    input_size: int = 32
    vocab: str = "french"
    workers: int = 4
    amp: bool = False
    early_stop: bool = False
    early_stop_epochs: int = 5
    early_stop_delta: float = 0.01
    output_dir: str | Path = Field(default=".")
    device: int | None = None
    pretrained: bool = True
    name: str | None = None


# ---------------------------------------------------------------------------
# ITrainingRunner Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ITrainingRunner(Protocol):
    """Contract for running DocTR detection and recognition training.

    Implementations yield ``TrainingEvent`` objects so callers can stream
    progress without polling.  Concrete implementations (e.g.
    ``LocalTrainingRunner``) bridge the existing ``progress_hook`` callback
    API into this iterator surface.

    Example::

        runner: ITrainingRunner = LocalTrainingRunner()
        cfg = DetectionConfig(train_path="...", val_path="...")
        for event in runner.train_detection("run-001", cfg):
            print(event.kind, event.message)
    """

    def train_detection(
        self,
        profile: str,
        config: DetectionConfig,
    ) -> Iterator[TrainingEvent]:
        """Run a detection training job and stream progress events.

        Args:
            profile: Logical identifier for this training run (used for
                logging and checkpoint naming when ``config.name`` is absent).
            config: Fully-specified detection training configuration.

        Yields:
            ``TrainingEvent`` objects during training; the final event has
            ``kind="done"`` on success or ``kind="error"`` on failure.
        """
        ...

    def train_recognition(
        self,
        profile: str,
        config: RecognitionConfig,
    ) -> Iterator[TrainingEvent]:
        """Run a recognition training job and stream progress events.

        Args:
            profile: Logical identifier for this training run.
            config: Fully-specified recognition training configuration.

        Yields:
            ``TrainingEvent`` objects during training; the final event has
            ``kind="done"`` on success or ``kind="error"`` on failure.
        """
        ...


# ---------------------------------------------------------------------------
# Glyph feature presence model
# ---------------------------------------------------------------------------


class GlyphFeatureSet(BaseModel):
    """Per-word glyph feature presence, decoupled from pd-book-tools.

    Carries only the three feature-presence facts that recognition eval needs.
    The caller (``pd-ocr-trainer-spa``) derives this from ``pd-book-tools``
    ``GlyphAnnotations``; ``pd-ocr-training`` never imports ``GlyphAnnotations``
    itself — that would add a heavy foundation-lib dependency edge.

    A JSON sidecar passed to :class:`RecognitionEvalConfig` is a single
    ``dict[str, GlyphFeatureSet]`` keyed by recognition crop id (the DocTR
    recognition val-set label key).

    Attributes:
        ligatures: Ligature kind strings present in this word, e.g.
            ``["fi", "long_st"]``.  Per-kind slicing is done on these values;
            they are never lumped into a single ``"ligatures-present"`` bucket.
        long_s: ``True`` when the word contains one or more long-s glyphs.
        swash: ``True`` when the word contains one or more swash glyphs.
    """

    ligatures: list[str] = []
    long_s: bool = False
    swash: bool = False


# ---------------------------------------------------------------------------
# Eval config models
# ---------------------------------------------------------------------------


class DetectionEvalConfig(BaseModel):
    """Configuration for a detection evaluation run.

    Attributes:
        val_path: Path to the validation data folder (must contain
            ``images/`` and ``labels.json``).
        model_path: Path to the trained model checkpoint file.
        arch: DocTR detection architecture name, e.g. ``"db_resnet50"``.
        batch_size: Evaluation batch size.
        input_size: Square input image size in pixels (height == width).
        rotation: Whether the model was trained with rotated bounding-box
            polygons (used to select the correct decode path).
        workers: Number of DataLoader worker processes.
        amp: Enable PyTorch Automatic Mixed Precision for inference.
        device: GPU device index; ``None`` selects the default device.
    """

    val_path: str | Path
    model_path: str | Path
    arch: str = "db_resnet50"
    batch_size: int = 2
    input_size: int = 1024
    rotation: bool = False
    workers: int = 4
    amp: bool = False
    device: int | None = None


class RecognitionEvalConfig(BaseModel):
    """Configuration for a recognition evaluation run.

    Attributes:
        val_path: Path to the validation data folder.
        model_path: Path to the trained model checkpoint file.
        arch: DocTR recognition architecture name, e.g. ``"crnn_vgg16_bn"``.
        batch_size: Evaluation batch size.
        input_size: Input image height in pixels (width is ``4 * input_size``).
        vocab: Vocabulary name (e.g. ``"french"``, ``"english"``) or
            ``"CUSTOM:<chars>"`` for a custom character set.
        workers: Number of DataLoader worker processes.
        amp: Enable PyTorch Automatic Mixed Precision for inference.
        device: GPU device index; ``None`` selects the default device.
        glyph_annotations_path: Optional path to a JSON sidecar file mapping
            recognition crop ids to :class:`GlyphFeatureSet` objects.  Required
            when ``slice_glyph_features`` is ``True``; ignored otherwise.
        slice_glyph_features: When ``True``, recognition eval emits per-feature
            :class:`EvalSlice` entries (``ligature:<kind>``, ``long_s``,
            ``swash``) in :attr:`RecognitionEvalResult.slices`.  Requires
            ``glyph_annotations_path`` to be set — a ``ValueError`` is raised at
            validation time when the flag is ``True`` but the path is ``None``.
    """

    val_path: str | Path
    model_path: str | Path
    arch: str = "crnn_vgg16_bn"
    batch_size: int = 64
    input_size: int = 32
    vocab: str = "french"
    workers: int = 4
    amp: bool = False
    device: int | None = None
    glyph_annotations_path: Path | None = None
    slice_glyph_features: bool = False

    @model_validator(mode="after")
    def _require_path_when_slicing(self) -> RecognitionEvalConfig:
        if self.slice_glyph_features and self.glyph_annotations_path is None:
            raise ValueError("glyph_annotations_path must be set when slice_glyph_features is True")
        return self


# ---------------------------------------------------------------------------
# Eval result models
# ---------------------------------------------------------------------------


class EvalSlice(BaseModel):
    """Per-feature slice of evaluation metrics.

    A slice breaks down overall metrics by a binary feature (e.g. italic,
    drop-cap, bold, header) to surface whether the model performs worse on
    documents with that feature present.  Downstream consumers (pd-ocr-trainer-
    spa M12 / M13) can render slices in a comparison table.

    Attributes:
        feature: Name of the binary feature this slice compares (e.g.
            ``"italic"``, ``"drop_cap"``).
        n_pos: Sample count where ``feature`` is present (positive class).
        n_neg: Sample count where ``feature`` is absent (negative class).
        n_excluded: Samples excluded from this slice (e.g. missing labels).
        cer_pos: Character Error Rate on the positive-feature subset.
        cer_neg: Character Error Rate on the negative-feature subset.
        wer_pos: Word Error Rate on the positive-feature subset.
        wer_neg: Word Error Rate on the negative-feature subset.
        delta_cer: ``cer_pos - cer_neg``; positive means feature hurts CER.
        delta_wer: ``wer_pos - wer_neg``; positive means feature hurts WER.
            ``None`` when either side is empty.  Mirrors ``delta_cer``.
        low_support: ``True`` when ``n_pos`` is below the support threshold
            and the delta should be interpreted with caution.
    """

    feature: str
    n_pos: int
    n_neg: int
    n_excluded: int = 0
    cer_pos: float | None = None
    cer_neg: float | None = None
    wer_pos: float | None = None
    wer_neg: float | None = None
    delta_cer: float | None = None
    delta_wer: float | None = None
    low_support: bool = False


class RecognitionEvalResult(BaseModel):
    """Overall + per-slice recognition evaluation results.

    Returned synchronously by ``IEvalRunner.evaluate_recognition``.  Field
    names and semantics are aligned with the pd-ocr-trainer-spa M7 worker
    so the adapter mapping is trivial.

    Attributes:
        cer: Overall Character Error Rate (lower is better).
        wer: Overall Word Error Rate (lower is better).
        exact_match_rate: Fraction of samples where the full predicted
            string exactly matches the ground truth (higher is better).
        slices: Per-feature breakdown; empty list when no slice features
            are configured (M7 baseline) -- populated by M12/M13.
        sample_count: Number of samples evaluated (before exclusions).
        excluded_count: Samples dropped due to missing / malformed labels.
        duration_seconds: Wall-clock seconds for the evaluation pass.
    """

    cer: float
    wer: float
    exact_match_rate: float
    slices: list[EvalSlice] = Field(default_factory=list)
    sample_count: int
    excluded_count: int
    duration_seconds: float


class DetectionEvalResult(BaseModel):
    """Overall + per-slice detection evaluation results.

    Returned synchronously by ``IEvalRunner.evaluate_detection``.

    Attributes:
        precision: Precision at the IoU-50 threshold.
        recall: Recall at the IoU-50 threshold.
        f1: F1 score at the IoU-50 threshold.
        iou_50: Mean Average Precision at IoU >= 0.50.
        iou_50_95: Mean Average Precision averaged over IoU 0.50-0.95.
        slices: Per-feature breakdown; empty list by default.
        sample_count: Number of images evaluated.
        excluded_count: Images dropped due to missing / malformed labels.
        duration_seconds: Wall-clock seconds for the evaluation pass.
    """

    precision: float
    recall: float
    f1: float
    iou_50: float
    iou_50_95: float
    slices: list[EvalSlice] = Field(default_factory=list)
    sample_count: int
    excluded_count: int
    duration_seconds: float


# ---------------------------------------------------------------------------
# IEvalRunner Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IEvalRunner(Protocol):
    """Contract for running DocTR detection and recognition evaluation.

    Unlike ``ITrainingRunner``, eval is a single synchronous forward pass
    with no epoch loop, so methods return result objects directly rather
    than yielding event streams.

    Concrete implementations (e.g. ``LocalEvalRunner``) wrap the underlying
    DocTR eval entry points (``evaluate_detection_from_config`` /
    ``evaluate_recognition_from_config``).

    Example::

        runner: IEvalRunner = LocalEvalRunner()
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
        """Run a detection evaluation pass and return metrics.

        Args:
            profile: Logical identifier for this evaluation run (used for
                logging).
            config: Fully-specified detection evaluation configuration.

        Returns:
            ``DetectionEvalResult`` with overall precision/recall/F1/IoU
            metrics and an (initially empty) slices list.

        Raises:
            Any exception raised by the underlying eval function propagates
            directly to the caller (no error-event wrapping -- caller decides
            how to handle).
        """
        ...

    def evaluate_recognition(
        self,
        profile: str,
        config: RecognitionEvalConfig,
    ) -> RecognitionEvalResult:
        """Run a recognition evaluation pass and return metrics.

        Args:
            profile: Logical identifier for this evaluation run.
            config: Fully-specified recognition evaluation configuration.

        Returns:
            ``RecognitionEvalResult`` with overall CER/WER/exact-match
            metrics and an (initially empty) slices list.

        Raises:
            Any exception raised by the underlying eval function propagates
            directly to the caller.
        """
        ...
