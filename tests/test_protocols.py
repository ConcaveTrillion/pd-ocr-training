"""Tests for pd_ocr_training.protocols — ITrainingRunner structural protocol."""

from collections.abc import Iterator

from pd_ocr_training.protocols import (
    DetectionConfig,
    ITrainingRunner,
    RecognitionConfig,
    TrainingEvent,
)

# ---------------------------------------------------------------------------
# TrainingEvent model
# ---------------------------------------------------------------------------


def test_training_event_minimal() -> None:
    """TrainingEvent can be constructed with only required fields."""
    event = TrainingEvent(kind="log", message="hello")
    assert event.kind == "log"
    assert event.message == "hello"
    assert event.progress is None


def test_training_event_with_progress() -> None:
    """TrainingEvent accepts an optional progress float."""
    event = TrainingEvent(kind="epoch", message="epoch 1/10", progress=0.1)
    assert event.progress == 0.1


def test_training_event_data_round_trips() -> None:
    """TrainingEvent.data round-trips a plain dict payload."""
    payload: dict[str, object] = {"loss": 0.42, "lr": 0.001}
    event = TrainingEvent(kind="metric", message="metrics", data=payload)
    assert event.data == payload


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


def test_detection_config_defaults() -> None:
    """DetectionConfig has sensible defaults for required data paths."""
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")
    assert cfg.arch == "db_resnet50"
    assert cfg.epochs == 100


def test_recognition_config_defaults() -> None:
    """RecognitionConfig has sensible defaults for required data paths."""
    cfg = RecognitionConfig(train_path="/tmp/train", val_path="/tmp/val")
    assert cfg.arch == "crnn_vgg16_bn"
    assert cfg.epochs == 10
    assert cfg.vocab == "french"


# ---------------------------------------------------------------------------
# ITrainingRunner Protocol — runtime_checkable
# ---------------------------------------------------------------------------


def test_protocol_is_runtime_checkable_with_both_methods() -> None:
    """A class implementing both methods satisfies ITrainingRunner."""

    class Stub:
        def train_detection(
            self,
            profile: str,
            config: DetectionConfig,
        ) -> Iterator[TrainingEvent]: ...

        def train_recognition(
            self,
            profile: str,
            config: RecognitionConfig,
        ) -> Iterator[TrainingEvent]: ...

    assert isinstance(Stub(), ITrainingRunner)


def test_protocol_missing_method_not_instance() -> None:
    """A class missing train_recognition is NOT an ITrainingRunner instance."""

    class IncompleteStub:
        def train_detection(
            self,
            profile: str,
            config: DetectionConfig,
        ) -> Iterator[TrainingEvent]: ...

    assert not isinstance(IncompleteStub(), ITrainingRunner)


def test_protocol_missing_both_methods_not_instance() -> None:
    """An empty class is NOT an ITrainingRunner instance."""

    class EmptyStub:
        pass

    assert not isinstance(EmptyStub(), ITrainingRunner)
