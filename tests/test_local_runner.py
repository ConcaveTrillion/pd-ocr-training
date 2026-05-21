"""Tests for pd_ocr_training.local — LocalTrainingRunner.

All tests monkeypatch the real training functions so no GPU is required.
Monkeypatch targets:
  - ``pd_ocr_training.local.detect_from_config``  (imported into local.py)
  - ``pd_ocr_training.local.train_from_config``   (imported into local.py)
"""

import threading
from collections.abc import Callable
from typing import Any

import pytest

from pd_ocr_training.local import LocalTrainingRunner
from pd_ocr_training.protocols import (
    DetectionConfig,
    ITrainingRunner,
    RecognitionConfig,
    TrainingEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_noop_detect(*, emit_events: list[dict[str, Any]] | None = None) -> Callable[..., None]:
    """Return a fake detect_from_config that optionally fires progress_hook events."""

    def fake_detect_from_config(*_args: Any, progress_hook: Callable[[dict[str, Any]], None] | None = None, **_kwargs: Any) -> None:
        if progress_hook is not None and emit_events:
            for ev in emit_events:
                progress_hook(ev)

    return fake_detect_from_config


def _make_noop_recog(*, emit_events: list[dict[str, Any]] | None = None) -> Callable[..., None]:
    """Return a fake train_from_config that optionally fires progress_hook events."""

    def fake_train_from_config(*_args: Any, progress_hook: Callable[[dict[str, Any]], None] | None = None, **_kwargs: Any) -> None:
        if progress_hook is not None and emit_events:
            for ev in emit_events:
                progress_hook(ev)

    return fake_train_from_config


def _make_raising_detect(exc: Exception) -> Callable[..., None]:
    """Return a fake detect_from_config that raises the given exception."""

    def fake(*_args: Any, **_kwargs: Any) -> None:
        raise exc

    return fake


def _make_raising_recog(exc: Exception) -> Callable[..., None]:
    """Return a fake train_from_config that raises the given exception."""

    def fake(*_args: Any, **_kwargs: Any) -> None:
        raise exc

    return fake


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_runner_satisfies_protocol() -> None:
    """LocalTrainingRunner is an instance of ITrainingRunner (runtime_checkable)."""
    assert isinstance(LocalTrainingRunner(), ITrainingRunner)


# ---------------------------------------------------------------------------
# train_detection — basic contract
# ---------------------------------------------------------------------------


def test_train_detection_yields_done_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """train_detection yields at least one event with kind='done'."""
    monkeypatch.setattr("pd_ocr_training.local.detect_from_config", _make_noop_detect())
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("demo", cfg))

    assert any(e.kind == "done" for e in events)
    assert all(isinstance(e, TrainingEvent) for e in events)


def test_train_detection_final_event_is_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last event yielded by train_detection has kind='done' on success."""
    monkeypatch.setattr("pd_ocr_training.local.detect_from_config", _make_noop_detect())
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("run-001", cfg))

    assert events[-1].kind == "done"


# ---------------------------------------------------------------------------
# train_recognition — basic contract
# ---------------------------------------------------------------------------


def test_train_recognition_yields_done_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """train_recognition yields at least one event with kind='done'."""
    monkeypatch.setattr("pd_ocr_training.local.train_from_config", _make_noop_recog())
    cfg = RecognitionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_recognition("demo", cfg))

    assert any(e.kind == "done" for e in events)
    assert all(isinstance(e, TrainingEvent) for e in events)


def test_train_recognition_final_event_is_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last event yielded by train_recognition has kind='done' on success."""
    monkeypatch.setattr("pd_ocr_training.local.train_from_config", _make_noop_recog())
    cfg = RecognitionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_recognition("run-001", cfg))

    assert events[-1].kind == "done"


# ---------------------------------------------------------------------------
# Event forwarding — progress_hook events are translated and forwarded
# ---------------------------------------------------------------------------


def test_train_detection_forwards_log_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """'log' events from the hook are forwarded as kind='log' TrainingEvents."""
    monkeypatch.setattr(
        "pd_ocr_training.local.detect_from_config",
        _make_noop_detect(emit_events=[{"event": "log", "message": "starting up"}]),
    )
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("fwd-test", cfg))
    log_events = [e for e in events if e.kind == "log"]

    assert len(log_events) >= 1


def test_train_detection_forwards_train_batch_as_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """'train_batch' hook events are forwarded as kind='metric' TrainingEvents."""
    monkeypatch.setattr(
        "pd_ocr_training.local.detect_from_config",
        _make_noop_detect(emit_events=[
            {"event": "train_batch", "loss": 0.5, "lr": 0.001, "batch": 1, "total_batches": 10},
        ]),
    )
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("batch-test", cfg))
    metric_events = [e for e in events if e.kind == "metric"]

    assert len(metric_events) >= 1
    assert metric_events[0].data is not None


def test_train_detection_forwards_val_batch_as_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """'val_batch' hook events are forwarded as kind='metric' TrainingEvents."""
    monkeypatch.setattr(
        "pd_ocr_training.local.detect_from_config",
        _make_noop_detect(emit_events=[
            {"event": "val_batch", "loss": 0.3, "batch": 1, "total_batches": 5},
        ]),
    )
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("val-test", cfg))
    metric_events = [e for e in events if e.kind == "metric"]

    assert len(metric_events) >= 1


def test_train_detection_forwards_epoch_end_as_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """'epoch_end' hook events are forwarded as kind='epoch' TrainingEvents."""
    monkeypatch.setattr(
        "pd_ocr_training.local.detect_from_config",
        _make_noop_detect(emit_events=[
            {"event": "epoch_end", "epoch": 1, "total_epochs": 10,
             "train_loss": 0.4, "val_loss": 0.35, "lr": 0.001},
        ]),
    )
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("epoch-test", cfg))
    epoch_events = [e for e in events if e.kind == "epoch"]

    assert len(epoch_events) >= 1
    # epoch events carry progress
    assert epoch_events[0].progress is not None


def test_train_recognition_forwards_epoch_end_as_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """'epoch_end' hook events for recognition are forwarded as kind='epoch'."""
    monkeypatch.setattr(
        "pd_ocr_training.local.train_from_config",
        _make_noop_recog(emit_events=[
            {"event": "epoch_end", "epoch": 2, "total_epochs": 10,
             "train_loss": 0.6, "val_loss": 0.5, "lr": 0.001},
        ]),
    )
    cfg = RecognitionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_recognition("recog-epoch", cfg))
    epoch_events = [e for e in events if e.kind == "epoch"]

    assert len(epoch_events) >= 1


# ---------------------------------------------------------------------------
# Error surfacing — exception in worker yields kind='error', does not raise
# ---------------------------------------------------------------------------


def test_train_detection_exception_yields_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the training function is surfaced as kind='error'."""
    monkeypatch.setattr(
        "pd_ocr_training.local.detect_from_config",
        _make_raising_detect(RuntimeError("GPU exploded")),
    )
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    # Must NOT raise — exception is surfaced as an event
    events = list(LocalTrainingRunner().train_detection("err-test", cfg))
    error_events = [e for e in events if e.kind == "error"]

    assert len(error_events) == 1
    assert "GPU exploded" in error_events[0].message


def test_train_recognition_exception_yields_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the recognition training function yields kind='error'."""
    monkeypatch.setattr(
        "pd_ocr_training.local.train_from_config",
        _make_raising_recog(ValueError("bad vocab")),
    )
    cfg = RecognitionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_recognition("err-recog", cfg))
    error_events = [e for e in events if e.kind == "error"]

    assert len(error_events) == 1
    assert "bad vocab" in error_events[0].message


def test_train_detection_no_done_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the worker raises, the final event is 'error', not 'done'."""
    monkeypatch.setattr(
        "pd_ocr_training.local.detect_from_config",
        _make_raising_detect(RuntimeError("oops")),
    )
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    events = list(LocalTrainingRunner().train_detection("err-final", cfg))

    assert events[-1].kind == "error"
    assert not any(e.kind == "done" for e in events)


# ---------------------------------------------------------------------------
# Thread safety — multiple independent instances do not share state
# ---------------------------------------------------------------------------


def test_two_runners_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two LocalTrainingRunner instances running concurrently don't share state."""
    monkeypatch.setattr("pd_ocr_training.local.detect_from_config", _make_noop_detect())
    cfg = DetectionConfig(train_path="/tmp/train", val_path="/tmp/val")

    results: dict[int, list[TrainingEvent]] = {}
    errors: list[Exception] = []

    def run(idx: int) -> None:
        try:
            results[idx] = list(LocalTrainingRunner().train_detection(f"run-{idx}", cfg))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for idx in range(3):
        assert any(e.kind == "done" for e in results[idx])
