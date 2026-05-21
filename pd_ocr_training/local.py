"""Concrete local implementation of ``ITrainingRunner``.

``LocalTrainingRunner`` bridges the callback-style training functions in
``detect.py`` / ``recog.py`` into the ``Iterator[TrainingEvent]`` surface
defined by ``ITrainingRunner``.

Thread model
------------
The training functions (``detect_from_config`` / ``train_from_config``) are
blocking and report progress via a synchronous ``progress_hook`` callback.
To expose them as generators without imposing async machinery on callers,
each ``train_*`` method:

1. Starts the training function in a daemon ``threading.Thread``.
2. Provides a ``progress_hook`` that translates raw event dicts and puts
   ``TrainingEvent`` objects onto a ``queue.Queue``.
3. Drains the queue on the calling thread, yielding each event.
4. After the worker thread finishes, yields a final ``"done"`` (success) or
   ``"error"`` (exception) event.

A sentinel value (``_DONE``) is put on the queue by the worker's finally-block
so the generator knows when to stop draining.

Raw-event-kind → TrainingEvent-kind mapping
--------------------------------------------
The raw ``event`` keys emitted by ``detect.py`` / ``recog.py`` progress hooks
are translated to the public ``Literal`` kinds as follows:

+------------------+-----------+--------------------------------------------+
| Raw event key    | Kind      | Notes                                      |
+==================+===========+============================================+
| ``"log"``        | ``"log"`` | Passthrough; ``message`` field preserved.  |
+------------------+-----------+--------------------------------------------+
| ``"train_batch"``| ``"metric"`` | Carries ``loss``, ``lr``, ``batch``,    |
|                  |           | ``total_batches`` in ``data``.             |
+------------------+-----------+--------------------------------------------+
| ``"val_batch"``  | ``"metric"`` | Carries ``loss``, ``batch``,            |
|                  |           | ``total_batches`` in ``data``.             |
+------------------+-----------+--------------------------------------------+
| ``"epoch_end"``  | ``"epoch"``  | ``progress`` = ``epoch / total_epochs``;|
|                  |           | ``data`` carries all epoch-end fields.     |
+------------------+-----------+--------------------------------------------+
| *anything else*  | ``"log"`` | Unknown raw kinds are logged verbatim.     |
+------------------+-----------+--------------------------------------------+
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import TYPE_CHECKING, Any

from pd_ocr_training.detect import detect_from_config
from pd_ocr_training.protocols import (
    DetectionConfig,
    RecognitionConfig,
    TrainingEvent,
)
from pd_ocr_training.recog import train_from_config

if TYPE_CHECKING:
    from collections.abc import Iterator

# Sentinel placed on the queue by the worker thread to signal completion.
_DONE = object()


def _translate_event(raw: dict[str, Any]) -> TrainingEvent:
    """Translate a raw progress-hook dict into a typed ``TrainingEvent``.

    Args:
        raw: Dict emitted by ``_emit_progress`` in ``detect.py`` / ``recog.py``.
             Must contain an ``"event"`` key with the raw kind string.

    Returns:
        A ``TrainingEvent`` with the translated ``kind`` and appropriate
        ``message``, ``progress``, and ``data`` fields.
    """
    raw_kind: str = raw.get("event", "")

    if raw_kind == "log":
        return TrainingEvent(
            kind="log",
            message=str(raw.get("message", "")),
            data=None,
        )

    if raw_kind == "train_batch":
        batch: int = int(raw.get("batch", 0))
        total: int = int(raw.get("total_batches", 0))
        return TrainingEvent(
            kind="metric",
            message=f"train batch {batch}/{total}",
            data={k: v for k, v in raw.items() if k != "event"},
        )

    if raw_kind == "val_batch":
        batch = int(raw.get("batch", 0))
        total = int(raw.get("total_batches", 0))
        return TrainingEvent(
            kind="metric",
            message=f"val batch {batch}/{total}",
            data={k: v for k, v in raw.items() if k != "event"},
        )

    if raw_kind == "epoch_end":
        epoch: int = int(raw.get("epoch", 0))
        total_epochs: int = int(raw.get("total_epochs", 1))
        progress: float = epoch / total_epochs if total_epochs > 0 else 0.0
        return TrainingEvent(
            kind="epoch",
            message=f"epoch {epoch}/{total_epochs}",
            progress=progress,
            data={k: v for k, v in raw.items() if k != "event"},
        )

    # Unknown raw kind — surface as a log event for observability.
    return TrainingEvent(
        kind="log",
        message=f"[{raw_kind}] {raw}",
        data={k: v for k, v in raw.items() if k != "event"},
    )


def _run_in_thread(
    fn: Any,
    kwargs: dict[str, Any],
    event_queue: queue.Queue[object],
) -> None:
    """Launch ``fn(**kwargs)`` in a background thread.

    ``kwargs`` must include a ``progress_hook`` key pointing to a callable
    that enqueues translated ``TrainingEvent`` objects.  When the function
    finishes (normally or via exception) the sentinel ``_DONE`` is placed
    on the queue so the generator can stop draining.

    Args:
        fn: The training function to call (``detect_from_config`` or
            ``train_from_config``).
        kwargs: Keyword arguments to pass to ``fn``.  The ``progress_hook``
            key will be present; it is the hook that enqueues events.
        event_queue: Queue shared with the generator; worker puts
            ``TrainingEvent`` or ``_DONE`` sentinel here.
    """
    exc_info: BaseException | None = None
    try:
        fn(**kwargs)
    except BaseException as exc:  # noqa: BLE001 — must capture *all* exceptions from worker thread
        exc_info = exc
    finally:
        event_queue.put(exc_info if exc_info is not None else _DONE)


class LocalTrainingRunner:
    """Concrete ``ITrainingRunner`` that runs training locally.

    Detection training delegates to ``detect.detect_from_config``; recognition
    training delegates to ``recog.train_from_config``.  Both are blocking
    functions that report progress via a synchronous callback; this class
    bridges them into ``Iterator[TrainingEvent]`` using a background thread and
    a thread-safe queue.

    Each ``train_*`` call creates a fresh queue and worker thread, so multiple
    instances (or sequential calls on the same instance) are fully independent.

    Example::

        runner = LocalTrainingRunner()
        cfg = DetectionConfig(train_path="data/train", val_path="data/val")
        for event in runner.train_detection("my-run", cfg):
            print(event.kind, event.message)
    """

    def train_detection(
        self,
        profile: str,
        config: DetectionConfig,
    ) -> Iterator[TrainingEvent]:
        """Run a detection training job locally and stream progress events.

        Delegates to ``detect.detect_from_config``, running it in a background
        thread.  Progress hook events are translated (see module docstring for
        the raw→kind mapping) and yielded in order.  Finishes with a
        ``kind="done"`` event on success or ``kind="error"`` on failure; the
        iterator never raises.

        Args:
            profile: Logical run identifier; used as the experiment ``name``
                when ``config.name`` is ``None``.
            config: Fully-specified detection training configuration.

        Yields:
            ``TrainingEvent`` objects; the final event has ``kind="done"``
            (success) or ``kind="error"`` (failure).
        """
        event_queue: queue.Queue[object] = queue.Queue()

        def hook(raw: dict[str, Any]) -> None:
            event_queue.put(_translate_event(raw))

        name = config.name if config.name is not None else profile
        kwargs: dict[str, Any] = {
            "train_path": config.train_path,
            "val_path": config.val_path,
            "arch": config.arch,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "input_size": config.input_size,
            "rotation": config.rotation,
            "workers": config.workers,
            "amp": config.amp,
            "early_stop": config.early_stop,
            "early_stop_epochs": config.early_stop_epochs,
            "early_stop_delta": config.early_stop_delta,
            "output_dir": str(config.output_dir),
            "device": config.device,
            "pretrained": config.pretrained,
            "name": name,
            "progress_hook": hook,
        }

        worker = threading.Thread(
            target=_run_in_thread,
            args=(detect_from_config, kwargs, event_queue),
            daemon=True,
        )
        worker.start()

        yield from _drain_queue(event_queue, worker)

    def train_recognition(
        self,
        profile: str,
        config: RecognitionConfig,
    ) -> Iterator[TrainingEvent]:
        """Run a recognition training job locally and stream progress events.

        Delegates to ``recog.train_from_config``, running it in a background
        thread.  See ``train_detection`` for the threading and event-translation
        model.

        Args:
            profile: Logical run identifier; used as the experiment ``name``
                when ``config.name`` is ``None``.
            config: Fully-specified recognition training configuration.

        Yields:
            ``TrainingEvent`` objects; the final event has ``kind="done"``
            (success) or ``kind="error"`` (failure).
        """
        event_queue: queue.Queue[object] = queue.Queue()

        def hook(raw: dict[str, Any]) -> None:
            event_queue.put(_translate_event(raw))

        name = config.name if config.name is not None else profile
        kwargs: dict[str, Any] = {
            "train_path": config.train_path,
            "val_path": config.val_path,
            "arch": config.arch,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "input_size": config.input_size,
            "vocab": config.vocab,
            "workers": config.workers,
            "amp": config.amp,
            "early_stop": config.early_stop,
            "early_stop_epochs": config.early_stop_epochs,
            "early_stop_delta": config.early_stop_delta,
            "output_dir": str(config.output_dir),
            "device": config.device,
            "pretrained": config.pretrained,
            "name": name,
            "progress_hook": hook,
        }

        worker = threading.Thread(
            target=_run_in_thread,
            args=(train_from_config, kwargs, event_queue),
            daemon=True,
        )
        worker.start()

        yield from _drain_queue(event_queue, worker)


def _drain_queue(
    event_queue: queue.Queue[object],
    worker: threading.Thread,
) -> Iterator[TrainingEvent]:
    """Drain *event_queue* until the worker sentinel arrives, then yield a final event.

    Args:
        event_queue: Queue populated by the worker thread.  Contains
            ``TrainingEvent`` objects interspersed with a single terminal
            value: either ``_DONE`` (clean exit) or a ``BaseException``
            instance (worker raised).
        worker: The background thread running the training function.  Joined
            after the sentinel is received to ensure clean teardown.

    Yields:
        ``TrainingEvent`` objects from the queue, followed by a final
        ``kind="done"`` or ``kind="error"`` event.
    """
    while True:
        item = event_queue.get()
        if item is _DONE:
            worker.join()
            yield TrainingEvent(kind="done", message="Training completed successfully.")
            return
        if isinstance(item, BaseException):
            worker.join()
            tb = "".join(traceback.format_exception(type(item), item, item.__traceback__))
            yield TrainingEvent(
                kind="error",
                message=f"{type(item).__name__}: {item}\n{tb}".strip(),
            )
            return
        if isinstance(item, TrainingEvent):
            yield item
