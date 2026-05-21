# ADR: Add IEvalRunner -- sibling Protocol to ITrainingRunner

**Date:** 2026-05-21
**Status:** Accepted
**Issue:** ConcaveTrillion/pd-ocr-training#2

## Context

`ITrainingRunner` exposes only training entry points.  pd-ocr-trainer-spa M7
(models registry + eval) proved that the consumer needs an eval round-trip:
its `worker/evaluate.py` runs a single forward pass and returns overall metrics
plus a per-slice breakdown.  The production `_build_runner()` path raised a
clear not-implemented error because pd-ocr-training had no eval API.

## Decision

Add a sibling `IEvalRunner` Protocol alongside `ITrainingRunner`.  Keep
`ITrainingRunner` training-only (Single-Responsibility Principle).  Mirror the
multi-Protocol pattern from `pd-ocr-ops`
(`StageDispatcher` / `LongJobRunner`).

## Options considered

**Option A -- add `evaluate_detection` / `evaluate_recognition` to `ITrainingRunner`.**
Rejected: conflates two concerns (training is long-running + streaming; eval is
a single synchronous forward pass); forces all training stubs to implement eval
too.

**Option B (chosen) -- sibling `IEvalRunner` Protocol + `LocalEvalRunner` concrete.**
Keeps SRP; mirrors the workspace multi-Protocol idiom; allows training and eval
to be injected independently.

## Call model: synchronous, not streaming

Training returns `Iterator[TrainingEvent]` because it spans many epochs and
callers need to stream progress.  Eval is a single forward pass -- no epoch
loop, no progress stream needed.  `IEvalRunner` methods therefore return result
objects directly.  This avoids the thread-queue bridge machinery entirely.

## Result shape

Result objects carry fields aligned with the pd-ocr-trainer-spa M7 worker
shapes so the adapter mapping is trivial:

- **`RecognitionEvalResult`**: `cer`, `wer`, `exact_match_rate`, `slices`,
  `sample_count`, `excluded_count`, `duration_seconds`.
- **`DetectionEvalResult`**: `precision`, `recall`, `f1`, `iou_50`,
  `iou_50_95`, `slices`, `sample_count`, `excluded_count`, `duration_seconds`.
- **`EvalSlice`**: `feature`, `n_pos`, `n_neg`, `n_excluded`, `cer_pos`,
  `cer_neg`, `wer_pos`, `wer_neg`, `delta_cer`, `low_support`.

`slices: []` by default -- M7 keeps an empty list; M12/M13 will populate it.

## Error handling

Unlike `ITrainingRunner` (which wraps exceptions in `kind="error"` events),
`IEvalRunner` lets exceptions propagate directly.  The consumer decides how to
handle (log, surface in the API response, etc.).

## Torch-free contract

`LocalEvalRunner` imports only from `pd_ocr_training.protocols` (via
`TYPE_CHECKING`) -- no torch/DocTR at module import time.  The two stub entry
points (`evaluate_detection_from_config` / `evaluate_recognition_from_config`)
raise `NotImplementedError` as placeholders until the real DocTR eval wrappers
are implemented.  Tests monkeypatch these stubs.  The class is importable in
the base (torch-free) install, unlike `LocalTrainingRunner` which drags in
`detect.py` / `recog.py`.

## Consequences

- `pd-ocr-trainer-spa` M7 can now inject a real `IEvalRunner` instead of a
  stub; the adapter maps `LocalEvalRunner.evaluate_recognition(profile, cfg)`
  to its existing result schema.
- M12 (typeface classifier) and M13 (glyph eval slicing) can populate
  `result.slices` without any Protocol changes.
- The torch-free base install now exports all eval config / result models and
  `IEvalRunner`, enabling the SPA web process to type-check eval results
  without pulling in the training stack.
