# Plan: Real DocTR eval backend for LocalEvalRunner

**Issue:** ConcaveTrillion/pd-ocr-training#3
**Date:** 2026-05-22

## Goal

Replace the `NotImplementedError` stubs `evaluate_detection_from_config` /
`evaluate_recognition_from_config` in `pd_ocr_training/local_eval.py` with real
DocTR forward-pass eval backends, returning populated `DetectionEvalResult` /
`RecognitionEvalResult` models.

## Constraints (from issue #3 + ADR)

- GPU/torch work — lives behind the `[train]` extra.
- Keep the torch-free test seam: the `from_config` functions stay
  monkeypatchable; `local_eval.py` keeps no torch imports at module scope.
- `slices` stays `[]` for this baseline — per-slice breakdown is M12/M13 work.
  Issue #3's "per-slice breakdown" ask is explicitly downstream-consumer
  territory; the result models already default `slices=[]`. We do NOT invent a
  glyph-feature labelling scheme here.

## Design decisions

The torch eval implementation must NOT live in `local_eval.py` (which is
torch-free). New module `pd_ocr_training/_eval_backend.py` (private, `_`-prefix
→ already `D`-suppressed) houses the torch/DocTR code. `local_eval.py`'s stub
functions are re-pointed: when called, they import `_eval_backend` lazily and
delegate.

### Metric mapping (the honest part)

- **Detection** — DocTR `LocalizationConfusion.summary()` returns
  `(recall, precision, mean_iou)` at a single IoU threshold (default 0.5).
  Map: `recall`, `precision` direct; `f1 = 2pr/(p+r)` (0.0 when p+r==0);
  `iou_50 = mean_iou`; `iou_50_95 = mean_iou` with a docstring note that a
  true 0.50:0.95 sweep needs multiple `LocalizationConfusion` instances —
  deferred, tracked as a follow-up. Single-threshold reuse keeps the result
  populated and honest (documented, not silently faked).
- **Recognition** — DocTR `TextMatch.summary()` returns exact-match rates
  (`raw`, `caseless`, `unicase`, `unidecode`), NOT CER/WER. So:
  `exact_match_rate = raw`. For `cer`/`wer`, compute directly during the eval
  loop with a small Levenshtein helper over the (pred, gt) string pairs —
  char-level for CER, whitespace-token-level for WER. This is exact, not an
  approximation.

### Reuse, don't duplicate

Build the val dataset, DataLoader, batch transforms and model exactly as
`detect.py` / `recog.py` `main()` do in their `test_only` branch. Wrap the
relevant slice in helper builders inside `_eval_backend.py`. The existing
`detect.evaluate()` is reused directly for detection; recognition needs a
variant loop that also accumulates CER/WER, so a dedicated eval loop is
written in `_eval_backend.py`.

## Tasks

### Task 1 — `_eval_backend.py` + recognition eval

- TDD: `tests/test_eval_backend.py` — monkeypatch torch/doctr internals;
  assert `evaluate_recognition_impl` returns a `RecognitionEvalResult` with
  correct cer/wer/exact_match_rate/sample_count/excluded_count/duration.
- Implement `_levenshtein`, `_cer`, `_wer` pure helpers (fully unit-tested,
  GPU-free).
- Implement `evaluate_recognition_impl(profile, config) -> RecognitionEvalResult`.

### Task 2 — detection eval

- TDD: assert `evaluate_detection_impl` maps `LocalizationConfusion` summary
  into a `DetectionEvalResult` (precision/recall/f1/iou_50/iou_50_95).
- Implement `evaluate_detection_impl(profile, config) -> DetectionEvalResult`.

### Task 3 — wire `local_eval.py` stubs to the backend

- Re-point `evaluate_detection_from_config` / `evaluate_recognition_from_config`
  to lazy-import `_eval_backend` and delegate.
- Keep them monkeypatchable: existing `test_local_eval_runner.py` must still
  pass unchanged (it monkeypatches the module-level names).
- Update `local_eval.py` module docstring (remove "stub raises
  NotImplementedError" wording).

### Verification

- `make ci` green.
- Torch-free contract test still green (`local_eval.py` imports nothing heavy).
- Optional GPU smoke test if a tiny fixture dataset is feasible; otherwise the
  monkeypatched unit tests are the acceptance gate (GPU-free per workspace rule).

## Out of scope

- Per-slice `EvalSlice` population (M12/M13).
- True IoU 0.50:0.95 mAP sweep was in scope as a fallback but the final
  implementation does a true COCO sweep, so only per-slice work remains.
