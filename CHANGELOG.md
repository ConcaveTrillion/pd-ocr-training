# Changelog

## [Unreleased]

## [0.2.0] - 2026-05-22

### Added in 0.2.0

- `IEvalRunner` Protocol + `LocalEvalRunner` — synchronous DocTR eval wrapper
  with real detection/recognition backends (closes #2, #3).
- `DetectionEvalConfig`, `RecognitionEvalConfig`, `EvalSlice`,
  `DetectionEvalResult`, `RecognitionEvalResult` config and result models.
- Spec: glyph-feature eval slicing design (#5).
- Architecture overview doc (`docs/architecture/overview.md`).
- Lint-deviations documentation (`docs/process/lint-deviations.md`).

### Fixed in 0.2.0

- CI: resolve `pd-book-tools` from `pd-index-pip` (not editable path).
- CI: basedpyright `failOnWarnings` replaced with baseline file approach
  (grandfathers 118 pre-existing warnings via `.basedpyright/baseline.json`).
- 4 basedpyright type errors in test files (`test_local_runner.py` lines 445
  and 453; `test_protocols.py` line 34).

## [0.1.0] - 2026-05-21

### Added

- Initial extraction of DocTR training pipeline from `pd-ocr-trainer`.
- `detect.py` and `recog.py`: verbatim-moved DocTR detection and recognition
  training entry points.
- `datasets.py`: `ExportManager` for dataset export.
- `utils.py`: shared training utilities.
- `protocols.py`: `ITrainingRunner` Protocol + `TrainingEvent`,
  `DetectionConfig`, `RecognitionConfig` typed config models.
- `local.py`: `LocalTrainingRunner` — bridges callback-style training functions
  into `Iterator[TrainingEvent]` via a background thread and queue.
