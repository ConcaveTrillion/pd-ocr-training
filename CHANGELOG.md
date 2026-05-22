# Changelog

## [Unreleased]

## [0.2.1] - 2026-05-22

### Added in 0.2.1

- `GlyphFeatureSet` Pydantic model in `protocols.py` — torch-free, carries
  per-word glyph feature presence (`ligatures: list[str]`, `long_s: bool`,
  `swash: bool`); exported from package `__init__.py` (closes #7).
- `RecognitionEvalConfig.glyph_annotations_path` and
  `RecognitionEvalConfig.slice_glyph_features` optional fields for future
  glyph-feature eval slicing (closes #7).  A model validator raises
  `ValueError` when `slice_glyph_features=True` and `glyph_annotations_path`
  is `None`.
- `EvalSlice.delta_wer: float | None = None` — mirrors `delta_cer` (closes #7).
- `_run_recognition_inference` now threads per-sample crop ids alongside
  predictions and ground-truth strings (closes #8).  Crop ids are the join
  key for the glyph-feature sidecar in the upcoming #9 slice-emission work.

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
