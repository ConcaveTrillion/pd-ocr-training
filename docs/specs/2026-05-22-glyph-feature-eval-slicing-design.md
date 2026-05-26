# Glyph-feature eval slicing for recognition eval

> **Status**: Draft
> **Last updated**: 2026-05-22
> **Spec-Issue**: pdomain/pdomain-ocr-training#5

## TL;DR

Let `pdomain-ocr-training`'s recognition evaluation accept per-word glyph feature
data and emit per-feature accuracy slices (`ligature:<kind>`, `long_s`,
`swash`) in `RecognitionEvalResult.slices`. The feature data crosses the repo
boundary as a **plain serialized JSON sidecar** keyed by recognition crop id —
`pdomain-ocr-training` never imports `pdomain-book-tools`. This unblocks
`pdomain-ocr-trainer-spa` M13 (glyph-feature metric breakdown, trainer-spa spec
07 §4).

## Context

`pdomain-ocr-trainer-spa` M13 needs recognition eval metrics broken down by glyph
feature so a trainer can see whether the model regresses specifically on
ligatures, long-s, or swash glyphs. The slicing rules are fixed by
trainer-spa `specs/07-evaluation-and-metrics.md` §4:

- For each binary feature `f` in `{ligature:<kind>, long_s, swash}`:
  - `positive` = words where glyph data is present **and** `f` is present.
  - `negative` = words where glyph data is present **and** `f` is absent.
  - `excluded` = words where glyph data is absent.
- CER + WER computed per (positive, negative) set; excluded words are **never**
  in the denominator.
- Ligatures slice **per `kind`** — never a single lumped "ligatures-present"
  bucket.
- `low_support = (n_pos < 30)` — the row is still emitted; the SPA renders it
  greyed and tagged "low support".

The recognition eval surface today:

- `RecognitionEvalConfig` (`protocols.py:296`) — `val_path`, `model_path`,
  `arch`, `batch_size`, `input_size`, `vocab`, `workers`, `amp`, `device`.
- `EvalSlice` (`protocols.py:328`) — already a generic per-feature slice with
  `feature: str`, `n_pos`, `n_neg`, `n_excluded`, `cer_pos/neg`, `wer_pos/neg`,
  `delta_cer`, `low_support`. No `delta_wer`.
- `RecognitionEvalResult` (`protocols.py:363`) — already carries a `slices`
  field.
- `_run_recognition_inference` (`_eval_backend.py:193`) — the inference worker;
  currently threads only flat prediction/ground-truth strings.

The glyph data originates as `pdomain-book-tools` `GlyphAnnotations` (published in
pdomain-book-tools v0.13.0, post-#163). `pdomain-ocr-training` is a lean training library
with a torch-free protocol layer (`tests/test_torch_free_import.py` guards it).

## Constraints

- **No `pdomain-book-tools` dependency edge.** `pdomain-ocr-training` must not import
  `GlyphAnnotations` or add `pdomain-book-tools` to its dependency graph. The eval
  only needs feature-presence booleans, not the full annotation object.
- **Protocol layer stays torch-free.** New models live in `protocols.py` and
  must import without torch — `test_torch_free_import.py` must still pass.
- **Slicing rules are verbatim from trainer-spa spec 07 §4** — no deviation in
  positive/negative/excluded semantics, per-kind ligature slicing, or the
  `n_pos < 30` low-support threshold.
- **Backward compatible.** Both new config fields are optional and default to
  the no-slicing behavior; existing eval callers are unaffected.
- **Excluded words never enter a denominator** — CER/WER for a slice are
  computed only over its positive and negative sets.

## Decision

### 1. New lightweight feature model

Add to `protocols.py` a small torch-free Pydantic model carrying only
feature-presence data:

```python
class GlyphFeatureSet(BaseModel):
    """Per-word glyph feature presence, decoupled from pdomain-book-tools.

    The caller derives this from pdomain-book-tools GlyphAnnotations; pdomain-ocr-training
    never imports GlyphAnnotations itself.
    """
    ligatures: list[str] = []   # ligature kinds present, e.g. ["fi", "long_st"]
    long_s: bool = False
    swash: bool = False
```

The JSON sidecar is a single object `dict[str, GlyphFeatureSet]` — keys are
recognition crop ids (see §3).

### 2. Config surface

`RecognitionEvalConfig` gains two optional fields:

```python
glyph_annotations_path: Path | None = None  # JSON sidecar: crop-id -> GlyphFeatureSet
slice_glyph_features: bool = False          # gate
```

Glyph slicing runs only when `slice_glyph_features is True` **and**
`glyph_annotations_path` is set and readable. If `slice_glyph_features` is
`True` but `glyph_annotations_path` is `None`, eval raises a `ValueError` at
config-validation time (a Pydantic model validator) — a silent no-op would
hide a caller mistake.

### 3. Sample keying

`_run_recognition_inference` is extended to thread each sample's **crop id**
(the DocTR recognition val-set label key — the per-crop filename / relative
path) alongside its prediction and ground-truth strings. The crop id is the
join key into the loaded sidecar. Keying by crop id (not by iteration index)
is robust to filtering or reordering of the val set.

### 4. Slice emission

When glyph slicing is enabled, after inference completes:

1. Load and parse the sidecar into `dict[str, GlyphFeatureSet]`.
2. Build the feature universe: `{long_s, swash}` plus one `ligature:<kind>`
   entry for every distinct kind appearing across all sidecar entries.
3. For each feature `f`:
   - `positive` = samples whose crop id has a sidecar entry **and** `f` is
     present (for `ligature:<kind>`, `<kind>` is in `ligatures`; for `long_s` /
     `swash`, the matching bool is `True`).
   - `negative` = samples whose crop id has a sidecar entry **and** `f` is
     absent.
   - `excluded` = samples whose crop id has **no** sidecar entry.
   - Compute CER and WER over `positive` and over `negative` independently.
   - `delta_cer = cer_pos - cer_neg`; `delta_wer = wer_pos - wer_neg`
     (both `None` when either side is empty).
   - `low_support = n_pos < 30`.
4. Emit one `EvalSlice` per feature into `RecognitionEvalResult.slices`, with
   `feature` set to `"ligature:<kind>"`, `"long_s"`, or `"swash"`. Ligatures
   are emitted **per kind** — never lumped.

### 5. `EvalSlice` change

Keep the existing `EvalSlice` model. Add **one** field:

```python
delta_wer: float | None = None  # wer_pos - wer_neg; mirrors delta_cer
```

The generic `feature: str` field already accommodates the parameterized
`ligature:<kind>` form — no other shape change is needed.

## Contract / Acceptance

- `GlyphFeatureSet` exists in `protocols.py`, imports torch-free, round-trips
  through `model_validate` / `model_dump`.
- `RecognitionEvalConfig` has `glyph_annotations_path: Path | None = None` and
  `slice_glyph_features: bool = False`, both optional.
- `RecognitionEvalConfig` validation raises `ValueError` when
  `slice_glyph_features is True` and `glyph_annotations_path is None`.
- `EvalSlice` has a `delta_wer: float | None = None` field.
- With `slice_glyph_features=False` (default), `RecognitionEvalResult.slices`
  is unchanged from current behavior — existing eval tests still pass.
- With slicing enabled against a sidecar:
  - One `EvalSlice` is emitted per `ligature:<kind>` kind seen, plus one for
    `long_s` and one for `swash`.
  - Ligature kinds are never lumped into a single bucket.
  - `n_excluded` counts samples whose crop id is absent from the sidecar; those
    samples appear in no positive or negative set.
  - `cer_pos`/`cer_neg`/`wer_pos`/`wer_neg` are computed only over the slice's
    own positive/negative sets.
  - `delta_cer` and `delta_wer` equal `pos - neg`, or `None` when a side is
    empty.
  - `low_support` is `True` exactly when `n_pos < 30`.
- `test_torch_free_import.py` still passes (new model is torch-free).
- A crop id present in the sidecar but absent from the val set is ignored; a
  crop id in the val set but absent from the sidecar is `excluded`.

## Trade-offs considered

- **Plain serialized sidecar vs. direct `GlyphAnnotations` import vs. mirrored
  schema.** Direct import adds a training-lib → foundation-lib dependency edge
  and pulls a heavy OCR/layout library into a training package, risking the
  torch-free guarantee. A mirrored full schema duplicates a `pdomain-book-tools`
  type that can drift. The sidecar carries only the three feature-presence
  facts the eval actually needs, keeps `pdomain-ocr-training`'s dependency graph
  lean, and puts the `GlyphAnnotations` → `GlyphFeatureSet` extraction in the
  caller (`pdomain-ocr-trainer-spa`), which already depends on `pdomain-book-tools`.
- **Crop-id keying vs. parallel-index list.** A parallel list aligned to val
  iteration order is simpler but silently misaligns every downstream slice if
  any sample is filtered or reordered. The DocTR recognition val set already
  has a stable per-crop label key; keying by it is robust.
- **Sidecar path vs. inline dict on the config.** An inline
  `dict[str, GlyphFeatureSet]` avoids file I/O but bloats the config object and
  its serialized form for large val sets. A path mirrors the existing
  `val_path` pattern and lets `ExportManager` write the sidecar alongside the
  val set.
- **Error vs. silent no-op when the flag is set without a path.** Raising at
  validation time surfaces a caller mistake immediately rather than producing a
  silently empty `slices` list.

## Consequences

- `pdomain-ocr-trainer-spa` M13 becomes implementable: it maps
  `EvalRequest.slice_glyph_features` onto `RecognitionEvalConfig`, derives the
  sidecar from `pdomain-book-tools` `GlyphAnnotations`, and renders
  `RecognitionEvalResult.slices` in `EvalMetricsTable`.
- `pdomain-ocr-training` gains no new runtime dependency; the protocol layer stays
  torch-free.
- The sidecar JSON format becomes a contract between the dataset/export side
  and the eval side — it must be documented where the recognition val-set
  layout is documented (`ExportManager` dataset docs).
- `delta_wer` is additive on `EvalSlice`; no existing consumer breaks.

## Open questions

- Which component writes the sidecar — `ExportManager` at dataset-export time,
  or `pdomain-ocr-trainer-spa` just before invoking eval? This is a build-time
  decomposition decision and does not change this spec's contract; the eval
  side only consumes the sidecar.
- Exact crop-id string form (bare filename vs. path relative to `val_path`).
  To be pinned against the DocTR recognition val-set label format during
  implementation; the contract above only requires it to match the val-set
  label key.

## References

- Spec issue: pdomain/pdomain-ocr-training#5
- Parent feature: pdomain/pdomain-ocr-training#4
- Slicing rules: `pdomain-ocr-trainer-spa` `specs/07-evaluation-and-metrics.md` §4
- Eval surface: `pdomain_ocr_training/protocols.py`
  (`RecognitionEvalConfig`, `EvalSlice`, `RecognitionEvalResult`, `IEvalRunner`)
- Inference worker: `pdomain_ocr_training/_eval_backend.py`
  (`_run_recognition_inference`)
- IEvalRunner ADR: `docs/decisions/2026-05-21-ieval-runner-protocol.md`
- pdomain-book-tools v0.13.0 — publishes the `GlyphAnnotations` glyph module (#163)
