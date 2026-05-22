# Lint-rule Deviations — pd-ocr-training

Standing suppressions and per-file rule overrides in this repo.
Each entry records: the rule, the tool, the file(s) affected, and
the justification. Update this file whenever a new suppression is added.

---

## Python — ruff (project-wide ignores)

### 1. `E501` — ruff (line-too-long)

**Config:** `pyproject.toml` `[tool.ruff.lint] ignore = ["E501"]` (project-wide)

**Justification.** Many long docstrings, error messages, and URLs; enforcing
88-char wrapping everywhere adds noise without improving readability.

---

### 2. `D203`, `D212` — ruff (docstring style conflicts)

**Config:** project-wide ignore.

**Justification.** `D203` (1-blank-before-class-docstring) conflicts with
`D211` (no-blank-before-class-docstring). `D212`
(multi-line-summary-first-line) conflicts with `D213`
(multi-line-summary-second-line). ruff requires picking one of each pair;
the selected alternatives (`D211`, `D213`) are what the Google convention
implies.

---

### 3. `D100`, `D104`, `D107` — ruff (missing docstrings)

**Config:** project-wide ignore.

**Justification.** Missing docstrings on public modules, packages, and
magic methods. Large existing codebase; docstrings are being added
incrementally — a single global enforcement sweep would be invasive.

---

### 4. `PLR0913` — ruff (too-many-arguments)

**Config:** project-wide ignore.

**Justification.** OCR and pipeline functions legitimately need many
parameters. Enforcing this rule would require invasive config-object
refactors not warranted by the linting rollout.

---

### 5. `PLR2004` — ruff (magic-value-comparison)

**Config:** project-wide ignore.

**Justification.** Common in threshold/port/timeout code where literal
values are semantically clear from context.

---

### 6. `TRY003` — ruff (long-message-outside-exception-class)

**Config:** project-wide ignore.

**Justification.** The library uses f-string error messages everywhere;
requiring a custom exception class per message would be invasive without
readability gain.

---

### 7. `COM812` — ruff (missing-trailing-comma)

**Config:** project-wide ignore.

**Justification.** Conflicts with the ruff formatter's auto-style. Both
cannot be on simultaneously; the formatter wins.

---

### 8. `PLC0415` — ruff (import-not-at-top-level)

**Config:** project-wide ignore.

**Justification.** Deferred imports are a legitimate pattern — used to
break circular dependencies and avoid loading optional-heavy modules
(torch, DocTR) until needed. The `LocalTrainingRunner` lazy-import via
`__init__.__getattr__` is the canonical example.

---

### 9. `PLR0912`, `PLR0911`, `PLR0915` — ruff (PLR complexity rules)

**Config:** project-wide ignore (`too-many-branches`, `too-many-return-statements`,
`too-many-statements`).

**Justification.** DocTR training pipeline functions legitimately have high
branch and return counts. Enforcing these would require invasive refactors
of verbatim-moved legacy code.

---

### 10. `ANN401` — ruff (dynamically-typed-expressions)

**Config:** project-wide ignore.

**Justification.** Some functions legitimately accept or return `Any` (e.g.
JSON deserialisers, generic dispatch helpers). Enforcing this globally would
force spurious casts with no safety gain.

---

### 11. `D205` — ruff (1-blank-line-required-between-summary-and-description)

**Config:** project-wide ignore.

**Justification.** Enforcing this across the entire docstring backlog in the
verbatim-moved legacy modules is too noisy for one commit. Will be addressed
in the annotation follow-up pass.

---

### 12. `D105` — ruff (missing-docstring-in-magic-method)

**Config:** project-wide ignore.

**Justification.** `__repr__`, `__eq__`, etc. are self-documenting;
requiring docstrings here adds noise. Will be required incrementally on new
code only.

---

### 13. `B008` — ruff (function-call-in-default-argument)

**Config:** project-wide ignore.

**Justification.** FastAPI `Depends()` and Pydantic `field()` use this
pattern legitimately in all route and model definitions.

---

## Python — ruff (per-file-ignores)

### 14. `tests/**/*.py` — broad test-file relaxations

**Config:** `[tool.ruff.lint.per-file-ignores]`

**Rules suppressed:** `S101`, `S105`, `S106`, `S311`, `T201`, `ANN`, `D`,
`PLR2004`, `PT011`, `S108`, `PLR0133`, `PLW2901`, `PERF401`, `BLE001`,
`PLW1510`, `SIM117`.

**Justification.**

- `S101` — `assert` is the pytest idiom.
- `S105`/`S106` — hardcoded credentials are test fixtures, not real secrets.
- `S311` — random values in tests are intentional test data, not crypto.
- `T201` — `print()` used for test diagnostics.
- `ANN`/`D` — annotations and docstrings not required in test functions.
- `PLR2004` — magic numbers common in expected-value assertions.
- `PT011` — `pytest.raises` match= not required for every test.
- `S108` — `/tmp` paths are fine in tests.
- `PLR0133` — trivial comparisons can be intentional in parameterised tests.
- `PLW2901` — loop-var reassignment is a common test-setup pattern.
- `PERF401` — list-building loops in tests are fine.
- `BLE001` — concurrency stress tests collect thread errors via broad except.
- `PLW1510` — `subprocess.run check=` omitted intentionally; returncode
  asserted explicitly.
- `SIM117` — nested `with` blocks are clearer for `pytest.raises` + context combos.

---

### 15. `scripts/*.py` — script relaxations

**Config:** `[tool.ruff.lint.per-file-ignores]`

**Rules suppressed:** `T201`, `D`, `S607`.

**Justification.** `print()` is the output mechanism in scripts. No
docstrings required. `S607` (partial executable path) is idiomatic when
invoking system tools (`uv`, `git`, etc.).

---

### 16. `**/__init__.py` — re-export modules

**Config:** `[tool.ruff.lint.per-file-ignores]`

**Rules suppressed:** `D104`, `F401`, `TC`.

**Justification.** `__init__.py` re-exports public API names without
docstrings. `F401` (unused import) would fire on every re-export. `TC`
(move to `TYPE_CHECKING`) conflicts with runtime-importable re-exports.

---

### 17. `**/_*.py` — private modules

**Config:** `[tool.ruff.lint.per-file-ignores]`

**Rules suppressed:** `D`.

**Justification.** Internal convention — private modules do not require
docstrings.

---

### 18. Legacy verbatim-moved modules — broad annotation/docstring/safety debt

**Files:** `pd_ocr_training/utils.py`, `pd_ocr_training/datasets.py`

**Rules suppressed:** `ANN`, `D`, `BLE`, `S`.

**Justification.** These modules were moved verbatim from `pd-ocr-trainer`
without modification. Annotation and docstring debt is tracked for a
follow-up pass. `BLE` (blind except) and `S` (bandit security rules) flag
pre-existing patterns in the original code; cleaning them in the move
commit would conflate migration with refactoring.

---

### 19. Legacy verbatim-moved modules — extended families for detect/recog

**Files:** `pd_ocr_training/detect.py`, `pd_ocr_training/recog.py`

**Rules suppressed:** `ANN`, `D`, `BLE`, `S`, `PLW2901`, `RET506`,
`PLW1508`, `LOG015`, `PLW0603`, `RET504`.

**Justification.** Same rationale as §18 (verbatim move). Additional
families were required by patterns in the DocTR training entrypoints:

- `PLW2901` — loop-variable overwrite is idiomatic GPU tensor reassignment
  (`for batch in loader: batch = batch.to(device)`).
- `RET506`/`RET504` — cosmetic return-style patterns pre-existing in the
  original code.
- `PLW1508` — `os.environ.get` with integer defaults is a common config
  pattern.
- `LOG015` — root logger calls (`logging.info(...)` vs `logger.info(...)`)
  are pre-existing in the original code.
- `PLW0603` — global statement for a shared step counter is pre-existing.

These modules are also excluded from basedpyright type-checking for the same
reason (see `[tool.basedpyright] exclude` in `pyproject.toml`).

---

## Python — ruff (inline noqa)

### 20. `BLE001` — `pd_ocr_training/local.py:158`

**Suppression:** `# noqa: BLE001 — must capture *all* exceptions from worker thread`

**Justification.** The `LocalTrainingRunner` worker thread runs arbitrary
user-supplied training code. Catching `BaseException` here is intentional:
a bare `Exception` catch would miss `SystemExit`, `KeyboardInterrupt`, and
other signals that should be surfaced as a `"fail"` training event rather
than silently swallowed or propagated past the thread boundary.

---

### 21. `S603` — `tests/test_torch_free_import.py:42`

**Suppression:** `# noqa: S603 - fixed argv, test-controlled code`

**Justification.** `subprocess.run` in this test invokes a fixed,
test-controlled argv to verify the torch-free import contract via a
subprocess. There is no user-supplied shell input; `S603` fires on
`subprocess.run` with a list argument, which is the safe form.

---

### 22. `TC003` — `pd_ocr_training/protocols.py:56`

**Suppression:** `# noqa: TC003 — keep Path importable at runtime; pydantic resolves the annotation at model-build time`

**Justification.** `Path` is used as a Pydantic field annotation. `TC003`
suggests moving it to `TYPE_CHECKING`, but Pydantic needs `Path` importable
at runtime to resolve field types when building the model. Moving it would
cause a `NameError` at model instantiation.

---

### 23. `N817` — `pd_ocr_training/recog.py:15`, `pd_ocr_training/detect.py:15`

**Suppression:** `# noqa: N817 — PyTorch standard alias`

**Justification.** `from torch.nn.parallel import DistributedDataParallel as DDP`
is the PyTorch community standard alias (all uppercase). `N817` flags
non-PEP-8 alias casing. The verbatim-moved code uses the standard alias;
renaming it would diverge from DocTR conventions.

---

### 24. `N812` — `pd_ocr_training/recog.py:33`, `pd_ocr_training/detect.py:31`

**Suppression:** `# noqa: N812 — doctr library conventional alias`

**Justification.** `from doctr import transforms as T` is the DocTR
library conventional alias (single uppercase letter). `N812` flags
non-PEP-8 alias casing. The verbatim-moved code uses the convention;
renaming it would diverge from all DocTR examples and documentation.

---

## Python — basedpyright (tests)

### 25. `type: ignore[index]` — `tests/test_local_runner.py:445`

**Suppression:** `# type: ignore[index]`

**Note.** This uses mypy-style suppression syntax. The suppression targets
`e.data["epoch"]` where `e.data` is typed `dict[str, object] | None`; the
`if e.data is not None` list-comprehension guard narrows the list but
basedpyright cannot narrow the indexing into a `dict[str, object]` to
`int` without a cast. A `# pyright: ignore[reportIndexIssue]` would be
the tool-native form — tracked for cleanup in the annotation follow-up pass.

**Justification.** The guard ensures `data` is not `None`; the `int(...)`
cast is explicit. No runtime error is possible.

---

### 26. `type: ignore[operator]` — `tests/test_local_runner.py:455`

**Suppression:** `# type: ignore[operator]`

**Note.** Same mypy-style syntax note as §25. The suppression covers
`progresses[i] > progresses[i - 1]` where progress values are
`float | None`; the explicit `assert progresses[i] is not None` guards
above do not narrow the type within the same assert chain.
`# pyright: ignore[reportOperatorIssue]` would be the correct form —
tracked for cleanup.

**Justification.** Both `progresses[i]` and `progresses[i - 1]` are
guarded non-None before the comparison. The suppression silences a
false-positive narrowing gap.

---

## Python — basedpyright (source: DocTR stub gaps)

### 27. `reportArgumentType` — `pd_ocr_training/_eval_backend.py`

**Suppression:** `# pyright: ignore[reportArgumentType]` on the `val_set`
argument to `DataLoader(...)` and on the `SequentialSampler(val_set)` argument,
in both `_run_recognition_inference` and `_run_detection_inference` (4 sites).

**Justification.** DocTR's `RecognitionDataset` / `DetectionDataset` are
torch-compatible map-style datasets at runtime (they implement `__len__` /
`__getitem__` and a `collate_fn`), but DocTR's bundled type stubs derive them
from an internal `AbstractDataset` that does not declare the
`torch.utils.data.Dataset` base class. basedpyright therefore cannot prove the
assignment to `DataLoader`'s `dataset` parameter. This is a third-party stub
gap, not a real type error — the legacy verbatim-moved `detect.py` / `recog.py`
use the identical pattern but are excluded from basedpyright entirely. The
tool-native `# pyright: ignore[reportArgumentType]` form is used (not a
mypy-style `# type: ignore`). Remove if DocTR ships corrected stubs.
