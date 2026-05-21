# CLAUDE — pd-ocr-training

DocTR OCR model training pipeline (detection + recognition) for the `pd-*`
suite. Extracted from the legacy `pd-ocr-trainer` repo.

## Why a separate package

`torch` and `DocTR` are heavy ML dependencies. Isolating them here keeps every
other `pd-*` SPA backend torch-free. Only the future `pd-ocr-trainer-spa`
depends on this package.

## Install modes — torch is an optional extra

`torch` / `python-doctr` / `matplotlib` are **not** in `[project.dependencies]`.
They live in the `train` optional extra. This lets `pd-ocr-trainer-spa`'s
long-lived web process import the torch-free half (config models + Protocol)
without dragging multi-GB torch into the web process.

```bash
pip install pd-ocr-training            # base — torch-free
pip install 'pd-ocr-training[train]'   # full training stack
```

The base install exposes `DetectionConfig`, `RecognitionConfig`,
`TrainingEvent`, `ITrainingRunner`, `IEvalRunner`, all eval config models
(`DetectionEvalConfig`, `RecognitionEvalConfig`), result models (`EvalSlice`,
`DetectionEvalResult`, `RecognitionEvalResult`), and `LocalEvalRunner`.
`LocalTrainingRunner` is exported lazily via `__init__.__getattr__` — `import
pd_ocr_training` never imports torch. Accessing `LocalTrainingRunner` without
the `[train]` extra raises a helpful `ImportError`. `LocalEvalRunner` is
torch-free (its stub impl raises `NotImplementedError`) and importable without
the extra. The `dev` dependency-group pulls in `[train]` so `make ci` exercises
the full stack; the torch-free contract is covered separately by
`tests/test_torch_free_import.py` (subprocess with torch hidden).

## Package layout

```text
pd_ocr_training/
    __init__.py      # Public API re-exports
    protocols.py     # ITrainingRunner + IEvalRunner Protocols; all config + result models
    local.py         # LocalTrainingRunner — callback→iterator bridge (strict-lint-compliant)
    local_eval.py    # LocalEvalRunner — synchronous eval wrapper (strict-lint-compliant)
    detect.py        # Verbatim-moved DocTR detection training (legacy; per-file-ignores active)
    recog.py         # Verbatim-moved DocTR recognition training (legacy; per-file-ignores active)
    datasets.py      # ExportManager (legacy; per-file-ignores active)
    utils.py         # Shared training utilities (legacy; per-file-ignores active)
```

`detect.py`, `recog.py`, `datasets.py`, and `utils.py` are verbatim moves from
the legacy repo. They carry `ANN`, `D`, `BLE`, and `S` per-file-ignores in
`pyproject.toml` pending an annotation follow-up pass. `protocols.py`,
`local.py`, and `local_eval.py` are new, strict-lint-compliant code.

## Ops-style contract

Consumers depend on the `ITrainingRunner` Protocol, not the concrete modules:

```python
from pd_ocr_training import ITrainingRunner, LocalTrainingRunner, DetectionConfig

runner: ITrainingRunner = LocalTrainingRunner()
cfg = DetectionConfig(train_path="data/train", val_path="data/val")
for event in runner.train_detection("my-run", cfg):
    print(event.kind, event.message)
```

## Dev commands

```bash
make setup          # uv sync --group dev + install pre-commit hooks
make lint           # ruff check --fix
make lint-check     # ruff format --check + ruff check (CI-exact, no fix)
make format         # ruff format pd_ocr_training tests
make typecheck      # basedpyright pd_ocr_training --level error
make test           # uv run pytest -n auto
make ci             # setup → pre-commit → lint-check → typecheck → test
make build          # uv build
make clean          # rm dist .venv .pytest_cache .ruff_cache .ci-ai.log htmlcov
```

Run any command with `uv run` directly if preferred:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run basedpyright pd_ocr_training --level error
```

## docs/ folder

This repo follows the workspace docs/ template — see [`docs/README.md`](docs/README.md). Active
folders: `architecture/`, `decisions/`, `plans/`, `process/`, `research/`,
`runbooks/`, `specs/`, `templates/`, `usage/`, plus parallel `archive/`
subfolders.

**Superpowers redirect.** When a superpowers skill (e.g. `brainstorming`,
`writing-plans`) instructs you to save to `docs/superpowers/specs/<file>.md`
or `docs/superpowers/plans/<file>.md`, save to `docs/specs/<file>.md` or
`docs/plans/<file>.md` instead. There is no `docs/superpowers/` subdirectory
in this repo.

<!-- workspace-process:start -->

## Before coding

These steps are workspace defaults for any coding task. **User-level settings
override them** — a user's own `~/.claude/CLAUDE.md`, `settings.json`, or a
direct instruction in the conversation takes precedence and may waive or
change any step below.

### Working principles

- **Use skills.** Invoke the relevant superpowers skill before starting —
  process skills first (`brainstorming`, `systematic-debugging`,
  `writing-plans`, `test-driven-development`), then implementation skills.
  If a skill applies, using it is not optional.
- **Delegate by default.** Dispatch subagents for non-trivial work: per-repo
  agents for repo changes, `Explore` for code searches. This keeps large tool
  output out of the parent context.
- **Parallelize.** Run independent tasks as concurrent subagents — multiple
  agent calls in a single message. Set `model: sonnet` on implementers and
  reviewers.

### Steps

1. **Check the working tree.** `git status --short`. Surface or resolve stray
   uncommitted work before starting — don't build on it.
2. **Read repo guidance.** This repo's `CLAUDE.md` and `CONVENTIONS.md` for
   repo-specific rules.
3. **Consult `docs/` for authoritative context** (whichever folders exist):
   `plans/` (the work plan), `specs/` (design specs — follow any `Spec:`
   pointer from the issue), `research/` (prior investigations), `decisions/`
   (ADRs / constraints), `architecture/` (shipped design).
4. **Check live issue status.** `gh issue view <N> --repo <owner/repo>` —
   confirm it isn't already closed; note its milestone.
5. **Check for in-flight work.** Open PRs and existing branches touching the
   same area, to avoid colliding with work-in-progress.
6. **Consult agent memory.** `.claude/agent-memory/<repo>/feedback_*.md` for
   corrections not yet promoted to `CONVENTIONS.md`.
7. **Locate code with `Explore` first.** Use an `Explore` subagent to find
   relevant files before broad `Read`/grep.
8. **Isolate in a worktree.** Never work directly in the interactive checkout
   at `/workspaces/ocr-container/<repo>/`. Use the `using-git-worktrees` skill
   to set up an isolated worktree. When delegating to a full-power
   implementation agent, pass `isolation: "worktree"` on the `Agent` call
   (skip for `-docs` agents and the `driver` agent). When an agent returns a
   worktree path + branch, use the `finishing-a-development-branch` skill to
   decide how to integrate.
9. **TDD.** Write the failing test first where the plan calls for it.
10. **Verify before committing.** Focused verification plus `make ci`.
11. **Commit locally; do not push** without explicit say-so.

<!-- workspace-process:end -->
