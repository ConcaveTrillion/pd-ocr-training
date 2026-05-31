#!/usr/bin/env python3
"""Refresh reviewed GitHub Actions refs in workflow files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github/workflows"
MANAGED_ACTIONS = (
    "actions/checkout",
    "astral-sh/setup-uv",
    "actions/setup-python",
    "actions/setup-node",
    "actions/upload-artifact",
    "actions/download-artifact",
    "peter-evans/create-pull-request",
)


@dataclass(frozen=True)
class ActionRelease:
    """Latest release tag and immutable commit SHA."""

    tag: str
    sha: str


GhRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def resolve_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} executable not found on PATH")
    return executable


def run_gh(command: list[str]) -> subprocess.CompletedProcess[str]:
    resolved = [resolve_executable(command[0]), *command[1:]]
    return subprocess.run(  # noqa: S603
        resolved, cwd=ROOT, check=True, capture_output=True, text=True
    )


def gh_json(endpoint: str, *, runner: GhRunner = run_gh) -> dict[str, object]:
    result = runner(["gh", "api", endpoint])
    return cast("dict[str, object]", json.loads(result.stdout))


def latest_release(action: str, *, runner: GhRunner = run_gh) -> ActionRelease:
    """Return the latest release tag and target commit SHA for an action."""
    release = gh_json(f"repos/{action}/releases/latest", runner=runner)
    tag = release.get("tag_name")
    if not isinstance(tag, str):
        raise TypeError(f"latest release for {action} did not include tag_name")
    tag_ref = gh_json(f"repos/{action}/git/ref/tags/{tag}", runner=runner)
    raw_object = tag_ref.get("object")
    if not isinstance(raw_object, dict):
        raise TypeError(f"tag ref for {action}@{tag} did not include object")
    tag_object = cast("dict[str, object]", raw_object)
    sha = tag_object.get("sha")
    if tag_object.get("type") == "tag" and isinstance(sha, str):
        tag_payload = gh_json(f"repos/{action}/git/tags/{sha}", runner=runner)
        nested = tag_payload.get("object")
        if not isinstance(nested, dict):
            raise TypeError(f"annotated tag for {action}@{tag} did not include object")
        sha = cast("dict[str, object]", nested).get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise TypeError(f"tag ref for {action}@{tag} did not resolve to a commit SHA")
    return ActionRelease(tag=tag, sha=sha)


def latest_uv_version(*, runner: GhRunner = run_gh) -> str:
    """Return the latest uv release version string (e.g. '0.11.17')."""
    release = gh_json("repos/astral-sh/uv/releases/latest", runner=runner)
    tag = release.get("tag_name")
    if not isinstance(tag, str):
        raise TypeError("latest uv release did not include tag_name")
    version = tag.lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"unexpected uv release tag: {tag}")
    return version


def update_uv_version_refs(path: Path, *, version: str) -> bool:
    """Update the uv version string inside setup-uv with: blocks. Returns True if changed."""
    text = path.read_text(encoding="utf-8")
    updated = re.sub(
        r'(uses:\s+astral-sh/setup-uv@[^\n]+\n\s+with:\n\s+version:\s+")[^"]+(")',
        rf"\g<1>{version}\g<2>",
        text,
    )
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def update_workflow_refs(path: Path, *, releases: dict[str, ActionRelease]) -> bool:
    """Update managed action refs in one workflow file. Returns True if changed."""
    text = path.read_text(encoding="utf-8")
    updated = text
    for action, release in releases.items():
        updated = re.sub(
            rf"(?m)(uses:\s+{re.escape(action)}@)[^\s]+",
            rf"\g<1>{release.sha}",
            updated,
        )
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def update_pyproject_uv_version(path: Path, *, version: str) -> bool:
    """Update [tool.uv] required-version in pyproject.toml. Returns True if changed."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    updated = re.sub(
        r'(required-version\s*=\s*")[^"]+(")',
        rf"\g<1>{version}\g<2>",
        text,
    )
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def update_github_actions(
    *,
    workflow_dir: Path = WORKFLOW_DIR,
    pyproject: Path | None = None,
    runner: GhRunner = run_gh,
) -> list[Path]:
    """Refresh managed action refs and uv version, return changed workflow paths."""
    releases = {a: latest_release(a, runner=runner) for a in MANAGED_ACTIONS}
    uv_version = latest_uv_version(runner=runner)
    changed: set[Path] = set()
    for path in sorted(workflow_dir.glob("*.yml")):
        if update_workflow_refs(path, releases=releases):
            changed.add(path)
        if update_uv_version_refs(path, version=uv_version):
            changed.add(path)
    pyproject_path = pyproject if pyproject is not None else ROOT / "pyproject.toml"
    if update_pyproject_uv_version(pyproject_path, version=uv_version):
        changed.add(pyproject_path)
    return sorted(changed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in update_github_actions():
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
