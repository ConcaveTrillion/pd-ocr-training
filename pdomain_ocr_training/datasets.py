"""Dataset store: paths, constants, file I/O, and ExportManager.

Owns all knowledge of where training data lives on disk and how to move it
between the labeler export root and the ml-training / ml-validation trees.
"""

import json
import os
import platform
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
ML_TRAINING_DIR = Path(os.getenv("PD_OCR_TRAINER_ML_TRAINING_DIR", PROJECT_ROOT / "ml-training"))
ML_VALIDATION_DIR = Path(
    os.getenv("PD_OCR_TRAINER_ML_VALIDATION_DIR", PROJECT_ROOT / "ml-validation")
)
APP_NAME = "pd-ocr-labeler"
MODEL_STORE_DIRNAME = "pd-ml-models"
MODEL_NAME_PREFIX = "pd"
BASE_OCR_PROFILE = "all"
LEGACY_BASE_OCR_PROFILE = "base-ocr"
DATASET_TASKS = ("detection", "recognition")


def get_os_data_parent() -> Path:
    """Return OS-aware parent directory for application data roots."""
    system_name = platform.system()
    if system_name == "Linux":
        data_home = os.getenv("XDG_DATA_HOME")
        base_dir = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    elif system_name == "Darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    elif system_name == "Windows":
        appdata = os.getenv("APPDATA")
        base_dir = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        base_dir = Path.home() / ".local" / "share"
    return base_dir


APP_DATA_ROOT = Path(os.getenv("PD_OCR_TRAINER_APP_DATA_ROOT", get_os_data_parent() / APP_NAME))
SHARED_MODELS_DIR = Path(
    os.getenv("PD_OCR_TRAINER_SHARED_MODELS_DIR", get_os_data_parent() / MODEL_STORE_DIRNAME)
)
TRAINER_SETTINGS_PATH = APP_DATA_ROOT / "trainer_settings.json"

# Ensure directories exist at import time
ML_TRAINING_DIR.mkdir(exist_ok=True)
ML_VALIDATION_DIR.mkdir(exist_ok=True)
(ML_TRAINING_DIR / BASE_OCR_PROFILE).mkdir(parents=True, exist_ok=True)
(ML_VALIDATION_DIR / BASE_OCR_PROFILE).mkdir(parents=True, exist_ok=True)
SHARED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


def normalize_profile_name(name: str) -> str:
    value = (name or "").strip().lower().replace(" ", "-").replace("_", "-")
    if value == LEGACY_BASE_OCR_PROFILE:
        return BASE_OCR_PROFILE
    return value or BASE_OCR_PROFILE


def profile_model_root(profile: str) -> Path:
    return SHARED_MODELS_DIR / normalize_profile_name(profile)


def model_output_dir(profile: str, model_type: str) -> Path:
    return profile_model_root(profile) / model_type


def split_profile_root(split: str, profile: str = BASE_OCR_PROFILE) -> Path:
    split_map = {"train": ML_TRAINING_DIR, "val": ML_VALIDATION_DIR}
    root = split_map.get(split)
    if root is None:
        raise ValueError(f"Unknown split '{split}'")
    return root / normalize_profile_name(profile)


# ---------------------------------------------------------------------------
# Export-root scanning
# ---------------------------------------------------------------------------


def iter_export_profile_dirs(export_root: Path):
    """Yield (project_dir, subfolder) for each subfolder containing DocTR task data."""
    if not export_root.exists():
        return
    for project_dir in sorted(export_root.iterdir()):
        if not project_dir.is_dir():
            continue
        for subdir in sorted(project_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            if any((subdir / task / "labels.json").exists() for task in DATASET_TASKS):
                yield project_dir, subdir


def get_available_model_profiles() -> list[str]:
    """List all known training profiles (from export dirs, on-disk datasets, and shared models)."""
    profiles = {BASE_OCR_PROFILE}
    for split_root in (ML_TRAINING_DIR, ML_VALIDATION_DIR):
        if not split_root.exists():
            continue
        for profile_dir in split_root.iterdir():
            if not profile_dir.is_dir():
                continue
            if any((profile_dir / task).exists() for task in DATASET_TASKS):
                profiles.add(normalize_profile_name(profile_dir.name))
    if SHARED_MODELS_DIR.exists():
        for profile_dir in SHARED_MODELS_DIR.iterdir():
            if profile_dir.is_dir():
                profiles.add(normalize_profile_name(profile_dir.name))
    export_root = ExportManager.get_export_root()
    for _project_dir, subfolder in iter_export_profile_dirs(export_root):
        profiles.add(normalize_profile_name(subfolder.name))
    return sorted(profiles)


# ---------------------------------------------------------------------------
# Label / page helpers
# ---------------------------------------------------------------------------


def project_from_stem(stem: str) -> str:
    """Strip trailing digit-only segments from an image stem to recover the project ID."""
    parts = stem.split("_")
    end = len(parts)
    while end > 1 and parts[end - 1].isdigit():
        end -= 1
    return "_".join(parts[:end])


def detection_page_from_recognition_name(img_name: str) -> str:
    """Best-effort page filename from a recognition crop filename."""
    path = Path(img_name)
    parts = path.stem.split("_")
    if len(parts) > 4 and all(p.isdigit() for p in parts[-4:]):
        return f"{'_'.join(parts[:-4])}{path.suffix}"
    return img_name


def group_existing_by_project(split_root: Path) -> dict[str, list[str]]:
    """Return {project_id: [page_img_name, ...]} for an existing split directory."""
    groups: dict[str, list[str]] = defaultdict(list)
    detection_labels = ExportManager._load_json_map(split_root / "detection" / "labels.json")
    if detection_labels:
        for img_name in detection_labels:
            groups[project_from_stem(Path(img_name).stem)].append(img_name)
        return {k: sorted(v) for k, v in sorted(groups.items())}

    recognition_labels = ExportManager._load_json_map(split_root / "recognition" / "labels.json")
    if recognition_labels:
        by_project_pages: dict[str, set[str]] = defaultdict(set)
        for img_name in recognition_labels:
            page_name = detection_page_from_recognition_name(img_name)
            by_project_pages[project_from_stem(Path(page_name).stem)].add(page_name)
        return {k: sorted(v) for k, v in sorted(by_project_pages.items())}

    return {}


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


def _merge_profile_tree(src_root: Path, dest_root: Path) -> None:
    """Merge one profile tree into another and remove the source when done."""
    if not src_root.exists() or src_root == dest_root:
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    for child in src_root.iterdir():
        target = dest_root / child.name
        if child.is_dir():
            if target.exists():
                _merge_profile_tree(child, target)
            else:
                shutil.move(str(child), str(target))
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(child), str(target))
    shutil.rmtree(src_root, ignore_errors=True)


def migrate_legacy_dataset_layout() -> None:
    """Move legacy split/task datasets into the profile-scoped layout."""
    base_profile = normalize_profile_name(BASE_OCR_PROFILE)
    for split_root in (ML_TRAINING_DIR, ML_VALIDATION_DIR):
        has_legacy = any((split_root / task).exists() for task in DATASET_TASKS)
        if not has_legacy:
            continue
        profile_root = split_root / base_profile
        profile_root.mkdir(parents=True, exist_ok=True)
        for task in DATASET_TASKS:
            legacy_task_root = split_root / task
            if not legacy_task_root.exists():
                continue
            target_task_root = profile_root / task
            if not target_task_root.exists():
                shutil.move(str(legacy_task_root), str(target_task_root))
                continue
            target_images = target_task_root / "images"
            target_images.mkdir(parents=True, exist_ok=True)
            legacy_images = legacy_task_root / "images"
            if legacy_images.exists():
                for src_img in legacy_images.iterdir():
                    if src_img.is_file():
                        shutil.move(str(src_img), str(target_images / src_img.name))
            legacy_labels = ExportManager._load_json_map(legacy_task_root / "labels.json")
            target_labels_path = target_task_root / "labels.json"
            target_labels = ExportManager._load_json_map(target_labels_path)
            target_labels.update(legacy_labels)
            ExportManager._write_json_map(target_labels_path, target_labels)
            shutil.rmtree(legacy_task_root, ignore_errors=True)

    legacy_profile = LEGACY_BASE_OCR_PROFILE
    target_profile = normalize_profile_name(BASE_OCR_PROFILE)
    if normalize_profile_name(legacy_profile) == target_profile:
        for root in (ML_TRAINING_DIR, ML_VALIDATION_DIR, SHARED_MODELS_DIR):
            legacy_root = root / legacy_profile
            target_root = root / target_profile
            if legacy_root.exists() and legacy_root != target_root:
                _merge_profile_tree(legacy_root, target_root)


# ---------------------------------------------------------------------------
# ExportManager
# ---------------------------------------------------------------------------


class ExportManager:
    """Manages pd-ocr-labeler DocTR export assignments for training."""

    def __init__(self) -> None:
        self.active_profile = normalize_profile_name(BASE_OCR_PROFILE)
        self.assignments: dict[str, str | None] = {}
        self.page_assignments: dict[tuple[str, str], str] = {}
        self.changed_keys: set[str] = set()
        self.scan()

    def set_profile(self, profile: str) -> None:
        self.active_profile = normalize_profile_name(profile)

    def split_root(self, split: str) -> Path:
        root = split_profile_root(split, self.active_profile)
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def get_export_root() -> Path:
        """OS-aware path to the pd-ocr-labeler DocTR export root."""
        return APP_DATA_ROOT / "doctr-export"

    def scan(self) -> None:
        """Scan the export root and rebuild available exports, preserving existing assignments."""
        export_root = self.get_export_root()

        existing_image_names: set[str] = set()
        for split_root in (self.split_root("train"), self.split_root("val")):
            for task in DATASET_TASKS:
                images_dir = split_root / task / "images"
                if images_dir.exists():
                    for img in images_dir.iterdir():
                        existing_image_names.add(img.name)

        new_assignments: dict[str, str | None] = {}
        new_changed: set[str] = set()

        for _project_dir, subfolder in iter_export_profile_dirs(export_root):
            if normalize_profile_name(subfolder.name) != self.active_profile:
                continue
            key = subfolder.relative_to(export_root).as_posix()

            export_pages = self.get_export_pages(key)
            if export_pages and all(
                page_name in existing_image_names for page_name in export_pages
            ):
                continue

            new_assignments[key] = self.assignments.get(key)
            for task in DATASET_TASKS:
                src_images = subfolder / task / "images"
                if src_images.exists() and any(
                    img.name in existing_image_names for img in src_images.iterdir()
                ):
                    new_changed.add(key)
                    break

        self.assignments = new_assignments
        self.page_assignments = {
            (key, page): split
            for (key, page), split in self.page_assignments.items()
            if key in self.assignments
            and page in set(self.get_export_pages(key))
            and split in {"train", "val"}
        }
        self.changed_keys = new_changed

    def get_by_split(self) -> dict[str, dict[str, list[str]]]:
        """Return exports grouped by split then project: {split: {project: [keys]}}."""
        result: dict[str, dict[str, list[str]]] = {
            "unassigned": defaultdict(list),
            "train": defaultdict(list),
            "val": defaultdict(list),
        }
        for key, split in self.assignments.items():
            col = split if split in {"train", "val"} else "unassigned"
            project = key.split("/")[0]
            result[col][project].append(key)
        return {k: dict(v) for k, v in result.items()}

    def get_export_pages_by_split(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """Return pending export pages grouped as {split: {project: {key: [page_names]}}}."""
        result: dict[str, dict[str, dict[str, list[str]]]] = {
            "unassigned": defaultdict(lambda: defaultdict(list)),
            "train": defaultdict(lambda: defaultdict(list)),
            "val": defaultdict(lambda: defaultdict(list)),
        }
        for key in self.assignments:
            project = key.split("/")[0]
            pages = self.get_export_pages(key)
            split = self.assignments.get(key)
            if split in {"train", "val"}:
                result[split][project][key].extend(pages)
                continue
            for page_name in pages:
                page_split = self.page_assignments.get((key, page_name))
                col = page_split if page_split in {"train", "val"} else "unassigned"
                result[col][project][key].append(page_name)

        return {
            split: {
                project: {key: sorted(pages) for key, pages in sorted(keys.items())}
                for project, keys in sorted(projects.items())
            }
            for split, projects in result.items()
        }

    def assign(self, key: str, target: str | None) -> None:
        if key in self.assignments:
            self.assignments[key] = target if target in {"train", "val"} else None
            self.page_assignments = {
                (k, page): split for (k, page), split in self.page_assignments.items() if k != key
            }

    def assign_page(self, key: str, page_name: str, target: str | None) -> None:
        if key not in self.assignments:
            return
        pages = self.get_export_pages(key)
        full_split = self.assignments.get(key)
        if full_split in {"train", "val"}:
            for page in pages:
                self.page_assignments[(key, page)] = full_split
            self.assignments[key] = None
        if target in {"train", "val"}:
            self.page_assignments[(key, page_name)] = target
        else:
            self.page_assignments.pop((key, page_name), None)

    def assign_project(self, project_id: str, target: str | None) -> None:
        for key in self.assignments:
            if key.startswith(f"{project_id}/"):
                self.assignments[key] = target if target in {"train", "val"} else None
                self.page_assignments = {
                    (k, page): split
                    for (k, page), split in self.page_assignments.items()
                    if k != key
                }

    def assign_pages(self, pages: list[tuple[str, str]], target: str | None) -> None:
        for key, page_name in pages:
            self.assign_page(key, page_name, target)

    def clear_split(self, split: str) -> None:
        if split not in {"train", "val"}:
            return
        for key in list(self.assignments):
            if self.assignments.get(key) == split:
                self.assignments[key] = None
        self.page_assignments = {
            (k, page): s for (k, page), s in self.page_assignments.items() if s != split
        }

    def is_changed(self, key: str) -> bool:
        return key in self.changed_keys

    def export_path(self, key: str) -> Path:
        parts = key.split("/", 1)
        return self.get_export_root() / parts[0] / parts[1]

    def get_export_pages(self, key: str) -> list[str]:
        """Return sorted page image names for an export key."""
        export_root = self.export_path(key)
        for task in ("detection", "recognition"):
            labels = self._load_json_map(export_root / task / "labels.json")
            if labels:
                return sorted(labels)
        return []

    @staticmethod
    def _load_json_map(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _write_json_map(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def get_existing_projects(split_root: Path) -> dict[str, int]:
        """Return {project_id: page_count} already present in a split directory."""
        return {k: len(v) for k, v in group_existing_by_project(split_root).items()}

    @staticmethod
    def get_existing_pages(split_root: Path, project_id: str) -> list[str]:
        """Return sorted image names for project_id already present in split_root."""
        return group_existing_by_project(split_root).get(project_id, [])

    def move_existing_project(self, project_id: str, from_split: str, to_split: str) -> int:
        """Physically move an on-disk project between ml-training and ml-validation."""
        src_root = self.split_root(from_split) if from_split in {"train", "val"} else None
        dest_root = self.split_root(to_split) if to_split in {"train", "val"} else None
        if src_root is None or dest_root is None or src_root == dest_root:
            return 0
        moved = 0
        for task in DATASET_TASKS:
            src_lp = src_root / task / "labels.json"
            if not src_lp.exists():
                continue
            try:
                with open(src_lp) as f:
                    src_labels = json.load(f)
            except Exception:
                continue
            to_move = {
                k: v for k, v in src_labels.items() if project_from_stem(Path(k).stem) == project_id
            }
            if not to_move:
                continue
            dest_lp = dest_root / task / "labels.json"
            dest_images = dest_root / task / "images"
            dest_images.mkdir(parents=True, exist_ok=True)
            dest_labels: dict = {}
            if dest_lp.exists():
                try:
                    with open(dest_lp) as f:
                        dest_labels = json.load(f)
                except Exception:
                    pass
            src_images = src_root / task / "images"
            for img_name, meta in to_move.items():
                src_img = src_images / img_name
                if src_img.exists():
                    shutil.move(str(src_img), dest_images / img_name)
                del src_labels[img_name]
                dest_labels[img_name] = meta
                if task == "detection":
                    moved += 1
            with open(src_lp, "w") as f:
                json.dump(src_labels, f, indent=2)
            with open(dest_lp, "w") as f:
                json.dump(dest_labels, f, indent=2)
        return moved

    def move_existing_page(self, page_name: str, from_split: str, to_split: str) -> int:
        """Physically move one on-disk page (and its recognition crops) between splits."""
        src_root = self.split_root(from_split) if from_split in {"train", "val"} else None
        dest_root = self.split_root(to_split) if to_split in {"train", "val"} else None
        if src_root is None or dest_root is None or src_root == dest_root:
            return 0
        page_stem = Path(page_name).stem
        moved = 0
        for task in DATASET_TASKS:
            src_lp = src_root / task / "labels.json"
            if not src_lp.exists():
                continue
            try:
                with open(src_lp) as f:
                    src_labels = json.load(f)
            except Exception:
                continue
            if task == "detection":
                keys = [k for k in src_labels if Path(k).stem == page_stem]
            else:
                keys = [k for k in src_labels if Path(k).stem.startswith(f"{page_stem}_")]
            if not keys:
                continue
            dest_lp = dest_root / task / "labels.json"
            dest_images = dest_root / task / "images"
            dest_images.mkdir(parents=True, exist_ok=True)
            dest_labels: dict = {}
            if dest_lp.exists():
                try:
                    with open(dest_lp) as f:
                        dest_labels = json.load(f)
                except Exception:
                    pass
            src_images = src_root / task / "images"
            for key in keys:
                src_img = src_images / key
                if src_img.exists():
                    shutil.move(str(src_img), dest_images / key)
                dest_labels[key] = src_labels[key]
                del src_labels[key]
            if task == "detection":
                moved += len(keys)
            with open(src_lp, "w") as f:
                json.dump(src_labels, f, indent=2)
            with open(dest_lp, "w") as f:
                json.dump(dest_labels, f, indent=2)
        return moved

    def save_assignments(
        self,
        include_detection: bool = True,
        include_recognition: bool = True,
    ) -> dict[str, int]:
        """Merge assigned DocTR exports into ML_TRAINING_DIR / ML_VALIDATION_DIR."""
        full_copy = [(k, v) for k, v in self.assignments.items() if v in {"train", "val"}]
        page_copy = [
            ((k, page), split)
            for (k, page), split in self.page_assignments.items()
            if split in {"train", "val"}
        ]
        if not full_copy and not page_copy:
            return {"copied": 0}

        task_flags = {"detection": include_detection, "recognition": include_recognition}
        plan: dict[tuple[str, str], set[str] | None] = {}
        for key, split in full_copy:
            plan[(key, split)] = None
        for (key, page_name), split in page_copy:
            bucket = plan.get((key, split))
            if bucket is None and (key, split) in plan:
                continue
            if bucket is None:
                bucket = set()
                plan[(key, split)] = bucket
            bucket.add(page_name)

        count = 0
        for (key, split), selected_pages in plan.items():
            src_root = self.export_path(key)
            dest_root = self.split_root("train" if split == "train" else "val")
            copied_any_task = False
            for task, include in task_flags.items():
                if not include:
                    continue
                src_labels_path = src_root / task / "labels.json"
                if not src_labels_path.exists():
                    continue
                src_images_dir = src_root / task / "images"
                dest_images_dir = dest_root / task / "images"
                dest_images_dir.mkdir(parents=True, exist_ok=True)
                src_labels = self._load_json_map(src_labels_path)
                if selected_pages is not None:
                    if task == "detection":
                        src_labels = {k: v for k, v in src_labels.items() if k in selected_pages}
                    else:
                        selected_stems = {Path(name).stem for name in selected_pages}
                        src_labels = {
                            k: v
                            for k, v in src_labels.items()
                            if any(Path(k).stem.startswith(f"{stem}_") for stem in selected_stems)
                        }
                if not src_labels:
                    continue
                dest_labels_path = dest_root / task / "labels.json"
                dest_labels = self._load_json_map(dest_labels_path)
                for img_name in src_labels:
                    src_img = src_images_dir / img_name
                    if src_img.exists():
                        shutil.copy2(src_img, dest_images_dir / img_name)
                dest_labels.update(src_labels)
                self._write_json_map(dest_labels_path, dest_labels)
                copied_any_task = True
                count += 1

            if copied_any_task:
                if selected_pages is None:
                    self.assignments[key] = None
                    self.page_assignments = {
                        (k, page): s for (k, page), s in self.page_assignments.items() if k != key
                    }
                else:
                    for page_name in selected_pages:
                        self.page_assignments.pop((key, page_name), None)

        self.scan()
        return {"copied": count}
