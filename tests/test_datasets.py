"""Behavioural tests for pdomain_ocr_training.datasets (ExportManager + helpers)."""

import json

import pdomain_ocr_training.datasets as ds


def test_export_manager_discovers_profile_via_training_dir(tmp_path, monkeypatch):
    """A profile directory with a recognition/labels.json under ML_TRAINING_DIR is discoverable."""
    # Create a minimal training-tree profile: ml-training/demo/recognition/labels.json
    profile_dir = tmp_path / "ml-training" / "demo" / "recognition"
    profile_dir.mkdir(parents=True)
    labels_path = profile_dir / "labels.json"
    labels_path.write_text(json.dumps({}))

    val_dir = tmp_path / "ml-validation"
    val_dir.mkdir(parents=True)

    # Redirect module-level globals so get_available_model_profiles() uses tmp_path
    monkeypatch.setattr(ds, "ML_TRAINING_DIR", tmp_path / "ml-training")
    monkeypatch.setattr(ds, "ML_VALIDATION_DIR", val_dir)
    # Use a non-existent shared-models dir so it contributes nothing
    monkeypatch.setattr(ds, "SHARED_MODELS_DIR", tmp_path / "no-models")
    # Override get_export_root so the export scan finds nothing
    monkeypatch.setattr(
        ds.ExportManager, "get_export_root", staticmethod(lambda: tmp_path / "no-exports")
    )

    profiles = ds.get_available_model_profiles()
    assert "demo" in profiles


def test_export_manager_constructs_without_errors(tmp_path, monkeypatch):
    """ExportManager() should construct (and scan) without errors when directories are redirected."""
    train_dir = tmp_path / "ml-training"
    val_dir = tmp_path / "ml-validation"
    models_dir = tmp_path / "models"
    train_dir.mkdir()
    val_dir.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(ds, "ML_TRAINING_DIR", train_dir)
    monkeypatch.setattr(ds, "ML_VALIDATION_DIR", val_dir)
    monkeypatch.setattr(ds, "SHARED_MODELS_DIR", models_dir)
    monkeypatch.setattr(
        ds.ExportManager, "get_export_root", staticmethod(lambda: tmp_path / "no-exports")
    )

    mgr = ds.ExportManager()
    # An empty export root means no assignments
    assert mgr.assignments == {}
    assert mgr.active_profile == ds.BASE_OCR_PROFILE
