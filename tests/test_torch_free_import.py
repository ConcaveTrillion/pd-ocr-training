"""Verify the torch-free import contract of ``pd_ocr_training``.

A downstream web process (``pd-ocr-trainer-spa``) installs the base package
without the ``[train]`` extra and must be able to import the typed config
models and the ``ITrainingRunner`` Protocol without dragging in torch / DocTR.

These tests run in a subprocess so the import graph is measured from a clean
interpreter, and they hide ``torch`` / ``doctr`` from the module finder to
simulate a base (torch-free) install even though the dev environment has the
``[train]`` extra installed.
"""

import subprocess
import sys
import textwrap

# A blocker that makes ``import torch`` / ``import doctr`` fail, simulating a
# base install without the [train] extra.
_BLOCK_TORCH = """
import sys
import importlib.abc
import importlib.machinery


class _Blocker(importlib.abc.MetaPathFinder):
    _blocked = ("torch", "doctr", "torchvision", "matplotlib")

    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root in self._blocked:
            raise ModuleNotFoundError(f"No module named {root!r} (blocked)")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``_BLOCK_TORCH`` + ``body`` (dedented) in a clean subprocess."""
    code = _BLOCK_TORCH + textwrap.dedent(body)
    return subprocess.run(  # noqa: S603 - fixed argv, test-controlled code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_base_import_succeeds_without_torch() -> None:
    """``import pd_ocr_training`` works and does not pull torch into sys.modules."""
    result = _run(
        """
        import pd_ocr_training

        assert "torch" not in sys.modules, "torch was imported by base package"
        assert "doctr" not in sys.modules, "doctr was imported by base package"
        assert pd_ocr_training.__version__ == "0.2.1"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_torch_free_public_names_importable() -> None:
    """The torch-free public surface imports without the training stack."""
    result = _run(
        """
        from pd_ocr_training import (
            DetectionConfig,
            DetectionEvalConfig,
            DetectionEvalResult,
            EvalSlice,
            GlyphFeatureSet,
            IEvalRunner,
            ITrainingRunner,
            RecognitionConfig,
            RecognitionEvalConfig,
            RecognitionEvalResult,
            TrainingEvent,
        )

        assert "torch" not in sys.modules
        cfg = DetectionConfig(train_path="data/train", val_path="data/val")
        assert str(cfg.train_path) == "data/train"
        assert hasattr(ITrainingRunner, "train_detection")
        assert hasattr(IEvalRunner, "evaluate_detection")
        assert hasattr(IEvalRunner, "evaluate_recognition")
        assert RecognitionConfig is not None and TrainingEvent is not None
        eval_cfg = DetectionEvalConfig(val_path="data/val", model_path="model.pt")
        assert eval_cfg.arch == "db_resnet50"
        assert RecognitionEvalConfig is not None
        assert EvalSlice is not None
        assert DetectionEvalResult is not None
        assert RecognitionEvalResult is not None
        # GlyphFeatureSet is torch-free (#7)
        g = GlyphFeatureSet()
        assert g.ligatures == []
        assert g.long_s is False
        assert g.swash is False
        assert "torch" not in sys.modules, "GlyphFeatureSet pulled in torch"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_local_runner_access_raises_helpful_error_without_torch() -> None:
    """Accessing LocalTrainingRunner without torch raises a guiding ImportError."""
    result = _run(
        """
        import pd_ocr_training

        try:
            pd_ocr_training.LocalTrainingRunner
        except ImportError as exc:
            assert "pd-ocr-training[train]" in str(exc), str(exc)
            print("OK")
        else:
            raise AssertionError("expected ImportError for LocalTrainingRunner")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_local_eval_runner_importable_without_torch() -> None:
    """LocalEvalRunner resolves without torch -- its stub impl is torch-free.

    Unlike ``LocalTrainingRunner`` (which imports ``detect.py``/``recog.py``
    and therefore torch/DocTR), ``LocalEvalRunner`` only imports from
    ``protocols.py``.  The stub entry points raise ``NotImplementedError``
    rather than requiring torch.  This means the class can be imported in a
    base install; the real eval implementation is a follow-up task.
    """
    result = _run(
        """
        import pd_ocr_training

        runner_cls = pd_ocr_training.LocalEvalRunner
        assert runner_cls is not None, "LocalEvalRunner should be importable without torch"
        assert "torch" not in sys.modules, "torch was imported by LocalEvalRunner"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_local_runner_importable_with_train_stack() -> None:
    """With the [train] extra installed, LocalTrainingRunner resolves normally."""
    from pd_ocr_training import LocalTrainingRunner

    assert LocalTrainingRunner is not None


def test_local_eval_runner_importable_with_train_stack() -> None:
    """With the [train] extra installed, LocalEvalRunner resolves normally."""
    from pd_ocr_training import LocalEvalRunner

    assert LocalEvalRunner is not None
