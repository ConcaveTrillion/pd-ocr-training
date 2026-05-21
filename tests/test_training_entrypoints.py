"""Smoke tests: training entry-point modules import and expose main()."""

import importlib


def test_detect_module_imports():
    mod = importlib.import_module("pd_ocr_training.detect")
    assert hasattr(mod, "main")


def test_recog_module_imports():
    mod = importlib.import_module("pd_ocr_training.recog")
    assert hasattr(mod, "main")
