from pdomain_ocr_training.utils import EarlyStopper


def test_early_stopper_resets_counter_on_improvement() -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.1)

    assert not stopper.early_stop(1.0)
    assert not stopper.early_stop(0.8)
    assert stopper.counter == 0
    assert stopper.min_validation_loss == 0.8


def test_early_stopper_triggers_after_patience_exceeded() -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.1)

    assert not stopper.early_stop(1.0)
    assert not stopper.early_stop(1.2)
    assert stopper.early_stop(1.3)
