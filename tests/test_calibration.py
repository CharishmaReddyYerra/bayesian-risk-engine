from bayesian_risk_engine.calibration import (
    brier_score,
    confusion_at_threshold,
    expected_calibration_error,
)


def test_brier_score_rewards_better_probabilities():
    good = brier_score([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    bad = brier_score([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8])
    assert good < bad


def test_expected_calibration_error_is_bounded():
    ece = expected_calibration_error([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], n_bins=4)
    assert 0.0 <= ece <= 1.0


def test_confusion_at_threshold_counts_outcomes():
    assert confusion_at_threshold([1, 0, 1, 0], [0.8, 0.7, 0.4, 0.2]) == {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }

