def brier_score(y_true: list[int], probabilities: list[float]) -> float:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not y_true:
        raise ValueError("inputs cannot be empty")
    if any(label not in (0, 1) for label in y_true):
        raise ValueError("y_true must contain binary labels 0 or 1")
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("probabilities must be between 0 and 1")

    return sum((probability - label) ** 2 for label, probability in zip(y_true, probabilities)) / len(y_true)


def expected_calibration_error(
    y_true: list[int],
    probabilities: list[float],
    *,
    n_bins: int = 10,
) -> float:
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not y_true:
        raise ValueError("inputs cannot be empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    total = len(y_true)
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        in_bin = [
            (label, probability)
            for label, probability in zip(y_true, probabilities)
            if lower <= probability < upper or (bin_index == n_bins - 1 and probability == 1.0)
        ]
        if not in_bin:
            continue

        accuracy = sum(label for label, _ in in_bin) / len(in_bin)
        confidence = sum(probability for _, probability in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(accuracy - confidence)

    return ece


def confusion_at_threshold(
    y_true: list[int],
    probabilities: list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, int]:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")

    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return {
        "tp": sum(true == 1 and pred == 1 for true, pred in zip(y_true, predictions)),
        "fp": sum(true == 0 and pred == 1 for true, pred in zip(y_true, predictions)),
        "tn": sum(true == 0 and pred == 0 for true, pred in zip(y_true, predictions)),
        "fn": sum(true == 1 and pred == 0 for true, pred in zip(y_true, predictions)),
    }

