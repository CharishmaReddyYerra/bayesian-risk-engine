from __future__ import annotations

from math import exp


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1 / (1 + z)
    z = exp(value)
    return z / (1 + z)


def standardize_fit(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    columns = list(zip(*rows))
    means = [sum(column) / len(column) for column in columns]
    scales = []
    for mean, column in zip(means, columns):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scales.append(variance**0.5 or 1.0)
    return means, scales


def standardize_transform(rows: list[list[float]], means: list[float], scales: list[float]) -> list[list[float]]:
    return [[(value - mean) / scale for value, mean, scale in zip(row, means, scales)] for row in rows]


def train_logistic_regression(
    rows: list[list[float]],
    labels: list[int],
    *,
    learning_rate: float = 0.08,
    epochs: int = 900,
    l2: float = 0.01,
) -> dict[str, object]:
    means, scales = standardize_fit(rows)
    x = standardize_transform(rows, means, scales)
    weights = [0.0 for _ in range(len(x[0]))]
    bias = 0.0
    n = len(x)

    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, label in zip(x, labels):
            probability = sigmoid(sum(weight * value for weight, value in zip(weights, row)) + bias)
            error = probability - label
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        for index in range(len(weights)):
            grad_w[index] = grad_w[index] / n + l2 * weights[index]
            weights[index] -= learning_rate * grad_w[index]
        bias -= learning_rate * grad_b / n

    return {"weights": weights, "bias": bias, "means": means, "scales": scales}


def predict_proba(model: dict[str, object], rows: list[list[float]]) -> list[float]:
    weights = model["weights"]
    bias = model["bias"]
    means = model["means"]
    scales = model["scales"]
    x = standardize_transform(rows, means, scales)
    return [sigmoid(sum(weight * value for weight, value in zip(weights, row)) + bias) for row in x]

