from bayesian_risk_engine.model import predict_proba, train_logistic_regression


def test_logistic_model_learns_simple_boundary():
    rows = [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]]
    labels = [0, 0, 0, 1, 1, 1]
    model = train_logistic_regression(rows, labels, epochs=500)
    probabilities = predict_proba(model, [[0.05], [0.95]])
    assert probabilities[0] < probabilities[1]

