# Bayesian Risk Engine

Build a probabilistic risk model that returns full uncertainty instead of only point predictions.

## Why This Project Exists

Many applied ML projects return a class label or probability without explaining how stable the estimate is. This project is designed to show the difference between prediction and risk reasoning: priors, posterior uncertainty, calibration, and careful decision thresholds.

## Demo

- Static report: `docs/index.html`
- Metrics artifact: `artifacts/report_metrics.json`
- Next app target: FastAPI endpoint for risk scoring

## Dataset Plan

The public baseline uses the UCI Cleveland heart disease dataset. Rows with missing values are removed, and the disease target is converted to a binary risk outcome for calibration and threshold analysis.

## Method

The first implementation should compare:

1. Logistic regression baseline.
2. Bayesian logistic regression with weakly informative priors.
3. Optional tree-based baseline for predictive comparison.

The project should explain where uncertainty comes from and how it affects decisions.

## Architecture

```text
raw data -> validation -> features -> baseline model -> Bayesian model -> posterior checks -> API/demo
```

## Statistical Evaluation

- Posterior predictive checks
- Credible intervals
- Brier score
- Expected calibration error
- Decision threshold analysis

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/build_report.py
pytest
```

## Current Status

Baseline report builder created. It downloads the public UCI dataset, trains a dependency-free logistic risk model, and writes calibration and threshold diagnostics.

## Limitations

Risk scores are not decisions. Any real deployment would require dataset documentation, subgroup evaluation, calibration monitoring, and domain review.
