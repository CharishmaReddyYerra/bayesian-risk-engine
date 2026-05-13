from __future__ import annotations

import csv
import json
import subprocess
import sys
from html import escape
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_risk_engine.calibration import brier_score, confusion_at_threshold, expected_calibration_error
from bayesian_risk_engine.model import predict_proba, train_logistic_regression


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
ARTIFACT_PATH = ROOT / "artifacts" / "report_metrics.json"
DOCS_PATH = ROOT / "docs" / "index.html"


def download_dataset() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        try:
            urlretrieve(DATA_URL, DATA_PATH)
        except URLError:
            subprocess.run(["curl", "-L", "--fail", DATA_URL, "-o", str(DATA_PATH)], check=True)


def load_dataset() -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for record in reader:
            if not record or "?" in record:
                continue
            values = [float(value) for value in record]
            rows.append(values[:-1])
            labels.append(1 if values[-1] > 0 else 0)
    return rows, labels


def accuracy(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> float:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return sum(label == prediction for label, prediction in zip(labels, predictions)) / len(labels)


def html_table(records: list[dict[str, object]]) -> str:
    keys = list(records[0])
    header = "".join(f"<th>{escape(key)}</th>" for key in keys)
    rows = []
    for record in records:
        rows.append("<tr>" + "".join(f"<td>{escape(str(record[key]))}</td>" for key in keys) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_report(rows: list[list[float]], labels: list[int]) -> None:
    split = int(len(rows) * 0.7)
    train_rows, test_rows = rows[:split], rows[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    model = train_logistic_regression(train_rows, train_labels)
    probabilities = predict_proba(model, test_rows)
    confusion = confusion_at_threshold(test_labels, probabilities)
    metrics = {
        "dataset_url": DATA_URL,
        "rows_after_cleaning": len(rows),
        "features": len(rows[0]),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "event_rate": round(sum(labels) / len(labels), 4),
        "test_accuracy": round(accuracy(test_labels, probabilities), 4),
        "test_brier_score": round(brier_score(test_labels, probabilities), 4),
        "test_expected_calibration_error": round(expected_calibration_error(test_labels, probabilities), 4),
        "confusion_at_0_5": confusion,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    threshold_rows = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        counts = confusion_at_threshold(test_labels, probabilities, threshold=threshold)
        threshold_rows.append({"threshold": threshold, **counts})

    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bayesian Risk Engine</title>
  <style>
    body {{ font-family: Inter, system-ui, sans-serif; margin: 0; background: #f7f9fc; color: #172033; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 48px 20px; }}
    h1 {{ font-size: 42px; margin-bottom: 8px; }}
    .lede {{ color: #435268; font-size: 18px; line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 28px 0; }}
    .metric {{ background: white; border: 1px solid #dfe6ef; border-radius: 8px; padding: 18px; }}
    .metric strong {{ display: block; font-size: 28px; color: #7c3aed; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e6ebf2; text-align: left; }}
    th {{ background: #f0ebff; }}
  </style>
</head>
<body>
<main>
  <h1>Bayesian Risk Engine</h1>
  <p class="lede">A first risk-model baseline on the UCI Cleveland heart disease dataset. The current report establishes calibration and threshold diagnostics before adding Bayesian posterior uncertainty.</p>
  <section class="grid">
    <div class="metric"><span>Rows</span><strong>{metrics["rows_after_cleaning"]}</strong></div>
    <div class="metric"><span>Event rate</span><strong>{metrics["event_rate"]}</strong></div>
    <div class="metric"><span>Accuracy</span><strong>{metrics["test_accuracy"]}</strong></div>
    <div class="metric"><span>Brier score</span><strong>{metrics["test_brier_score"]}</strong></div>
    <div class="metric"><span>ECE</span><strong>{metrics["test_expected_calibration_error"]}</strong></div>
  </section>
  <h2>Threshold Audit</h2>
  {html_table(threshold_rows)}
  <h2>Next Modeling Step</h2>
  <p>Add Bayesian logistic regression with weakly informative priors, posterior predictive checks, and credible intervals around patient-level risk.</p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    download_dataset()
    rows, labels = load_dataset()
    write_report(rows, labels)
    print(f"Wrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
