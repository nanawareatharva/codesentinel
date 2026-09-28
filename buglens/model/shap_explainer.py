"""Produce developer-friendly SHAP explanations for BugLens predictions."""
from pathlib import Path

import joblib
import numpy as np

from buglens.config import FEATURES, MODEL_DIR

FEATURE_LABELS = {
    "cyclomatic_complexity": "High cyclomatic complexity", "loc": "Large function (high LOC)",
    "maintainability_index": "Low maintainability index", "churn": "High total code churn",
    "unique_developers": "Many contributors", "bug_fix_commits": "Frequent historical bug fixes",
    "file_age_days": "Old file with accumulated debt", "recent_churn": "Heavy recent changes (last 90 days)",
}


def _positive_class_values(raw_values) -> np.ndarray:
    """Normalize SHAP's version-dependent binary-class output shape."""
    values = np.asarray(raw_values)
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])[0]
    if values.ndim == 3:  # newer SHAP: samples × features × classes
        return values[0, :, 1]
    return values[0]


def explain_function(feature_values: dict) -> dict:
    """Explain one function using stored scaler and TreeExplainer artifacts."""
    scaler = joblib.load(Path(MODEL_DIR) / "scaler.pkl")
    explainer = joblib.load(Path(MODEL_DIR) / "shap_explainer.pkl")
    row = np.array([[feature_values.get(feature, 0) for feature in FEATURES]], dtype=float)
    values = _positive_class_values(explainer.shap_values(scaler.transform(row)))
    expected = np.asarray(explainer.expected_value)
    base = float(expected[1] if expected.ndim and expected.size > 1 else expected.ravel()[0])
    ranked = sorted(zip(FEATURES, values), key=lambda pair: abs(float(pair[1])), reverse=True)
    drivers = [{"feature": feature, "label": FEATURE_LABELS.get(feature, feature),
                "shap_value": round(float(value), 4), "raw_value": round(float(feature_values.get(feature, 0)), 2),
                "direction": "increases risk" if value > 0 else "decreases risk"}
               for feature, value in ranked[:4]]
    summary = ", ".join(FEATURE_LABELS.get(feature, feature) for feature, value in ranked[:3] if value > 0)
    return {"top_risk_drivers": drivers, "summary": summary or "No dominant risk factors identified",
            "base_value": round(base, 4), "prediction": round(float(np.sum(values)) + base, 4)}
