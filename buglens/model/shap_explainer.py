"""Produce developer-friendly SHAP explanations for BugLens predictions."""
from pathlib import Path

import joblib
import numpy as np

from buglens.config import FEATURES, MODEL_DIR

# Each entry is (risk_label, protective_label).  The first element is used
# when the SHAP value is positive (pushes toward defective), the second when
# negative (pushes toward clean).
FEATURE_RISK_LABELS = {
    "cyclomatic_complexity": ("High cyclomatic complexity", "Low complexity (simple logic)"),
    "loc":                   ("Large function (high LOC)", "Small compact function"),
    "maintainability_index": ("Low maintainability index", "High maintainability (protective)"),
    "churn":                 ("High total code churn", "Stable code with low churn"),
    "unique_developers":     ("Many contributors", "Few dedicated maintainers"),
    "bug_fix_commits":       ("Frequent historical bug fixes", "Few/no historical bug fixes"),
    "file_age_days":         ("Old file with accumulated debt", "Recently created file"),
    "recent_churn":          ("Heavy recent changes (last 90 days)", "Stable in recent 90 days"),
}


def _dynamic_label(feature: str, shap_value: float) -> str:
    """Return a human-readable label that matches the SHAP direction."""
    pair = FEATURE_RISK_LABELS.get(feature, (feature, feature))
    return pair[0] if shap_value > 0 else pair[1]


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
    norm_values = dict(feature_values)
    if float(norm_values.get("maintainability_index", 0.0)) > 1.0:
        norm_values["maintainability_index"] = float(norm_values["maintainability_index"]) / 100.0
    row = np.array([[norm_values.get(feature, 0) for feature in FEATURES]], dtype=float)
    values = _positive_class_values(explainer.shap_values(scaler.transform(row)))
    expected = np.asarray(explainer.expected_value)
    base = float(expected[1] if expected.ndim and expected.size > 1 else expected.ravel()[0])
    ranked = sorted(zip(FEATURES, values), key=lambda pair: abs(float(pair[1])), reverse=True)
    drivers = [{"feature": feature, "label": _dynamic_label(feature, float(value)),
                "shap_value": round(float(value), 4), "raw_value": round(float(feature_values.get(feature, 0)), 2),
                "direction": "increases risk" if value > 0 else "decreases risk"}
               for feature, value in ranked[:4]]
    summary = ", ".join(_dynamic_label(feature, float(value)) for feature, value in ranked[:3] if value > 0)
    return {"top_risk_drivers": drivers, "summary": summary or "No dominant risk factors identified",
            "base_value": round(base, 4), "prediction": round(float(np.sum(values)) + base, 4)}
