"""Inference pipeline for per-function BugLens risk predictions."""
from pathlib import Path

import joblib
import pandas as pd

from buglens.config import FEATURES, MODEL_DIR, RISK_THRESHOLDS
from buglens.mining.git_miner import mine_repository
from buglens.mining.metric_extractor import extract_repo_metrics


def _relative_path(path: object, root: Path) -> str:
    """Normalize an absolute or repository-relative path for joining."""
    value = str(path).replace("\\", "/")
    normalized_root = str(root).replace("\\", "/")
    return value[len(normalized_root):].lstrip("/") if value.startswith(normalized_root) else value.lstrip("/")


def _risk_category(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["HIGH"][0]:
        return "HIGH"
    if probability >= RISK_THRESHOLDS["MEDIUM"][0]:
        return "MEDIUM"
    return "LOW"


def predict_repo(repo_path: str) -> list[dict]:
    """Analyse a local Python repository and return high-risk functions first."""
    model_path = Path(MODEL_DIR) / "lightgbm_model.pkl"
    scaler_path = Path(MODEL_DIR) / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model artifacts are absent. Run: python buglens/model/train.py")
    root = Path(repo_path).resolve()
    print(f"\n[BugLens] Analysing: {root}")
    model, scaler = joblib.load(model_path), joblib.load(scaler_path)
    git_df = mine_repository(str(root))
    code_df = extract_repo_metrics(str(root))
    if code_df.empty:
        print("  No Python functions found.")
        return []
    code_df = code_df.copy()
    code_df["_rel"] = code_df["filepath"].map(lambda value: _relative_path(value, root))
    if not git_df.empty:
        git_df = git_df.copy()
        git_df["_rel"] = git_df["filepath"].map(lambda value: _relative_path(value, root))
        merged = code_df.merge(git_df.drop(columns=["filepath"]), on="_rel", how="left")
    else:
        merged = code_df
    for feature in FEATURES:
        if feature not in merged:
            merged[feature] = 0.0
        merged[feature] = pd.to_numeric(merged[feature], errors="coerce").fillna(0.0)
    probabilities = model.predict_proba(scaler.transform(merged[FEATURES]))[:, 1]
    results = []
    for probability, (_, row) in zip(probabilities, merged.iterrows()):
        value = float(probability)
        item = {"filepath": str(row["filepath"]), "function_name": str(row["function_name"]),
                "line_start": int(row["line_start"]), "line_end": int(row["line_end"]),
                "bug_probability": round(value * 100, 1), "risk_category": _risk_category(value)}
        item.update({feature: float(row[feature]) for feature in FEATURES})
        results.append(item)
    results.sort(key=lambda item: item["bug_probability"], reverse=True)
    counts = {category: sum(item["risk_category"] == category for item in results) for category in RISK_THRESHOLDS}
    print(f"  Done: {len(results)} functions — HIGH:{counts['HIGH']} MEDIUM:{counts['MEDIUM']} LOW:{counts['LOW']}")
    return results
