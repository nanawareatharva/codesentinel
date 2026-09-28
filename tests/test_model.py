from pathlib import Path
import subprocess

import numpy as np
import pytest

SAMPLE_FEATURES = {
    "cyclomatic_complexity": 8.0, "loc": 45.0, "maintainability_index": 55.0,
    "churn": 320.0, "unique_developers": 4.0, "bug_fix_commits": 2.0,
    "file_age_days": 400.0, "recent_churn": 80.0,
}


@pytest.fixture(scope="module")
def artifacts():
    import joblib
    from buglens.config import MODEL_DIR
    directory = Path(MODEL_DIR)
    if not (directory / "lightgbm_model.pkl").exists():
        pytest.skip("Run: python buglens/model/train.py first")
    return {"model": joblib.load(directory / "lightgbm_model.pkl"),
            "scaler": joblib.load(directory / "scaler.pkl"),
            "explainer": joblib.load(directory / "shap_explainer.pkl")}


def test_model_predicts_probability(artifacts):
    from buglens.config import FEATURES
    result = artifacts["model"].predict_proba(artifacts["scaler"].transform([[SAMPLE_FEATURES[f] for f in FEATURES]]))[0, 1]
    assert 0.0 <= result <= 1.0


def test_shap_returns_correct_structure(artifacts):
    from buglens.model.shap_explainer import explain_function
    result = explain_function(SAMPLE_FEATURES)
    assert {"top_risk_drivers", "summary", "base_value", "prediction"} <= result.keys()
    assert result["top_risk_drivers"]
    assert all({"label", "shap_value", "direction"} <= item.keys() for item in result["top_risk_drivers"])


def test_shap_values_are_finite(artifacts):
    from buglens.model.shap_explainer import explain_function
    assert all(np.isfinite(item["shap_value"]) for item in explain_function(SAMPLE_FEATURES)["top_risk_drivers"])


def test_feature_schema_is_eight(artifacts):
    from buglens.config import FEATURES
    assert len(FEATURES) == 8


def test_predict_repo_on_mini_repo(tmp_path, artifacts):
    from buglens.model.predict import predict_repo
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (["git", "init"], ["git", "config", "user.email", "t@t.com"], ["git", "config", "user.name", "T"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    (repo / "calc.py").write_text("""
def complex_calc(a, b, mode):
    if mode == 'add':
        if a < 0:
            return 0
        return a + b
    if mode == 'mul':
        return a * b
    raise ValueError('unknown mode')
""")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fix initial"], cwd=repo, check=True, capture_output=True)
    results = predict_repo(str(repo))
    assert results and {"bug_probability", "risk_category"} <= results[0].keys()
    assert results[0]["risk_category"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= results[0]["bug_probability"] <= 100
