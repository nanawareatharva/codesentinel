# tests/test_model.py
import pytest
import numpy as np
from pathlib import Path


SAMPLE_FEATURES = {
    "cyclomatic_complexity": 8.0,
    "loc": 45.0,
    "maintainability_index": 55.0,
    "churn": 320.0,
    "unique_developers": 4.0,
    "bug_fix_commits": 2.0,
    "file_age_days": 400.0,
    "recent_churn": 80.0,
}


@pytest.fixture(scope="module")
def artifacts():
    import joblib
    from buglens.config import MODEL_DIR
    m = Path(MODEL_DIR) / "lightgbm_model.pkl"
    if not m.exists():
        pytest.skip("Run: python buglens/model/train.py first")
    return {
        "model": joblib.load(m),
        "scaler": joblib.load(Path(MODEL_DIR) / "scaler.pkl"),
        "explainer": joblib.load(Path(MODEL_DIR) / "shap_explainer.pkl"),
    }


def test_model_predicts_probability(artifacts):
    from buglens.config import FEATURES
    X = np.array([[SAMPLE_FEATURES[f] for f in FEATURES]])
    X_sc = artifacts["scaler"].transform(X)
    proba = artifacts["model"].predict_proba(X_sc)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


def test_complex_function_riskier_than_simple(artifacts):
    from buglens.config import FEATURES
    simple = {f: 1.0 for f in FEATURES}
    simple.update({"cyclomatic_complexity": 1, "loc": 3, "churn": 5})
    risky  = {f: 1.0 for f in FEATURES}
    risky.update({"cyclomatic_complexity": 20, "loc": 250, "churn": 2000, "unique_developers": 12})

    X_s = artifacts["scaler"].transform(np.array([[simple[f] for f in FEATURES]]))
    X_r = artifacts["scaler"].transform(np.array([[risky[f] for f in FEATURES]]))

    p_s = artifacts["model"].predict_proba(X_s)[:, 1][0]
    p_r = artifacts["model"].predict_proba(X_r)[:, 1][0]
    assert p_r > p_s, f"Risky ({p_r:.3f}) should be > simple ({p_s:.3f})"


def test_shap_returns_correct_structure(artifacts):
    from buglens.model.shap_explainer import explain_function
    result = explain_function(SAMPLE_FEATURES)
    assert "top_risk_drivers" in result
    assert "summary" in result
    assert isinstance(result["top_risk_drivers"], list)
    assert len(result["top_risk_drivers"]) >= 1
    for d in result["top_risk_drivers"]:
        assert "label" in d and "shap_value" in d and "direction" in d


def test_shap_values_finite(artifacts):
    from buglens.model.shap_explainer import explain_function
    result = explain_function(SAMPLE_FEATURES)
    for d in result["top_risk_drivers"]:
        assert np.isfinite(d["shap_value"]), f"Non-finite SHAP: {d}"


def test_feature_schema_is_8(artifacts):
    from buglens.config import FEATURES
    assert len(FEATURES) == 8


def test_predict_repo_on_mini_repo(tmp_path):
    import subprocess
    from buglens.model.predict import predict_repo

    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in [["git","init"],["git","config","user.email","t@t.com"],["git","config","user.name","T"]]:
        subprocess.run(cmd, cwd=str(repo), capture_output=True)

    (repo / "calc.py").write_text("""
def complex_calc(a, b, mode):
    if mode == 'add':
        if a < 0:
            return 0
        return a + b
    elif mode == 'mul':
        return a * b
    else:
        raise ValueError('unknown mode')
""")
    subprocess.run(["git","add","."], cwd=str(repo), capture_output=True)
    subprocess.run(["git","commit","-m","fix initial"], cwd=str(repo), capture_output=True)

    results = predict_repo(str(repo))
    assert isinstance(results, list)
    if results:
        assert "bug_probability" in results[0]
        assert "risk_category" in results[0]
        assert results[0]["risk_category"] in ["LOW","MEDIUM","HIGH"]
        assert 0 <= results[0]["bug_probability"] <= 100
