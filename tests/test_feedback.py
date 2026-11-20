"""
CodeSentinel — Unit & Integration Tests for Closed-Loop Feedback Aggregator
===========================================================================
Verifies:
- Heuristic ground-truth defect labeling logic (y=1, y=0, inconclusive)
- Feedback CSV storage, deduplication, and schema adherence
- Dataset augmentation and feature matrix merging
- Retraining triggers and sample guards
- FastAPI feedback REST routes
"""
import pytest
from pathlib import Path
import pandas as pd
from starlette.testclient import TestClient

from backend.main import app
from buglens.config import FEATURES, LABEL
from feedback.aggregator import label_function_outcome, record_outcomes, get_feedback_stats
from feedback.model_updater import prepare_augmented_dataset, retrain_with_feedback


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_heuristic_labeling_logic():
    """Verify ground-truth label assignments under various execution outcomes."""
    dummy_func = {"function_name": "calc_total", "filepath": "calc.py"}

    # Case 1: Failures present -> Defective (1)
    res_fail = {"passed": 2, "failed": 1, "errors": 0, "coverage_percent": 85.0}
    assert label_function_outcome(dummy_func, res_fail) == 1

    # Case 2: Execution errors present -> Defective (1)
    res_err = {"passed": 0, "failed": 0, "errors": 1, "coverage_percent": 0.0}
    assert label_function_outcome(dummy_func, res_err) == 1

    # Case 3: Clean pass with high coverage (>= 75%) -> Clean (0)
    res_clean = {"passed": 3, "failed": 0, "errors": 0, "coverage_percent": 88.0}
    assert label_function_outcome(dummy_func, res_clean) == 0

    # Case 4: Low coverage pass (< 75%) -> Inconclusive (None)
    res_inconclusive = {"passed": 1, "failed": 0, "errors": 0, "coverage_percent": 45.0}
    assert label_function_outcome(dummy_func, res_inconclusive) is None


def test_record_outcomes_csv_persistence_and_deduplication(tmp_path: Path):
    """Verify appending to CSV, 8-feature alignment, and deduplication on (filepath, function_name)."""
    feedback_csv = tmp_path / "feedback_test.csv"

    funcs = [
        {
            "filepath": "app/math.py",
            "function_name": "divide",
            "cyclomatic_complexity": 3.0,
            "loc": 15.0,
            "maintainability_index": 80.0,
            "churn": 4.0,
            "unique_developers": 2.0,
            "bug_fix_commits": 1.0,
            "file_age_days": 100.0,
            "recent_churn": 1.0,
        },
        {
            "filepath": "app/auth.py",
            "function_name": "login",
            "cyclomatic_complexity": 5.0,
            "loc": 30.0,
            "maintainability_index": 70.0,
            "churn": 10.0,
            "unique_developers": 3.0,
            "bug_fix_commits": 2.0,
            "file_age_days": 200.0,
            "recent_churn": 3.0,
        },
    ]

    # First run: test_results has failures -> label=1
    test_results_1 = {"passed": 1, "failed": 1, "errors": 0, "coverage_percent": 60.0}
    count = record_outcomes(funcs, test_results_1, db_run_id=1, feedback_file=feedback_csv)
    assert count == 2
    assert feedback_csv.exists()

    df1 = pd.read_csv(feedback_csv)
    assert len(df1) == 2
    for feat in FEATURES:
        assert feat in df1.columns
    assert LABEL in df1.columns
    assert (df1[LABEL] == 1).all()

    # Second run: updating app/math.py::divide to passing with high coverage -> label=0
    test_results_2 = {"passed": 5, "failed": 0, "errors": 0, "coverage_percent": 90.0}
    single_func = [funcs[0]]
    count_2 = record_outcomes(single_func, test_results_2, db_run_id=2, feedback_file=feedback_csv)
    assert count_2 == 1

    df2 = pd.read_csv(feedback_csv)
    # Total row count should remain 2 because app/math.py::divide was deduplicated
    assert len(df2) == 2
    divide_row = df2[df2["function_name"] == "divide"].iloc[0]
    assert divide_row[LABEL] == 0


def test_feedback_stats(tmp_path: Path):
    """Verify statistics calculation on absent and populated feedback files."""
    missing_csv = tmp_path / "nonexistent.csv"
    stats_empty = get_feedback_stats(missing_csv)
    assert stats_empty["total_samples"] == 0
    assert stats_empty["ready_for_retraining"] is False

    pop_csv = tmp_path / "pop.csv"
    data = {
        "filepath": ["a.py", "b.py", "c.py"],
        "function_name": ["f1", "f2", "f3"],
        LABEL: [1, 0, 1],
    }
    pd.DataFrame(data).to_csv(pop_csv, index=False)

    stats = get_feedback_stats(pop_csv)
    assert stats["total_samples"] == 3
    assert stats["defective_count"] == 2
    assert stats["clean_count"] == 1
    assert stats["defect_ratio"] == round(2 / 3, 4)


def test_prepare_augmented_dataset(tmp_path: Path):
    """Verify merging baseline features with feedback records."""
    base_csv = tmp_path / "base.csv"
    fb_csv = tmp_path / "fb.csv"
    out_csv = tmp_path / "out.csv"

    # Minimal baseline rows
    base_data = {feat: [1.0, 2.0] for feat in FEATURES}
    base_data[LABEL] = [0, 1]
    pd.DataFrame(base_data).to_csv(base_csv, index=False)

    # Minimal feedback row
    fb_data = {feat: [3.0] for feat in FEATURES}
    fb_data[LABEL] = [1]
    pd.DataFrame(fb_data).to_csv(fb_csv, index=False)

    result_path = prepare_augmented_dataset(
        baseline_path=base_csv,
        feedback_path=fb_csv,
        output_path=out_csv,
    )
    assert result_path.exists()
    aug_df = pd.read_csv(result_path)
    assert len(aug_df) == 3


def test_retrain_sample_guard(tmp_path: Path):
    """Verify retrain guard when minimum feedback samples threshold is not met."""
    fb_csv = tmp_path / "fb_empty.csv"
    res = retrain_with_feedback(min_feedback_samples=10, feedback_path=fb_csv)
    assert res["retrained"] is False
    assert "Insufficient feedback samples" in res["message"]


def test_feedback_api_endpoints(client: TestClient):
    """Verify REST endpoints for recording feedback, checking stats, and requesting retraining."""
    # 1. Record feedback endpoint
    rec_res = client.post("/feedback/record", json={
        "target_functions": [
            {
                "filepath": "service/user.py",
                "function_name": "validate_email",
                "cyclomatic_complexity": 2.0,
                "loc": 10.0,
                "maintainability_index": 85.0,
            }
        ],
        "test_results": {
            "passed": 4,
            "failed": 0,
            "errors": 0,
            "coverage_percent": 95.0,
        },
        "run_id": 999
    })
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["samples_recorded"] == 1

    # 2. Check stats endpoint
    stats_res = client.get("/feedback/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_samples"] >= 1

    # 3. Retrain endpoint with high guard
    retrain_res = client.post("/feedback/retrain", json={"min_samples": 99999})
    assert retrain_res.status_code == 200
    retrain_data = retrain_res.json()
    assert retrain_data["retrained"] is False
