"""
CodeSentinel — Closed-Loop Feedback Aggregator
=============================================
Translates dynamic unit test execution results from AutoTestArmy into
ground-truth defect labels for continuous retraining:
- Persistent Failures / Errors -> Confirmed Defect (label = 1)
- Clean Passes with High Coverage (>= 75%) -> Confirmed Clean (label = 0)
- Inconclusive / Low Coverage -> Discarded (prevents label noise)
"""
import os
import logging
from pathlib import Path
from typing import Optional, Any
import pandas as pd

from buglens.config import FEATURES, LABEL

logger = logging.getLogger("codesentinel.feedback")

FEEDBACK_DATA_PATH = Path("data") / "processed" / "feedback_labels.csv"


def label_function_outcome(func_info: dict[str, Any], test_results: dict[str, Any]) -> Optional[int]:
    """
    Apply heuristic decision rules to determine ground-truth defect label:
    - Returns 1 if tests failed or threw errors.
    - Returns 0 if tests passed cleanly with >= 75.0% coverage.
    - Returns None if results are inconclusive.
    """
    failed = test_results.get("failed", 0)
    errors = test_results.get("errors", 0)
    passed = test_results.get("passed", 0)
    coverage = float(test_results.get("coverage_percent", 0.0))

    if failed > 0 or errors > 0:
        return 1

    if passed > 0 and coverage >= 75.0:
        return 0

    return None


def record_outcomes(
    target_functions: list[dict[str, Any]],
    test_results: dict[str, Any],
    db_run_id: Optional[int] = None,
    feedback_file: Path = FEEDBACK_DATA_PATH,
) -> int:
    """
    Extracts features for analyzed functions, assigns ground-truth defect labels
    based on test outcomes, and persists deduplicated records to feedback_labels.csv.

    Returns:
        Number of valid labeled records newly recorded or updated.
    """
    feedback_file = Path(feedback_file)
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for func in target_functions:
        label = label_function_outcome(func, test_results)
        if label is None:
            continue

        # Extract 8-feature representation
        row = {
            "filepath": str(func.get("filepath", "")),
            "function_name": str(func.get("function_name", "")),
            "cyclomatic_complexity": float(func.get("cyclomatic_complexity", 0.0) or 0.0),
            "loc": float(func.get("loc", 0.0) or 0.0),
            "maintainability_index": float(func.get("maintainability_index", 100.0) or 100.0),
            "churn": float(func.get("churn", 0.0) or 0.0),
            "unique_developers": float(func.get("unique_developers", 1.0) or 1.0),
            "bug_fix_commits": float(func.get("bug_fix_commits", 0.0) or 0.0),
            "file_age_days": float(func.get("file_age_days", 1.0) or 1.0),
            "recent_churn": float(func.get("recent_churn", 0.0) or 0.0),
            LABEL: int(label),
            "run_id": db_run_id,
            "source": "autotestarmy_feedback",
        }
        rows.append(row)

    if not rows:
        logger.info("No conclusive feedback labels extracted from test run.")
        return 0

    new_df = pd.DataFrame(rows)

    if feedback_file.exists():
        try:
            existing_df = pd.read_csv(feedback_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate by filepath and function_name, keeping newest outcome
            combined_df = combined_df.drop_duplicates(
                subset=["filepath", "function_name"],
                keep="last"
            )
        except Exception as exc:
            logger.warning("Failed to read existing feedback file %s: %s", feedback_file, exc)
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df.to_csv(feedback_file, index=False)
    logger.info("Recorded %d feedback samples -> %s", len(rows), feedback_file)
    return len(rows)


def get_feedback_stats(feedback_file: Path = FEEDBACK_DATA_PATH) -> dict[str, Any]:
    """Retrieve statistical summary of accumulated feedback records."""
    feedback_file = Path(feedback_file)
    if not feedback_file.exists():
        return {
            "total_samples": 0,
            "defective_count": 0,
            "clean_count": 0,
            "defect_ratio": 0.0,
            "feedback_file": str(feedback_file),
            "ready_for_retraining": False,
        }

    try:
        df = pd.read_csv(feedback_file)
        if df.empty or LABEL not in df.columns:
            return {
                "total_samples": 0,
                "defective_count": 0,
                "clean_count": 0,
                "defect_ratio": 0.0,
                "feedback_file": str(feedback_file),
                "ready_for_retraining": False,
            }

        total = len(df)
        defective = int((df[LABEL] == 1).sum())
        clean = int((df[LABEL] == 0).sum())
        defect_ratio = round(defective / total, 4) if total > 0 else 0.0

        return {
            "total_samples": total,
            "defective_count": defective,
            "clean_count": clean,
            "defect_ratio": defect_ratio,
            "feedback_file": str(feedback_file),
            "ready_for_retraining": total >= 5,
        }
    except Exception as exc:
        return {
            "total_samples": 0,
            "error": str(exc),
            "feedback_file": str(feedback_file),
            "ready_for_retraining": False,
        }
