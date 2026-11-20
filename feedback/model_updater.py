"""
CodeSentinel — Automated Model Updater & Feedback Retrainer
===========================================================
Merges accumulated dynamic feedback labels with baseline benchmark data
(NASA MDP + PROMISE) and retrains the LightGBM classifier and SHAP explainer.
Refreshes saved production artifacts in buglens/model/saved/.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import pandas as pd

from buglens.config import FEATURES, LABEL, MODEL_DIR
from buglens.model.train import train
from feedback.aggregator import FEEDBACK_DATA_PATH

logger = logging.getLogger("codesentinel.model_updater")

BASELINE_FEATURES_PATH = Path("data") / "processed" / "features.csv"
AUGMENTED_FEATURES_PATH = Path("data") / "processed" / "features_with_feedback.csv"


def prepare_augmented_dataset(
    baseline_path: Path = BASELINE_FEATURES_PATH,
    feedback_path: Path = FEEDBACK_DATA_PATH,
    output_path: Path = AUGMENTED_FEATURES_PATH,
) -> Path:
    """
    Merges baseline benchmark features with newly accumulated feedback records.
    Ensures 8-feature schema alignment and writes features_with_feedback.csv.
    """
    baseline_path = Path(baseline_path)
    feedback_path = Path(feedback_path)
    output_path = Path(output_path)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline feature matrix not found at: {baseline_path}")

    base_df = pd.read_csv(baseline_path)
    required_cols = FEATURES + [LABEL]

    # Verify baseline columns
    missing_base = set(required_cols) - set(base_df.columns)
    if missing_base:
        raise ValueError(f"Baseline dataset missing required columns: {sorted(missing_base)}")

    base_clean = base_df[required_cols].copy()

    if feedback_path.exists():
        try:
            fb_df = pd.read_csv(feedback_path)
            # Filter and align feedback columns
            missing_fb = set(required_cols) - set(fb_df.columns)
            if not missing_fb:
                fb_clean = fb_df[required_cols].copy()
                combined_df = pd.concat([base_clean, fb_clean], ignore_index=True)
                logger.info(
                    "Augmented dataset prepared: %d baseline rows + %d feedback rows = %d total",
                    len(base_clean),
                    len(fb_clean),
                    len(combined_df),
                )
            else:
                logger.warning("Feedback file %s missing columns %s; using baseline only.", feedback_path, missing_fb)
                combined_df = base_clean
        except Exception as exc:
            logger.warning("Failed to parse feedback file %s: %s; using baseline only.", feedback_path, exc)
            combined_df = base_clean
    else:
        combined_df = base_clean

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    return output_path


def retrain_with_feedback(
    min_feedback_samples: int = 1,
    baseline_path: Path = BASELINE_FEATURES_PATH,
    feedback_path: Path = FEEDBACK_DATA_PATH,
    output_path: Path = AUGMENTED_FEATURES_PATH,
) -> dict[str, Any]:
    """
    Checks feedback thresholds, prepares the augmented feature matrix,
    and executes full cross-validated LightGBM training and SHAP artifact generation.
    """
    feedback_path = Path(feedback_path)
    feedback_count = 0

    if feedback_path.exists():
        try:
            fb_df = pd.read_csv(feedback_path)
            feedback_count = len(fb_df)
        except Exception:
            feedback_count = 0

    if feedback_count < min_feedback_samples:
        return {
            "retrained": False,
            "message": f"Insufficient feedback samples ({feedback_count} < {min_feedback_samples}). Retraining skipped.",
            "feedback_samples": feedback_count,
            "timestamp": datetime.now().isoformat(),
        }

    augmented_csv = prepare_augmented_dataset(
        baseline_path=baseline_path,
        feedback_path=feedback_path,
        output_path=output_path,
    )

    # Trigger core training pipeline
    train_results = train(data_path=str(augmented_csv))

    return {
        "retrained": True,
        "message": f"LightGBM successfully retrained with {feedback_count} feedback samples.",
        "feedback_samples": feedback_count,
        "cv_auc_mean": train_results.get("cv_auc_mean"),
        "cv_auc_std": train_results.get("cv_auc_std"),
        "artifacts_directory": str(MODEL_DIR),
        "timestamp": datetime.now().isoformat(),
    }
