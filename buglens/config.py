"""
CodeSentinel — Central Configuration
=====================================
THE FEATURE SCHEMA IS DEFINED HERE AND ONLY HERE.
Import FEATURES from this module in every file that needs the feature list.
Never hardcode the feature names anywhere else.

If you change FEATURES, you MUST update:
  - buglens/model/train.py
  - buglens/model/predict.py
  - buglens/model/shap_explainer.py
  - feedback/model_updater.py  (Phase 6)
"""
from pathlib import Path

# ── Feature Schema ────────────────────────────────────────────────────────────
# 8 features used for both training (on NASA MDP + PROMISE)
# and inference (on live Python repositories).
FEATURES = [
    "cyclomatic_complexity",   # Radon: number of independent paths through a function
    "loc",                     # Radon: lines of code in the function
    "maintainability_index",   # Radon: composite score 0-100 (higher = more maintainable)
    "churn",                   # PyDriller: total lines added + deleted across all commits
    "unique_developers",       # PyDriller: number of distinct author emails
    "bug_fix_commits",         # PyDriller: commits containing 'fix/bug/patch/error' keywords
    "file_age_days",           # PyDriller: days between first and most recent commit
    "recent_churn",            # PyDriller: churn in the last 90 days
]

LABEL = "defects"             # Binary label: 0 = clean, 1 = defective

# ── Risk Thresholds ──────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "LOW": (0.0, 0.33),
    "MEDIUM": (0.33, 0.66),
    "HIGH": (0.66, 1.0),
}

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = "buglens/model/saved"
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FEATURES_CSV = "data/processed/features.csv"

# ── Model Hyperparameters ─────────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
    "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
    "verbose": -1, "n_estimators": 300, "class_weight": "balanced",
    "random_state": 42,
}

# ── Keywords for Bug-Fix Commit Detection ─────────────────────────────────────
BUG_FIX_KEYWORDS = [
    "fix", "bug", "patch", "error", "defect", "hotfix", "repair",
    "resolve", "issue", "crash",
]
