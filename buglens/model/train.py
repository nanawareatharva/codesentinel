"""Train the BugLens LightGBM classifier and persist its inference artifacts."""
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from buglens.config import FEATURES, LABEL, LIGHTGBM_PARAMS, MODEL_DIR

warnings.filterwarnings("ignore")


def _smote(y: pd.Series) -> SMOTE:
    """Select a valid number of SMOTE neighbours for small datasets."""
    minority_count = int(y.value_counts().min())
    if minority_count < 2:
        raise ValueError("At least two examples of each class are required for SMOTE")
    return SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))


def _cross_validate(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Evaluate folds without leaking SMOTE/scaler information into validation data."""
    smallest_class = int(y.value_counts().min())
    splits = min(5, smallest_class)
    if splits < 2:
        raise ValueError("At least two samples in each class are required for CV")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, validation_index) in enumerate(cv.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[validation_index]
        X_resampled, y_resampled = _smote(y_train).fit_resample(X_train, y_train)
        scaler = StandardScaler().fit(X_resampled)
        model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
        model.fit(scaler.transform(X_resampled), y_resampled)
        score = roc_auc_score(y_valid, model.predict_proba(scaler.transform(X_valid))[:, 1])
        scores.append(score)
        print(f"    Fold {fold}: AUC = {score:.4f} {'OK' if score >= .70 else 'WARN'}")
    return np.asarray(scores)


def train(data_path: str = "data/processed/features.csv") -> dict:
    """Fit, evaluate and save a LightGBM model, scaler and SHAP explainer."""
    print("=" * 60 + "\nCodeSentinel — LightGBM Training Pipeline\n" + "=" * 60)
    df = pd.read_csv(data_path)
    missing = set(FEATURES + [LABEL]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    X = df[FEATURES].fillna(0).astype(float)
    y = df[LABEL].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training requires both clean and defective labels")
    print(f"\n[1] Loaded {len(df)} rows; defective={y.mean():.1%}")

    print("\n[2] Stratified cross-validation (SMOTE only in each training fold)...")
    cv_scores = _cross_validate(X, y)
    print(f"    Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    if cv_scores.mean() >= .75:
        print("    Meets target threshold AUC >= 0.75")
    elif cv_scores.mean() < .70:
        print("    WARNING: Mean AUC < 0.70; inspect source metric mapping and dataset quality")

    print("\n[3] Applying SMOTE and fitting final scaler/model...")
    X_resampled, y_resampled = _smote(y).fit_resample(X, y)
    scaler = StandardScaler().fit(X_resampled)
    X_scaled = scaler.transform(X_resampled)
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(X_scaled, y_resampled)

    y_prediction = model.predict(X_scaled)
    y_probability = model.predict_proba(X_scaled)[:, 1]
    print("\n[4] Resampled training-set evaluation:")
    print(f"    AUC-ROC: {roc_auc_score(y_resampled, y_probability):.4f}")
    print(f"    F1: {f1_score(y_resampled, y_prediction):.4f}")
    print(f"    Precision: {precision_score(y_resampled, y_prediction, zero_division=0):.4f}")
    print(f"    Recall: {recall_score(y_resampled, y_prediction, zero_division=0):.4f}")
    print(classification_report(y_resampled, y_prediction, target_names=["Clean", "Defective"], zero_division=0))
    print("    Confusion matrix:\n", confusion_matrix(y_resampled, y_prediction))
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n[5] Feature importance:\n", importance.to_string())

    directory = Path(MODEL_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "lightgbm_model.pkl")
    joblib.dump(scaler, directory / "scaler.pkl")
    print("\n[6] Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    # Perform a small eager validation so corrupt/incompatible artifacts fail early.
    _ = explainer.shap_values(X_scaled[: min(3, len(X_scaled))])
    joblib.dump(explainer, directory / "shap_explainer.pkl")
    print(f"    Saved artifacts in {directory}")
    return {"model": model, "scaler": scaler, "explainer": explainer,
            "cv_auc_mean": float(cv_scores.mean()), "cv_auc_std": float(cv_scores.std())}


if __name__ == "__main__":
    train()
