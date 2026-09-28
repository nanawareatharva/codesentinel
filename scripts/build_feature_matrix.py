"""Build the unified CodeSentinel training feature matrix from ARFF files."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buglens.config import FEATURES, LABEL
from buglens.mining.dataset_loader import load_all_datasets


def build() -> pd.DataFrame:
    """Load, validate, clean and persist the model feature matrix."""
    print("=" * 60 + "\nBuilding CodeSentinel Feature Matrix\n" + "=" * 60)
    df = load_all_datasets()
    missing = [feature for feature in FEATURES if feature not in df]
    if missing:
        print(f"  WARNING: missing mapped features {missing}; filling with 0")
        for feature in missing:
            df[feature] = 0.0
    if LABEL not in df:
        raise ValueError(f"Label column {LABEL!r} missing from loaded datasets")
    columns = FEATURES + [LABEL, "source"]
    output = df[[column for column in columns if column in df]].copy()
    output[FEATURES] = output[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    output[LABEL] = output[LABEL].astype(int)
    if output[LABEL].nunique() != 2:
        raise ValueError("Training data must contain clean and defective samples")
    destination = Path("data/processed/features.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Saved {destination} — shape {output.shape}; defective={output[LABEL].mean():.1%}")
    return output


if __name__ == "__main__":
    build()
