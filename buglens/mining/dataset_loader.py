"""Load ARFF defect datasets into CodeSentinel's central feature schema."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from buglens.config import FEATURES, LABEL

# NASA MDP uses McCabe and Halstead metrics.
NASA_COLUMN_MAP = {
    "v(g)": "cyclomatic_complexity", "loc": "loc", "l": "maintainability_index",
    "lOCode": "churn", "uniq_Op": "unique_developers", "b": "bug_fix_commits",
    "n": "file_age_days", "t": "recent_churn", "defects": LABEL,
    # Current klainfo/NASADefectDataset files use these descriptive names.
    "CYCLOMATIC_COMPLEXITY": "cyclomatic_complexity",
    "LOC_TOTAL": "loc",
    "HALSTEAD_LEVEL": "maintainability_index",
    "LOC_CODE_AND_COMMENT": "churn",
    "NUM_UNIQUE_OPERATORS": "unique_developers",
    "HALSTEAD_ERROR_EST": "bug_fix_commits",
    "HALSTEAD_LENGTH": "file_age_days",
    "HALSTEAD_PROG_TIME": "recent_churn",
    "Defective": LABEL,
}

# PROMISE data commonly uses CK object-oriented metrics.
PROMISE_COLUMN_MAP = {
    "wmc": "cyclomatic_complexity", "loc": "loc", "cam": "maintainability_index",
    "ce": "churn", "npm": "unique_developers", "lcom3": "bug_fix_commits",
    "dam": "file_age_days", "moa": "recent_churn", "bug": LABEL,
}


def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Decode categorical values emitted by scipy's ARFF reader."""
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(
                lambda value: value.decode("utf-8").strip() if isinstance(value, bytes) else value
            )
    return df


def _encode_label(series: pd.Series) -> pd.Series:
    """Normalize known binary and count-style defect labels to binary integers."""
    normalized = series.astype(str).str.lower().str.strip()
    numeric = pd.to_numeric(normalized, errors="coerce")
    # NASA labels are often numeric defect counts; a positive count is defective.
    return np.where(numeric.notna(), (numeric > 0).astype(int),
                    normalized.isin({"true", "yes", "y", "buggy", "defective", "1"}).astype(int))


def _find_column(columns: pd.Index, desired: str) -> str | None:
    """Find a column without making ARFF capitalization significant."""
    lookup = {str(column).lower(): str(column) for column in columns}
    return lookup.get(desired.lower())


def load_arff(filepath: str, column_map: dict[str, str]) -> pd.DataFrame:
    """Load one ARFF or current PROMISE CSV file into the unified schema."""
    try:
        if Path(filepath).suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        else:
            raw, _metadata = arff.loadarff(filepath)
            df = _decode_bytes(pd.DataFrame(raw))
    except Exception as exc:
        print(f"  Cannot load {filepath}: {exc}")
        return pd.DataFrame()

    rename_map = {}
    for source, target in column_map.items():
        actual = _find_column(df.columns, source)
        if actual is not None:
            rename_map[actual] = target
    if not rename_map:
        print(f"  No mappable columns in {Path(filepath).name}. Check column_map.")
        return pd.DataFrame()

    mapped = df[list(rename_map)].rename(columns=rename_map)
    if LABEL not in mapped:
        print(f"  WARNING: No label column found in {Path(filepath).name}")
        return pd.DataFrame()
    mapped[LABEL] = _encode_label(mapped[LABEL]).astype(int)
    return mapped


def load_all_datasets() -> pd.DataFrame:
    """Load all locally downloaded NASA MDP and PROMISE ARFF datasets."""
    loaded: list[pd.DataFrame] = []
    for label, directory, mapping in (
        ("NASA MDP", Path("data/raw/nasa_mdp"), NASA_COLUMN_MAP),
        ("PROMISE", Path("data/raw/promise"), PROMISE_COLUMN_MAP),
    ):
        if not directory.exists():
            print(f"  {directory} not found. Run scripts/download_datasets.py")
            continue
        files = sorted([*directory.glob("*.arff"), *directory.glob("*.csv")])
        for file in files:
            frame = load_arff(str(file), mapping)
            if frame.empty:
                continue
            frame["source"] = file.stem
            loaded.append(frame)
            print(f"  Loaded {label}: {file.name:20s} — {len(frame):5d} rows, defective={frame[LABEL].sum()}")
    if not loaded:
        raise FileNotFoundError("No datasets loaded. Run: python scripts/download_datasets.py")
    combined = pd.concat(loaded, ignore_index=True)
    print(f"\n  TOTAL: {len(combined)} rows from {len(loaded)} datasets")
    return combined
