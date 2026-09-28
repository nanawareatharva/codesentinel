"""Print a compact inspection report for downloaded ARFF datasets."""
from pathlib import Path

from scipy.io import arff


def main() -> None:
    files = sorted(Path("data/raw").glob("*/*.arff"))
    if not files:
        raise FileNotFoundError("No ARFF datasets found. Run scripts/download_datasets.py first.")
    for file in files:
        data, metadata = arff.loadarff(file)
        print(f"\n{file}: {len(data)} rows")
        print("Columns:", ", ".join(metadata.names()))


if __name__ == "__main__":
    main()
