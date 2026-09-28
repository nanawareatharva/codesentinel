"""Download NASA MDP and PROMISE defect-prediction ARFF datasets.

Run with ``python scripts/download_datasets.py``. Files are placed in
``data/raw/nasa_mdp`` and ``data/raw/promise``.
"""
from pathlib import Path
import urllib.request

from tqdm import tqdm

NASA_BASE = "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/OriginalData/MDP"
NASA_FILES = ["CM1.arff", "KC1.arff", "KC2.arff", "MC1.arff", "MW1.arff", "PC1.arff", "PC3.arff", "PC4.arff"]
PROMISE_BASE = "https://raw.githubusercontent.com/klainfo/DefectData/master/inst/extdata/ck"
PROMISE_FILES = ["ant-1.7.arff", "camel-1.6.arff", "jedit-4.3.arff", "log4j-1.2.arff", "poi-3.0.arff"]


def download_file(url: str, dest: str) -> bool:
    """Download a file, returning False instead of aborting a batch on errors."""
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as exc:  # network failures should not hide remaining files
        print(f"  FAILED {url}: {exc}")
        return False


def _download_group(label: str, base: str, files: list[str], directory: str) -> None:
    print(f"Downloading {label} datasets...")
    successful = sum(
        download_file(f"{base}/{filename}", str(Path(directory) / filename))
        for filename in tqdm(files, desc=label)
    )
    print(f"  Downloaded {successful}/{len(files)} {label} files")


def main() -> None:
    _download_group("NASA MDP", NASA_BASE, NASA_FILES, "data/raw/nasa_mdp")
    _download_group("PROMISE", PROMISE_BASE, PROMISE_FILES, "data/raw/promise")
    print("\nDone. Check data/raw/ for files.")


if __name__ == "__main__":
    main()
