# scripts/download_datasets.py
"""
Downloads NASA MDP and PROMISE defect prediction datasets.
Run: python scripts/download_datasets.py
Places files in data/raw/nasa_mdp/ and data/raw/promise/
"""
import urllib.request
from pathlib import Path
from tqdm import tqdm

NASA_BASE = "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/OriginalData/MDP"
KC2_URL = "https://raw.githubusercontent.com/klainfo/DefectData/master/inst/extdata/terapromise/mccabe/kc2.arff"
NASA_FILES = [
    "CM1.arff", "KC1.arff", "KC2.arff", "MC1.arff",
    "MW1.arff", "PC1.arff", "PC3.arff", "PC4.arff"
]

PROMISE_BASE = "https://raw.githubusercontent.com/klainfo/DefectData/master/inst/extdata/terapromise/ck"
PROMISE_FILES = [
    "ant-1.7.csv", "camel-1.6.csv", "jedit-4.3.csv",
    "log4j-1.2.csv", "poi-3.0.csv"
]


def download_file(url: str, dest: str) -> bool:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        return False


def main():
    print("Downloading NASA MDP datasets...")
    success = 0
    for fname in tqdm(NASA_FILES, desc="NASA MDP"):
        url = KC2_URL if fname == "KC2.arff" else f"{NASA_BASE}/{fname}"
        dest = f"data/raw/nasa_mdp/{fname}"
        if download_file(url, dest):
            success += 1
    print(f"  Downloaded {success}/{len(NASA_FILES)} NASA MDP files")

    print("\nDownloading PROMISE datasets...")
    success = 0
    for fname in tqdm(PROMISE_FILES, desc="PROMISE"):
        url = f"{PROMISE_BASE}/{fname}"
        dest = f"data/raw/promise/{fname}"
        if download_file(url, dest):
            success += 1
    print(f"  Downloaded {success}/{len(PROMISE_FILES)} PROMISE files")

    print("\nDone. Check data/raw/ for files.")


if __name__ == "__main__":
    main()
