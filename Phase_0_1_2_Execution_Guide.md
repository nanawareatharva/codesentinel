# CodeSentinel — Phase 0 · Phase 1 · Phase 2
## Complete Step-by-Step Execution Guide

**Project:** CodeSentinel — AI-Powered Code Quality Platform
**Group 10 | FR. CRCE, Dept. of AI & Data Science | B.E. Final Year Project 2026–27**
**Team:** Atharva Nanaware (10511) · Suhera Siddiqui (10381) · Aum Singh (10383)

---

## Before You Start — Read This

These three phases are **purely local**. No cloud services. No paid APIs. Everything runs on your machine.

| Phase | What It Builds | Time Estimate |
|---|---|---|
| Phase 0 | Project structure, environment, all tools installed | 1–2 days |
| Phase 1 | Data pipeline — git mining + metric extraction + feature CSV | 4–5 days |
| Phase 2 | LightGBM model + SHAP explainer — trained and saved as `.pkl` files | 3–4 days |

**After these three phases are complete, you will have:**
- A fully structured Python project with correct package layout
- A working data pipeline that can mine any Python Git repository
- A trained defect prediction model saved to disk
- A SHAP explainer that explains why any function is flagged as high-risk
- All three verified with passing tests and checkpoint commands

---

# PHASE 0 — Environment Setup & Project Scaffold

## What Phase 0 Produces
- Empty but correctly structured project directory
- Python virtual environment with all 20+ libraries installed
- Git repository initialized with first commit
- Ollama running with phi3:mini model downloaded
- All `__init__.py` files in place so Python treats folders as packages
- `requirements.txt`, `requirements-dev.txt`, `.gitignore` committed

---

## Step 0.1 — Verify System Prerequisites

Run each of these before doing anything else. Fix any that fail before proceeding.

### Check Python Version
```bash
python --version
```
**Expected:** `Python 3.10.x` or higher (3.11 or 3.12 also fine)

If Python < 3.10, download from https://python.org/downloads/ and install.

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### Check Git
```bash
git --version
```
**Expected:** `git version 2.x.x`

If not installed:
```bash
# Ubuntu/Debian
sudo apt install git

# Mac
brew install git

# Windows: download from https://git-scm.com/
```

### Check Node.js (needed for VS Code extension in Phase 7, install now)
```bash
node --version
npm --version
```
**Expected:** `v18.x.x` or higher for node, `9.x.x` or higher for npm

If not installed, download from https://nodejs.org/ (LTS version).

### Install Ollama
Visit https://ollama.ai and download the installer for your OS.

After installing, verify it is running:
```bash
ollama --version
```
**Expected:** `ollama version 0.x.x`

---

## Step 0.2 — Create the Root Project Directory

```bash
mkdir codesentinel
cd codesentinel
```

From this point on, every command in this guide is run from inside `codesentinel/` unless stated otherwise.

---

## Step 0.3 — Initialize Git Repository

```bash
git init
git config user.name "Your Name"
git config user.email "your@email.com"
```

---

## Step 0.4 — Create Python Virtual Environment

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# On Linux / Mac:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

**How to verify it is activated:**
Your terminal prompt should now show `(.venv)` at the beginning. Example:
```
(.venv) atharva@machine:~/codesentinel$
```

Upgrade pip immediately after activating:
```bash
pip install --upgrade pip
```

**Important:** Every time you open a new terminal to work on this project, you must run `source .venv/bin/activate` again. The virtual environment is not automatically activated.

---

## Step 0.5 — Create the Full Directory Structure

Run this block as-is. It creates every folder the project needs across all 8 phases.

```bash
# Core module directories
mkdir -p buglens/mining
mkdir -p buglens/model/saved
mkdir -p buglens/api
mkdir -p autotestarmy/agents
mkdir -p autotestarmy/graph
mkdir -p autotestarmy/tools
mkdir -p autotestarmy/api
mkdir -p feedback
mkdir -p backend

# Data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/demo_repos

# Utility directories
mkdir -p scripts
mkdir -p tests
mkdir -p vscode-extension/src
```

Now create all `__init__.py` files that make Python treat these as packages:

```bash
# Linux / Mac
touch buglens/__init__.py
touch buglens/mining/__init__.py
touch buglens/model/__init__.py
touch buglens/api/__init__.py
touch autotestarmy/__init__.py
touch autotestarmy/agents/__init__.py
touch autotestarmy/graph/__init__.py
touch autotestarmy/tools/__init__.py
touch autotestarmy/api/__init__.py
touch feedback/__init__.py
touch backend/__init__.py
touch tests/__init__.py
```

```bash
# Windows PowerShell (run one by one if needed)
New-Item buglens/__init__.py -type file
New-Item buglens/mining/__init__.py -type file
New-Item buglens/model/__init__.py -type file
New-Item buglens/api/__init__.py -type file
New-Item autotestarmy/__init__.py -type file
New-Item autotestarmy/agents/__init__.py -type file
New-Item autotestarmy/graph/__init__.py -type file
New-Item autotestarmy/tools/__init__.py -type file
New-Item autotestarmy/api/__init__.py -type file
New-Item feedback/__init__.py -type file
New-Item backend/__init__.py -type file
New-Item tests/__init__.py -type file
```

**Verify the structure:**
```bash
# Linux/Mac
find . -type d | grep -v ".venv" | grep -v "__pycache__" | sort

# Windows
tree /F /A
```

**Expected output (directories only):**
```
./autotestarmy
./autotestarmy/agents
./autotestarmy/api
./autotestarmy/graph
./autotestarmy/tools
./backend
./buglens
./buglens/api
./buglens/mining
./buglens/model
./buglens/model/saved
./data
./data/demo_repos
./data/processed
./data/raw
./feedback
./scripts
./tests
./vscode-extension
./vscode-extension/src
```

---

## Step 0.6 — Create requirements.txt

Create a file named `requirements.txt` in the root `codesentinel/` directory with this exact content:

```
# ML & Explainability
lightgbm==4.3.0
scikit-learn==1.4.2
shap==0.45.1
pandas==2.2.2
numpy==1.26.4
radon==6.0.1
imbalanced-learn==0.12.3

# Repository Mining
pydriller==2.6
gitpython==3.1.43

# Agentic AI / LLM
langgraph==0.1.19
langchain==0.2.12
langchain-community==0.2.12
ollama==0.2.1

# Testing & Code Analysis
pytest==8.2.2
pytest-cov==5.0.0
libcst==1.4.0

# Backend
fastapi==0.111.1
uvicorn[standard]==0.30.3
sqlalchemy==2.0.31
pydantic==2.8.2
python-dotenv==1.0.1
httpx==0.27.0

# Utilities
joblib==1.4.2
tqdm==4.66.4
scipy==1.13.1
```

---

## Step 0.7 — Create requirements-dev.txt

Create `requirements-dev.txt` in the root directory:

```
-r requirements.txt
black==24.4.2
isort==5.13.2
pytest-asyncio==0.23.7
```

---

## Step 0.8 — Install All Dependencies

```bash
pip install -r requirements-dev.txt
```

This will take 3–8 minutes depending on your internet speed. You will see many lines of output as packages download.

**After installation, verify the key packages:**
```bash
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import shap; print('SHAP:', shap.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import langgraph; print('LangGraph OK')"
python -c "import pydriller; print('PyDriller OK')"
python -c "import radon; print('Radon OK')"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

Every line should print without error. If any fail, run:
```bash
pip install <package-name> --force-reinstall
```

---

## Step 0.9 — Download Ollama Models

```bash
# Start Ollama server (keep this terminal open, or run in background)
ollama serve &

# Pull the lightweight model (runs on 4GB RAM) — use this for development
ollama pull phi3:mini

# Pull the capable model (runs on 8GB RAM) — use this for final evaluation
ollama pull llama3.1:8b
```

The downloads will take 5–20 minutes depending on internet speed.
- `phi3:mini` is approximately 2.3 GB
- `llama3.1:8b` is approximately 4.7 GB

**Verify models are available:**
```bash
ollama list
```

**Expected:**
```
NAME            ID              SIZE    MODIFIED
llama3.1:8b     <hash>          4.7 GB  ...
phi3:mini       <hash>          2.3 GB  ...
```

**Quick test — confirm LLM responds:**
```bash
ollama run phi3:mini "Write one pytest function for a function called add(a, b)"
```

You should see generated Python test code in the terminal.

---

## Step 0.10 — Create .gitignore

Create `.gitignore` in the root directory:

```gitignore
# Virtual environment — never commit this
.venv/

# Python bytecode
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment variables
.env
.env.local

# Raw datasets — too large for git
data/raw/
data/processed/feedback_labels.csv

# Trained model artifacts — regenerated by train.py
buglens/model/saved/*.pkl
buglens/model/saved/*.joblib

# VS Code extension build artifacts
vscode-extension/node_modules/
vscode-extension/out/
*.vsix

# Test coverage artifacts
.coverage
htmlcov/
coverage.xml

# SQLite database — local data
*.db
*.sqlite3

# OS files
.DS_Store
Thumbs.db

# IDE files
.idea/
.vscode/settings.json
```

---

## Step 0.11 — Create README.md

Create `README.md` in the root:

```markdown
# CodeSentinel

AI-Powered Code Quality Platform: Defect Prediction and Targeted Test Generation

**B.E. Final Year Project 2026–27**
FR. Conceicao Rodrigues College of Engineering, Mumbai
Dept. of Artificial Intelligence & Data Science
Group 10: Atharva Nanaware · Suhera Siddiqui · Aum Singh

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
ollama pull phi3:mini
```

## Run Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

BugLens (Defect Prediction) + AutoTestArmy (Multi-Agent Test Generation) + Feedback Loop
```

---

## Step 0.12 — Create .env.example

Create `.env.example` in the root:

```
# Copy this file to .env and fill in values
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
DATABASE_URL=sqlite:///./codesentinel.db
API_HOST=0.0.0.0
API_PORT=8000
```

Then create the actual `.env`:
```bash
cp .env.example .env
```

---

## Step 0.13 — Initial Git Commit

```bash
git add .
git commit -m "Phase 0: Project scaffold, requirements, directory structure"
```

---

## ✅ Phase 0 Checkpoint — Run All These Before Moving to Phase 1

```bash
# 1. All key imports work
python -c "import lightgbm, shap, fastapi, langgraph, pydriller, radon; print('ALL IMPORTS OK')"

# 2. Ollama has models
ollama list

# 3. Ollama responds
python -c "import ollama; r = ollama.chat(model='phi3:mini', messages=[{'role':'user','content':'say ok'}]); print(r['message']['content'])"

# 4. Directory structure exists
ls buglens/mining/ buglens/model/ buglens/api/ autotestarmy/agents/ data/raw/

# 5. __init__.py files exist
ls buglens/__init__.py buglens/mining/__init__.py backend/__init__.py

# 6. Git has first commit
git log --oneline
```

**Every one of these must pass before starting Phase 1.**

---
---

# PHASE 1 — Data Collection & Feature Engineering

## What Phase 1 Produces
- `data/raw/` — NASA MDP + PROMISE dataset files downloaded
- `buglens/mining/git_miner.py` — mines any Python Git repo for commit/churn/developer features
- `buglens/mining/metric_extractor.py` — extracts cyclomatic complexity, LOC, and other code metrics
- `scripts/build_feature_matrix.py` — combines dataset columns into a unified 8-feature + label CSV
- `data/processed/features.csv` — the training data for Phase 2
- `tests/test_miner.py` — unit tests for the mining scripts

---

## Step 1.1 — Understand the Feature Schema (Read First)

**This schema is the single most important constant in the entire project.** It must be identical in Phase 1 (extraction), Phase 2 (training), Phase 2 (prediction), and Phase 6 (retraining). Write it down and never change it without updating all files.

```python
FEATURES = [
    "cyclomatic_complexity",   # From Radon — number of independent code paths
    "loc",                     # Lines of code in this function
    "maintainability_index",   # Radon composite score (0-100, higher = more maintainable)
    "churn",                   # Total lines added + deleted across all commits (PyDriller)
    "unique_developers",       # Number of distinct author emails (PyDriller)
    "bug_fix_commits",         # Commits with 'fix/bug/patch/error' in message (PyDriller)
    "file_age_days",           # Days between first and most recent commit (PyDriller)
    "recent_churn",            # Churn in the last 90 days (PyDriller)
]
LABEL = "defects"             # Binary: 0 = clean, 1 = defective
```

Save this list in a config file so every script imports it from one place.

Create `buglens/config.py`:

```python
# buglens/config.py
# THE FEATURE SCHEMA — never change column names or order without updating:
# - buglens/model/train.py
# - buglens/model/predict.py
# - buglens/model/shap_explainer.py
# - feedback/model_updater.py

FEATURES = [
    "cyclomatic_complexity",
    "loc",
    "maintainability_index",
    "churn",
    "unique_developers",
    "bug_fix_commits",
    "file_age_days",
    "recent_churn",
]

LABEL = "defects"

RISK_THRESHOLDS = {
    "LOW": (0.0, 0.33),
    "MEDIUM": (0.33, 0.66),
    "HIGH": (0.66, 1.0),
}

MODEL_DIR = "buglens/model/saved"
```

---

## Step 1.2 — Download NASA MDP Datasets

NASA MDP (Metrics Data Program) datasets are the primary training data. These are public datasets from real NASA software systems.

### Option A — Manual Download (Recommended)

Visit: **https://github.com/klainfo/NASADefectDataset**

Download these `.arff` files and place them in `data/raw/nasa_mdp/`:
- `CM1.arff`
- `KC1.arff`
- `KC2.arff`
- `MC1.arff`
- `MW1.arff`
- `PC1.arff`
- `PC3.arff`
- `PC4.arff`

```bash
mkdir -p data/raw/nasa_mdp
# Place downloaded files here
```

### Option B — Download Script

Create `scripts/download_datasets.py`:

```python
# scripts/download_datasets.py
"""
Downloads NASA MDP defect datasets from GitHub.
Run: python scripts/download_datasets.py
"""
import urllib.request
import os
from pathlib import Path
from tqdm import tqdm

NASA_BASE = "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/data"
NASA_FILES = ["CM1.arff", "KC1.arff", "KC2.arff", "MC1.arff", "MW1.arff",
              "PC1.arff", "PC3.arff", "PC4.arff"]

PROMISE_BASE = "https://raw.githubusercontent.com/klainfo/DefectData/master"
PROMISE_FILES = {
    "ant-1.7.arff": "ant-1.7.arff",
    "camel-1.6.arff": "camel-1.6.arff",
    "jedit-4.3.arff": "jedit-4.3.arff",
    "log4j-1.2.arff": "log4j-1.2.arff",
    "poi-3.0.arff": "poi-3.0.arff",
}

def download(url, dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to: {dest}")
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    for fname in tqdm(NASA_FILES, desc="NASA MDP"):
        download(f"{NASA_BASE}/{fname}", f"data/raw/nasa_mdp/{fname}")

    for fname in tqdm(PROMISE_FILES, desc="PROMISE"):
        download(f"{PROMISE_BASE}/{fname}", f"data/raw/promise/{fname}")

    print("\nDownload complete. Check data/raw/ for files.")
```

Run it:
```bash
python scripts/download_datasets.py
```

**Verify files are downloaded:**
```bash
ls data/raw/nasa_mdp/
ls data/raw/promise/
```

---

## Step 1.3 — Load and Inspect a Dataset

Before writing the full pipeline, understand the data format.

Create `scripts/inspect_dataset.py`:

```python
# scripts/inspect_dataset.py
"""Run this to understand the NASA MDP .arff format before building the pipeline."""
from scipy.io import arff
import pandas as pd

# Load one dataset
data, meta = arff.loadarff("data/raw/nasa_mdp/KC1.arff")
df = pd.DataFrame(data)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 3 rows:\n", df.head(3))
print("\nDefect distribution:\n", df["defects"].value_counts())
print("\nNull counts:\n", df.isnull().sum())
```

Run it:
```bash
python scripts/inspect_dataset.py
```

**What to note from the output:**
- The column names in the NASA MDP dataset use different names than our feature schema. You will map them.
- The `defects` column may contain `b'true'`/`b'false'` (byte strings) — these must be converted to 1/0.
- There will be some null values — these are handled in the preprocessing step.

---

## Step 1.4 — Build the Dataset Loader

Create `buglens/mining/dataset_loader.py`:

```python
# buglens/mining/dataset_loader.py
"""
Loads NASA MDP and PROMISE .arff files and maps their columns
to the CodeSentinel 8-feature schema.
"""
from scipy.io import arff
import pandas as pd
import numpy as np
from pathlib import Path


# NASA MDP column mapping → our feature schema
# NASA MDP uses Halstead + McCabe complexity metrics
# We map the closest equivalents to our 8 features
NASA_COLUMN_MAP = {
    # NASA column name    →    our feature name
    "v(g)": "cyclomatic_complexity",     # McCabe cyclomatic complexity
    "loc": "loc",                         # Lines of code
    "l": "maintainability_index",         # Halstead level (proxy for MI)
    "lOCode": "churn",                    # Proxy: lines of code change
    "uniq_Op": "unique_developers",       # Proxy: unique operators
    "b": "bug_fix_commits",               # Halstead estimated bugs (proxy)
    "n": "file_age_days",                 # Halstead vocabulary (proxy)
    "t": "recent_churn",                  # Halstead time (proxy)
    "defects": "defects",
}

# PROMISE .arff datasets use slightly different column names
PROMISE_COLUMN_MAP = {
    "wmc": "cyclomatic_complexity",       # Weighted methods per class
    "loc": "loc",
    "cam": "maintainability_index",       # Cohesion among methods (proxy)
    "ce": "churn",                        # Efferent coupling (proxy)
    "npm": "unique_developers",           # Number of public methods (proxy)
    "lcom3": "bug_fix_commits",           # LCOM3 (proxy)
    "dam": "file_age_days",               # Data access metric (proxy)
    "moa": "recent_churn",               # Measure of aggregation (proxy)
    "bug": "defects",
}


def load_arff(filepath: str, column_map: dict) -> pd.DataFrame:
    """Load one .arff file and return a DataFrame with our 8-feature schema."""
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Decode byte strings (arff files encode strings as bytes)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    # Keep only columns we can map
    available = {k: v for k, v in column_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Convert defect label to binary integer
    if "defects" in df.columns:
        df["defects"] = df["defects"].apply(
            lambda x: 1 if str(x).lower() in ("true", "yes", "1", "1.0") else 0
        )

    return df


def load_all_datasets() -> pd.DataFrame:
    """Load all NASA MDP and PROMISE datasets and concatenate into one DataFrame."""
    all_dfs = []

    # Load NASA MDP
    nasa_dir = Path("data/raw/nasa_mdp")
    for arff_file in nasa_dir.glob("*.arff"):
        try:
            df = load_arff(str(arff_file), NASA_COLUMN_MAP)
            df["source"] = arff_file.stem  # Track which dataset each row came from
            all_dfs.append(df)
            print(f"  Loaded NASA MDP: {arff_file.name} — {len(df)} rows")
        except Exception as e:
            print(f"  SKIP {arff_file.name}: {e}")

    # Load PROMISE
    promise_dir = Path("data/raw/promise")
    for arff_file in promise_dir.glob("*.arff"):
        try:
            df = load_arff(str(arff_file), PROMISE_COLUMN_MAP)
            df["source"] = arff_file.stem
            all_dfs.append(df)
            print(f"  Loaded PROMISE: {arff_file.name} — {len(df)} rows")
        except Exception as e:
            print(f"  SKIP {arff_file.name}: {e}")

    if not all_dfs:
        raise FileNotFoundError("No dataset files found in data/raw/. Run download_datasets.py first.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    return combined
```

---

## Step 1.5 — Write the Git Miner

Create `buglens/mining/git_miner.py`:

```python
# buglens/mining/git_miner.py
"""
Mines a Git repository's commit history using PyDriller.
Returns per-file features: churn, developer count, bug-fix commits, file age, recent churn.
These features are used BOTH in training (via proxy mapping) and at inference time.
"""
from pydriller import Repository
from collections import defaultdict
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path


BUG_FIX_KEYWORDS = ["fix", "bug", "patch", "error", "defect", "hotfix", "repair", "resolve"]


def mine_repository(repo_path: str, since=None, to=None) -> pd.DataFrame:
    """
    Traverses all commits in a local Git repository.
    Returns one row per Python file with commit-level features.

    Parameters:
        repo_path : str — absolute or relative path to the git repository
        since     : datetime (optional) — only count commits after this date
        to        : datetime (optional) — only count commits before this date

    Returns:
        pd.DataFrame with columns:
            filepath, commit_count, churn, unique_developers,
            bug_fix_commits, file_age_days, recent_churn
    """
    stats = defaultdict(lambda: {
        "commit_count": 0,
        "churn": 0,
        "developers": set(),
        "bug_fix_commits": 0,
        "first_commit_date": None,
        "last_commit_date": None,
        "recent_churn": 0,
    })

    ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)

    print(f"Mining repository: {repo_path}")
    commit_count = 0

    for commit in Repository(repo_path, since=since, to=to).traverse_commits():
        commit_count += 1
        if commit_count % 100 == 0:
            print(f"  Processed {commit_count} commits...")

        for mod in commit.modified_files:
            # Only track Python files
            if not (mod.filename.endswith(".py")):
                continue

            fp = mod.new_path or mod.old_path
            if fp is None:
                continue

            s = stats[fp]
            s["commit_count"] += 1
            s["churn"] += (mod.added_lines or 0) + (mod.deleted_lines or 0)
            s["developers"].add(commit.author.email.lower())

            # Check if this is a bug-fix commit
            msg_lower = commit.msg.lower()
            if any(kw in msg_lower for kw in BUG_FIX_KEYWORDS):
                s["bug_fix_commits"] += 1

            # Track first and last commit dates
            commit_date = commit.author_date
            if commit_date.tzinfo is None:
                commit_date = commit_date.replace(tzinfo=timezone.utc)

            if s["first_commit_date"] is None:
                s["first_commit_date"] = commit_date
            s["last_commit_date"] = commit_date

            # Recent churn — last 90 days only
            if commit_date >= ninety_days_ago:
                s["recent_churn"] += (mod.added_lines or 0) + (mod.deleted_lines or 0)

    print(f"  Total commits processed: {commit_count}")
    print(f"  Total Python files found: {len(stats)}")

    # Build result DataFrame
    rows = []
    for filepath, s in stats.items():
        age_days = 0
        if s["first_commit_date"] and s["last_commit_date"]:
            delta = s["last_commit_date"] - s["first_commit_date"]
            age_days = max(int(delta.days), 1)

        rows.append({
            "filepath": filepath,
            "commit_count": s["commit_count"],
            "churn": s["churn"],
            "unique_developers": len(s["developers"]),
            "bug_fix_commits": s["bug_fix_commits"],
            "file_age_days": age_days,
            "recent_churn": s["recent_churn"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: No Python files found in repository history.")
    return df
```

---

## Step 1.6 — Write the Metric Extractor

Create `buglens/mining/metric_extractor.py`:

```python
# buglens/mining/metric_extractor.py
"""
Extracts per-function code complexity metrics from Python source files.
Uses Radon for cyclomatic complexity and maintainability index.
Uses Python's built-in ast module for structural metrics.
"""
import ast
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw
from pathlib import Path
import pandas as pd


def extract_file_metrics(filepath: str) -> list[dict]:
    """
    Analyzes one Python file and returns per-function metric dictionaries.

    Parameters:
        filepath : str — path to a .py file

    Returns:
        list of dicts, one per function/method found in the file.
        Each dict has: filepath, function_name, line_start, line_end,
                       cyclomatic_complexity, loc, maintainability_index,
                       max_nesting_depth, num_returns, num_try_except
    """
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError) as e:
        print(f"  Cannot read {filepath}: {e}")
        return []

    if not source.strip():
        return []

    results = []

    try:
        # Radon cyclomatic complexity — returns list of Function/Class blocks
        cc_results = radon_cc.cc_visit(source)

        # Radon maintainability index — one score per file
        try:
            mi_score = radon_metrics.mi_visit(source, multi=True)
            mi_score = round(float(mi_score), 2)
        except Exception:
            mi_score = 50.0  # Default mid-range if MI fails

        # AST-based structural metrics
        ast_metrics = _extract_ast_metrics(source)

    except SyntaxError as e:
        # File has syntax errors — skip it
        return []
    except Exception as e:
        return []

    for block in cc_results:
        # Only include functions and methods (not classes)
        if block.type not in ("function", "method"):
            continue

        # Skip very small helper functions (< 3 lines) — too trivial to predict
        func_loc = (block.endline or block.lineno) - block.lineno + 1
        if func_loc < 3:
            continue

        func_ast = ast_metrics.get(block.name, {})

        results.append({
            "filepath": str(filepath),
            "function_name": block.name,
            "line_start": block.lineno,
            "line_end": block.endline or block.lineno,
            "cyclomatic_complexity": block.complexity,
            "loc": func_loc,
            "maintainability_index": mi_score,
            "max_nesting_depth": func_ast.get("max_nesting_depth", 0),
            "num_returns": func_ast.get("num_returns", 0),
            "num_try_except": func_ast.get("num_try_except", 0),
        })

    return results


def _extract_ast_metrics(source: str) -> dict:
    """
    Walk the AST to get per-function structural metrics that Radon doesn't provide.
    Returns dict keyed by function name.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    function_metrics = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        metrics = {"max_nesting_depth": 0, "num_returns": 0, "num_try_except": 0}

        # Walk this function's body to collect metrics
        class _Visitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.max_depth = 0

            def _enter_scope(self, node):
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                self.generic_visit(node)
                self.depth -= 1

            visit_If = _enter_scope
            visit_For = _enter_scope
            visit_While = _enter_scope
            visit_With = _enter_scope

            def visit_Try(self, node):
                metrics["num_try_except"] += 1
                self.generic_visit(node)

            def visit_Return(self, node):
                metrics["num_returns"] += 1

        v = _Visitor()
        v.visit(node)
        metrics["max_nesting_depth"] = v.max_depth
        function_metrics[node.name] = metrics

    return function_metrics


def extract_repo_metrics(repo_path: str) -> pd.DataFrame:
    """
    Scan all .py files in a repository and extract per-function metrics.

    Parameters:
        repo_path : str — path to root of Python project

    Returns:
        pd.DataFrame with all functions across all files
    """
    all_functions = []
    repo = Path(repo_path)

    py_files = list(repo.rglob("*.py"))
    print(f"Found {len(py_files)} Python files in {repo_path}")

    for py_file in py_files:
        # Skip test files, migrations, and virtual envs
        parts = py_file.parts
        if any(skip in parts for skip in [".venv", "venv", "__pycache__", "migrations", "node_modules"]):
            continue
        if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
            continue

        funcs = extract_file_metrics(str(py_file))
        all_functions.extend(funcs)

    df = pd.DataFrame(all_functions)
    print(f"Extracted metrics for {len(df)} functions")
    return df
```

---

## Step 1.7 — Build the Feature Matrix

Create `scripts/build_feature_matrix.py`:

```python
# scripts/build_feature_matrix.py
"""
Combines NASA MDP + PROMISE dataset features into the unified 8-feature matrix
that LightGBM will be trained on.

Run: python scripts/build_feature_matrix.py

Output: data/processed/features.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from buglens.mining.dataset_loader import load_all_datasets
from buglens.config import FEATURES, LABEL


def build_feature_matrix():
    print("=" * 60)
    print("Building CodeSentinel Feature Matrix")
    print("=" * 60)

    # Load all datasets
    print("\n1. Loading datasets...")
    df = load_all_datasets()

    # Ensure all required feature columns exist
    print("\n2. Validating feature columns...")
    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
        print("  These will be filled with 0 — check column mapping in dataset_loader.py")
        for col in missing_cols:
            df[col] = 0
    else:
        print("  All 8 feature columns present ✓")

    # Ensure label column exists
    if LABEL not in df.columns:
        raise ValueError(f"Label column '{LABEL}' not found. Check column mapping.")

    # Select only feature columns + label
    df_clean = df[FEATURES + [LABEL, "source"]].copy()

    # Handle missing values
    print("\n3. Handling missing values...")
    null_counts = df_clean[FEATURES].isnull().sum()
    if null_counts.sum() > 0:
        print(f"  Filling nulls:\n{null_counts[null_counts > 0]}")
        df_clean[FEATURES] = df_clean[FEATURES].fillna(0)
    else:
        print("  No missing values found ✓")

    # Convert all feature columns to numeric
    for col in FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    # Ensure label is integer 0 or 1
    df_clean[LABEL] = df_clean[LABEL].astype(int)

    # Print class distribution
    print("\n4. Class distribution:")
    dist = df_clean[LABEL].value_counts()
    total = len(df_clean)
    print(f"  Non-defective (0): {dist.get(0, 0)} ({dist.get(0, 0)/total*100:.1f}%)")
    print(f"  Defective    (1): {dist.get(1, 0)} ({dist.get(1, 0)/total*100:.1f}%)")
    print(f"  Total rows       : {total}")

    # Print feature statistics
    print("\n5. Feature statistics:")
    print(df_clean[FEATURES].describe().round(2).to_string())

    # Check for feature correlation
    print("\n6. Feature correlation with defect label:")
    correlations = df_clean[FEATURES].corrwith(df_clean[LABEL]).sort_values(ascending=False)
    print(correlations.round(3).to_string())

    # Save to disk
    output_path = Path("data/processed/features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Feature matrix saved: {output_path}")
    print(f"  Shape: {df_clean.shape}")

    return df_clean


if __name__ == "__main__":
    build_feature_matrix()
```

Run it:
```bash
python scripts/build_feature_matrix.py
```

**Expected output (approximate):**
```
Building CodeSentinel Feature Matrix
Loading datasets...
  Loaded NASA MDP: KC1.arff — 2109 rows
  Loaded NASA MDP: PC1.arff — 1107 rows
  ...
Total rows loaded: 8791

Validating feature columns...
  All 8 feature columns present ✓

Class distribution:
  Non-defective (0): 7235 (82.3%)
  Defective    (1): 1556 (17.7%)
  Total rows: 8791

✓ Feature matrix saved: data/processed/features.csv
  Shape: (8791, 10)
```

---

## Step 1.8 — Write Unit Tests for Mining

Create `tests/test_miner.py`:

```python
# tests/test_miner.py
"""Unit tests for git_miner.py and metric_extractor.py"""
import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from buglens.mining.metric_extractor import extract_file_metrics, extract_repo_metrics


@pytest.fixture
def simple_python_file(tmp_path):
    """Creates a temp .py file with a known function."""
    code = '''
def calculate_risk(score, threshold=0.5):
    """Returns risk category."""
    if score > 0.8:
        return "HIGH"
    elif score > threshold:
        return "MEDIUM"
    else:
        return "LOW"

def simple_add(a, b):
    return a + b
'''
    f = tmp_path / "sample.py"
    f.write_text(code)
    return str(f)


def test_extract_file_metrics_returns_list(simple_python_file):
    results = extract_file_metrics(simple_python_file)
    assert isinstance(results, list)
    assert len(results) >= 1


def test_extract_file_metrics_function_names(simple_python_file):
    results = extract_file_metrics(simple_python_file)
    names = [r["function_name"] for r in results]
    assert "calculate_risk" in names


def test_extract_file_metrics_has_required_fields(simple_python_file):
    results = extract_file_metrics(simple_python_file)
    required = ["filepath", "function_name", "line_start", "line_end",
                "cyclomatic_complexity", "loc", "maintainability_index"]
    for func in results:
        for field in required:
            assert field in func, f"Missing field: {field}"


def test_cyclomatic_complexity_is_positive(simple_python_file):
    results = extract_file_metrics(simple_python_file)
    for func in results:
        assert func["cyclomatic_complexity"] >= 1


def test_calculate_risk_has_higher_complexity(simple_python_file):
    """calculate_risk has 3 branches, simple_add has 1 — CC should differ."""
    results = extract_file_metrics(simple_python_file)
    func_dict = {r["function_name"]: r for r in results}
    if "calculate_risk" in func_dict and "simple_add" in func_dict:
        assert func_dict["calculate_risk"]["cyclomatic_complexity"] > \
               func_dict["simple_add"]["cyclomatic_complexity"]


def test_extract_file_metrics_invalid_syntax(tmp_path):
    """Should return empty list for files with syntax errors."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def broken(:\n    pass")
    results = extract_file_metrics(str(bad_file))
    assert results == []


def test_extract_file_metrics_empty_file(tmp_path):
    """Should return empty list for empty files."""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    results = extract_file_metrics(str(empty_file))
    assert results == []


@pytest.fixture
def mini_git_repo(tmp_path):
    """Creates a minimal git repository with one Python file and one commit."""
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=str(repo_dir), capture_output=True)
    py_file = repo_dir / "app.py"
    py_file.write_text("def hello():\n    return 'hello'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"],
                   cwd=str(repo_dir), capture_output=True)
    return str(repo_dir)


def test_mine_repository_returns_dataframe(mini_git_repo):
    from buglens.mining.git_miner import mine_repository
    df = mine_repository(mini_git_repo)
    assert hasattr(df, "columns")
    assert "churn" in df.columns
    assert "unique_developers" in df.columns
    assert len(df) >= 1


def test_mine_repository_commit_count(mini_git_repo):
    from buglens.mining.git_miner import mine_repository
    df = mine_repository(mini_git_repo)
    assert df["commit_count"].iloc[0] >= 1
```

Run the tests:
```bash
pytest tests/test_miner.py -v
```

**Expected:**
```
tests/test_miner.py::test_extract_file_metrics_returns_list PASSED
tests/test_miner.py::test_extract_file_metrics_function_names PASSED
tests/test_miner.py::test_extract_file_metrics_has_required_fields PASSED
tests/test_miner.py::test_cyclomatic_complexity_is_positive PASSED
...
8 passed in 3.24s
```

---

## Step 1.9 — Clone a Demo Repository for Testing

```bash
cd data/demo_repos
git clone https://github.com/pallets/flask.git
cd flask
git log --oneline | head -10   # Verify it has commit history
cd ../../..   # Back to codesentinel/
```

**Test the full mining pipeline on Flask:**
```bash
python -c "
from buglens.mining.git_miner import mine_repository
from buglens.mining.metric_extractor import extract_repo_metrics
import pandas as pd

# Mine git history
git_df = mine_repository('data/demo_repos/flask')
print('Git features shape:', git_df.shape)
print(git_df.head(3).to_string())

# Extract code metrics
code_df = extract_repo_metrics('data/demo_repos/flask')
print('\nCode metrics shape:', code_df.shape)
print(code_df.head(3).to_string())
"
```

---

## ✅ Phase 1 Checkpoint — All Must Pass Before Phase 2

```bash
# 1. Feature matrix exists and has correct shape
python -c "
import pandas as pd
from buglens.config import FEATURES, LABEL
df = pd.read_csv('data/processed/features.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
assert all(f in df.columns for f in FEATURES), 'MISSING FEATURE COLUMNS'
assert LABEL in df.columns, 'MISSING LABEL COLUMN'
assert df[LABEL].nunique() == 2, 'LABEL NOT BINARY'
assert len(df) >= 1000, 'TOO FEW ROWS'
print('Feature matrix OK ✓')
"

# 2. Mining tests pass
pytest tests/test_miner.py -v

# 3. Dataset has both defective and clean examples
python -c "
import pandas as pd
df = pd.read_csv('data/processed/features.csv')
dist = df['defects'].value_counts()
print('Defective:', dist.get(1, 0))
print('Clean:', dist.get(0, 0))
assert dist.get(1, 0) > 100, 'NOT ENOUGH DEFECTIVE EXAMPLES'
print('Class distribution OK ✓')
"

# 4. Commit Phase 1
git add .
git commit -m "Phase 1: Data pipeline — git miner, metric extractor, feature matrix"
```

---
---

# PHASE 2 — LightGBM Model Training & SHAP Explainability

## What Phase 2 Produces
- `buglens/model/train.py` — training script
- `buglens/model/predict.py` — inference function for live repositories
- `buglens/model/shap_explainer.py` — per-function SHAP explanation generator
- `buglens/model/saved/lightgbm_model.pkl` — trained LightGBM classifier
- `buglens/model/saved/scaler.pkl` — fitted StandardScaler
- `buglens/model/saved/shap_explainer.pkl` — TreeExplainer
- `tests/test_model.py` — unit tests for prediction and explanation
- Cross-validation AUC-ROC ≥ 0.75 confirmed

---

## Step 2.1 — Write the Training Script

Create `buglens/model/train.py`:

```python
# buglens/model/train.py
"""
Trains the LightGBM defect prediction model on the features.csv produced in Phase 1.
Saves model, scaler, and SHAP explainer to buglens/model/saved/.

Run: python buglens/model/train.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import shared config — same feature list used everywhere
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from buglens.config import FEATURES, LABEL, MODEL_DIR


def train(data_path: str = "data/processed/features.csv"):
    """
    Full training pipeline:
    1. Load feature matrix
    2. Apply SMOTE for class balance
    3. Scale features
    4. Cross-validate LightGBM
    5. Train final model
    6. Generate classification report
    7. Save model + scaler + SHAP explainer
    """
    print("=" * 60)
    print("CodeSentinel — LightGBM Training")
    print("=" * 60)

    # ── Step 1: Load Data ──────────────────────────────────────────
    print(f"\n1. Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    X = df[FEATURES].fillna(0).astype(float)
    y = df[LABEL].astype(int)

    print(f"   Features: {FEATURES}")
    print(f"   Class distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    # ── Step 2: SMOTE ─────────────────────────────────────────────
    print("\n2. Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"   After SMOTE — 0: {(y_res==0).sum()}, 1: {(y_res==1).sum()}")

    # ── Step 3: Scale Features ────────────────────────────────────
    print("\n3. Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    print("   Scaler fitted ✓")

    # ── Step 4: LightGBM Hyperparameters ──────────────────────────
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "n_estimators": 300,
        "class_weight": "balanced",
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)

    # ── Step 5: Cross-Validation ──────────────────────────────────
    print("\n4. Running 5-fold Stratified Cross-Validation (AUC-ROC)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y_res, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"   Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"   Mean AUC:    {cv_scores.mean():.4f}")
    print(f"   Std AUC:     {cv_scores.std():.4f}")

    if cv_scores.mean() < 0.70:
        print("   ⚠ WARNING: AUC below 0.70. Check data quality and feature mapping.")
    elif cv_scores.mean() >= 0.75:
        print("   ✓ AUC meets target threshold (≥ 0.75)")

    # ── Step 6: Final Training on Full Dataset ────────────────────
    print("\n5. Training final model on full (SMOTE-balanced) dataset...")
    model.fit(X_scaled, y_res)
    print("   Training complete ✓")

    # ── Step 7: Evaluation Report ─────────────────────────────────
    print("\n6. Evaluation on training data (upper bound):")
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    print(f"   AUC-ROC:   {roc_auc_score(y_res, y_prob):.4f}")
    print(f"   F1-Score:  {f1_score(y_res, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_res, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_res, y_pred):.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_res, y_pred, target_names=["Clean", "Defective"]))

    # Feature importance
    print("7. Feature Importance (LightGBM split-based):")
    importances = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "█" * int(imp / importances.max() * 20)
        print(f"   {feat:<30} {bar} ({imp})")

    # ── Step 8: Save Artifacts ────────────────────────────────────
    print(f"\n8. Saving model artifacts to {MODEL_DIR}/")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    model_path = f"{MODEL_DIR}/lightgbm_model.pkl"
    scaler_path = f"{MODEL_DIR}/scaler.pkl"
    explainer_path = f"{MODEL_DIR}/shap_explainer.pkl"

    joblib.dump(model, model_path)
    print(f"   Saved: {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"   Saved: {scaler_path}")

    # ── Step 9: SHAP Explainer ────────────────────────────────────
    print("\n9. Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, explainer_path)
    print(f"   Saved: {explainer_path}")

    # Quick SHAP test on 5 samples
    sample = X_scaled[:5]
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    print(f"   SHAP output shape: {shap_vals.shape} ✓")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Model : {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  SHAP  : {explainer_path}")
    print("=" * 60)

    return model, scaler, explainer


if __name__ == "__main__":
    train()
```

**Run the training:**
```bash
python buglens/model/train.py
```

Training should complete in 1–5 minutes on CPU. You will see fold AUC scores and a classification report.

---

## Step 2.2 — Write the Prediction Module

Create `buglens/model/predict.py`:

```python
# buglens/model/predict.py
"""
Runs inference on a live Python repository.
Called by the FastAPI backend when a developer triggers analysis in VS Code.

Loads the saved model artifacts from Phase 2 training.
Returns per-function risk predictions with probability scores.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from buglens.mining.metric_extractor import extract_repo_metrics
from buglens.mining.git_miner import mine_repository
from buglens.config import FEATURES, LABEL, MODEL_DIR, RISK_THRESHOLDS


def load_artifacts():
    """Load model, scaler, and explainer from disk. Call once at server startup."""
    model = joblib.load(f"{MODEL_DIR}/lightgbm_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return model, scaler


def predict_repo(repo_path: str) -> list[dict]:
    """
    Full prediction pipeline for a live repository.

    1. Mine git history → commit/churn/developer features
    2. Extract code metrics → cyclomatic complexity, LOC, MI
    3. Join features by filepath
    4. Scale features using saved StandardScaler
    5. Predict bug probability using saved LightGBM model
    6. Categorize as LOW / MEDIUM / HIGH risk
    7. Return list of dicts (one per function)

    Parameters:
        repo_path : str — path to root of a Python Git repository

    Returns:
        list of dicts with keys:
            filepath, function_name, line_start, line_end,
            bug_probability, risk_category,
            + all 8 feature values
    """
    print(f"\nRunning BugLens prediction on: {repo_path}")

    # Load model artifacts
    model, scaler = load_artifacts()

    # ── Step 1: Mine Git History ───────────────────────────────────
    print("Step 1: Mining git history...")
    git_df = mine_repository(repo_path)

    if git_df.empty:
        print("  WARNING: No git history found. Using default values for git features.")
        # Will be handled by fillna(0) below

    # ── Step 2: Extract Code Metrics ──────────────────────────────
    print("Step 2: Extracting code metrics...")
    code_df = extract_repo_metrics(repo_path)

    if code_df.empty:
        print("  No Python functions found.")
        return []

    # ── Step 3: Join Git Features with Code Metrics ───────────────
    print("Step 3: Joining features...")

    # Normalize filepath for joining
    # code_df has absolute paths, git_df has relative paths from repo root
    repo_root = str(Path(repo_path).resolve())

    def normalize_path(p: str) -> str:
        p = str(p)
        if p.startswith(repo_root):
            return p[len(repo_root):].lstrip("/\\")
        return p

    code_df["rel_path"] = code_df["filepath"].apply(normalize_path)

    # Merge — left join keeps all functions even if no git history found
    merged = code_df.merge(
        git_df,
        left_on="rel_path",
        right_on="filepath",
        how="left",
        suffixes=("_code", "_git")
    )

    # Fill missing git features with 0
    for col in ["churn", "unique_developers", "bug_fix_commits", "file_age_days", "recent_churn"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0)

    # ── Step 4: Build Feature Matrix ──────────────────────────────
    for col in FEATURES:
        if col not in merged.columns:
            merged[col] = 0

    X = merged[FEATURES].fillna(0).astype(float)

    # ── Step 5: Scale & Predict ────────────────────────────────────
    print("Step 4: Running LightGBM prediction...")
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # ── Step 6: Build Results ──────────────────────────────────────
    results = []
    for i, row in merged.iterrows():
        prob = float(probabilities[i - merged.index[0]])
        bug_probability = round(prob * 100, 1)

        # Determine risk category
        if prob >= RISK_THRESHOLDS["HIGH"][0]:
            risk_category = "HIGH"
        elif prob >= RISK_THRESHOLDS["MEDIUM"][0]:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"

        result = {
            "filepath": row.get("filepath_code", row.get("filepath", "")),
            "function_name": row["function_name"],
            "line_start": int(row["line_start"]),
            "line_end": int(row["line_end"]),
            "bug_probability": bug_probability,
            "risk_category": risk_category,
        }
        # Include feature values for SHAP explanation
        for feat in FEATURES:
            result[feat] = float(row.get(feat, 0))

        results.append(result)

    # Sort by bug probability descending (highest risk first for priority queue)
    results.sort(key=lambda x: x["bug_probability"], reverse=True)

    # Summary
    high = sum(1 for r in results if r["risk_category"] == "HIGH")
    medium = sum(1 for r in results if r["risk_category"] == "MEDIUM")
    low = sum(1 for r in results if r["risk_category"] == "LOW")
    print(f"  Total functions: {len(results)}")
    print(f"  HIGH risk: {high} | MEDIUM: {medium} | LOW: {low}")

    return results
```

---

## Step 2.3 — Write the SHAP Explainer

Create `buglens/model/shap_explainer.py`:

```python
# buglens/model/shap_explainer.py
"""
Generates per-function SHAP explanations.
Called after prediction to explain WHY a function is flagged as high-risk.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from buglens.config import FEATURES, MODEL_DIR


# Human-readable labels for each feature
FEATURE_LABELS = {
    "cyclomatic_complexity": "High cyclomatic complexity",
    "loc":                   "Large function size (LOC)",
    "maintainability_index": "Low maintainability index",
    "churn":                 "High total code churn",
    "unique_developers":     "Many contributors touched this file",
    "bug_fix_commits":       "Frequent bug-fix commits in history",
    "file_age_days":         "Old file with accumulated debt",
    "recent_churn":          "Heavy recent churn (last 90 days)",
}


def explain_function(feature_values: dict) -> dict:
    """
    Given a dict of feature values for one function,
    returns a SHAP-based explanation with top risk drivers.

    Parameters:
        feature_values : dict — keys must include all 8 FEATURES

    Returns:
        dict with:
            top_risk_drivers : list of top contributing features
            summary          : human-readable one-line explanation
            base_value       : model's average prediction
            prediction       : this function's predicted probability
    """
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")

    # Build feature array — same order as FEATURES list
    feature_array = np.array([[feature_values.get(f, 0) for f in FEATURES]], dtype=float)
    X_scaled = scaler.transform(feature_array)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_scaled)

    # For binary LightGBM, shap_values may be list [class0, class1] or just one array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Positive class (defective)
    else:
        sv = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    else:
        base_value = float(base_value)

    # Rank features by absolute SHAP value
    contributions = sorted(
        zip(FEATURES, sv),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Build top risk drivers (show top 4)
    top_drivers = []
    for feat, shap_val in contributions[:4]:
        top_drivers.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "shap_value": round(float(shap_val), 4),
            "raw_value": round(float(feature_values.get(feat, 0)), 2),
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
        })

    # Build one-line summary for hover tooltip
    positive_drivers = [
        FEATURE_LABELS.get(f, f)
        for f, v in contributions[:3]
        if v > 0
    ]
    summary = ", ".join(positive_drivers) if positive_drivers else "No dominant risk factors"

    return {
        "top_risk_drivers": top_drivers,
        "summary": summary,
        "base_value": round(base_value, 4),
        "prediction": round(float(np.sum(sv)) + base_value, 4),
    }
```

---

## Step 2.4 — Write Model Unit Tests

Create `tests/test_model.py`:

```python
# tests/test_model.py
"""Unit tests for model training, prediction, and SHAP explanation."""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def trained_artifacts():
    """Load saved model artifacts — requires Phase 2 training to have run."""
    import joblib
    from buglens.config import MODEL_DIR

    model_path = Path(MODEL_DIR) / "lightgbm_model.pkl"
    scaler_path = Path(MODEL_DIR) / "scaler.pkl"
    explainer_path = Path(MODEL_DIR) / "shap_explainer.pkl"

    if not model_path.exists():
        pytest.skip("Model not yet trained. Run: python buglens/model/train.py")

    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "explainer": joblib.load(explainer_path),
    }


@pytest.fixture
def sample_features():
    """A realistic set of feature values for one function."""
    return {
        "cyclomatic_complexity": 8,
        "loc": 45,
        "maintainability_index": 55.0,
        "churn": 320,
        "unique_developers": 4,
        "bug_fix_commits": 2,
        "file_age_days": 400,
        "recent_churn": 80,
    }


def test_model_loads(trained_artifacts):
    assert trained_artifacts["model"] is not None
    assert trained_artifacts["scaler"] is not None


def test_prediction_returns_probability(trained_artifacts, sample_features):
    from buglens.config import FEATURES
    import numpy as np

    model = trained_artifacts["model"]
    scaler = trained_artifacts["scaler"]

    X = np.array([[sample_features[f] for f in FEATURES]], dtype=float)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]

    assert 0.0 <= float(proba[0]) <= 1.0


def test_prediction_high_complexity_is_riskier(trained_artifacts):
    """A function with very high complexity should have higher risk than a simple one."""
    from buglens.config import FEATURES
    import numpy as np

    model = trained_artifacts["model"]
    scaler = trained_artifacts["scaler"]

    simple = {f: 1.0 for f in FEATURES}
    simple["cyclomatic_complexity"] = 1
    simple["loc"] = 5

    complex_func = {f: 1.0 for f in FEATURES}
    complex_func["cyclomatic_complexity"] = 15
    complex_func["loc"] = 200
    complex_func["churn"] = 1000

    X_simple = scaler.transform(np.array([[simple[f] for f in FEATURES]]))
    X_complex = scaler.transform(np.array([[complex_func[f] for f in FEATURES]]))

    prob_simple = model.predict_proba(X_simple)[:, 1][0]
    prob_complex = model.predict_proba(X_complex)[:, 1][0]

    assert prob_complex > prob_simple, (
        f"Complex function ({prob_complex:.3f}) should be riskier than simple ({prob_simple:.3f})"
    )


def test_shap_explanation_structure(trained_artifacts, sample_features):
    from buglens.model.shap_explainer import explain_function

    result = explain_function(sample_features)

    assert "top_risk_drivers" in result
    assert "summary" in result
    assert isinstance(result["top_risk_drivers"], list)
    assert len(result["top_risk_drivers"]) > 0
    assert "label" in result["top_risk_drivers"][0]
    assert "shap_value" in result["top_risk_drivers"][0]
    assert "direction" in result["top_risk_drivers"][0]


def test_shap_explanation_values_are_finite(trained_artifacts, sample_features):
    from buglens.model.shap_explainer import explain_function

    result = explain_function(sample_features)

    for driver in result["top_risk_drivers"]:
        assert np.isfinite(driver["shap_value"]), f"SHAP value not finite: {driver}"


def test_feature_schema_consistency():
    """Critical: verify feature list in config matches training and prediction."""
    from buglens.config import FEATURES

    assert len(FEATURES) == 8, f"Expected 8 features, got {len(FEATURES)}"
    assert "cyclomatic_complexity" in FEATURES
    assert "churn" in FEATURES
    assert "unique_developers" in FEATURES
```

Run the tests:
```bash
pytest tests/test_model.py -v
```

---

## Step 2.5 — Quick End-to-End Prediction Test

After training, test the full prediction pipeline on Flask:

```bash
python -c "
from buglens.model.predict import predict_repo
from buglens.model.shap_explainer import explain_function

# Run prediction on Flask
results = predict_repo('data/demo_repos/flask')

print(f'Total functions analyzed: {len(results)}')
print(f'HIGH risk: {sum(1 for r in results if r[\"risk_category\"]==\"HIGH\")}')
print(f'MEDIUM risk: {sum(1 for r in results if r[\"risk_category\"]==\"MEDIUM\")}')
print(f'LOW risk: {sum(1 for r in results if r[\"risk_category\"]==\"LOW\")}')

# Show top 5 highest-risk functions
print('\nTop 5 Highest-Risk Functions:')
for r in results[:5]:
    print(f'  {r[\"function_name\"]:30s} {r[\"bug_probability\"]:5.1f}%  [{r[\"risk_category\"]}]')

# Show SHAP explanation for the riskiest function
if results:
    top = results[0]
    explanation = explain_function(top)
    print(f'\nSHAP Explanation for: {top[\"function_name\"]}')
    print(f'Summary: {explanation[\"summary\"]}')
    for driver in explanation['top_risk_drivers'][:3]:
        print(f'  {driver[\"label\"]}: SHAP={driver[\"shap_value\"]:+.4f} ({driver[\"direction\"]})')
"
```

---

## ✅ Phase 2 Checkpoint — All Must Pass Before Phase 3

```bash
# 1. All model files exist
ls -lh buglens/model/saved/

# 2. Model tests pass
pytest tests/test_model.py -v

# 3. CV AUC is in acceptable range (check training output log)
# It was printed during training — re-run train.py if you need to see it again

# 4. Prediction runs on Flask without crash
python -c "
from buglens.model.predict import predict_repo
results = predict_repo('data/demo_repos/flask')
assert len(results) > 0, 'No results returned'
assert all('bug_probability' in r for r in results), 'Missing bug_probability'
assert all(r['risk_category'] in ['LOW','MEDIUM','HIGH'] for r in results), 'Invalid risk category'
print(f'Prediction OK ✓ — {len(results)} functions analyzed')
"

# 5. SHAP explanation works
python -c "
from buglens.model.shap_explainer import explain_function
sample = {'cyclomatic_complexity': 8, 'loc': 45, 'maintainability_index': 55,
          'churn': 320, 'unique_developers': 4, 'bug_fix_commits': 2,
          'file_age_days': 400, 'recent_churn': 80}
result = explain_function(sample)
assert 'top_risk_drivers' in result
assert len(result['top_risk_drivers']) > 0
print('SHAP explanation OK ✓')
print('Summary:', result['summary'])
"

# 6. Commit Phase 2
git add .
git commit -m "Phase 2: LightGBM training, prediction, SHAP explainer — model artifacts saved"
```

---

## Summary — What You Now Have After Phases 0, 1, 2

| Artifact | Location | Purpose |
|---|---|---|
| Virtual environment | `.venv/` | Isolated Python with all 20+ packages |
| Feature schema | `buglens/config.py` | Single source of truth for 8-feature list |
| Dataset loader | `buglens/mining/dataset_loader.py` | Loads NASA MDP + PROMISE `.arff` files |
| Git miner | `buglens/mining/git_miner.py` | Extracts commit/churn/developer features |
| Metric extractor | `buglens/mining/metric_extractor.py` | Extracts cyclomatic complexity, LOC, MI |
| Feature matrix | `data/processed/features.csv` | Training data (8 features + label) |
| Training script | `buglens/model/train.py` | Runs LightGBM with SMOTE, 5-fold CV |
| Trained model | `buglens/model/saved/lightgbm_model.pkl` | LightGBM classifier |
| Scaler | `buglens/model/saved/scaler.pkl` | StandardScaler (fit on training data) |
| SHAP explainer | `buglens/model/saved/shap_explainer.pkl` | TreeExplainer for per-function SHAP |
| Prediction module | `buglens/model/predict.py` | Runs inference on any Python repo |
| SHAP module | `buglens/model/shap_explainer.py` | Generates risk explanations |
| Tests | `tests/test_miner.py` + `tests/test_model.py` | Verification tests |

**Next phase (Phase 3):** FastAPI backend that serves these prediction results over HTTP to the VS Code extension.

---

*Phase 0 · Phase 1 · Phase 2 Execution Guide — CodeSentinel Group 10, FR. CRCE 2026–27*
