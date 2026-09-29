# CodeSentinel: AI-Powered Code Quality Platform

> **Defect Prediction and Targeted Test Generation**  
> **B.E. Final Year Project 2026–27**  
> **Department of AI & Data Science | FR. Conceicao Rodrigues College of Engineering, Mumbai**  
> **Team:** Group 10 — Atharva Nanaware (10511), Suhera Siddiqui (10381), Aum Singh (10383)

---

## 📌 System Overview

CodeSentinel is an AI-powered code quality platform designed to predict defects in software repositories and automatically generate targeted unit tests for high-risk code paths:

1. **BugLens (Module 1 - Current Implementation)**: Mines Git commit history and AST complexity metrics, predicts defect probability per function using a trained **LightGBM** classifier, and attributes root causes using **SHAP (SHapley Additive exPlanations)**.
2. **AutoTestArmy (Module 2 - Next Phase)**: A multi-agent pipeline (Analyzer → Generator → Critic) that generates, executes, and iteratively refines `pytest` unit tests for prioritized high-risk functions.
3. **Feedback Loop**: Incorporates test execution outcomes back into training data to refine defect prediction accuracy over time.

---

## 📂 Repository Structure

```
codesentinel/
├── buglens/
│   ├── config.py                 # Central 8-feature schema & model hyperparameters
│   ├── mining/
│   │   ├── dataset_loader.py     # Maps NASA MDP & PROMISE datasets to unified schema
│   │   ├── git_miner.py          # PyDriller engine (churn, developers, bug-fix commits)
│   │   └── metric_extractor.py   # Radon & AST engine (cyclomatic complexity, LOC, MI)
│   └── model/
│       ├── train.py              # SMOTE + StandardScaler + 5-Fold Stratified CV LightGBM
│       ├── predict.py            # Live repository inference engine
│       ├── shap_explainer.py     # Natural language SHAP root-cause explanations
│       └── saved/                # Saved models (.pkl files)
├── data/
│   ├── raw/                      # Downloaded NASA MDP & PROMISE datasets
│   ├── processed/                # Unified features.csv matrix
│   └── demo_repos/               # Cloned test repositories (e.g. Flask)
├── scripts/
│   ├── download_datasets.py      # Downloads all 13 defect prediction datasets
│   ├── build_feature_matrix.py   # Assembles 19,980-row training matrix
│   └── demo_buglens.py           # Live interactive inference & SHAP demo
├── tests/
│   ├── test_miner.py             # Unit tests for Git and AST metric extraction
│   └── test_model.py             # Unit tests for LightGBM prediction & SHAP
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt          # Development & test tooling
└── README.md
```

---

## ⚙️ Prerequisites

Before starting, ensure you have the following installed on your machine:

- **Python**: **3.10, 3.11, or 3.12** *(Python 3.12 is recommended. Do not use Python 3.13 as scientific packages lack pre-built Windows wheels).*
- **Git**: Installed and available in your system `PATH`.
- **Ollama** *(Optional for BugLens, required for upcoming AutoTestArmy)*: Download from [ollama.ai](https://ollama.ai).

---

## 🚀 Quick Start Guide (For New Users)

Follow these step-by-step instructions to set up and run the project on a new computer.

### 1. Clone the Repository

```bash
git clone https://github.com/nanawareatharva/codesentinel.git
cd codesentinel
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled on your system, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first).*

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Configure Environment Variables

Create your local `.env` file from the provided template:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

### 4. Install Dependencies

Upgrade `pip` and install all required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

---

## 🧪 Running the Complete ML Pipeline

If you want to reproduce the data pipeline and model training from scratch:

### Step 1: Download Defect Datasets
Downloads 8 NASA MDP datasets (`CM1`, `KC1`, `KC2`, `MC1`, `MW1`, `PC1`, `PC3`, `PC4`) and 5 PROMISE datasets (`ant-1.7`, `camel-1.6`, `jedit-4.3`, `log4j-1.2`, `poi-3.0`):
```bash
python scripts/download_datasets.py
```

### Step 2: Assemble the Unified Feature Matrix
Extracts the 8 standard features and creates `data/processed/features.csv` (19,980 rows):
```bash
python scripts/build_feature_matrix.py
```

### Step 3: Train the LightGBM Classifier
Trains the model with SMOTE class balancing and 5-fold stratified cross-validation (achieves **Mean AUC ~ 0.88**), saving the model, scaler, and SHAP explainer to `buglens/model/saved/`:
```bash
python buglens/model/train.py
```

### Step 4: Run the Test Suite
Verify that all 12 unit tests pass:
```bash
pytest tests/ -v
```

---

## 🖥️ Live Working Demo

To see BugLens in action analyzing a real-world Python codebase:

### 1. Ensure Demo Repository Exists
A demo clone of `Flask` is used for testing. If not already present in `data/demo_repos/flask`, clone it:
```bash
git clone https://github.com/pallets/flask.git data/demo_repos/flask
```

### 2. Run the Demo Script
Run the interactive inference script:
```bash
python scripts/demo_buglens.py
```

> **Tip:** You can analyze any Python repository on your machine by passing its path:
> ```bash
> python scripts/demo_buglens.py path/to/any/python-repo
> ```

### 3. What the Demo Shows
The demo executes the full end-to-end defect prediction engine:
1. **Git History Mining**: Scans commit history using PyDriller to determine commit count, total churn, author diversity, file age, and historical bug fixes.
2. **AST & Complexity Extraction**: Calculates cyclomatic complexity, lines of code, and maintainability index across all functions.
3. **Risk Categorization**: Classifies every function into `LOW` (< 33%), `MEDIUM` (33–66%), or `HIGH` (≥ 66%) defect risk.
4. **Priority Queue**: Displays the top most defect-prone functions sorted by probability.
5. **SHAP Root-Cause Attribution**: Deconstructs the #1 highest-risk function to explain *why* it was flagged (e.g. high historical bug fixes, author churn, or elevated cyclomatic complexity).

---

## 📊 Central Feature Schema

CodeSentinel uses a strictly standardized 8-feature schema defined in [`buglens/config.py`](buglens/config.py):

| Feature Name | Source | Description |
|---|---|---|
| `cyclomatic_complexity` | Radon | Number of independent execution paths in the function |
| `loc` | Radon | Function lines of code |
| `maintainability_index` | Radon | Composite maintainability score (0–100) |
| `churn` | PyDriller | Total lines added and deleted over repository history |
| `unique_developers` | PyDriller | Distinct author emails who modified the enclosing file |
| `bug_fix_commits` | PyDriller | Commits with bug-fix keywords in their message |
| `file_age_days` | PyDriller | Days between first and latest commit |
| `recent_churn` | PyDriller | Code churn in the enclosing file over the last 90 days |

---

## 👥 Contributors

- **Atharva Nanaware** (10511)
- **Suhera Siddiqui** (10381)
- **Aum Singh** (10383)

*Department of Artificial Intelligence & Data Science*  
*FR. Conceicao Rodrigues College of Engineering, Bandra, Mumbai*
