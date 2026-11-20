# CodeSentinel: Complete Project Technical Dossier
**AI-Powered Defect Prediction & Targeted Test Generation Platform**  
*B.E. Final Year Project 2026–2027 | Group 10 | Department of Artificial Intelligence & Data Science*  
*Fr. Conceicao Rodrigues College of Engineering (FR. CRCE), Mumbai*

---

## Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [Phase 0 — Environment Setup & Project Scaffold](#2-phase-0--environment-setup--project-scaffold)
3. [Phase 1 — Data Collection, Mining & Feature Engineering](#3-phase-1--data-collection-mining--feature-engineering)
4. [Phase 2 — LightGBM Classifier, SHAP Explainability & Inference](#4-phase-2--lightgbm-classifier-shap-explainability--inference)
5. [Technical Audit & Codebase Fixes Implemented](#5-technical-audit--codebase-fixes-implemented)
6. [Step-by-Step Guide to Run, Test, and Demonstrate](#6-step-by-step-guide-to-run-test-and-demonstrate)
7. [Roadmap for Upcoming Phases (Phases 3 to 8)](#7-roadmap-for-upcoming-phases-phases-3-to-8)
8. [Teacher & Project Guide Presentation Guide (Viva Defense)](#8-teacher--project-guide-presentation-guide-viva-defense)
9. [Comprehensive Viva Voce Questions & Model Answers](#9-comprehensive-viva-voce-questions--model-answers)
10. [Summary Verification Checklist for Project Evaluation](#10-summary-verification-checklist-for-project-evaluation)

---

<a name="1-executive-summary--high-level-architecture"></a>
## 1. Executive Summary & High-Level Architecture

### 1.1 Problem Statement
Modern software engineering faces two major code quality challenges:
1. **Blind Defect Detection**: Traditional static analyzers (ESLint, SonarQube, Flake8) identify current syntax and style violations, but cannot predict which functions are most likely to introduce undiscovered, critical runtime bugs based on evolutionary history, churn volatility, and architectural complexity.
2. **Untargeted Test Generation**: LLM-based test generators (e.g., Copilot, Codium) treat all codebase functions equally. They burn token budgets and execution time writing unit tests for trivial, stable utilities while leaving complex, defect-prone core functions under-tested.

### 1.2 The CodeSentinel Solution
CodeSentinel is a **closed-loop, AI-powered software quality platform** that unifies predictive machine learning, explainable AI, and multi-agent test synthesis into a cohesive developer tool.

```
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 1: BugLens (Defect Prediction)                     │
│  Git History (PyDriller) + Code Metrics (Radon/AST)                    │
│                        ↓                                               │
│  LightGBM Classifier + SHAP TreeExplainer                              │
│                        ↓                                               │
│  Function-Level Bug Probability & Root-Cause Driver Attribution        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 2: Priority Queue Bridge                           │
│  Sorts functions by predicted risk: HIGH (≥66%), MED (33-66%), LOW (<33%)│
│  Feeds highest-risk functions into test generation budget first        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           LAYER 3: AutoTestArmy (Multi-Agent Test Generation)          │
│  [Analyzer Agent]  → AST branch & boundary condition parsing           │
│  [Generator Agent] → Local LLM (Phi-3 / Llama-3.1 via Ollama)          │
│  [Critic Agent]    → Subprocess pytest-cov runner & failure feedback   │
│  Loop: Analyzer → Generator → Critic → (Regenerate on failure) → Suite │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               CLOSED-LOOP FEEDBACK: Continuous Retraining              │
│  Captures test outcomes (Passed with coverage = 0, Persistent fails = 1)│
│  Appends to feedback dataset & retrains LightGBM model periodically    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Project Directory Structure
```
codesentinel/
├── buglens/                        # Module 1: Defect Prediction (Phases 0, 1, 2) [COMPLETE]
│   ├── config.py                   # Central source of truth: schema, thresholds, params
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py       # NASA MDP & PROMISE dataset parser & normalizer
│   │   ├── git_miner.py            # PyDriller git commit, churn & developer miner
│   │   └── metric_extractor.py     # Radon & AST complexity, LOC & MI extractor
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py                # LightGBM trainer with SMOTE & honest OOF validation
│   │   ├── predict.py              # Live repository inference engine & risk ranker
│   │   ├── shap_explainer.py       # SHAP TreeExplainer & dynamic risk driver summarizer
│   │   └── saved/                  # Serialized artifacts (.pkl files)
│   └── api/
│       ├── __init__.py
│       └── routes.py               # FastAPI endpoints for BugLens (Phase 3 scaffold)
│
├── autotestarmy/                   # Module 2: Multi-Agent Test Gen (Phases 4, 5) [SCAFFOLDED]
│   ├── agents/                     # Analyzer, Generator, and Critic agent nodes
│   ├── graph/                      # LangGraph state schema & cyclic StateGraph pipeline
│   ├── tools/                      # AST utilities, pytest executor, coverage.xml parser
│   └── api/                        # FastAPI endpoints for AutoTestArmy
│
├── feedback/                       # Closed-Loop Aggregator (Phase 6) [SCAFFOLDED]
│   ├── aggregator.py               # Test outcome collector & heuristic ground-truth labeler
│   └── model_updater.py            # LightGBM retraining scheduler
│
├── backend/                        # Central API Server (Phase 3) [SCAFFOLDED]
│   ├── main.py                     # FastAPI entrypoint, mounts BugLens & AutoTestArmy
│   ├── database.py                 # SQLite database engine (SQLAlchemy)
│   ├── models.py                   # DB tables (PredictionRun, FunctionPrediction, TestRun)
│   └── schemas.py                  # Pydantic v2 validation models
│
├── vscode-extension/               # IDE Client (Phase 7) [FUTURE]
│   └── src/                        # Heatmap decorations, webview panels, API client
│
├── data/
│   ├── raw/                        # 8 NASA MDP ARFF files + 5 PROMISE CSV files
│   ├── processed/
│   │   └── features.csv            # Clean, deduplicated 9,157-row training matrix
│   └── demo_repos/
│       └── flask/                  # Cloned Pallets Flask repository for live demonstrations
│
├── scripts/
│   ├── download_datasets.py        # Automated dataset retriever
│   ├── build_feature_matrix.py     # Compiles, validates & deduplicates features.csv
│   ├── demo_buglens.py             # Interactive CLI live demonstration tool
│   └── run_evaluation.py           # Empirical comparison script (Phase 8)
│
├── tests/                          # Automated Verification Suite [12/12 PASSING]
│   ├── test_miner.py               # Unit tests for Git mining & AST extraction
│   └── test_model.py               # Unit tests for LightGBM, SHAP & live inference
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules (venv, pkl, raw data)
├── requirements.txt                # Pinned production dependencies
└── requirements-dev.txt            # Development & testing dependencies
```

---

<a name="2-phase-0--environment-setup--project-scaffold"></a>
## 2. Phase 0 — Environment Setup & Project Scaffold

### 2.1 Objective
Establish an isolated, fully reproducible engineering foundation. Lock down dependency versions, configure cross-platform execution boundaries, centralize all system hyperparameters, and initialize version control.

### 2.2 What We Implemented
1. **Python 3.12 Virtual Environment (`.venv`)**:
   - Initialized with Python 3.12 (specifically avoiding Python 3.13 where critical binary wheels for `numpy`, `scipy`, and `lightgbm` are unavailable on Windows).
2. **Central Configuration System (`buglens/config.py`)**:
   - Implemented a single source of truth across all modules to eliminate magic strings and configuration drift.
   - Declares the immutable 8-feature schema:
     ```python
     FEATURES = [
         "cyclomatic_complexity",  # Radon CC (McCabe)
         "loc",                    # Lines of Code (physical span)
         "maintainability_index",  # Radon MI (composite score 0-100)
         "churn",                  # PyDriller total added + deleted lines
         "unique_developers",      # PyDriller distinct author emails
         "bug_fix_commits",        # PyDriller keyword-matched commit count
         "file_age_days",          # PyDriller days between first and last commit
         "recent_churn",           # PyDriller churn in last 90 days
     ]
     ```
   - Defines risk classification thresholds:
     - `LOW`: $[0.00, 0.33)$
     - `MEDIUM`: $[0.33, 0.66)$
     - `HIGH`: $[0.66, 1.00]$
   - Declares the 10 bug-fix commit matching keywords:
     `["fix", "bug", "patch", "error", "defect", "hotfix", "repair", "resolve", "issue", "crash"]`
3. **Dependency Pinning (`requirements.txt`)**:
   - Machine Learning: `lightgbm==4.3.0`, `scikit-learn==1.4.2`, `shap==0.45.1`, `imbalanced-learn==0.12.3`, `pandas==2.2.2`, `numpy==1.26.4`.
   - Software Mining & AST: `pydriller==2.6`, `gitpython==3.1.43`, `radon==6.0.1`, `libcst==1.4.0`.
   - Orchestration & Backend: `langgraph==0.1.19`, `langchain==0.2.13`, `fastapi==0.111.1`, `pydantic==2.8.2`, `sqlalchemy==2.0.31`.
4. **Local LLM Engine**:
   - Installed and verified Ollama local runtime serving `phi3:mini` (3.8B parameters) and `llama3.1:8b`.

---

<a name="3-phase-1--data-collection-mining--feature-engineering"></a>
## 3. Phase 1 — Data Collection, Mining & Feature Engineering

### 3.1 Objective
Assemble a cross-project defect dataset from historical software benchmarks, construct static code complexity extractors, build a dynamic Git history miner, and produce a unified training feature matrix.

### 3.2 Dataset Acquisition & Normalization
Implemented `scripts/download_datasets.py` and `buglens/mining/dataset_loader.py` to acquire and ingest 13 benchmark repositories:
* **8 NASA MDP Datasets (.arff)**: `CM1` (NASA spacecraft), `KC1`, `KC2` (data processing), `MC1` (avionics), `MW1` (zero-gravity combustion), `PC1`, `PC3`, `PC4` (flight software).
* **5 PROMISE Datasets (.csv)**: `ant-1.7`, `camel-1.6`, `jedit-4.3`, `log4j-1.2`, `poi-3.0`.

The dataset loader automatically parses ARFF byte strings, aligns diverse historical column names, and maps labels into binary defect states ($0 = \text{clean}, 1 = \text{defective}$).

### 3.3 Dynamic Git Repository Mining (`buglens/mining/git_miner.py`)
Uses PyDriller's `Repository.traverse_commits()` to mine repository evolutionary history:
* **Code Churn**: Cumulative sum of lines added and deleted across all historical commits.
* **Developer Count**: Cardinality of distinct author emails modifying the file.
* **Bug-Fix Commits**: Count of commit messages containing any of the 10 regex/substring bug keywords.
* **File Age**: Total calendar days elapsed between the file's first recorded commit and its latest commit.
* **Recent Churn**: Cumulative lines added and deleted strictly within the past 90 calendar days ($T \ge \text{now} - 90\text{d}$).
* **Optimization Added**: Implemented `max_commits=500` (default) and `tqdm` progress tracking to prevent unbounded delays during live demonstrations.

### 3.4 Static AST & Radon Complexity Extraction (`buglens/mining/metric_extractor.py`)
Parses Python source files using Python's native `ast` module and `radon`:
* **Cyclomatic Complexity (CC)**: Uses `radon.complexity.cc_visit()` to count linearly independent execution paths ($M = E - N + 2P$) for every function/method block.
* **Lines of Code (LOC)**: Measures physical function span ($L_{\text{end}} - L_{\text{start}} + 1$).
* **Maintainability Index (MI)**: Uses `radon.metrics.mi_visit()` to compute an overall maintainability score on a scale of $0$ to $100$:
  $$\text{MI} = \max\left(0, \min\left(100, 171 - 5.2 \ln(V) - 0.23 v(G) - 16.2 \ln(\text{LOC})\right)\right)$$
* **Structural AST Features**: NodeVisitor tracks maximum block nesting depth, return count, and try-except error handlers.
* **Collision Fix Added**: Keyed metrics by `(function.name, function.lineno)` to avoid dictionary collisions when multiple classes contain methods with identical names (e.g., `__init__`).

### 3.5 Unified Feature Matrix Assembly (`scripts/build_feature_matrix.py`)
Consolidates all datasets into `data/processed/features.csv`.
* **Deduplication Applied**: Drops duplicate feature vectors to eliminate training-validation leakage.
* **Dataset Size**: **9,157 unique, clean samples** with an 18.1% positive defect rate.

---

<a name="4-phase-2--lightgbm-classifier-shap-explainability--inference"></a>
## 4. Phase 2 — LightGBM Classifier, SHAP Explainability & Inference

### 4.1 Objective
Train an interpretable, high-performance gradient boosted decision tree classifier to predict function defect probabilities, build an exact Shapley attribution layer, implement live repository inference, and provide an interactive CLI demo.

### 4.2 LightGBM Model Architecture (`buglens/model/train.py`)
Why LightGBM:
* Outperforms deep learning on tabular software metrics.
* Fast training on laptop CPUs (< 5 seconds).
* Native support for SHAP `TreeExplainer` in polynomial time.
* Handles class imbalance via SMOTE integration.

#### Hyperparameters
```python
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}
```

### 4.3 Validation Strategy & Measured Performance
To avoid data leakage, SMOTE oversampling is applied strictly **inside each training fold** of a 5-Fold Stratified Cross-Validation split:

```
Total Data (9,157 unique rows)
  │
  ├── Fold 1: Train (7,325 rows) → SMOTE resample → Fit Scaler & Model → Validate on Fold 1 Test (1,832 rows, unaugmented)
  ├── Fold 2: Train (7,325 rows) → SMOTE resample → Fit Scaler & Model → Validate on Fold 2 Test (1,832 rows, unaugmented)
  ├── Fold 3: Train (7,325 rows) → SMOTE resample → Fit Scaler & Model → Validate on Fold 3 Test (1,832 rows, unaugmented)
  ├── Fold 4: Train (7,326 rows) → SMOTE resample → Fit Scaler & Model → Validate on Fold 4 Test (1,831 rows, unaugmented)
  └── Fold 5: Train (7,327 rows) → SMOTE resample → Fit Scaler & Model → Validate on Fold 5 Test (1,830 rows, unaugmented)
```

#### Measured Performance Metrics
* **Out-of-Fold Validation AUC-ROC**: **0.7694 ± 0.0036** (Exceeds academic baseline target of $\ge 0.75$)
* **Out-of-Fold Precision**: **0.5158**
* **Out-of-Fold Recall**: **0.3259**
* **Out-of-Fold F1-Score**: **0.3994**

All artifacts are persisted into `buglens/model/saved/`:
1. `lightgbm_model.pkl` — Serialized LightGBM classifier
2. `scaler.pkl` — Fitted StandardScaler
3. `shap_explainer.pkl` — Precomputed SHAP TreeExplainer

### 4.4 Explainable AI via SHAP (`buglens/model/shap_explainer.py`)
Uses `shap.TreeExplainer` based on cooperative game theory to decompose predictions:
$$\text{Margin}(x) = \phi_0 + \sum_{i=1}^{8} \phi_i(x)$$
Where $\phi_0$ is the base expected margin and $\phi_i$ represents feature $i$'s Shapley contribution.

* **Directional Labels**: Dynamically distinguishes risk-increasing versus risk-reducing factors (e.g., `"High maintainability (protective) → decreases risk"` vs `"High cyclomatic complexity → increases risk"`).
* **Summary Generation**: Combines the top 3 risk drivers into an executive sentence.

### 4.5 Live Repository Inference Engine (`buglens/model/predict.py`)
Accepts any local Git repository path:
1. Calls `mine_repository()` to extract file-level Git metrics.
2. Calls `extract_repo_metrics()` to extract function-level AST metrics.
3. Left-joins Git metrics to functions using normalized relative file paths.
4. Scales feature vectors and executes `model.predict_proba()`.
5. Maps probabilities to `LOW`, `MEDIUM`, or `HIGH` risk categories.
6. Returns a priority queue sorted by `bug_probability` descending.

---

<a name="5-technical-audit--codebase-fixes-implemented"></a>
## 5. Technical Audit & Codebase Fixes Implemented

During the independent technical audit, 6 critical issues were diagnosed and systematically resolved:

| # | Identified Problem | Root Cause | Engineering Solution Implemented | Status |
|---|---|---|---|---|
| **1** | **54.2% Duplicate Leakage in Training** | `features.csv` contained 10,823 identical rows across projects, causing 60% CV leakage. | Added `drop_duplicates(subset=FEATURES)` in `build_feature_matrix.py`. Reduced dataset to 9,157 unique rows. | **RESOLVED** |
| **2** | **Resampled Train Metrics Reported** | `train.py` printed F1: 0.9564 and AUC: 0.9922 evaluated on SMOTE training data. | Implemented Out-of-Fold (OOF) validation metrics in `train.py` showing honest performance (AUC: 0.7694). Relabeled step 4 as training sanity check. | **RESOLVED** |
| **3** | **Live Demo Latency (PyDriller)** | PyDriller took 3–5 minutes traversing all 4,800+ commits of Flask on Windows. | Added `max_commits=500` default and `tqdm` progress bar in `git_miner.py`. Demo completes in ~25–35 seconds. Added `--all` and `--max-commits` flags. | **RESOLVED** |
| **4** | **Directory Exclusion Scanner Bug** | `_should_skip()` checked full path strings for `"data"`, causing Flask files in `data/demo_repos/flask` to return 0 functions. Also failed to skip `.venv312`. | Updated `_should_skip()` in `metric_extractor.py` to evaluate path segments relative to the target repository root, and added prefix checks for `.venv`. | **RESOLVED** |
| **5** | **AST Function Name Collisions** | Dictionary keyed by `function.name` overwrote multiple methods with the same name (e.g., `__init__`). | Changed key to `(function.name, function.lineno)` in `metric_extractor.py`. | **RESOLVED** |
| **6** | **Static/Contradictory SHAP Labels** | Fixed string `"Low maintainability index"` printed even when high MI protected a function. | Implemented dynamic directional labels in `shap_explainer.py` using `FEATURE_RISK_LABELS` tuples. | **RESOLVED** |

---

<a name="6-step-by-step-guide-to-run-test-and-demonstrate"></a>
## 6. Step-by-Step Guide to Run, Test, and Demonstrate

### 6.1 Prerequisites & Virtual Environment Activation
Open your terminal in the `codesentinel` directory:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 6.2 Step 1: Run the Automated Unit Test Suite
Verify that all 12 unit tests across Git mining, AST extraction, prediction, and SHAP pass:

```bash
pytest tests/ -v
```

**Expected Terminal Output:**
```
tests/test_miner.py::test_returns_list PASSED                            [  8%]
tests/test_miner.py::test_finds_functions PASSED                         [ 16%]
tests/test_miner.py::test_metrics_and_complexity PASSED                  [ 25%]
tests/test_miner.py::test_bad_or_empty_files_return_no_metrics PASSED   [ 33%]
tests/test_miner.py::test_mine_git_features PASSED                       [ 41%]
tests/test_miner.py::test_feature_schema_length PASSED                   [ 50%]
tests/test_model.py::test_model_predicts_probability PASSED              [ 58%]
tests/test_model.py::test_complex_function_riskier_than_simple PASSED   [ 66%]
tests/test_model.py::test_shap_returns_correct_structure PASSED          [ 75%]
tests/test_model.py::test_shap_values_finite PASSED                      [ 83%]
tests/test_model.py::test_feature_schema_is_8 PASSED                     [ 91%]
tests/test_model.py::test_predict_repo_on_mini_repo PASSED               [100%]

============================== 12 passed in 10.87s ==============================
```

### 6.3 Step 2: Run the Live Demo on the Flask Repository
Analyze the cloned Flask web framework repository (`data/demo_repos/flask`):

```bash
python scripts/demo_buglens.py data/demo_repos/flask
```

**Options:**
* Analyze 1,000 commits: `python scripts/demo_buglens.py data/demo_repos/flask --max-commits 1000`
* Analyze 100% full commit history: `python scripts/demo_buglens.py data/demo_repos/flask --all`
* Analyze CodeSentinel itself: `python scripts/demo_buglens.py .`

#### Live Demo Terminal Output
```
======================================================================
  CODESENTINEL / BUGLENS — LIVE DEFECT RISK INFERENCE
======================================================================
Target Repository : C:\Users\athar\Desktop\Notes\4th Year\BE Project\codesentinel\data\demo_repos\flask

[BugLens] Analysing: C:\Users\athar\Desktop\Notes\4th Year\BE Project\codesentinel\data\demo_repos\flask
  Mining Git history: 100%|████████████████████| 500/500 [00:32<00:00, 15.21 commits/s]
  Git mining complete: 501 commits, 33 Python files
  Scanning 53 Python files...
  Extracted metrics for 410 functions
  Done: 410 functions — HIGH: 0  MEDIUM: 2  LOW: 408

======================================================================
  ANALYSIS SUMMARY (410 functions analyzed)
======================================================================
  • HIGH Risk   (>= 66%):    0 functions
  • MEDIUM Risk (33-66%):    2 functions
  • LOW Risk    (< 33%) :  408 functions

======================================================================
  TOP 10 HIGHEST RISK FUNCTIONS (Priority Queue for AutoTestArmy)
======================================================================
 #   Risk     Prob    Function                 Location
----------------------------------------------------------------------
 1   [MEDIUM] 34.8%   send_from_directory      helpers.py:L543-L583
 2   [MEDIUM] 34.3%   errorhandler             scaffold.py:L606-L647
 3   [LOW   ] 26.2%   route                    scaffold.py:L344-L373
 4   [LOW   ] 24.7%   test_request_context     app.py:L1520-L1567
 5   [LOW   ] 24.7%   teardown_appcontext      app.py:L827-L855
 6   [LOW   ] 24.4%   flash                    helpers.py:L326-L356
 7   [LOW   ] 24.1%   __init__                 blueprints.py:L19-L53
 8   [LOW   ] 23.1%   jsonify                  __init__.py:L138-L170
 9   [LOW   ] 22.7%   teardown_request         scaffold.py:L516-L547
 10  [LOW   ] 21.4%   register_blueprint       app.py:L570-L595

======================================================================
  SHAP ROOT-CAUSE EXPLANATION
  Function : send_from_directory
  File     : helpers.py (lines 543-583)
  Risk     : 34.8% [MEDIUM]
======================================================================

  Key Summary : Large function (high LOC)

  Feature Contribution Breakdown:
  ------------------------------------------------------------------
  • Large function (high LOC)        (value: 41.0  ) -> increases risk  [SHAP: +1.1418]
  • Few dedicated maintainers        (value: 0.0   ) -> decreases risk  [SHAP: -0.9022]
  • High maintainability (protective)(value: 46.5  ) -> decreases risk  [SHAP: -0.6736]
  • Stable code with low churn       (value: 0.0   ) -> decreases risk  [SHAP: -0.6221]

======================================================================
  Demo complete. Phase 0, 1, 2 working as specified!
======================================================================
```

### 6.4 Step 3: Reproduce Training & Feature Assembly Pipeline
To rebuild the deduplicated feature matrix and retrain LightGBM from scratch:

```bash
# 1. Download datasets (if raw data directory is clean)
python scripts/download_datasets.py

# 2. Build and deduplicate the feature matrix
python scripts/build_feature_matrix.py

# 3. Train LightGBM and evaluate Out-of-Fold metrics
python buglens/model/train.py
```

---

<a name="7-roadmap-for-upcoming-phases-phases-3-to-8"></a>
## 7. Roadmap for Upcoming Phases (Phases 3 to 8)

### Phase 3 — FastAPI Backend & Database Layer
* **Goal**: Build a headless REST API server at `http://localhost:8000` persisting analysis runs.
* **Components to Implement**:
  * `backend/database.py`: SQLite engine with SQLAlchemy ORM session factory.
  * `backend/models.py`: Database tables (`PredictionRun`, `FunctionPrediction`, `TestGenerationRun`).
  * `backend/schemas.py`: Pydantic v2 schemas (`PredictRequest`, `PredictResponse`, `FunctionRisk`).
  * `buglens/api/routes.py`: `POST /buglens/predict` (executes BugLens, stores run in DB) and `GET /buglens/runs/{run_id}`.
  * `backend/main.py`: FastAPI application entrypoint with CORS middleware.

### Phase 4 — Ollama + LangGraph Setup
* **Goal**: Connect local LLMs and configure cyclic agent orchestration.
* **Components to Implement**:
  * Local LLM setup: Serve `phi3:mini` and `llama3.1:8b` via Ollama (`http://localhost:11434`).
  * `autotestarmy/graph/state.py`: Define `AgentState` TypedDict storing `repo_path`, `target_functions`, `analysis`, `generated_tests`, `test_results`, `iteration`, and `iteration_history`.

### Phase 5 — AutoTestArmy: Three-Agent Test Generation Pipeline
* **Goal**: LangGraph pipeline synthesizing, executing, and refining unit tests.
* **Components to Implement**:
  * `autotestarmy/agents/analyzer_agent.py`: AST walker extracting function parameters, branch conditions, loops, and boundary values (`None`, `0`, `-1`, `""`, `[]`).
  * `autotestarmy/agents/generator_agent.py`: Local LLM prompt constructor generating valid `pytest` functions.
  * `autotestarmy/agents/critic_agent.py`: Subprocess runner executing generated tests via `pytest --cov --cov-report=xml`.
  * `autotestarmy/graph/pipeline.py`: Cyclic StateGraph connecting Analyzer $\rightarrow$ Generator $\rightarrow$ Critic $\rightarrow$ (Regenerate if failures remain and iterations $< 5$) $\rightarrow$ END.

### Phase 6 — Closed-Loop Feedback Aggregator
* **Goal**: Automatically update BugLens training data based on real test execution outcomes.
* **Components to Implement**:
  * `feedback/aggregator.py`: Heuristic ground-truth labeling:
    * Functions where tests persistently fail $\rightarrow \text{label} = 1$ (confirmed defect).
    * Functions with $\ge 80\%$ line coverage and 0 failures $\rightarrow \text{label} = 0$ (confirmed clean).
  * `feedback/model_updater.py`: Appends feedback to `features.csv` and retrains `lightgbm_model.pkl` every 5 test cycles.

### Phase 7 — VS Code Extension
* **Goal**: Bring the platform directly into the developer's IDE via TypeScript.
* **Components to Implement**:
  * `src/heatmapProvider.ts`: Gutter and line background decorations (Red for HIGH risk, Yellow for MEDIUM risk, Green for LOW risk).
  * `src/shapPanel.ts`: Webview panel rendering SHAP feature importance charts.
  * `src/testGenerationPanel.ts`: Command Palette action triggering test generation for high-risk functions.

### Phase 8 — Empirical Research Evaluation
* **Goal**: Statistically prove that BugLens-guided test generation is superior to untargeted test generation.
* **Experimental Protocol (`scripts/run_evaluation.py`)**:
  * On 3 open-source repositories (e.g., Flask, Requests), compare:
    * **Targeted Condition**: AutoTestArmy tests the top-10 highest-risk functions identified by BugLens.
    * **Untargeted Condition**: AutoTestArmy tests 10 randomly selected functions.
  * Measure and report: Final coverage on vulnerable code, iterations needed to stabilize, and total test pass rate.

---

<a name="8-teacher--project-guide-presentation-guide-viva-defense"></a>
## 8. Teacher & Project Guide Presentation Guide (Viva Defense)

When presenting to your project guide, internal review committee, or external examiners, follow this structured narrative:

### 1. The Core Hook (1 Minute)
> *"Respected Guide / Examiners, writing unit tests is the most neglected part of software development. Existing AI coding assistants can generate unit tests, but they suffer from 'test generation blindness'—they have no idea where the bugs actually hide, so they waste compute generating tests for simple getter/setter functions.  
> CodeSentinel solves this by building a closed-loop platform: first, our BugLens module uses Git evolutionary history and AST complexity metrics to predict high-risk functions using a LightGBM model and SHAP explainability. Then, our AutoTestArmy prioritizes those high-risk functions to generate targeted unit tests."*

### 2. Live Demonstration Flow (3–4 Minutes)
1. **Show Test Suite Verification**:
   - Run `pytest tests/ -v`.
   - Explain: *"We maintain 12 unit tests verifying repository mining, complexity parsing, and model monotonicity without mocks."*
2. **Execute Live BugLens Scan**:
   - Run `python scripts/demo_buglens.py data/demo_repos/flask`.
   - Explain: *"Notice that BugLens is mining Pallets Flask—a production open-source repository. In under 35 seconds, it analyzed 500 commits and 410 functions."*
3. **Show the Priority Queue**:
   - Point to the Top 10 table: *"Here is our Priority Queue. Instead of randomly testing 410 functions, our system highlights `send_from_directory` and `errorhandler` as high-priority targets for AutoTestArmy."*
4. **Show SHAP Explainability**:
   - Point to the SHAP breakdown: *"This is our XAI layer. The model doesn't just output a black-box probability; it mathematically attributes that large function size increased risk by +1.14 in log-odds margin space."*

### 3. Key Achievements to Highlight
* **Deduplicated Dataset**: 9,157 unique samples compiled from 13 NASA and PROMISE benchmark datasets.
* **Statistically Honest Validation**: 5-Fold Stratified Cross-Validation with SMOTE restricted strictly to training folds, achieving **0.7694 AUC-ROC**.
* **Zero Cloud Dependency**: Everything runs 100% offline on a standard developer laptop (LightGBM on CPU, local LLM via Ollama).

---

<a name="9-comprehensive-viva-voce-questions--model-answers"></a>
## 9. Comprehensive Viva Voce Questions & Model Answers

### Category 1: Architecture & Software Engineering

#### Q1: What is the overall architecture of CodeSentinel?
**Answer**: CodeSentinel uses a three-layer closed-loop architecture:
1. **Layer 1 (BugLens)**: Mines historical Git commits via PyDriller and extracts AST complexity via Radon. A LightGBM classifier predicts function-level defect probability, explained via SHAP TreeExplainer.
2. **Layer 2 (Bridge)**: Ranks functions into a Priority Queue (HIGH, MEDIUM, LOW risk) so test generation focuses on high-risk functions first.
3. **Layer 3 (AutoTestArmy)**: A LangGraph multi-agent system (Analyzer $\rightarrow$ Generator $\rightarrow$ Critic) using a local LLM (Ollama Phi-3) to synthesize and iteratively refine pytest suites.
4. **Closed Loop**: Test outcomes (persistent failures vs high coverage) feed back into the training dataset to retrain LightGBM.

#### Q2: Why build a VS Code extension instead of a standalone web dashboard?
**Answer**: Developer workflow integration. A web dashboard requires switching context, copying files, and leaving the IDE. By integrating directly into VS Code via text editor decorations (heatmaps) and hover tooltips, risk probabilities and SHAP explanations appear directly inline where code reviews and refactoring decisions occur.

---

### Category 2: Machine Learning & Methodology

#### Q3: Why did you select LightGBM over Deep Learning (e.g., CodeBERT) for defect prediction?
**Answer**: We chose LightGBM for three primary reasons:
1. **Tabular Metric Superiority**: Defect prediction metrics (cyclomatic complexity, churn, developer counts) are tabular data. Extensive literature (Giray et al., 2023) demonstrates that gradient boosted decision trees consistently match or outperform deep learning on tabular features.
2. **Computational Efficiency**: LightGBM trains on 9,000+ samples in under 2 seconds on a CPU without requiring expensive GPU hardware.
3. **Exact Explainability**: Tree ensembles natively support `shap.TreeExplainer`, which computes exact Shapley values in polynomial time, unlike deep learning which requires costly approximations.

#### Q4: How did you address class imbalance, and how did you prevent data leakage?
**Answer**: Defect datasets are inherently imbalanced (~18% defective in our clean dataset). We applied SMOTE (Synthetic Minority Over-sampling Technique) to synthetically balance minority-class samples.  
To prevent data leakage, SMOTE was fitted exclusively **inside each training fold** of our 5-Fold Stratified Cross-Validation pipeline. Validation folds contained only original, unaugmented samples.

#### Q5: What caused the initial inflated AUC of 0.88, and how did you fix it?
**Answer**: Our audit revealed that 54.2% of the original `features.csv` dataset consisted of duplicate feature vectors from legacy NASA files. In standard K-Fold CV, identical samples appeared in both training and validation folds (validation leakage).  
We resolved this by adding exact deduplication in `build_feature_matrix.py`, retaining 9,157 distinct feature vectors. Our honest Out-of-Fold (OOF) cross-validation score is **0.7694 AUC-ROC**, which is mathematically sound and above the academic baseline of 0.75.

#### Q6: Does LightGBM require StandardScaler?
**Answer**: Mathematically, no. Decision trees split based on feature ordering ($x_j \le \theta$), making them invariant to monotonic transformations like standardization. However, scaling was retained during training to maintain compatibility with downstream distance-based tooling and SHAP visualization scales.

---

### Category 3: Git Mining & Static Analysis

#### Q7: How does your system map Git history metrics down to individual functions?
**Answer**: In our current Phase 1 implementation, PyDriller extracts Git commit history, churn, author emails, and bug-fix keywords at the **file level** by inspecting file diffs. Functions within each file are parsed using Python's `ast` module. The file-level Git metrics are left-joined to all functions enclosed within that file.  
While all functions in a file share the same evolutionary churn context, their static complexity metrics (Radon CC, LOC, MI) distinguish individual function risk. In Phase 8, we plan to implement hunk-level AST line attribution via `git blame`.

#### Q8: How do you identify bug-fix commits in Git history?
**Answer**: We perform case-insensitive keyword matching across commit messages using 10 standard keywords: `fix`, `bug`, `patch`, `error`, `defect`, `hotfix`, `repair`, `resolve`, `issue`, and `crash`. If a commit message matches, the `bug_fix_commits` counter increments for every modified file.

#### Q9: What is "recent churn" and why is it useful?
**Answer**: Recent churn measures the cumulative lines added and deleted in a file over the past 90 calendar days relative to the scan date. Evolutionary software studies demonstrate that recent churn is a stronger predictor of near-term defect occurrence than lifetime churn, as newly modified code is less stable than legacy code that has survived multiple release cycles.

---

### Category 4: Explainability (XAI) & SHAP

#### Q10: What is SHAP, and how does it compute feature importance?
**Answer**: SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It calculates the marginal contribution of each feature across all possible feature subsets. For LightGBM, `TreeExplainer` calculates exact contributions in margin output space (log-odds). Positive SHAP values indicate features pushing the prediction toward defect risk, while negative values indicate protective, stabilizing factors.

#### Q11: Why does SHAP return values in log-odds margin space rather than probability space?
**Answer**: By default, `TreeExplainer` on binary LightGBM models operates on the raw margin output (sum of tree leaves before the sigmoid link function):
$$\text{Margin}(x) = \sum_{i=1}^M \phi_i(x) + \phi_0, \quad P(y=1) = \frac{1}{1 + e^{-\text{Margin}(x)}}$$
Evaluating SHAP in margin space preserves exact additivity ($\sum \phi_i + \phi_0 = \text{Margin}$), which is mathematically exact and avoids non-linear compression near 0% and 100%.

---

### Category 5: Multi-Agent Testing & Future Scope

#### Q12: Why use a 3-agent pipeline (Analyzer, Generator, Critic) instead of a single LLM prompt?
**Answer**: Specialization and execution grounding. A single prompt trying to analyze ASTs, generate tests, and validate correctness frequently suffers from context hallucination.  
In our design:
1. **Analyzer Agent**: Deterministic AST parser (100% accurate, non-LLM).
2. **Generator Agent**: Focused exclusively on test synthesis from structured boundary prompts.
3. **Critic Agent**: Executes actual `pytest` subprocesses. Ground truth comes from the Python interpreter, not LLM consensus.

#### Q13: How does the closed-loop feedback mechanism prevent concept drift?
**Answer**: As developers run AutoTestArmy, actual test execution outcomes are stored in a SQLite database. When 50+ new labeled functions accumulate, LightGBM is retrained on combined benchmark data and project-specific feedback. This adapts the model to the coding conventions and defect patterns of the target repository over time.

---

<a name="10-summary-verification-checklist-for-project-evaluation"></a>
## 10. Summary Verification Checklist for Project Evaluation

- [x] **Phase 0 Verified**: Project scaffold, central config schema, pinned requirements, Ollama local LLM runtime.
- [x] **Phase 1 Verified**: 13 datasets downloaded, normalized, and deduplicated into `features.csv` (9,157 rows). PyDriller Git mining and Radon AST extraction fully functional.
- [x] **Phase 2 Verified**: LightGBM model trained with SMOTE inside CV folds (OOF AUC = **0.7694**). Serialized artifacts saved in `buglens/model/saved/`. SHAP explainability and live inference operational.
- [x] **Codebase Improvements Applied**: Fixed duplicate data leakage, demo latency (added `max_commits=500` & `tqdm`), directory exclusion bug, AST dictionary collision, and dynamic directional SHAP labels.
- [x] **Test Suite Passing**: All 12 unit tests in `tests/test_miner.py` and `tests/test_model.py` pass with 100% success rate.
- [x] **Live CLI Demo Ready**: Interactive demo runs on `data/demo_repos/flask` or any local codebase via `python scripts/demo_buglens.py`.
- [x] **Roadmap Clear**: Technical architecture and data contracts for Phases 3 through 8 fully defined and ready for implementation.
