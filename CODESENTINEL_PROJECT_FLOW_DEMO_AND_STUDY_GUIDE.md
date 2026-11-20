# CodeSentinel: Project Flow, Live Demonstration Guide & Comprehensive Viva Study Guide

> **AI-Powered Defect Prediction, Targeted Multi-Agent Test Generation & Closed-Loop Learning**  
> **B.E. Final Year Project 2026–2027 | Group 10**  
> **Department of Artificial Intelligence & Data Science**  
> **Fr. Conceicao Rodrigues College of Engineering (FR. CRCE), Bandra (W), Mumbai**  
> **Affiliated with the University of Mumbai**  
> **Project Team:** Atharva Nanaware (10511) · Suhera Siddiqui (10381) · Aum Singh (10383)

---

## Document Purpose & Overview

This document is prepared specifically for your **College Project Presentation, Evaluation Defense, and Technical Viva**. It is divided into three comprehensive sections:

1. **[Part 1: The Complete Project Flow](#part-1-the-complete-project-flow)** — The architectural and logical flow of CodeSentinel from raw source code to predictions, agentic test generation, and continuous retraining.
2. **[Part 2: Step-by-Step Live Demonstration Guide](#part-2-step-by-step-live-demonstration-guide)** — Exact walkthrough of how to demo the project to teachers/evaluators, including commands, screen layouts, and word-for-word talking points.
3. **[Part 3: Comprehensive Viva Study Guide & Q&A](#part-3-comprehensive-viva-study-guide--qa)** — Deep theoretical explanations, mathematical formulations, architectural justifications, and 20 critical questions examiners will ask.

---

# Part 1: The Complete Project Flow

CodeSentinel does not simply generate tests or predict bugs in isolation. It implements a **continuous, closed-loop software intelligence pipeline**:

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer / User
    participant Ext as Layer 1: Ingestion & Mining (Git + AST)
    participant ML as Layer 2: BugLens (LightGBM + SHAP)
    participant API as Layer 3: FastAPI Backend & SQLite
    participant Agents as Layer 4: AutoTestArmy (LangGraph + Ollama)
    participant Runner as Layer 4: Pytest Subprocess Sandbox
    participant Loop as Layer 5: Closed-Loop Feedback Engine

    Dev->>API: 1. Submit Repository Path (POST /buglens/predict)
    API->>Ext: 2. Mine Git Commits & Parse AST Metrics
    Ext-->>API: 3. Return 8-Feature Numerical Vectors
    API->>ML: 4. Predict Defect Probabilities & Compute SHAP Values
    ML-->>API: 5. Return Probabilities + Risk Tiers (HIGH, MED, LOW)
    API->>API: 6. Save Run & Build Priority Queue in SQLite

    Dev->>API: 7. Request Test Generation (POST /autotestarmy/generate)
    API->>Agents: 8. Trigger LangGraph Loop for HIGH/MED functions
    Note over Agents: Analyzer Agent extracts AST boundaries & branches
    Note over Agents: Generator Agent prompts Ollama (phi3:mini)
    Agents->>Runner: 9. Execute Generated Pytest Code in Subprocess
    Runner-->>Agents: 10. Return Pass/Fail Counts, Tracebacks & Coverage XML
    
    alt If Failures Found & Iterations < Max
        Agents->>Agents: Self-Correction: Inject failure traceback into LLM prompt
        Agents->>Runner: Re-run regenerated test suite
    else If Coverage >= 80% or Stabilised
        Agents-->>API: 11. Final State (status=DONE, final_coverage, test_file)
    end

    API->>Loop: 12. Hook Test Outcomes into Feedback Aggregator
    Loop->>Loop: 13. Label Outcomes: failed -> y=1, clean >= 75% -> y=0
    Loop->>Loop: 14. Append to feedback_labels.csv & Retrain LightGBM
    API-->>Dev: 15. Deliver Completed Test Suites & Updated Model
```

---

### Phase-by-Phase Technical Flow

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEM EXECUTION PIPELINE                              │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DUAL-ENGINE REPOSITORY MINING (Phase 1)                                      │
│ • PyDriller scans commit history: computes churn, author count, bug-fix keywords.     │
│ • Radon scans Python AST: calculates Cyclomatic Complexity, LOC, Maintainability.     │
│ • Assembles standardized 8-feature vector: [CC, LOC, MI, Churn, Devs, Fixes, Age, RCh]│
└───────────────────────────────────┬───────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: BUGLENS DEFECT RISK INFERENCE & SHAP EXPLAINABILITY (Phase 2)                │
│ • StandardScaler normalizes metrics against 9,157 NASA/PROMISE benchmark samples.    │
│ • LightGBM classifier calculates P(defect) for every individual function.             │
│ • TreeSHAP computes exact local Shapley values: isolates top 3 root-cause drivers.    │
│ • Sorts functions into Priority Queue: HIGH (>=66%), MEDIUM (33-66%), LOW (<33%).     │
└───────────────────────────────────┬───────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: BACKEND REST PERSISTENCE & QUEUE DISPATCH (Phase 3)                          │
│ • FastAPI records full execution to SQLite (`codesentinel.db`).                       │
│ • Tables populated: `PredictionRun` and `FunctionPrediction` (with SHAP JSON).        │
│ • Exposes `GET /buglens/runs/{id}/queue` providing targeted feed to generative loop.  │
└───────────────────────────────────┬───────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: AUTOTESTARMY MULTI-AGENT TEST SYNTHESIS (Phases 4 & 5)                       │
│ • Analyzer Agent: Deterministically extracts parameter names, branches, boundaries.   │
│ • Generator Agent: Prompts Ollama (`phi3:mini`) with memory guard (num_ctx=2048).     │
│ • Critic Agent: Runs `pytest --cov` in isolated subprocess, parses coverage & errors. │
│ • Self-Correction Loop: If tests fail, tracebacks loop back to Generator for repair.  │
└───────────────────────────────────┬───────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: CLOSED-LOOP CONTINUOUS FEEDBACK & ADAPTATION (Phase 6)                       │
│ • Dynamic Ground-Truth Labeling: failed tests -> y=1 (defect), high coverage -> y=0.  │
│ • Records deduplicated samples in `data/processed/feedback_labels.csv`.               │
│ • Merges baseline with feedback data into `features_with_feedback.csv`.              │
│ • Retrains LightGBM + Scaler + SHAP to adapt to target repository coding styles.      │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 2: Step-by-Step Live Demonstration Guide

Use this exact structure during your project viva or teacher presentation. It shows a complete, working live system in **6 to 8 minutes**.

---

## 2.1 Pre-Presentation Setup Checklist (Do This 5 Minutes Before)

Open your terminal, navigate to the project directory, and activate your environment:

```powershell
cd "c:\Users\athar\Desktop\Notes\4th Year\BE Project\codesentinel"
.\.venv\Scripts\Activate.ps1
```

Confirm Ollama is running in the background:
```powershell
ollama list
```
*(Ensure `phi3:mini` is listed).*

---

## 2.2 Live Demo Step 1: Automated Full-System Verification Audit (1 Minute)

### What to execute:
```powershell
python scripts/verify_all_phases.py
```

### What appears on the screen:
A live audit testing Phase 0 through Phase 6 with green `[OK] PASSED` markers.

### What to say to your teachers:
> *"Respected teachers, before demonstrating individual features, we begin by executing our automated verification audit. This script tests every layer of our platform: our local Python 3.12 environment, the 9,157-sample NASA/PROMISE benchmark matrix, our trained LightGBM model, the SQLite database, the Ollama local LLM runtime, the LangGraph multi-agent loop, and the continuous feedback retrainer. As you can see on the scorecard, all 6 implemented phases are verified and 100% operational."*

---

## 2.3 Live Demo Step 2: BugLens Defect Prediction & SHAP on Flask (2 Minutes)

### What to execute:
```powershell
python scripts/demo_buglens.py data/demo_repos/flask
```

### What appears on the screen:
1. `tqdm` progress bar mining 500 Git commits in the Pallets Flask open-source framework.
2. AST complexity extraction across Python functions.
3. Formatted ASCII table ranking functions into **HIGH**, **MEDIUM**, and **LOW** risk tiers.
4. Top 3 SHAP root-cause feature drivers for the most defect-prone function.

### What to say to your teachers:
> *"Here, BugLens is analyzing a real-world, open-source repository: Pallets Flask. Notice how it operates in two steps:*
> *First, it mines historical commit churn and author diversity using PyDriller, alongside AST cyclomatic complexity using Radon.*
> *Second, our LightGBM model evaluates the 8-feature representation, classifying every function into HIGH, MEDIUM, or LOW risk.*
> *Crucially, we do not treat the model as a black box. Look at this SHAP breakdown: for the highest-risk function, it mathematically demonstrates that high code churn (+0.245) and elevated cyclomatic complexity (+0.312) are the exact root causes driving the defect probability."*

---

## 2.4 Live Demo Step 3: Interactive FastAPI Backend & Swagger UI (2 Minutes)

### What to execute:
```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Open browser at: 👉 **`http://127.0.0.1:8000/docs`**

### What to demonstrate in browser:
1. **`GET /health`**:
   - Click "Try it out" $\to$ "Execute".
   - Show `models_loaded: true`.
2. **`POST /buglens/predict`**:
   - Execute with `{"repo_path": "data/demo_repos/flask", "max_commits": 50}`.
   - Show the JSON response returning function risks and `run_id`.
3. **`GET /buglens/runs/{run_id}/queue`**:
   - Enter `run_id: 1` $\to$ click "Execute".
   - **Point to the screen**: Show that it filters *only* `HIGH` and `MEDIUM` risk functions.
4. **`POST /autotestarmy/generate`**:
   - Execute with `{"prediction_run_id": 1, "max_iterations": 3}`.
   - Show the response: `status: "PENDING"`, task enqueued to background workers.
5. **`GET /feedback/stats`**:
   - Execute to show accumulated samples and defect ratios.

### What to say to your teachers:
> *"Our platform is built on an enterprise-grade FastAPI REST microservice architecture. Every analysis run and function prediction is persisted into a local SQLite database (`codesentinel.db`).*
> *Notice our priority queue endpoint: instead of blindly prompting an LLM to generate tests for hundreds of boilerplate functions, our backend feeds only the high-risk and medium-risk functions directly into our multi-agent test generator."*

---

## 2.5 Live Demo Step 4: AutoTestArmy's Generated Unit Tests (1.5 Minutes)

### What to show:
Open the file in VS Code or your terminal:
👉 **[data/generated_tests/run_7/generated_tests.py](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/generated_tests/run_7/generated_tests.py)**

Or run:
```powershell
Get-Content data\generated_tests\run_7\generated_tests.py -Head 35
```

### What appears on the screen:
Real, synthesized `pytest` functions created by local `phi3:mini` testing normal paths, edge cases (`None`, `""`, `[]`), and `with pytest.raises(...)`.

### What to say to your teachers:
> *"Here is the output of AutoTestArmy. This is not static code written by us—this was synthesized entirely by our local Ollama model (`phi3:mini`) running on local hardware with zero paid API keys.*
> *Notice three key technical achievements:*
> 1. *It does not just test happy paths; our Analyzer Agent extracted boundary inputs, so the LLM explicitly tested `None`, empty strings, and empty lists.*
> 2. *It includes exception assertions using `pytest.raises`.*
> 3. *Our Critic Agent executed these tests in a sandboxed subprocess and measured over 90% branch coverage."*

---

## 2.6 Live Demo Step 5: Closed-Loop Feedback Data & Retraining (1 Minute)

### What to show:
Open the file:
👉 **[data/processed/feedback_labels.csv](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/processed/feedback_labels.csv)**

Or run:
```powershell
Get-Content data\processed\feedback_labels.csv
```

### What appears on the screen:
Rows showing functions with `defects: 1` or `defects: 0` and `source: autotestarmy_feedback`.

### What to say to your teachers:
> *"This is the closed-loop continuous learning engine—the crowning contribution of CodeSentinel. When tests fail repeatedly during execution, the system dynamically labels that function as defective ($y = 1$). When tests pass cleanly with high coverage, it is labeled clean ($y = 0$).*
> *These records are stored in `feedback_labels.csv`, merged with our baseline training set, and used to retrain LightGBM. Over time, CodeSentinel automatically adapts to any repository's specific codebase conventions without human intervention."*

---

# Part 3: Comprehensive Viva Study Guide & Q&A

Use this section to prepare for any technical question your guide or external examiner might ask.

---

## 3.1 Mathematical & Theoretical Formulations

### 1. McCabe Cyclomatic Complexity ($M$)
- **Formula**:
  $$M = E - N + 2P$$
  - $E$ = Number of edges in the function's Control Flow Graph (CFG).
  - $N$ = Number of nodes (sequential statements).
  - $P$ = Number of connected components (typically 1 for a single function).
- **Practical Interpretation**: Represents the number of linearly independent paths through the code. High $M$ ($\ge 10$) indicates complex conditional branching and strongly correlates with defect density.

### 2. Maintainability Index ($\text{MI}$)
- **Formula**:
  $$\text{MI} = \max\left(0, \frac{171 - 5.2 \ln(V) - 0.23(M) - 16.2 \ln(\text{LOC})}{171} \times 100\right)$$
  - $V$ = Halstead Volume (program vocabulary and length).
  - $M$ = Cyclomatic Complexity.
  - $\text{LOC}$ = Executable lines of code.
- **Range**: 0 to 100. Values $< 65$ represent difficult-to-maintain, error-prone code.

### 3. TreeSHAP Formulation (Cooperative Game Theory)
- **Formula**:
  $$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
  - $\phi_i$ = Shapley value of feature $i$ (its additive contribution to the prediction).
  - $F$ = Total feature set (8 features).
  - $S$ = Subset of features excluding feature $i$.
  - $f(S)$ = Expected model prediction given only features in subset $S$.
- **Efficiency Property**: The sum of all feature attributions equals the difference between the model output and the base expected value:
  $$\sum_{i=1}^{|F|} \phi_i(f, x) = f(x) - \mathbb{E}[f(X)]$$

---

## 3.2 Machine Learning Architecture Details

| Property | Implementation Detail | Justification |
| :--- | :--- | :--- |
| **Model** | `LightGBM (LGBMClassifier)` | Gradient Boosted Decision Tree using Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). Outperforms Deep Learning on tabular regimes. |
| **Objective** | `binary` (Log Loss) | Predicts calibrated posterior probability $P(\text{defect} = 1 \mid x)$. |
| **Class Balancing** | `SMOTE` (In-Fold Only) | Resamples minority class only inside each training fold to eliminate synthetic data leakage into validation splits. |
| **Validation Metric** | `ROC-AUC` (OOF = 0.7694) | Threshold-independent metric essential for imbalanced defect datasets (18.1% positive). Exceeds target threshold of 0.75. |
| **Training Corpus** | 13 NASA MDP & PROMISE datasets | 9,157 unique, deduplicated, normalized historical records. |

---

## 3.3 Top 20 Viva Questions & Exact Answers

### Q1: What is the core problem CodeSentinel solves?
> **Answer**: CodeSentinel addresses the inefficiency and unreliability of software testing. Developers spend 50% of their time writing tests, but human bias often misses edge cases. Meanwhile, LLMs generate tests blindly without knowing where bugs actually hide. CodeSentinel uses machine learning to predict which specific functions are defect-prone, explains why using SHAP, and deploys a multi-agent loop to generate targeted unit tests for those high-risk functions.

### Q2: Why did you choose LightGBM over Deep Learning (like Transformers or GNNs)?
> **Answer**: For tabular software metrics (churn, complexity, LOC), tree-based gradient boosting models consistently outperform neural architectures in empirical studies (e.g., Grinsztajn et al., NeurIPS 2022). LightGBM trains in seconds, handles imbalanced class ratios efficiently, runs inference on commodity CPUs with sub-millisecond latency, and allows exact, polynomial-time TreeSHAP explainability ($O(TLD^2)$), which is impossible with black-box neural networks.

### Q3: What is the significance of the 8 features in your schema?
> **Answer**: Our 8-feature schema balances two complementary dimensions of software risk:
> 1. **Internal Code Complexity**: Cyclomatic complexity, lines of code, and maintainability index (extracted via AST parsing).
> 2. **External Process Churn**: Total churn, unique author count, historical bug fixes, file age, and recent 90-day churn (mined via Git commit history).
> Combining structural and process metrics significantly increases defect prediction accuracy compared to using code metrics alone.

### Q4: What is data leakage and how did your team prevent it in Phase 2?
> **Answer**: Data leakage occurs when information from the validation or test set is inadvertently shared with the model during training. In defect prediction, applying SMOTE oversampling or `StandardScaler` to the entire dataset *before* cross-validation creates synthetic samples based on validation data, artificially inflating performance. We prevented this by implementing **in-fold resampling**: splitting into 5 stratified folds first, and fitting SMOTE and the scaler strictly on the 4 training folds during each cross-validation step.

### Q5: What does Out-of-Fold (OOF) AUC mean and what is your score?
> **Answer**: Out-of-Fold AUC is computed by aggregating predictions made on each holdout validation split during cross-validation, meaning every prediction evaluated was made on data the model never saw during training. Our model achieves an **honest OOF AUC of 0.7694**, exceeding the IEEE benchmark target of 0.75.

### Q6: Why do you need SHAP? Can't you just look at global feature importances?
> **Answer**: Global feature importances (like Gini gain) tell you which features were important across the entire training set, but they cannot tell you why a *specific* function in a repository was flagged. SHAP provides **local explainability**: for a specific function `calculate_tax()`, it calculates the exact mathematical contribution of each feature (e.g., +0.32 due to complexity, +0.15 due to churn), allowing developers to verify the diagnosis and feeding actionable root causes into the LLM.

### Q7: What is the architecture of AutoTestArmy?
> **Answer**: AutoTestArmy is an autonomous three-agent state machine built with **LangGraph**:
> 1. **Analyzer Agent**: Deterministically parses function ASTs to extract parameters, branches, and boundary inputs.
> 2. **Generator Agent**: Prompts our local LLM (`phi3:mini`) to write pytest suites.
> 3. **Critic Agent**: Executes tests in a sandboxed subprocess using `pytest --cov`, parses pass/fail counts and coverage XML, and either terminates or loops back with failure tracebacks for self-correction.

### Q8: Why use LangGraph instead of a simple sequential Python script?
> **Answer**: LangGraph provides a cyclical state machine (`StateGraph`) with typed shared state (`AgentState`). A linear script cannot cleanly manage dynamic conditional loops where the Critic agent evaluates test failures, injects error tracebacks, and routes execution back to the Generator agent for iterative repair while tracking telemetry history.

### Q9: Why run Ollama locally instead of using the OpenAI API?
> **Answer**: Running locally via Ollama (`phi3:mini`) guarantees:
> 1. **Zero Financial Cost**: Completely free and independent of paid API quotas.
> 2. **Code Privacy & Security**: Proprietary enterprise source code never leaves the developer's local machine.
> 3. **Offline Reliability**: Works in air-gapped or restricted development environments without network dependencies.

### Q10: What memory bug did you discover in Ollama and how was it solved?
> **Answer**: Ollama's `phi3:mini` defaults to a 128k context window, which triggers a **51 GB RAM allocation** for the KV cache on CPU-backed runtimes, causing an immediate out-of-memory error. We resolved this in `autotestarmy/tools/llm.py` by strictly enforcing `options={"num_ctx": 2048}`, capping memory consumption to negligible levels and ensuring smooth execution on standard PCs.

### Q11: How does the self-correction loop work in AutoTestArmy?
> **Answer**: When the Critic Agent executes the synthesized pytest file via subprocess, if any assertion fails or an error is thrown, the Critic extracts the failure traceback (`FAILED test_func - AssertionError:...`). It increments the iteration counter, changes the state status to `GENERATING`, and loops back. The Generator Agent receives these tracebacks in its prompt and instructs the LLM to fix the specific broken assertion.

### Q12: What are the stopping conditions for the agentic loop?
> **Answer**: The Critic terminates the loop and sets `status = "DONE"` under three conditions:
> 1. **Full Coverage**: Line/branch coverage reaches $\ge 80\%$.
> 2. **Stabilised**: 0 failures and 0 errors are achieved over $\ge 2$ iterations.
> 3. **Budget Exhausted**: The iteration count reaches `max_iterations` (default: 5).

### Q13: How does the system handle infinite loops or dangerous code in generated tests?
> **Answer**: In `autotestarmy/tools/pytest_runner.py`, tests are executed inside an isolated `subprocess.run` with a strict `timeout=45` seconds. If a test contains an infinite loop, the subprocess is killed, a timeout error is recorded, and the process safely continues without freezing the application.

### Q14: What is the Closed-Loop Feedback Aggregator (Phase 6)?
> **Answer**: It is our continuous learning engine. Defect prediction models suffer from distribution drift when applied to new repositories. When AutoTestArmy executes tests, the outcomes provide dynamic ground truth:
> - Persistent test failures indicate confirmed defects ($y = 1$).
> - Clean passes with high coverage indicate robust code ($y = 0$).
> These labeled records are appended to `feedback_labels.csv` and used to retrain LightGBM, allowing the model to adapt to specific repository conventions over time.

### Q15: How do you prevent label noise in the feedback loop?
> **Answer**: Inconclusive test runs—such as tests that pass with low coverage ($< 75\%$)—are discarded. Only high-confidence signals (verified runtime exceptions or high-coverage clean executions) are converted into training labels, protecting the model from noisy data.

### Q16: How is the backend structured?
> **Answer**: Built using FastAPI with CORS middleware and SQLite (`codesentinel.db`) via SQLAlchemy ORM. It provides endpoints for defect prediction (`/buglens/*`), multi-agent generation (`/autotestarmy/*`), and feedback retraining (`/feedback/*`), with interactive documentation at `/docs`.

### Q17: What is the priority queue endpoint and why is it important?
> **Answer**: The endpoint is `GET /buglens/runs/{run_id}/queue`. It filters functions belonging to `HIGH` ($\ge 66\%$) and `MEDIUM` ($33\% - 66\%$) risk tiers, sorted descending by defect probability. This serves as the architectural bridge between BugLens and AutoTestArmy, ensuring LLM compute is focused only where defects are likely to occur.

### Q18: What is the role of SQLite in the architecture?
> **Answer**: It provides local, persistent storage across three relational models:
> 1. `PredictionRun`: Records historical analysis runs and overall risk statistics.
> 2. `FunctionPrediction`: Stores function line coordinates, defect probabilities, and SHAP drivers.
> 3. `TestGenerationRun`: Tracks agent iteration history, final coverage percentages, and generated test file paths.

### Q19: How many tests does your project have and what is the pass rate?
> **Answer**: We have an automated regression suite of **30 comprehensive tests** across 5 test modules (`test_miner.py`, `test_model.py`, `test_api.py`, `test_agents.py`, `test_feedback.py`). The suite executes in **54.28 seconds with a 100% pass rate** (30/30 passed).

### Q20: What are the upcoming Phases 7 and 8?
> **Answer**:
> - **Phase 7**: The Visual Studio Code extension (`vscode-extension/`), which integrates gutter heatmaps (red/yellow/green icons), hover risk tooltips, and an interactive SHAP webview sidebar directly into the developer's IDE.
> - **Phase 8**: An empirical research evaluation comparing targeted test generation against untargeted generation on three open-source repositories (`Flask`, `Requests`, `Click`) for our final thesis and conference publication.

---

# Part 4: Presentation Team Roles & Division of Responsibilities

When presenting as Group 10, structure your verbal defense by dividing the functional modules:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              GROUP 10 PRESENTATION ROLES                               │
├────────────────────┬────────────────────────────────────┬──────────────────────────────┤
│ Team Member        │ Primary Technical Ownership        │ Key Presentation Segments    │
├────────────────────┼────────────────────────────────────┼──────────────────────────────┤
│ Atharva Nanaware   │ System Architecture, Mining Engine │ • Project Introduction       │
│ (10511)            │ & Multi-Agent Test Loop            │ • Dual Mining Pipeline       │
│                    │                                    │ • AutoTestArmy LangGraph Loop│
│                    │                                    │ • Live Demo Command Runner   │
├────────────────────┼────────────────────────────────────┼──────────────────────────────┤
│ Suhera Siddiqui    │ Machine Learning, SMOTE Training   │ • Literature Baseline        │
│ (10381)            │ & SHAP Explainability Engine       │ • LightGBM Model & OOF AUC   │
│                    │                                    │ • TreeSHAP Math & Root Causes│
│                    │                                    │ • Model Retraining Pipeline  │
├────────────────────┼────────────────────────────────────┼──────────────────────────────┤
│ Aum Singh          │ FastAPI Backend, SQLite DB Layer   │ • REST API & Swagger UI Demo │
│ (10383)            │ & Closed-Loop Feedback Aggregator  │ • Database Schema & Queue    │
│                    │                                    │ • Heuristic Labeling Logic   │
│                    │                                    │ • Automated Test Suite (30)  │
└────────────────────┴────────────────────────────────────┴──────────────────────────────┘
```

---

### Project Repository & Reference Links
- **GitHub Repository**: [https://github.com/nanawareatharva/codesentinel.git](https://github.com/nanawareatharva/codesentinel.git)
- **Master Technical Specification**: [CODESENTINEL_MASTER_ARCHITECTURE_AND_SYSTEM_SPECIFICATION.md](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/CODESENTINEL_MASTER_ARCHITECTURE_AND_SYSTEM_SPECIFICATION.md)
- **Interactive API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
