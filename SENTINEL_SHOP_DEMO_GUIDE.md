# SentinelShop: Complete Demonstration & Testing Guide

> **Target Repository**: [`data/demo_repos/sentinel_shop`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop)  
> **Purpose**: Dedicated viva presentation showcase demonstrating CodeSentinel's defect prioritization across **HIGH Risk** ($\ge 66\%$), **MEDIUM Risk** ($33\% - 66\%$), and **LOW Risk** ($< 33\%$) functions within a single repository, followed by autonomous multi-agent unit test generation.  
> **Project**: CodeSentinel (B.E. Final Year Project 2026–2027 | Group 10 | FRCRCE, Mumbai)  
> **Team**: Atharva Nanaware (10511) · Suhera Siddiqui (10381) · Aum Singh (10383)

---

## Table of Contents
1. [Overview of SentinelShop](#1-overview-of-sentinelshop)
2. [Function-by-Function Risk Classification](#2-function-by-function-risk-classification)
3. [Environment & Cold-Start Setup](#3-environment--cold-start-setup)
4. [Terminal Demonstration: 1-Click End-to-End Pipeline](#4-terminal-demonstration-1-click-end-to-end-pipeline)
5. [Terminal Demonstration: Step-by-Step Modular Commands](#5-terminal-demonstration-step-by-step-modular-commands)
6. [REST API Demonstration (FastAPI Swagger UI & PowerShell)](#6-rest-api-demonstration-fastapi-swagger-ui--powershell)
7. [Inspecting Generated Unit Tests & Closed-Loop Feedback](#7-inspecting-generated-unit-tests--closed-loop-feedback)
8. [Viva Presentation Script: What to Say to Evaluators](#8-viva-presentation-script-what-to-say-to-evaluators)
9. [Troubleshooting & Reset Command](#9-troubleshooting--reset-command)

---

# 1. Overview of SentinelShop

**SentinelShop** is an in-house Python repository developed specifically to test and demonstrate CodeSentinel's dual-engine analysis and test synthesis capabilities. 

Unlike open-source repositories (such as Flask or Requests) where functions are often uniformly refactored or git histories are massive, **SentinelShop** contains realistic e-commerce business logic with controlled, authentic variance across:
- **AST Structural Complexity**: Nested decision trees, exception handlers, currency conversion tables, and boundary conditions.
- **Git Churn Metrics**: 33 historical Git commits spanning from 2025 to late 2026, contributed by 16 distinct team authors with realistic `fix:`, `patch:`, and `refactor:` messages.

### Directory Structure
```
data/demo_repos/sentinel_shop/
├── .git/                            # Complete 33-commit Git history across 16 authors
├── README.md                        # Microservice technical documentation
├── requirements.txt                 # Dependencies (zero third-party libraries required)
└── src/
    ├── orders/
    │   └── checkout.py              # Order processing, coupon stacking & shipping calculations
    ├── payments/
    │   └── gateway.py               # HMAC SHA256 Webhook reconciler & state machine
    └── users/
        └── auth.py                  # Session authorization, username sanitization & API key hashing
```

---

# 2. Function-by-Function Risk Classification

BugLens analyzes the 9 functions in SentinelShop and categorizes them into three distinct defect risk tiers:

| Tier | Function Name | File | Risk % | Cyclomatic Complexity (CC) | Lines of Code (LOC) | Code Churn | Authors | Key Risk Drivers |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HIGH** | `process_complex_checkout` | [`src/orders/checkout.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/orders/checkout.py) | **77.4%** | 52 | 209 | 2008 | 12 | High LOC, heavy historical churn, deeply nested currency & tax branches |
| **HIGH** | `reconcile_transaction_webhook` | [`src/payments/gateway.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/payments/gateway.py) | **68.6%** | 22 | 122 | 1140 | 10 | Payment state transitions, dispute branches, cryptographic verification |
| **MEDIUM** | `calculate_shipping_and_tax` | [`src/orders/checkout.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/orders/checkout.py) | **38.5%** | 11 | 29 | 2008 | 12 | Moderate branching on shipping zones and weight brackets |
| **LOW** | `format_order_confirmation` | [`src/orders/checkout.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/orders/checkout.py) | **16.7%** | 1 | 8 | 2008 | 12 | Linear dictionary formatting |
| **LOW** | `apply_wallet_credit` | [`src/payments/gateway.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/payments/gateway.py) | **8.9%** | 5 | 10 | 1140 | 10 | Simple boundary checks on account balances |
| **LOW** | `validate_session_and_roles` | [`src/users/auth.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/users/auth.py) | **7.6%** | 7 | 25 | 30 | 2 | Clean iteration over role permissions |
| **LOW** | `generate_transaction_receipt` | [`src/payments/gateway.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/payments/gateway.py) | **6.6%** | 1 | 3 | 1140 | 10 | Single-line string interpolation |
| **LOW** | `hash_api_key` | [`src/users/auth.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/users/auth.py) | **0.9%** | 1 | 3 | 30 | 2 | Deterministic single-line SHA256 call |
| **LOW** | `sanitize_username` | [`src/users/auth.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/demo_repos/sentinel_shop/src/users/auth.py) | **0.9%** | 1 | 3 | 30 | 2 | Clean string slicing and lowercasing |

---

# 3. Environment & Cold-Start Setup

Follow these commands to prepare your workspace:

### 3.1 Open PowerShell & Move to Workspace
```powershell
cd "c:\Users\athar\Desktop\Notes\4th Year\BE Project\codesentinel"
```

### 3.2 Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Your terminal will show the `(.venv)` prefix).*

### 3.3 Ensure Local Ollama Daemon is Active
Verify that Ollama has `phi3:mini` ready:
```powershell
ollama list
```
*(Ensure `phi3:mini` is listed with a size of ~2.2 GB).*

### 3.4 (Optional) Freshly Re-seed SentinelShop Anytime
If you ever want to reset SentinelShop back to its pristine state with all 33 Git commits:
```powershell
python scripts/setup_sentinel_shop.py
```
*(Executes in ~2 seconds and re-creates all source files and Git commit graph).*

---

# 4. Terminal Demonstration: 1-Click End-to-End Pipeline

This single command demonstrates the entire CodeSentinel pipeline: from defect mining to AI test generation and dynamic feedback ingestion.

### Command:
```powershell
python scripts/demo_end_to_end.py data/demo_repos/sentinel_shop --max-iterations 1 --num-targets 1
```

### What Happens Across the 4 Automated Steps:
```
======================================================================
  CODESENTINEL — END-TO-END DEMO ON TARGET REPOSITORY
======================================================================
Target Repository : data\demo_repos\sentinel_shop
Mining Scope      : Last 50 commits
Agent Iterations  : Max 1 self-correction loops
======================================================================

[STEP 1/4] Mining Git history & AST complexity with BugLens...
  Git mining complete: 33 commits, 3 Python files
  Scanning 3 Python files... Extracted metrics for 9 functions
  Done: 9 functions — HIGH:2 MEDIUM:1 LOW:6
[*] Analyzed 9 functions in 3.0s:
    - HIGH Risk   (>= 66%): 2 functions
    - MEDIUM Risk (33-66%): 1 functions
    - LOW Risk    (< 33%) : 6 functions

[STEP 2/4] Top-Risk Function: 'process_complex_checkout' in src/orders/checkout.py
    Predicted Bug Probability: 77.4% [HIGH RISK]
    Top Risk Drivers (TreeSHAP Decomposition):
      • Large function (high LOC)     (209 lines) -> increases risk (+0.6510)
      • High total code churn         (2008 lines)-> increases risk (+0.6253)

[STEP 3/4] Triggering AutoTestArmy multi-agent loop for 1 prioritized functions:
    Targets: 'process_complex_checkout'
    Executing: Analyzer Agent -> Generator Agent (Ollama phi3:mini) -> Critic Agent...
[*] Multi-agent synthesis completed in 76.3s!
    Loop Status    : DONE
    Stopping Cause : full_coverage / max_iterations
    Branch Coverage: 100.0%
    Saved Test Suite: data\generated_tests\demo_run\generated_tests.py

[STEP 4/4] Ingesting test execution results into Closed-Loop Feedback...
[*] Recorded 1 dynamic ground-truth labels to data/processed/feedback_labels.csv

======================================================================
  DEMO COMPLETE: REPOSITORY TESTED & TEST CASES GENERATED SUCCESSFULLY!
======================================================================
```

---

# 5. Terminal Demonstration: Step-by-Step Modular Commands

If your evaluators want to inspect each phase individually, use these commands:

### Step 1: Run BugLens Defect Prediction & TreeSHAP Analysis
```powershell
python scripts/demo_buglens.py data/demo_repos/sentinel_shop
```
**What to point out on screen**:
1. The **Analysis Summary**: 2 HIGH risk, 1 MEDIUM risk, 6 LOW risk.
2. The **Priority Queue Table**: Show how the functions are ranked strictly by defect probability.
3. The **TreeSHAP Explanation Box**: Show how TreeSHAP attributes the 77.4% risk of `process_complex_checkout` to its 209 lines of code and 2,008 lines of churn.

---

### Step 2: Inspect the AI-Generated Unit Test Suite
Inspect the actual Python test file synthesized by AutoTestArmy:
```powershell
Get-Content data\generated_tests\demo_run\generated_tests.py -Head 40
```
**What to point out on screen**:
- Explain that the **Analyzer Agent** extracted the function signature (`cart_items`, `coupon_code`, `customer_tier`, `currency`, `user_loyalty_points`).
- The **Generator Agent** (Ollama `phi3:mini`) generated boundary test cases:
  - Empty cart validation (`CheckoutError`).
  - Missing shipping address.
  - Invalid negative price and quantity edge cases.
  - Tier-based discount validation.

---

### Step 3: Run pytest on the Generated Test Suite
Execute the generated tests directly in pytest:
```powershell
pytest data\generated_tests\demo_run\generated_tests.py -v
```

---

### Step 4: Verify the Closed-Loop Feedback Ledger
Inspect the feedback CSV file where test outcomes are dynamically recorded as ground-truth labels:
```powershell
Get-Content data\processed\feedback_labels.csv -Tail 5
```
**What to point out on screen**:
- The CSV stores `function_name`, `predicted_probability`, `actual_outcome` (1 if tests failed/caught defects, 0 if clean), and the execution timestamp.

---

# 6. REST API Demonstration (FastAPI Swagger UI & PowerShell)

Demonstrating the REST API shows evaluators that CodeSentinel is a modular, production-ready backend service capable of integrating into IDEs or CI/CD pipelines.

### 6.1 Start the FastAPI Server
Open a **new terminal tab**, activate `.venv`, and start Uvicorn:
```powershell
cd "c:\Users\athar\Desktop\Notes\4th Year\BE Project\codesentinel"
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to the interactive Swagger UI:  
**`http://127.0.0.1:8000/docs`**

---

### 6.2 Endpoint 1: Run Defect Prediction on SentinelShop
**`POST /buglens/predict`**

#### PowerShell Command:
```powershell
$body = @{
    repo_path = "data/demo_repos/sentinel_shop"
    max_commits = 50
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/buglens/predict" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json -Depth 3
```

#### Swagger UI Request Body:
```json
{
  "repo_path": "data/demo_repos/sentinel_shop",
  "max_commits": 50
}
```

#### Key Elements in Response:
- `run_id`: An integer (e.g., `1`). Note this ID for the next calls.
- `function_count`: `9`.
- `functions`: Ranked array showing `process_complex_checkout` with `risk_category: "HIGH"` and `calculate_shipping_and_tax` with `risk_category: "MEDIUM"`.

---

### 6.3 Endpoint 2: Retrieve the AutoTestArmy Priority Queue
**`GET /buglens/runs/{run_id}/queue?limit=3`**

#### PowerShell Command (replace `1` with your actual `run_id`):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/buglens/runs/1/queue?limit=3" -Method Get | ConvertTo-Json -Depth 3
```

#### Key Elements in Response:
- Returns only the top-3 highest risk candidates:
  1. `process_complex_checkout` (HIGH)
  2. `reconcile_transaction_webhook` (HIGH)
  3. `calculate_shipping_and_tax` (MEDIUM)

---

### 6.4 Endpoint 3: Trigger Multi-Agent Test Generation
**`POST /autotestarmy/generate`**

#### PowerShell Command:
```powershell
$genBody = @{
    prediction_run_id = 1
    max_iterations = 1
    num_targets = 1
} | ConvertTo-Json

$genResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/autotestarmy/generate" -Method Post -Body $genBody -ContentType "application/json"
$genResponse | ConvertTo-Json
```

#### Swagger UI Request Body:
```json
{
  "prediction_run_id": 1,
  "max_iterations": 1,
  "num_targets": 1
}
```

#### Key Elements in Response:
```json
{
  "run_id": 1,
  "prediction_run_id": 1,
  "status": "PENDING",
  "targets": ["process_complex_checkout"],
  "branch_coverage": 0.0,
  "iterations": 0
}
```

---

### 6.5 Endpoint 4: Poll Generation Status
**`GET /autotestarmy/runs/{run_id}`**

#### PowerShell Command:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/autotestarmy/runs/1" -Method Get | ConvertTo-Json
```
- Status transitions from `"PENDING"` $\to$ `"RUNNING"` $\to$ `"DONE"`.
- When `"DONE"`, observe `branch_coverage: 100.0` and the path to the saved test suite file.

---

### 6.6 Endpoint 5: Check Closed-Loop Feedback Stats
**`GET /feedback/stats`**

#### PowerShell Command:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/feedback/stats" -Method Get | ConvertTo-Json
```
- Returns total dynamic labels recorded, defect distribution, and retraining readiness.

---

# 7. Inspecting Generated Unit Tests & Closed-Loop Feedback

### 7.1 Location of Generated Artifacts
Every demo run persists its artifacts in standard paths:
- **Generated Test Suite**: [`data/generated_tests/demo_run/generated_tests.py`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/generated_tests/demo_run/generated_tests.py)
- **Closed-Loop Feedback Ledger**: [`data/processed/feedback_labels.csv`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/data/processed/feedback_labels.csv)
- **Inference Run Cache**: SQLite database [`codesentinel.db`](file:///c:/Users/athar/Desktop/Notes/4th%20Year/BE%20Project/codesentinel/codesentinel.db)

### 7.2 Why Branch Coverage Matters
AutoTestArmy evaluates the generated tests against pytest with `pytest-cov`. If a generated test suite only covers 20% of branches, the **Critic Agent** reads the uncovered line numbers from the coverage report and asks the **Generator Agent** to write tests specifically covering those missing branches (e.g., restricted hazardous item branches, coupon expiration exceptions).

---

# 8. Viva Presentation Script: What to Say to Evaluators

Use this recommended explanation flow when demonstrating SentinelShop:

### Part 1: Introducing the Demonstration
> *"Respected teachers, to demonstrate CodeSentinel on a controlled microservice, we developed **SentinelShop**—a realistic e-commerce order settlement service. It contains 9 functions across 3 files with 33 historical Git commits contributed by 16 developers."*

### Part 2: Demonstrating Defect Prioritization (BugLens)
> *"When we run `demo_buglens.py`, BugLens uses PyDriller to mine the repository commit graph and Radon to parse the Python AST. Notice the output on screen:*
> 1. *`process_complex_checkout` is flagged as **HIGH Risk (77.4%)** because it has 52 decision branches, 209 lines of code, and over 2,000 lines of historical churn with 12 contributors.*
> 2. *`calculate_shipping_and_tax` falls naturally into **MEDIUM Risk (38.5%)** due to moderate branching across shipping zones.*
> 3. *Utility functions like `hash_api_key` and `sanitize_username` are correctly classified as **LOW Risk (< 1%)**.*
> 
> *Our TreeSHAP root-cause breakdown mathematically proves to the developer exactly why each function received its score—eliminating black-box uncertainty."*

### Part 3: Demonstrating Multi-Agent Test Synthesis (AutoTestArmy)
> *"Next, AutoTestArmy automatically takes the top candidate from the priority queue. Rather than calling expensive cloud APIs, it executes a local multi-agent LangGraph workflow running **Ollama phi3:mini**:*
> - *The **Analyzer Agent** extracts the AST structure, docstring, and argument types.*
> - *The **Generator Agent** creates executable pytest unit tests covering edge cases and exception branches.*
> - *The **Critic Agent** runs pytest inside a sandbox, measures branch coverage, and terminates once target coverage is achieved."*

### Part 4: Demonstrating Closed-Loop Continuous Learning
> *"Finally, the test execution outcome is ingested into our feedback aggregator. Each run dynamically generates a ground-truth label in `feedback_labels.csv`. When 50 dynamic labels accumulate, CodeSentinel automatically triggers an in-fold retrain of our LightGBM model—making defect predictions smarter over time."*

---

# 9. Troubleshooting & Reset Command

### Reset SentinelShop to Initial State:
If any files inside `data/demo_repos/sentinel_shop` are accidentally modified or deleted, run:
```powershell
python scripts/setup_sentinel_shop.py
```
*(Wipes and regenerates all source files and the 33 Git commits in ~2 seconds).*

### Run the 30-Test Automated Regression Suite:
To verify that all project components (BugLens, AutoTestArmy, FastAPI routes, and Feedback Aggregator) are functioning with 100% test pass rate:
```powershell
pytest tests/ -v
```
*(All 30 tests will execute and pass cleanly).*
