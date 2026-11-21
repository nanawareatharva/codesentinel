# CodeSentinel: Semester VII Project Log Book (Weeks 1 to 14)

> **Department of Artificial Intelligence & Data Science**  
> **Fr. Conceicao Rodrigues College of Engineering (FR. CRCE), Bandra (W), Mumbai**  
> **Affiliated with the University of Mumbai**  
> **Academic Year:** 2026–2027 | **Semester:** VII  
> **Project Title:** CodeSentinel: AI-Powered Defect Prediction & Targeted Test Generation  
> **Group Number:** Group 10  
>
> **Team Members:**
> 1. **Team Member 1:** Atharva Nanaware (Roll No. 10511)
> 2. **Team Member 2:** Suhera Siddiqui (Roll No. 10381)
> 3. **Team Member 3:** Aum Singh (Roll No. 10383)
> 4. **Team Member 4:** N/A (3-Member Project Group)

---

# Part 1: Semester VII — Action Plan (Summary Table)

*Fill this into the two-page Action Plan table at the beginning of your Log Book (matching Images 1 & 2).*

| Week | From – to (dates) | Planned Activity |
| :--- | :--- | :--- |
| **Week 1** | `18/07/2026 – 25/07/2026` | Finalised project topic and problem statement. Started literature survey. |
| **Week 2** | `25/07/2026 – 01/08/2026` | Studied research papers and existing tools; identified gaps. Finalized tech stack. |
| **Week 3** | `01/08/2026 – 08/08/2026` | Prepared and presented first review: problem, limitations, methodology, research papers, tech stack and novelty. |
| **Week 4** | `08/08/2026 – 15/08/2026` | Phase 0: environment setup and project structure, Phase 1: collected datasets. |
| **Week 5** | `15/08/2026 – 22/08/2026` | Built git miner and code metric extractor. Trained LightGBM model with SMOTE and 5-fold validation, added SHAP. |
| **Week 6** | `22/08/2026 – 29/08/2026` | Audited and fixed bugs, final AUC 0.769, Built live demo on flask repo. Prepared ISE 2 presentation. |
| **Week 7** | `29/08/2026 – 05/09/2026` | Worked on panel feedback from ISE 2. Started Phase 3: FastAPI backend & SQLite database layer. |
| **Week 8** | `05/09/2026 – 12/09/2026` | Implemented FastAPI service layer, Pydantic schemas, database models, and BugLens REST endpoints. |
| **Week 9** | `12/09/2026 – 19/09/2026` | Built Priority Queue Bridge endpoint (`/buglens/runs/{id}/queue`) and wrote automated API tests. |
| **Week 10** | `19/09/2026 – 26/09/2026` | Phase 4: Local Ollama runtime setup (`phi3:mini`), fixed 51GB context memory crash, designed LangGraph AgentState. |
| **Week 11** | `26/09/2026 – 03/10/2026` | Phase 5: Built AutoTestArmy multi-agent pipeline (Analyzer AST, Generator phi3:mini, Critic test runner). |
| **Week 12** | `03/10/2026 – 10/10/2026` | Phase 6: Built Closed-Loop dynamic feedback aggregator and on-demand LightGBM model retraining with SMOTE. |
| **Week 13** | `10/10/2026 – 17/10/2026` | Executed 30-test regression suite, tested live on 3 repositories (Flask, Requests, Movie Rec), built audit script. |
| **Week 14** | `17/10/2026 – 24/10/2026` | Compiled master documentation, live demonstration runbook, pushed to GitHub, prepared for final evaluation. |

---

# Part 2: Weekly Log Sheets (Week 1 to Week 14)

*Fill each of these sections into the corresponding weekly log sheet in your book (matching Images 3 & 4).*

---

### Week 1
* **Dates : From** `18/07/2026` **To** `25/07/2026`
* **Progress Achieved :**  
  Finalized the project domain as AI-driven software engineering and intelligent quality assurance. Formulated the project problem statement: bridging the gap between machine learning defect prediction and generative unit test synthesis. Commenced literature survey on defect prediction benchmarks (NASA MDP, PROMISE repository) and LLM code generation tools.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Literature survey on LLM-based test generation tools (ChatTester, TestPilot) and identified lack of defect targeting.
  * **Team Member 2 (Suhera Siddiqui):** Literature survey on classical software defect prediction using machine learning (Decision Trees, Random Forest, SVM).
  * **Team Member 3 (Aum Singh):** Researched backend architecture, Git repository mining approaches, and drafted initial problem statement.
  * **Team Member 4:** N/A (3-member project group)
* **Guide's Suggestions :**  
  Narrow down the research scope. Focus on combining defect prediction with autonomous test generation rather than building a standalone prediction tool. Ensure the system relies on local open-source LLMs to eliminate cloud API costs.

---

### Week 2
* **Dates : From** `25/07/2026` **To** `01/08/2026`
* **Progress Achieved :**  
  Reviewed 15+ IEEE and ACM research papers on defect prediction and automated testing. Identified the critical research gap: existing LLM test generators generate unit tests blindly for entire codebases without knowing where bugs hide. Finalized the technology stack: Python 3.12, PyDriller, Radon, LightGBM, TreeSHAP, FastAPI, SQLite, LangGraph, and Ollama (`phi3:mini`).
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Investigated LangGraph state machine framework and Ollama local model inference feasibility on consumer hardware.
  * **Team Member 2 (Suhera Siddiqui):** Evaluated ML algorithms for tabular defect data; selected LightGBM due to fast training, low memory, and TreeSHAP compatibility.
  * **Team Member 3 (Aum Singh):** Finalized backend architecture (FastAPI + SQLite) and static code extraction pipeline (PyDriller + Radon).
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Prepare formal slides for Review 1 (ISE 1). Clearly articulate the mathematical justification for selecting LightGBM and how TreeSHAP will provide transparent explainability for developers.

---

### Week 3
* **Dates : From** `01/08/2026` **To** `08/08/2026`
* **Progress Achieved :**  
  Prepared and presented the Review 1 (ISE 1) presentation before the departmental evaluation panel. Presented the problem statement, literature survey findings, identified limitations, proposed two-stage hybrid architecture (BugLens + AutoTestArmy), standardized 8-feature schema, and project milestones. Received formal panel approval to proceed.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Presented problem motivation, existing LLM testing inefficiencies, and the end-to-end multi-agent execution pipeline.
  * **Team Member 2 (Suhera Siddiqui):** Presented defect prediction methodology, class imbalance resolution via SMOTE, and TreeSHAP mathematical foundations.
  * **Team Member 3 (Aum Singh):** Presented system architectural diagram, REST API microservice design, and project implementation schedule.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Incorporate panel feedback regarding cross-validation rigor. Ensure that SMOTE oversampling is applied strictly within cross-validation folds to eliminate data leakage.

---

### Week 4
* **Dates : From** `08/08/2026` **To** `15/08/2026`
* **Progress Achieved :**  
  Completed Phase 0 (Environment Setup) and initiated Phase 1 (Dataset Engineering). Initialized Python 3.12 virtual environment, project directory hierarchy, and Git version control. Downloaded and standardized 13 NASA MDP & PROMISE datasets (CM1, JM1, KC1, MC1, PC1, etc.). Created a clean, deduplicated baseline training dataset (`features.csv`) with 9,157 rows.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Configured repository structure, `.gitignore`, dependency configuration, and virtual environment scripts.
  * **Team Member 2 (Suhera Siddiqui):** Cleaned, standardized, and merged NASA MDP datasets; mapped legacy features into unified schema.
  * **Team Member 3 (Aum Singh):** Developed automated dataset validation scripts to verify feature distributions and check for duplicate samples.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Maintain a clear separation between code complexity metrics and process churn metrics to ensure model generalizability across diverse codebases.

---

### Week 5
* **Dates : From** `15/08/2026` **To** `22/08/2026`
* **Progress Achieved :**  
  Completed Phase 1 and Phase 2. Implemented PyDriller Git commit history miner (`git_miner.py`) and Radon AST complexity extractor (`metric_extractor.py`). Defined the standardized 8-feature vector schema. Implemented 5-fold Stratified Cross-Validation with in-fold SMOTE for LightGBM training. Integrated TreeSHAP explainer for computing additive feature attributions.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Built Radon AST metric extractor for McCabe Cyclomatic Complexity, LOC, and Maintainability Index.
  * **Team Member 2 (Suhera Siddiqui):** Developed LightGBM training pipeline with in-fold SMOTE and StandardScaler; generated `lightgbm_model.pkl`.
  * **Team Member 3 (Aum Singh):** Built PyDriller Git mining pipeline to extract historical churn, unique authors, bug-fix commits, and file age.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Verify that the trained LightGBM model outperforms baseline benchmark metrics (target ROC-AUC > 0.75). Test feature attribution outputs on real functions.

---

### Week 6
* **Dates : From** `22/08/2026` **To** `29/08/2026`
* **Progress Achieved :**  
  Audited model performance: achieved an Out-of-Fold (OOF) ROC-AUC of 0.7694, exceeding the IEEE benchmark target. Fixed Git shallow clone traversal errors in PyDriller. Tested live BugLens scanning on the Pallets Flask repository (410 functions scanned in 13 seconds). Built `demo_buglens.py` and prepared slides for ISE 2 review.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Created `demo_buglens.py` live terminal demonstration script; tested scanning on open-source repositories.
  * **Team Member 2 (Suhera Siddiqui):** Computed ROC curves, confusion matrices, and verified TreeSHAP waterfall/summary plots for top-risk functions.
  * **Team Member 3 (Aum Singh):** Handled edge cases in Git mining (detached HEAD, shallow clones); authored ISE 2 presentation deck.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Well done on achieving 0.7694 AUC. For ISE 2, emphasize how TreeSHAP makes the model explainable. Next, transition to Phase 3 by building the FastAPI backend.

---

### Week 7
* **Dates : From** `29/08/2026` **To** `05/09/2026`
* **Progress Achieved :**  
  Presented the ISE 2 review to the faculty panel. Conducted a live terminal demonstration of BugLens scanning Flask and displaying TreeSHAP explanations. Received positive evaluation and permission to proceed with Phase 3 (FastAPI Backend) and Phase 4 (AutoTestArmy). Initiated SQLite database schema design.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Delivered the live system demonstration on the Flask repository during the ISE 2 presentation.
  * **Team Member 2 (Suhera Siddiqui):** Addressed examiner questions on class imbalance, in-fold SMOTE mathematical rigor, and SHAP Shapley values.
  * **Team Member 3 (Aum Singh):** Designed SQLite database schema (`PredictionRun`, `FunctionPrediction`, `TestGenerationRun`) in SQLAlchemy.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Focus on designing the Priority Queue endpoint to cleanly bridge BugLens prediction outputs with the upcoming AutoTestArmy multi-agent generator.

---

### Week 8
* **Dates : From** `05/09/2026` **To** `12/09/2026`
* **Progress Achieved :**  
  Implemented Phase 3 (FastAPI Service Layer & SQLite Persistence). Built `backend/database.py` with SQLAlchemy engine and session dependency. Created Pydantic v2 schemas (`backend/schemas.py`) for request/response serialization. Implemented BugLens REST endpoints: `GET /health`, `GET /`, `POST /buglens/predict`, and `GET /buglens/runs`.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Built FastAPI route handlers in `buglens/api/routes.py` and connected live `predict_repo()` inference.
  * **Team Member 2 (Suhera Siddiqui):** Implemented Pydantic data validation schemas for function risks, metrics, and SHAP drivers.
  * **Team Member 3 (Aum Singh):** Configured SQLite database tables, connection pooling with `check_same_thread=False`, and CORS middleware.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Ensure Swagger UI (`/docs`) provides interactive documentation and clear error responses for invalid repository paths.

---

### Week 9
* **Dates : From** `12/09/2026` **To** `19/09/2026`
* **Progress Achieved :**  
  Built `GET /buglens/runs/{id}/queue`—the Priority Queue Bridge isolating HIGH and MEDIUM risk functions. Implemented `GET /buglens/runs/{id}/functions/{name}` for function-level SHAP explanations. Developed 7 automated API tests in `tests/test_api.py` verifying database persistence, queue extraction, and 400 error handling.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Implemented Priority Queue SQL queries with fallback handling for low-risk codebases.
  * **Team Member 2 (Suhera Siddiqui):** Verified SHAP explanation endpoint serialization and tested pagination on historical runs.
  * **Team Member 3 (Aum Singh):** Authored `tests/test_api.py` regression suite using FastAPI `TestClient`; achieved 100% test pass rate.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  The Priority Queue bridge is a key contribution. Begin integrating local Ollama LLM execution and benchmark token generation speed.

---

### Week 10
* **Dates : From** `19/09/2026` **To** `26/09/2026`
* **Progress Achieved :**  
  Initiated Phase 4 (Ollama Runtime & LangGraph State Schema). Integrated local Ollama daemon running `phi3:mini` (3.8B parameters). Diagnosed and fixed the 51 GB context memory bug by enforcing `num_ctx: 2048` in `autotestarmy/tools/llm.py`. Designed centralized LangGraph `AgentState` TypedDict memory schema in `autotestarmy/graph/state.py`.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Diagnosed the Ollama out-of-memory crash; implemented strict context window and timeout guards in `llm.py`.
  * **Team Member 2 (Suhera Siddiqui):** Designed the multi-agent shared state structure (`AgentState`) tracking inputs, analysis, and test outcomes.
  * **Team Member 3 (Aum Singh):** Implemented Markdown code fence stripper (`clean_code_fence()`) to ensure generated test code executes cleanly.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Ensure the Analyzer agent deterministic AST parsing extract concrete boundary values so that the LLM does not hallucinate unrealistic inputs.

---

### Week 11
* **Dates : From** `26/09/2026` **To** `03/10/2026`
* **Progress Achieved :**  
  Completed Phase 5 (AutoTestArmy Multi-Agent Loop). Developed the 3 collaborating agents: Analyzer Agent (AST parameters, branches, boundary payloads), Generator Agent (phi3:mini prompt with failure injection), and Critic Agent (subprocess `pytest --cov` runner). Compiled LangGraph `StateGraph` pipeline (`autotestarmy/graph/pipeline.py`).
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Implemented Generator Agent prompting and Critic convergence rules ($\ge 80\%$ coverage, zero failures, max iterations).
  * **Team Member 2 (Suhera Siddiqui):** Implemented Analyzer Agent AST visitor extracting boundary inputs (`None`, `0`, `-1`, `""`, `[]`).
  * **Team Member 3 (Aum Singh):** Implemented isolated subprocess pytest execution in `pytest_runner.py` with XML coverage extraction.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Ensure the multi-agent generator runs asynchronously using background tasks so that the HTTP API remains responsive during test generation.

---

### Week 12
* **Dates : From** `03/10/2026` **To** `10/10/2026`
* **Progress Achieved :**  
  Connected `POST /autotestarmy/generate` with FastAPI `BackgroundTasks`. Completed Phase 6 (Closed-Loop Feedback & Retraining). Built feedback aggregator (`feedback/aggregator.py`): dynamically converts test failures to $y=1$ and high-coverage passes to $y=0$. Built model updater (`feedback/model_updater.py`) for live LightGBM retraining with feedback data.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Integrated background task execution with SQLite status updates (`PENDING` $\to$ `RUNNING` $\to$ `DONE`).
  * **Team Member 2 (Suhera Siddiqui):** Formulated heuristic labeling rules and implemented LightGBM retraining pipeline on augmented data.
  * **Team Member 3 (Aum Singh):** Implemented feedback REST endpoints (`/feedback/record`, `/feedback/stats`, `/feedback/retrain`).
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Run end-to-end integration tests across the complete feedback loop to confirm that model weights update properly on feedback data.

---

### Week 13
* **Dates : From** `10/10/2026` **To** `17/10/2026`
* **Progress Achieved :**  
  Conducted full-system testing. Automated regression suite achieved a 100% pass rate (30/30 tests in `pytest tests/ -v`). Developed full-system audit script `scripts/verify_all_phases.py` and end-to-end repository demo script `scripts/demo_end_to_end.py`. Verified multi-agent test synthesis on 3 repositories: Pallets Flask, PSF Requests, and Atharva's Movie Recommendation System.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Built `scripts/demo_end_to_end.py`; tested defect prediction and test generation on Movie Recommendation System.
  * **Team Member 2 (Suhera Siddiqui):** Evaluated branch code coverage and SHAP feature drivers across all 3 test repositories.
  * **Team Member 3 (Aum Singh):** Built `scripts/verify_all_phases.py` automated 6-phase health audit script and verified SQLite database integrity.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Prepare exhaustive documentation covering every single command, API walkthrough, and presentation script for the final evaluation.

---

### Week 14
* **Dates : From** `17/10/2026` **To** `24/10/2026`
* **Progress Achieved :**  
  Authored comprehensive documentation: Master Architecture Specification (466 lines), Live Demonstration Runbook (880 lines), and Concepts Explainer. Synchronized entire codebase, tests, and documentation to GitHub repository (`main` branch). Completed Log Book entries, finalized presentation slides, and conducted final dry run of live demonstration.
* **Teams Member's Contribution :**
  * **Team Member 1 (Atharva Nanaware):** Authored Master Demonstration Runbook (`CODESENTINEL_COMPLETE_DEMO_AND_RUNBOOK.md`) and pushed to GitHub.
  * **Team Member 2 (Suhera Siddiqui):** Authored Concepts & Theory Explainer guide; finalized ML and TreeSHAP presentation talking points.
  * **Team Member 3 (Aum Singh):** Structured presentation slide deck, verified Swagger UI endpoints, and coordinated team speaking flow.
  * **Team Member 4:** N/A
* **Guide's Suggestions :**  
  Ensure all team members are confident presenting their specific modules during the viva. Rehearse the live 3-repository demo to ensure seamless execution.
