# CodeSentinel

**AI-Powered Code Quality Platform: Defect Prediction and Targeted Test Generation**

CodeSentinel is the B.E. Final Year Project (2026–27) of Group 10 at FR. Conceicao Rodrigues College of Engineering, Mumbai, Department of AI & Data Science.

Current implementation covers BugLens Phases 0–2: dataset collection, feature engineering, Git/Radon repository mining, LightGBM defect prediction, and SHAP explanations. AutoTestArmy, the backend, and the VS Code extension are reserved for later phases.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts/download_datasets.py
python scripts/build_feature_matrix.py
python buglens/model/train.py
pytest -v
```

To analyse a repository after training:

```python
from buglens.model.predict import predict_repo
from buglens.model.shap_explainer import explain_function

results = predict_repo("path/to/repository")
if results:
    print(results[0])
    print(explain_function(results[0]))
```

The eight model features are defined only in `buglens/config.py`; all pipeline components import that schema.
