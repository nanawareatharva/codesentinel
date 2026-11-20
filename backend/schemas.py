"""
CodeSentinel — Pydantic Validation & Serialization Schemas
==========================================================
Defines request and response schemas for BugLens defect prediction,
SHAP explainability, historical runs, and test generation orchestration.
"""
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class PredictRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to a local Git repository")
    max_commits: Optional[int] = Field(
        default=500,
        description="Maximum commits to traverse for Git mining (None or 0 for complete history)"
    )


class SHAPDriver(BaseModel):
    feature: str
    label: str
    raw_value: float
    shap_value: float
    direction: str


class SHAPExplanation(BaseModel):
    top_risk_drivers: list[SHAPDriver] = []
    summary: str
    base_value: Optional[float] = None
    prediction: Optional[float] = None


class FunctionRisk(BaseModel):
    filepath: str
    function_name: str
    line_start: int
    line_end: int
    bug_probability: float
    risk_category: str
    cyclomatic_complexity: Optional[float] = None
    loc: Optional[float] = None
    maintainability_index: Optional[float] = None
    churn: Optional[float] = None
    unique_developers: Optional[float] = None
    bug_fix_commits: Optional[float] = None
    file_age_days: Optional[float] = None
    recent_churn: Optional[float] = None
    shap_explanation: Optional[Any] = None

    model_config = ConfigDict(from_attributes=True)


class PredictResponse(BaseModel):
    run_id: int
    repo_path: str
    created_at: Optional[datetime] = None
    function_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    functions: list[FunctionRisk]

    model_config = ConfigDict(from_attributes=True)


class RunSummary(BaseModel):
    id: int
    repo_path: str
    created_at: datetime
    function_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    status: str

    model_config = ConfigDict(from_attributes=True)


class GenerateTestsRequest(BaseModel):
    prediction_run_id: int = Field(..., description="ID of a previous successful prediction run")
    max_iterations: int = Field(default=5, ge=1, le=10, description="Maximum self-correction loops for agents")


class GenerateTestsResponse(BaseModel):
    run_id: int
    status: str
    message: str


class TestRunStatusResponse(BaseModel):
    run_id: int = Field(validation_alias=AliasChoices("run_id", "id"))
    prediction_run_id: int
    status: str
    iterations: int
    final_coverage: Optional[float] = None
    test_file_path: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: bool


class FeedbackTarget(BaseModel):
    filepath: str
    function_name: str
    cyclomatic_complexity: Optional[float] = 0.0
    loc: Optional[float] = 0.0
    maintainability_index: Optional[float] = 100.0
    churn: Optional[float] = 0.0
    unique_developers: Optional[float] = 1.0
    bug_fix_commits: Optional[float] = 0.0
    file_age_days: Optional[float] = 1.0
    recent_churn: Optional[float] = 0.0


class FeedbackTestResults(BaseModel):
    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage_percent: float = 0.0


class RecordFeedbackRequest(BaseModel):
    target_functions: list[FeedbackTarget]
    test_results: FeedbackTestResults
    run_id: Optional[int] = None


class RecordFeedbackResponse(BaseModel):
    samples_recorded: int
    message: str


class RetrainRequest(BaseModel):
    min_samples: int = Field(default=1, ge=1)


class RetrainResponse(BaseModel):
    retrained: bool
    message: str
    feedback_samples: int
    cv_auc_mean: Optional[float] = None
    cv_auc_std: Optional[float] = None
    timestamp: str


class FeedbackStatsResponse(BaseModel):
    total_samples: int
    defective_count: int
    clean_count: int
    defect_ratio: float
    feedback_file: str
    ready_for_retraining: bool

