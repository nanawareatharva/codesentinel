"""
CodeSentinel — Database ORM Models
==================================
Defines database schemas for repository prediction runs, function-level
defect probabilities with metrics and SHAP explanations, and test generation runs.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id = Column(Integer, primary_key=True, index=True)
    repo_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    function_count = Column(Integer, default=0)
    high_risk_count = Column(Integer, default=0)
    medium_risk_count = Column(Integer, default=0)
    low_risk_count = Column(Integer, default=0)
    status = Column(String, default="COMPLETED")  # COMPLETED | FAILED

    predictions = relationship("FunctionPrediction", back_populates="run", cascade="all, delete-orphan")
    test_runs = relationship("TestGenerationRun", back_populates="prediction_run", cascade="all, delete-orphan")


class FunctionPrediction(Base):
    __tablename__ = "function_predictions"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("prediction_runs.id"), nullable=False, index=True)
    filepath = Column(String, nullable=False)
    function_name = Column(String, nullable=False, index=True)
    line_start = Column(Integer, nullable=False)
    line_end = Column(Integer, nullable=False)
    bug_probability = Column(Float, nullable=False)
    risk_category = Column(String, nullable=False, index=True)  # HIGH | MEDIUM | LOW

    # Static and evolutionary metrics snapshot
    cyclomatic_complexity = Column(Float, nullable=True)
    loc = Column(Float, nullable=True)
    maintainability_index = Column(Float, nullable=True)
    churn = Column(Float, nullable=True)
    unique_developers = Column(Float, nullable=True)
    bug_fix_commits = Column(Float, nullable=True)
    file_age_days = Column(Float, nullable=True)
    recent_churn = Column(Float, nullable=True)

    # Precomputed SHAP attribution payload
    shap_explanation = Column(JSON, nullable=True)

    run = relationship("PredictionRun", back_populates="predictions")


class TestGenerationRun(Base):
    __tablename__ = "test_generation_runs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_run_id = Column(Integer, ForeignKey("prediction_runs.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String, default="PENDING")  # PENDING | RUNNING | DONE | FAILED
    iterations = Column(Integer, default=0)
    final_coverage = Column(Float, nullable=True)
    test_file_path = Column(String, nullable=True)

    prediction_run = relationship("PredictionRun", back_populates="test_runs")
