"""
CodeSentinel — BugLens API Endpoints
====================================
Exposes REST endpoints for live repository defect prediction,
historical prediction runs, priority queue retrieval, and SHAP explainability.
"""
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from buglens.model.predict import predict_repo
from buglens.model.shap_explainer import explain_function

router = APIRouter(prefix="/buglens", tags=["BugLens"])


@router.post("/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest, db: Session = Depends(get_db)):
    """
    Run BugLens defect risk prediction and SHAP explainability on a local repository.
    Persists the run and all predicted functions in SQLite.
    """
    repo = Path(req.repo_path).resolve()
    if not repo.exists() or not repo.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Target directory does not exist or is not a directory: {req.repo_path}"
        )

    try:
        results = predict_repo(str(repo), max_commits=req.max_commits)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing BugLens analysis: {str(e)}"
        )

    if not results:
        raise HTTPException(
            status_code=400,
            detail=f"No Python functions found or analyzed in repository: {req.repo_path}"
        )

    # Compute SHAP root-cause explanations for all functions
    for r in results:
        try:
            r["shap_explanation"] = explain_function(r)
        except Exception:
            r["shap_explanation"] = None

    # Count risk categories
    high_risk = [r for r in results if r["risk_category"] == "HIGH"]
    med_risk = [r for r in results if r["risk_category"] == "MEDIUM"]
    low_risk = [r for r in results if r["risk_category"] == "LOW"]

    # Persist PredictionRun record
    run = models.PredictionRun(
        repo_path=str(repo),
        function_count=len(results),
        high_risk_count=len(high_risk),
        medium_risk_count=len(med_risk),
        low_risk_count=len(low_risk),
        status="COMPLETED",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Persist all individual function predictions
    for r in results:
        fp = models.FunctionPrediction(
            run_id=run.id,
            filepath=r["filepath"],
            function_name=r["function_name"],
            line_start=r["line_start"],
            line_end=r["line_end"],
            bug_probability=r["bug_probability"],
            risk_category=r["risk_category"],
            cyclomatic_complexity=r.get("cyclomatic_complexity"),
            loc=r.get("loc"),
            maintainability_index=r.get("maintainability_index"),
            churn=r.get("churn"),
            unique_developers=r.get("unique_developers"),
            bug_fix_commits=r.get("bug_fix_commits"),
            file_age_days=r.get("file_age_days"),
            recent_churn=r.get("recent_churn"),
            shap_explanation=r.get("shap_explanation"),
        )
        db.add(fp)
    db.commit()

    return schemas.PredictResponse(
        run_id=run.id,
        repo_path=run.repo_path,
        created_at=run.created_at,
        function_count=run.function_count,
        high_risk_count=run.high_risk_count,
        medium_risk_count=run.medium_risk_count,
        low_risk_count=run.low_risk_count,
        functions=[schemas.FunctionRisk.model_validate(r) for r in results],
    )


@router.get("/runs", response_model=List[schemas.RunSummary])
def get_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """Retrieve historical prediction runs sorted by creation date descending."""
    runs = (
        db.query(models.PredictionRun)
        .order_by(models.PredictionRun.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [schemas.RunSummary.model_validate(r) for r in runs]


@router.get("/runs/{run_id}", response_model=schemas.PredictResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Retrieve full results and functions for a specific historical run."""
    run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Prediction run {run_id} not found.")

    functions = (
        db.query(models.FunctionPrediction)
        .filter(models.FunctionPrediction.run_id == run_id)
        .order_by(models.FunctionPrediction.bug_probability.desc())
        .all()
    )

    return schemas.PredictResponse(
        run_id=run.id,
        repo_path=run.repo_path,
        created_at=run.created_at,
        function_count=run.function_count,
        high_risk_count=run.high_risk_count,
        medium_risk_count=run.medium_risk_count,
        low_risk_count=run.low_risk_count,
        functions=[schemas.FunctionRisk.model_validate(f) for f in functions],
    )


@router.get("/runs/{run_id}/queue", response_model=List[schemas.FunctionRisk])
def get_priority_queue(
    run_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Retrieve the prioritized high-risk functions queue for a run.
    Used by AutoTestArmy to direct test generation effort toward vulnerable code.
    """
    run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Prediction run {run_id} not found.")

    # High & Medium risk functions first, ordered by probability descending
    queue = (
        db.query(models.FunctionPrediction)
        .filter(
            models.FunctionPrediction.run_id == run_id,
            models.FunctionPrediction.risk_category.in_(["HIGH", "MEDIUM"])
        )
        .order_by(models.FunctionPrediction.bug_probability.desc())
        .limit(limit)
        .all()
    )

    # Fallback: if no HIGH/MEDIUM functions exist, take the top functions overall
    if not queue:
        queue = (
            db.query(models.FunctionPrediction)
            .filter(models.FunctionPrediction.run_id == run_id)
            .order_by(models.FunctionPrediction.bug_probability.desc())
            .limit(limit)
            .all()
        )

    return [schemas.FunctionRisk.model_validate(f) for f in queue]


@router.get("/runs/{run_id}/functions/{function_name}", response_model=List[schemas.FunctionRisk])
def get_function_detail(run_id: int, function_name: str, db: Session = Depends(get_db)):
    """Retrieve detailed predictions and SHAP explanations for functions matching a name."""
    funcs = (
        db.query(models.FunctionPrediction)
        .filter(
            models.FunctionPrediction.run_id == run_id,
            models.FunctionPrediction.function_name == function_name
        )
        .all()
    )
    if not funcs:
        raise HTTPException(
            status_code=404,
            detail=f"Function '{function_name}' not found in run {run_id}."
        )
    return [schemas.FunctionRisk.model_validate(f) for f in funcs]
