"""
CodeSentinel — Closed-Loop Feedback REST Endpoints
===================================================
Endpoints for ingesting test feedback outcomes, monitoring accumulated samples,
and triggering LightGBM model retraining with real-world feedback data.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend import schemas
from feedback.aggregator import record_outcomes, get_feedback_stats
from feedback.model_updater import retrain_with_feedback

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/record", response_model=schemas.RecordFeedbackResponse)
def record_test_feedback(req: schemas.RecordFeedbackRequest):
    """
    Ingest unit test execution results for a set of target functions
    and record heuristic ground-truth defect labels.
    """
    target_dicts = [t.model_dump() for t in req.target_functions]
    results_dict = req.test_results.model_dump()

    count = record_outcomes(
        target_functions=target_dicts,
        test_results=results_dict,
        db_run_id=req.run_id,
    )

    return schemas.RecordFeedbackResponse(
        samples_recorded=count,
        message=f"Successfully processed {len(target_dicts)} functions; {count} conclusive feedback samples recorded.",
    )


@router.get("/stats", response_model=schemas.FeedbackStatsResponse)
def get_feedback_statistics():
    """Retrieve sample counts, defect distribution, and retraining readiness."""
    stats = get_feedback_stats()
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])

    return schemas.FeedbackStatsResponse(
        total_samples=stats["total_samples"],
        defective_count=stats["defective_count"],
        clean_count=stats["clean_count"],
        defect_ratio=stats["defect_ratio"],
        feedback_file=stats["feedback_file"],
        ready_for_retraining=stats["ready_for_retraining"],
    )


@router.post("/retrain", response_model=schemas.RetrainResponse)
def trigger_retraining(req: schemas.RetrainRequest = schemas.RetrainRequest()):
    """
    Triggers LightGBM retraining with augmented feedback data.
    Refreshes saved production model, scaler, and SHAP explainer artifacts.
    """
    try:
        results = retrain_with_feedback(min_feedback_samples=req.min_samples)
        return schemas.RetrainResponse(
            retrained=results["retrained"],
            message=results["message"],
            feedback_samples=results["feedback_samples"],
            cv_auc_mean=results.get("cv_auc_mean"),
            cv_auc_std=results.get("cv_auc_std"),
            timestamp=results["timestamp"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model retraining failed: {exc}"
        )
