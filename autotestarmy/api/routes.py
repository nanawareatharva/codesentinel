"""
CodeSentinel — AutoTestArmy API Endpoints
=========================================
Endpoints for triggering and monitoring multi-agent test generation.
Integrates BugLens priority queues with the LangGraph self-healing pipeline.
"""
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from backend.database import get_db, SessionLocal
from backend import models, schemas
from autotestarmy.graph.pipeline import run_pipeline
from feedback.aggregator import record_outcomes

router = APIRouter(prefix="/autotestarmy", tags=["AutoTestArmy"])


def _execute_agent_generation(
    gen_run_id: int,
    repo_path: str,
    target_functions: list[dict[str, Any]],
    max_iterations: int,
) -> None:
    """Background worker executing the LangGraph multi-agent test generation loop."""
    db = SessionLocal()
    try:
        run = db.query(models.TestGenerationRun).filter(models.TestGenerationRun.id == gen_run_id).first()
        if not run:
            return

        run.status = "RUNNING"
        db.commit()

        # Target output directory for saved test artifacts
        out_dir = Path("data") / "generated_tests" / f"run_{gen_run_id}"
        final_state = run_pipeline(
            repo_path=repo_path,
            target_functions=target_functions,
            max_iterations=max_iterations,
            save_output_dir=str(out_dir),
        )

        run.status = final_state.get("status", "DONE")
        run.iterations = final_state.get("iteration", 1)
        if final_state.get("test_results"):
            run.final_coverage = final_state["test_results"].get("coverage_percent", 0.0)
            run.test_file_path = final_state["test_results"].get("test_file")
            # Automatically record closed-loop feedback outcomes
            try:
                record_outcomes(
                    target_functions=target_functions,
                    test_results=final_state["test_results"],
                    db_run_id=gen_run_id,
                )
            except Exception:
                pass

        db.commit()
    except Exception:
        db.rollback()
        run = db.query(models.TestGenerationRun).filter(models.TestGenerationRun.id == gen_run_id).first()
        if run:
            run.status = "FAILED"
            db.commit()
    finally:
        db.close()


@router.post("/generate", response_model=schemas.GenerateTestsResponse)
def generate_tests(
    req: schemas.GenerateTestsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Triggers an asynchronous multi-agent test generation run for high/medium-risk functions
    identified by a previous BugLens prediction run.
    """
    pred_run = db.query(models.PredictionRun).filter(models.PredictionRun.id == req.prediction_run_id).first()
    if not pred_run:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction run {req.prediction_run_id} not found."
        )

    # Fetch prioritized functions (prefer HIGH/MEDIUM, or all if none are high/medium)
    funcs = (
        db.query(models.FunctionPrediction)
        .filter(
            models.FunctionPrediction.run_id == req.prediction_run_id,
            models.FunctionPrediction.risk_category.in_(["HIGH", "MEDIUM"])
        )
        .order_by(models.FunctionPrediction.bug_probability.desc())
        .all()
    )

    if not funcs:
        funcs = (
            db.query(models.FunctionPrediction)
            .filter(models.FunctionPrediction.run_id == req.prediction_run_id)
            .order_by(models.FunctionPrediction.bug_probability.desc())
            .all()
        )

    target_dicts = [
        {
            "filepath": f.filepath,
            "function_name": f.function_name,
            "line_start": f.line_start,
            "line_end": f.line_end,
            "bug_probability": f.bug_probability,
            "risk_category": f.risk_category,
        }
        for f in funcs
    ]

    gen_run = models.TestGenerationRun(
        prediction_run_id=req.prediction_run_id,
        status="PENDING",
    )
    db.add(gen_run)
    db.commit()
    db.refresh(gen_run)

    # Enqueue background agent task
    background_tasks.add_task(
        _execute_agent_generation,
        gen_run.id,
        pred_run.repo_path,
        target_dicts,
        req.max_iterations,
    )

    return schemas.GenerateTestsResponse(
        run_id=gen_run.id,
        status=gen_run.status,
        message=f"Test generation task queued for {len(target_dicts)} functions via AutoTestArmy multi-agent pipeline.",
    )


@router.get("/runs/{run_id}", response_model=schemas.TestRunStatusResponse)
def get_generation_run(run_id: int, db: Session = Depends(get_db)):
    """Retrieve the status and results of a test generation run."""
    run = db.query(models.TestGenerationRun).filter(models.TestGenerationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Test generation run {run_id} not found.")

    return schemas.TestRunStatusResponse(
        run_id=run.id,
        prediction_run_id=run.prediction_run_id,
        status=run.status,
        iterations=run.iterations,
        final_coverage=run.final_coverage,
        test_file_path=run.test_file_path,
    )
