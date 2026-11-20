"""
CodeSentinel — Main FastAPI Application Entrypoint
==================================================
Initializes database tables, mounts API routers, configures CORS,
and provides system health check endpoints.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import engine
from backend import models, schemas
from buglens.api.routes import router as buglens_router
from autotestarmy.api.routes import router as ata_router
from feedback.api.routes import router as feedback_router

load_dotenv()

# Automatically ensure all tables are created in SQLite
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CodeSentinel API",
    description="AI-Powered Code Quality Platform — Defect Prediction & Targeted Test Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration allowing IDE extensions and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(buglens_router)
app.include_router(ata_router)
app.include_router(feedback_router)


@app.get("/", tags=["General"])
def root():
    """Root entrypoint linking to API documentation."""
    return {
        "service": "CodeSentinel API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", response_model=schemas.HealthResponse, tags=["General"])
def health():
    """System health check verifying database and model artifact availability."""
    model_dir = Path("buglens/model/saved")
    models_ready = (
        (model_dir / "lightgbm_model.pkl").exists() and
        (model_dir / "scaler.pkl").exists() and
        (model_dir / "shap_explainer.pkl").exists()
    )
    return schemas.HealthResponse(
        status="ok",
        service="CodeSentinel",
        version="1.0.0",
        models_loaded=models_ready,
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
