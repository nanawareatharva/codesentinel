"""
CodeSentinel — API Endpoint Tests
=================================
Automated test suite verifying FastAPI endpoints, SQLite ORM persistence,
BugLens live prediction integration, error handling, priority queue, and SHAP responses.
"""
import subprocess
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import get_db, Base, engine, SessionLocal


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient connected to the FastAPI application."""
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_git_repo(tmp_path_factory):
    """Creates a temporary Git repository containing sample Python functions."""
    repo_dir = tmp_path_factory.mktemp("api_sample_repo")

    sample_code = """
def calculate_grade(score: float) -> str:
    \"\"\"Calculates letter grade with boundary logic.\"\"\"
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def process_data(items: list) -> list:
    \"\"\"Processes list of items with exception safety.\"\"\"
    results = []
    for item in items:
        try:
            if item is not None and len(str(item)) > 0:
                results.append(str(item).strip().upper())
        except Exception:
            continue
    return results
"""
    (repo_dir / "service.py").write_text(sample_code, encoding="utf-8")

    # Initialize Git repository and commit files
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@codesentinel.io"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "CodeSentinel Tester"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "fix: initial service implementation"], cwd=repo_dir, capture_output=True, check=True)

    return repo_dir


def test_root_endpoint(client):
    """Verify root endpoint responds with online status and documentation links."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
    assert data["redoc"] == "/redoc"


def test_health_endpoint(client):
    """Verify health check endpoint reports status and model availability."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "CodeSentinel"
    assert data["models_loaded"] is True


def test_predict_nonexistent_repo(client):
    """Verify that targeting a non-existent folder returns HTTP 400."""
    res = client.post("/buglens/predict", json={"repo_path": "/path/that/does/not/exist/99999"})
    assert res.status_code == 400
    assert "does not exist" in res.json()["detail"]


def test_predict_empty_repo(client, tmp_path):
    """Verify that targeting an empty repository with no Python code returns HTTP 400."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "README.txt").write_text("No python code here", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial empty commit"], cwd=tmp_path, capture_output=True, check=True)

    res = client.post("/buglens/predict", json={"repo_path": str(tmp_path)})
    assert res.status_code == 400
    assert "No Python functions found" in res.json()["detail"]


def test_predict_valid_repo_and_persistence(client, sample_git_repo):
    """
    Verify complete prediction on sample repository:
    - Returns 200 with structured JSON.
    - Accurately captures functions.
    - Enriches with SHAP explanations.
    - Persists record in database.
    """
    res = client.post("/buglens/predict", json={
        "repo_path": str(sample_git_repo),
        "max_commits": 10
    })
    assert res.status_code == 200
    data = res.json()

    assert "run_id" in data
    assert data["run_id"] > 0
    assert data["function_count"] >= 2
    assert len(data["functions"]) == data["function_count"]

    # Verify function attributes and SHAP enrichment
    first_func = data["functions"][0]
    assert "function_name" in first_func
    assert "bug_probability" in first_func
    assert first_func["risk_category"] in ("HIGH", "MEDIUM", "LOW")
    assert "shap_explanation" in first_func
    assert first_func["shap_explanation"] is not None
    assert "top_risk_drivers" in first_func["shap_explanation"]
    assert "summary" in first_func["shap_explanation"]

    # Test GET /buglens/runs (Historical runs list)
    runs_res = client.get("/buglens/runs")
    assert runs_res.status_code == 200
    runs = runs_res.json()
    assert len(runs) >= 1
    assert any(r["id"] == data["run_id"] for r in runs)

    # Test GET /buglens/runs/{run_id} (Full run detail)
    run_detail_res = client.get(f"/buglens/runs/{data['run_id']}")
    assert run_detail_res.status_code == 200
    run_detail = run_detail_res.json()
    assert run_detail["run_id"] == data["run_id"]
    assert run_detail["function_count"] == data["function_count"]

    # Test GET /buglens/runs/{run_id}/queue (Priority queue)
    queue_res = client.get(f"/buglens/runs/{data['run_id']}/queue?limit=5")
    assert queue_res.status_code == 200
    queue = queue_res.json()
    assert len(queue) >= 1
    # Verify probability descending order
    probs = [q["bug_probability"] for q in queue]
    assert probs == sorted(probs, reverse=True)

    # Test GET /buglens/runs/{run_id}/functions/{function_name}
    func_name = first_func["function_name"]
    func_detail_res = client.get(f"/buglens/runs/{data['run_id']}/functions/{func_name}")
    assert func_detail_res.status_code == 200
    funcs = func_detail_res.json()
    assert len(funcs) >= 1
    assert funcs[0]["function_name"] == func_name


def test_get_nonexistent_run(client):
    """Verify that requesting an invalid run ID returns HTTP 404."""
    res = client.get("/buglens/runs/9999999")
    assert res.status_code == 404


def test_autotestarmy_routes(client, sample_git_repo):
    """Verify AutoTestArmy integration endpoint stubs and status tracking."""
    # First create a prediction run
    pred_res = client.post("/buglens/predict", json={
        "repo_path": str(sample_git_repo),
        "max_commits": 5
    })
    run_id = pred_res.json()["run_id"]

    # Trigger test generation registration
    gen_res = client.post("/autotestarmy/generate", json={
        "prediction_run_id": run_id,
        "max_iterations": 3
    })
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert "run_id" in gen_data
    assert gen_data["status"] == "PENDING"

    # Query generation run status
    status_res = client.get(f"/autotestarmy/runs/{gen_data['run_id']}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["run_id"] == gen_data["run_id"]
    assert status_data["prediction_run_id"] == run_id
    assert status_data["status"] in ["PENDING", "RUNNING", "DONE"]
