"""
CodeSentinel — Unit & Integration Tests for AutoTestArmy Multi-Agent System
==========================================================================
Verifies:
- AST Analyzer extraction of parameters, branches, boundaries, and exceptions
- LLM code fence cleaning
- Pytest subprocess runner execution and failure parsing
- Critic Agent stopping conditions (coverage >= 80%, stabilised, max iterations)
- LangGraph StateGraph execution loop
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autotestarmy.graph.state import AgentState
from autotestarmy.tools.llm import clean_markdown_fences
from autotestarmy.tools.pytest_runner import run_tests
from autotestarmy.agents.analyzer_agent import analyze_function_ast, analyzer_agent
from autotestarmy.agents.critic_agent import critic_agent
from autotestarmy.graph.pipeline import build_pipeline, run_pipeline


@pytest.fixture
def sample_code_file(tmp_path: Path) -> Path:
    """Creates a python file with various language features for AST analysis."""
    code = '''
def process_order(price: float, tier: str, coupon: str = None) -> float:
    """Calculates final price based on customer tier and coupon."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    
    discount = 0.0
    if tier == "VIP":
        discount = 0.20
    elif tier == "GOLD":
        discount = 0.10
    
    if coupon:
        discount += 0.05

    for _ in range(1):
        pass

    return price * (1.0 - min(discount, 0.5))
'''
    f = tmp_path / "orders.py"
    f.write_text(code, encoding="utf-8")
    return f


def test_clean_markdown_fences():
    """Verify markdown code fence stripping from LLM outputs."""
    raw_1 = "```python\ndef test_add(): assert 1 + 1 == 2\n```"
    assert clean_markdown_fences(raw_1) == "def test_add(): assert 1 + 1 == 2"

    raw_2 = "```\ndef test_sub(): assert 3 - 1 == 2\n```"
    assert clean_markdown_fences(raw_2) == "def test_sub(): assert 3 - 1 == 2"

    clean = "def test_clean(): assert True"
    assert clean_markdown_fences(clean) == clean


def test_analyzer_ast_extraction(sample_code_file: Path):
    """Verify deterministic extraction of AST structures."""
    source = sample_code_file.read_text(encoding="utf-8")
    analysis = analyze_function_ast(source, "process_order", line_start=2)

    assert "price" in analysis["parameters"]
    assert "tier" in analysis["parameters"]
    assert "coupon" in analysis["parameters"]
    assert analysis["raises_exceptions"] is True
    assert analysis["has_loops"] is True
    assert len(analysis["branches"]) >= 3
    assert len(analysis["boundary_inputs"]) >= 3

    # Check analyzer_agent node execution
    state: AgentState = {
        "repo_path": str(sample_code_file.parent),
        "target_functions": [
            {
                "filepath": str(sample_code_file),
                "function_name": "process_order",
                "line_start": 2,
                "line_end": 20,
            }
        ],
        "max_iterations": 3,
        "iteration": 0,
        "status": "ANALYZING",
    }
    new_state = analyzer_agent(state)
    assert new_state["status"] == "GENERATING"
    assert "process_order" in new_state["analysis"]


def test_pytest_runner_execution(tmp_path: Path):
    """Verify isolated subprocess execution and pass/fail/error parsing."""
    module_code = "def add(a, b): return a + b\n"
    (tmp_path / "math_mod.py").write_text(module_code, encoding="utf-8")

    # Passing test
    passing_test = """
from math_mod import add
def test_add_positive():
    assert add(2, 3) == 5
"""
    res_pass = run_tests(passing_test, str(tmp_path))
    assert res_pass["passed"] >= 1
    assert res_pass["failed"] == 0

    # Failing test
    failing_test = """
from math_mod import add
def test_add_fail():
    assert add(2, 2) == 5
"""
    res_fail = run_tests(failing_test, str(tmp_path))
    assert res_fail["failed"] >= 1
    assert len(res_fail["failure_msgs"]) >= 1


def test_critic_agent_stopping_conditions():
    """Verify Critic convergence rules: coverage >= 80%, stabilised, max iterations."""
    base_state: AgentState = {
        "repo_path": ".",
        "target_functions": [],
        "max_iterations": 3,
        "iteration": 0,
        "status": "CRITIQUING",
        "generated_tests": "def test_stub(): assert True",
    }

    # Case 1: High coverage (>= 80%) terminates immediately
    with patch("autotestarmy.agents.critic_agent.run_tests") as mock_run:
        mock_run.return_value = {
            "passed": 3, "failed": 0, "errors": 0,
            "coverage_percent": 88.5, "failure_msgs": [], "test_file": None
        }
        res_state = critic_agent(base_state)
        assert res_state["status"] == "DONE"
        assert res_state["stop_reason"] == "full_coverage"

    # Case 2: Failures on iteration 1 causes loop back to GENERATING
    with patch("autotestarmy.agents.critic_agent.run_tests") as mock_run:
        mock_run.return_value = {
            "passed": 1, "failed": 1, "errors": 0,
            "coverage_percent": 40.0, "failure_msgs": ["FAILED test_err - AssertionError"], "test_file": None
        }
        res_state = critic_agent(base_state)
        assert res_state["status"] == "GENERATING"
        assert res_state["iteration"] == 1

    # Case 3: Reached max iterations terminates
    max_iter_state = dict(base_state)
    max_iter_state["iteration"] = 2  # next iteration will be 3 (which equals max_iterations)
    with patch("autotestarmy.agents.critic_agent.run_tests") as mock_run:
        mock_run.return_value = {
            "passed": 1, "failed": 1, "errors": 0,
            "coverage_percent": 45.0, "failure_msgs": ["AssertionError"], "test_file": None
        }
        res_state = critic_agent(max_iter_state)
        assert res_state["status"] == "DONE"
        assert res_state["stop_reason"] == "max_iterations"


def test_full_pipeline_execution(sample_code_file: Path):
    """Verify end-to-end execution of LangGraph pipeline with simulated LLM generation."""
    test_gen_code = f"""
from orders import process_order
import pytest

def test_process_order_vip():
    assert process_order(100.0, "VIP") == 80.0

def test_process_order_negative_price():
    with pytest.raises(ValueError):
        process_order(-5.0, "VIP")
"""
    targets = [
        {
            "filepath": str(sample_code_file),
            "function_name": "process_order",
            "line_start": 2,
            "line_end": 20,
            "bug_probability": 0.85,
            "risk_category": "HIGH",
        }
    ]

    # Patch LLM generation to test full graph flow reliably without waiting on model load
    with patch("autotestarmy.agents.generator_agent.generate_code", return_value=test_gen_code):
        final_state = run_pipeline(
            repo_path=str(sample_code_file.parent),
            target_functions=targets,
            max_iterations=2,
            save_output_dir=str(sample_code_file.parent / "out_tests"),
        )
        assert final_state["status"] == "DONE"
        assert final_state["iteration"] >= 1
        assert final_state["test_results"]["passed"] >= 2
        assert final_state["test_results"]["failed"] == 0
