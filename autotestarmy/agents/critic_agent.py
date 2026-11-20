"""
CodeSentinel — Critic Agent (Module 2: AutoTestArmy)
====================================================
Evaluator agent that executes synthesized tests via pytest_runner, analyzes
code coverage and failure messages, and evaluates stopping conditions:
- Full Coverage: Target code coverage >= 80%
- Stabilised: 0 failures / errors achieved
- Max Iterations: Iteration budget exhausted
"""
from typing import Any
from autotestarmy.graph.state import AgentState, IterationLog
from autotestarmy.tools.pytest_runner import run_tests


def critic_agent(state: AgentState) -> AgentState:
    """
    Critic Node for LangGraph.
    Executes tests in subprocess, calculates metrics, and updates loop status.
    """
    test_code = state.get("generated_tests", "")
    repo_path = state.get("repo_path", "")
    current_iter = state.get("iteration", 0) + 1
    max_iters = state.get("max_iterations", 5)

    # Run tests in subprocess
    results = run_tests(
        test_code=test_code,
        repo_path=repo_path,
        timeout_seconds=45,
        save_test_file=False,
    )

    history_entry: IterationLog = {
        "iteration": current_iter,
        "passed": results.get("passed", 0),
        "failed": results.get("failed", 0),
        "errors": results.get("errors", 0),
        "coverage_percent": results.get("coverage_percent", 0.0),
        "status": "EVALUATED",
    }

    new_state = dict(state)
    new_state["test_results"] = results
    new_state["iteration"] = current_iter
    new_state["iteration_history"] = state.get("iteration_history", []) + [history_entry]

    coverage = results.get("coverage_percent", 0.0)
    failures = results.get("failed", 0) + results.get("errors", 0)
    passed = results.get("passed", 0)

    # Stopping condition evaluation
    if coverage >= 80.0:
        new_state["status"] = "DONE"
        new_state["stop_reason"] = "full_coverage"
    elif current_iter >= max_iters:
        new_state["status"] = "DONE"
        new_state["stop_reason"] = "max_iterations"
    elif failures == 0 and (current_iter >= 2 or (passed > 0 and coverage >= 50.0)):
        new_state["status"] = "DONE"
        new_state["stop_reason"] = "stabilised"
    else:
        # Failures remain or low coverage, loop back to generator agent
        new_state["status"] = "GENERATING"

    return new_state
