"""
CodeSentinel — LangGraph Multi-Agent Test Generation Pipeline
=============================================================
Assembles and compiles the StateGraph connecting:
    Analyzer Agent -> Generator Agent -> Critic Agent -> [Self-Correction Loop]
"""
import os
from typing import Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from autotestarmy.graph.state import AgentState
from autotestarmy.agents.analyzer_agent import analyzer_agent
from autotestarmy.agents.generator_agent import generator_agent
from autotestarmy.agents.critic_agent import critic_agent


def analyzer_router(state: AgentState) -> str:
    """Decide next node after Analyzer."""
    if not state.get("target_functions"):
        return END
    return "generator"


def critic_router(state: AgentState) -> str:
    """Decide whether to self-correct (loop back) or terminate."""
    status = state.get("status", "DONE")
    if status == "GENERATING":
        return "generator"
    return END


def build_pipeline():
    """Build and compile the LangGraph multi-agent pipeline."""
    workflow = StateGraph(AgentState)

    # Register agent nodes
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("generator", generator_agent)
    workflow.add_node("critic", critic_agent)

    # Set entry point
    workflow.set_entry_point("analyzer")

    # Routing edges
    workflow.add_conditional_edges(
        "analyzer",
        analyzer_router,
        {
            "generator": "generator",
            END: END,
        },
    )

    workflow.add_edge("generator", "critic")

    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "generator": "generator",
            END: END,
        },
    )

    return workflow.compile()


# Compile global pipeline instance
PIPELINE = build_pipeline()


def run_pipeline(
    repo_path: str,
    target_functions: list[dict[str, Any]],
    max_iterations: int = 5,
    save_output_dir: str | None = None,
) -> AgentState:
    """
    Execute the AutoTestArmy multi-agent test generation loop.

    Args:
        repo_path: Absolute or relative path to target repository.
        target_functions: List of prioritized functions from BugLens.
        max_iterations: Maximum generator-critic iterations.
        save_output_dir: Optional directory to persist generated test file.

    Returns:
        Final AgentState after convergence or budget termination.
    """
    initial_state: AgentState = {
        "repo_path": str(Path(repo_path).resolve()),
        "target_functions": target_functions,
        "max_iterations": max_iterations,
        "analysis": None,
        "generated_tests": None,
        "test_results": None,
        "iteration": 0,
        "status": "ANALYZING",
        "stop_reason": None,
        "iteration_history": [],
    }

    final_state = PIPELINE.invoke(initial_state)

    # Optionally persist the generated test code
    if save_output_dir and final_state.get("generated_tests"):
        out_dir = Path(save_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "generated_tests.py"
        out_file.write_text(final_state["generated_tests"], encoding="utf-8")
        if final_state.get("test_results") is not None:
            final_state["test_results"]["test_file"] = str(out_file)

    return final_state
