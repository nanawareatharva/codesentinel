"""
CodeSentinel — LangGraph State Schema for AutoTestArmy
======================================================
Defines the central shared state TypedDict that flows through the
multi-agent test generation graph:
    Analyzer -> Generator -> Critic -> (Self-Correction Loop)
"""
from typing import TypedDict, Optional, Any


class TargetFunctionInfo(TypedDict, total=False):
    """Schema for a prioritized function from BugLens."""
    filepath: str
    function_name: str
    line_start: int
    line_end: int
    bug_probability: float
    risk_category: str
    cyclomatic_complexity: Optional[float]
    loc: Optional[int]


class FunctionAnalysis(TypedDict, total=False):
    """AST breakdown extracted by the Analyzer Agent."""
    parameters: list[str]
    branches: list[str]
    has_loops: bool
    raises_exceptions: bool
    boundary_inputs: list[dict[str, Any]]


class TestExecutionResults(TypedDict, total=False):
    """Execution outcomes parsed by the pytest runner and Critic Agent."""
    passed: int
    failed: int
    errors: int
    coverage_percent: float
    failure_msgs: list[str]
    test_file: Optional[str]


class IterationLog(TypedDict, total=False):
    """Historical telemetry for one iteration."""
    iteration: int
    passed: int
    failed: int
    errors: int
    coverage_percent: float
    status: str


class AgentState(TypedDict, total=False):
    """
    Central shared state flowing across LangGraph nodes.
    Maintains all intermediate artifacts, execution outputs, and iteration history.
    """
    # Inputs
    repo_path: str
    target_functions: list[dict[str, Any]]
    max_iterations: int

    # Agent Outputs
    analysis: Optional[dict[str, Any]]
    generated_tests: Optional[str]
    test_results: Optional[dict[str, Any]]

    # Flow Control
    iteration: int
    status: str  # ANALYZING | GENERATING | CRITIQUING | DONE | FAILED
    stop_reason: Optional[str]  # full_coverage | stabilised | max_iterations | error

    # Telemetry Log
    iteration_history: list[dict[str, Any]]
