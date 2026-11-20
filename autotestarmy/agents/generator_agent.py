"""
CodeSentinel — Generator Agent (Module 2: AutoTestArmy)
======================================================
LLM-driven unit test generation agent. Uses local Ollama model (phi3:mini)
to synthesize targeted pytest suites based on AST structural analysis.
Incorporates execution feedback and failure tracebacks across iterations.
"""
from pathlib import Path
from typing import Optional, Any
from autotestarmy.graph.state import AgentState
from autotestarmy.tools.llm import generate_code, DEFAULT_SYSTEM_PROMPT


def _resolve_filepath(repo_path: str, filepath: str) -> Path:
    p = Path(filepath)
    if not p.is_absolute():
        p = Path(repo_path) / p
    return p


def _build_test_prompt(
    func_info: dict[str, Any],
    analysis: dict[str, Any],
    previous_failures: Optional[list[str]] = None,
    repo_path: str = "",
) -> str:
    """Build a detailed prompt guiding the LLM to write targeted test cases."""
    func_name = func_info.get("function_name", "unknown")
    filepath = func_info.get("filepath", "")
    line_start = func_info.get("line_start", 1)
    line_end = func_info.get("line_end", line_start + 20)

    # Fetch source code snippet
    func_source = ""
    resolved = _resolve_filepath(repo_path, filepath)
    if resolved.exists() and resolved.is_file():
        try:
            lines = resolved.read_text(encoding="utf-8", errors="ignore").splitlines()
            s_idx = max(0, line_start - 1)
            e_idx = min(len(lines), line_end)
            func_source = "\n".join(lines[s_idx:e_idx])
        except Exception:
            func_source = f"# Could not read lines {line_start}-{line_end} from {filepath}"

    # Lookup analysis
    a = analysis.get(func_name) or analysis.get(f"{func_name}@{filepath}") or {}

    prompt = f"""Write high-coverage pytest test cases for the following Python function.

FILE PATH: {filepath}
FUNCTION NAME: {func_name}

TARGET FUNCTION SOURCE CODE:
```python
{func_source}
```

AST CODE ANALYSIS:
- Parameters: {a.get('parameters', [])}
- Identified Branches: {a.get('branches', [])}
- Contains Loops: {a.get('has_loops', False)}
- Raises Exceptions: {a.get('raises_exceptions', False)}
- Boundary Values to Test: {a.get('boundary_inputs', [])}

INSTRUCTIONS:
1. Write 2 to 4 focused pytest functions.
2. Include at least one boundary value test (e.g. None, empty list/string, or 0).
3. Test exception handling using `with pytest.raises(...):` if applicable.
4. Output executable Python test code only. Do NOT include markdown code fences or conversational text.
"""
    if previous_failures:
        prompt += "\nPRIOR ITERATION FAILURES (FIX THESE ISSUES):\n"
        for fail_msg in previous_failures[:5]:
            prompt += f"- {fail_msg}\n"
        prompt += "\nEnsure the generated tests fix the above failures.\n"

    return prompt


def generator_agent(state: AgentState) -> AgentState:
    """
    Generator Node for LangGraph.
    Synthesizes test functions for each target function using local LLM.
    """
    repo_path = state.get("repo_path", "")
    target_functions = state.get("target_functions", [])
    analysis = state.get("analysis", {})

    failures: list[str] = []
    if state.get("test_results"):
        failures = state["test_results"].get("failure_msgs", [])

    generated_sections: list[str] = [
        "import pytest",
        "import sys",
        "from pathlib import Path",
        "",
    ]

    for func_info in target_functions:
        prompt = _build_test_prompt(
            func_info=func_info,
            analysis=analysis,
            previous_failures=failures if failures else None,
            repo_path=repo_path,
        )

        try:
            test_code = generate_code(
                prompt=prompt,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )
            func_name = func_info.get("function_name", "target")
            generated_sections.append(f"# ===========================================")
            generated_sections.append(f"# Auto-Generated Tests for: {func_name}")
            generated_sections.append(f"# ===========================================")
            generated_sections.append(test_code)
            generated_sections.append("")
        except Exception as exc:
            func_name = func_info.get("function_name", "target")
            generated_sections.append(f"# Generation fallback for {func_name}: {exc}")
            generated_sections.append(f"def test_{func_name}_fallback_pass():")
            generated_sections.append(f"    assert True, 'Fallback test stub'")
            generated_sections.append("")

    new_state = dict(state)
    new_state["generated_tests"] = "\n".join(generated_sections)
    new_state["status"] = "CRITIQUING"
    return new_state
