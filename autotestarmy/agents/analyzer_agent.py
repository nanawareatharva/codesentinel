"""
CodeSentinel — Analyzer Agent (Module 2: AutoTestArmy)
======================================================
Deterministic AST analysis agent. Inspects target functions to discover:
- Function parameters and default values
- Conditional branches (if, elif, ternary, match)
- Iteration constructs (for, while)
- Explicit exception raises
- Boundary and edge-case input candidates (None, 0, -1, empty structures)
"""
import ast
import os
from pathlib import Path
from typing import Any
from autotestarmy.graph.state import AgentState, FunctionAnalysis


def _resolve_filepath(repo_path: str, filepath: str) -> Path:
    """Resolve file path whether relative to repo or absolute."""
    p = Path(filepath)
    if not p.is_absolute():
        p = Path(repo_path) / p
    return p


def analyze_function_ast(source_code: str, function_name: str, line_start: int = 1) -> FunctionAnalysis:
    """Analyze the AST of a specific function within source code."""
    try:
        tree = ast.parse(source_code)
    except Exception:
        return {
            "parameters": [],
            "branches": [],
            "has_loops": False,
            "raises_exceptions": False,
            "boundary_inputs": [],
        }

    matched_node: Any = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                # If line_start is specified, prefer the node closest to that line
                if matched_node is None or abs(node.lineno - line_start) < abs(matched_node.lineno - line_start):
                    matched_node = node

    if matched_node is None:
        return {
            "parameters": [],
            "branches": [],
            "has_loops": False,
            "raises_exceptions": False,
            "boundary_inputs": [],
        }

    # Extract parameters (excluding self and cls)
    params = [arg.arg for arg in matched_node.args.args]
    boundary_inputs = []
    for param in params:
        if param not in ("self", "cls"):
            boundary_inputs.append({
                "param": param,
                "values": [None, 0, -1, "", [], {}, "edge_case_string"],
            })

    # Detect branches, loops, and raises
    branches: list[str] = []
    has_loops = False
    raises_exceptions = False

    for child in ast.walk(matched_node):
        if isinstance(child, ast.If):
            branches.append(f"line {child.lineno}: if-condition")
        elif isinstance(child, ast.IfExp):
            branches.append(f"line {child.lineno}: inline-ternary")
        elif hasattr(ast, "Match") and isinstance(child, ast.Match):
            branches.append(f"line {child.lineno}: match-pattern")
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            has_loops = True
        elif isinstance(child, ast.Raise):
            raises_exceptions = True

    return {
        "parameters": params,
        "branches": branches,
        "has_loops": has_loops,
        "raises_exceptions": raises_exceptions,
        "boundary_inputs": boundary_inputs,
    }


def analyzer_agent(state: AgentState) -> AgentState:
    """
    Analyzer Node for LangGraph.
    Scans AST for all functions in state["target_functions"] and records analysis.
    """
    repo_path = state.get("repo_path", "")
    target_functions = state.get("target_functions", [])
    analysis_map: dict[str, Any] = {}

    for func_info in target_functions:
        filepath = func_info.get("filepath", "")
        func_name = func_info.get("function_name", "")
        line_start = func_info.get("line_start", 1)

        resolved_path = _resolve_filepath(repo_path, filepath)
        if resolved_path.exists() and resolved_path.is_file():
            try:
                source = resolved_path.read_text(encoding="utf-8", errors="ignore")
                func_analysis = analyze_function_ast(source, func_name, line_start)
            except Exception:
                func_analysis = {
                    "parameters": [],
                    "branches": [],
                    "has_loops": False,
                    "raises_exceptions": False,
                    "boundary_inputs": [],
                }
        else:
            func_analysis = {
                "parameters": [],
                "branches": [],
                "has_loops": False,
                "raises_exceptions": False,
                "boundary_inputs": [],
            }

        # Key by function_name or filepath::function_name if unique
        key = f"{func_name}@{filepath}" if filepath else func_name
        analysis_map[key] = func_analysis
        # Also store direct func_name for easy lookup
        if func_name not in analysis_map:
            analysis_map[func_name] = func_analysis

    new_state = dict(state)
    new_state["analysis"] = analysis_map
    new_state["status"] = "GENERATING"
    return new_state
