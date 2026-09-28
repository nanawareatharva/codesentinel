"""Extract per-function Python metrics using Radon and the standard AST."""
import ast
from pathlib import Path

import pandas as pd
import radon.complexity as radon_cc
import radon.metrics as radon_metrics

SKIP_DIRS = {".venv", "venv", "__pycache__", "migrations", "node_modules", ".git", "build", "dist", ".tox"}


def _extract_ast_metrics(source: str) -> dict[str, dict[str, int]]:
    """Compute structural metrics for each function, including async functions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result = {}
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        metrics = {"max_nesting_depth": 0, "num_returns": 0, "num_try_except": 0}

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.depth = 0
                self.maximum = 0

            def _visit_nested(self, node):
                self.depth += 1
                self.maximum = max(self.maximum, self.depth)
                self.generic_visit(node)
                self.depth -= 1

            visit_If = visit_For = visit_AsyncFor = visit_While = visit_With = visit_AsyncWith = _visit_nested

            def visit_Try(self, node):
                metrics["num_try_except"] += 1
                self.generic_visit(node)

            def visit_Return(self, node):
                metrics["num_returns"] += 1

        visitor = Visitor()
        # Visit the body, not the FunctionDef itself, so nested functions retain own metrics.
        for statement in function.body:
            visitor.visit(statement)
        metrics["max_nesting_depth"] = visitor.maximum
        result[function.name] = metrics
    return result


def extract_file_metrics(filepath: str) -> list[dict]:
    """Return metric dictionaries for each non-trivial function in one file."""
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError):
        return []
    if not source.strip():
        return []
    try:
        blocks = radon_cc.cc_visit(source)
    except (SyntaxError, ValueError):
        return []
    except Exception:
        return []
    try:
        mi = round(max(0.0, min(100.0, float(radon_metrics.mi_visit(source, multi=True)))), 2)
    except Exception:
        mi = 50.0
    ast_metrics = _extract_ast_metrics(source)
    functions = []
    for block in blocks:
        # Radon 6 represents functions with visitor.Function objects; it does
        # not expose the older ``type`` string on those objects.
        if not hasattr(block, "is_method"):
            continue
        end = block.endline or block.lineno
        loc = max(end - block.lineno + 1, 1)
        if loc < 2:
            continue
        structural = ast_metrics.get(block.name, {})
        functions.append({"filepath": str(filepath), "function_name": block.name,
                          "line_start": block.lineno, "line_end": end,
                          "cyclomatic_complexity": block.complexity, "loc": loc,
                          "maintainability_index": mi,
                          "max_nesting_depth": structural.get("max_nesting_depth", 0),
                          "num_returns": structural.get("num_returns", 0),
                          "num_try_except": structural.get("num_try_except", 0)})
    return functions


def extract_repo_metrics(repo_path: str) -> pd.DataFrame:
    """Scan a repository and return per-function source metrics, excluding tests."""
    repo = Path(repo_path)
    files = [file for file in repo.rglob("*.py") if not any(part in SKIP_DIRS for part in file.parts)
             and not file.name.startswith("test_") and not file.name.endswith("_test.py")
             and not file.name.startswith("conftest")]
    print(f"  Scanning {len(files)} Python files...")
    records = [record for file in files for record in extract_file_metrics(str(file))]
    result = pd.DataFrame(records)
    print(f"  Extracted metrics for {len(result)} functions")
    return result
