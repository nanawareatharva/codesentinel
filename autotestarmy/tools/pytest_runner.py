"""
CodeSentinel — Pytest & Coverage Test Runner
============================================
Executes generated test suites inside an isolated subprocess with pytest
and pytest-cov. Parses pass/fail counts, execution errors, failure tracebacks,
and branch/line coverage percentages.
"""
import os
import re
import sys
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Any
from autotestarmy.graph.state import TestExecutionResults


def run_tests(
    test_code: str,
    repo_path: str,
    timeout_seconds: int = 45,
    save_test_file: bool = False,
) -> TestExecutionResults:
    """
    Executes a generated pytest test suite against target repository code.

    Args:
        test_code: Full Python pytest source code string.
        repo_path: Base directory of the repository under test.
        timeout_seconds: Subprocess execution timeout in seconds.
        save_test_file: If True, keeps the test file on disk and returns its path.

    Returns:
        TestExecutionResults dictionary with passed, failed, errors, coverage, and failure_msgs.
    """
    repo = Path(repo_path).resolve()
    temp_dir = repo if repo.exists() else Path(tempfile.gettempdir())

    # Create temporary test file inside target repo so module imports resolve correctly
    tf = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="codesentinel_test_",
        dir=str(temp_dir),
        delete=False,
        encoding="utf-8",
    )
    test_filepath = Path(tf.name)
    try:
        tf.write(test_code)
        tf.flush()
    finally:
        tf.close()

    coverage_xml_path = temp_dir / f"cov_{test_filepath.stem}.xml"

    results: TestExecutionResults = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "coverage_percent": 0.0,
        "failure_msgs": [],
        "test_file": str(test_filepath),
    }

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(repo)}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(repo)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_filepath),
        "-v",
        "--tb=short",
        f"--cov={str(repo)}",
        f"--cov-report=xml:{str(coverage_xml_path)}",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        _parse_pytest_output(combined_output, results)

        # Parse coverage from XML if generated
        if coverage_xml_path.exists():
            try:
                tree = ET.parse(str(coverage_xml_path))
                root = tree.getroot()
                line_rate = float(root.attrib.get("line-rate", 0.0))
                results["coverage_percent"] = round(line_rate * 100, 1)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        results["errors"] += 1
        results["failure_msgs"].append(
            f"TIMEOUT: Test suite exceeded execution limit ({timeout_seconds}s)."
        )
    except Exception as exc:
        results["errors"] += 1
        results["failure_msgs"].append(f"EXECUTION ERROR: {exc}")
    finally:
        # Clean up coverage XML
        if coverage_xml_path.exists():
            try:
                coverage_xml_path.unlink()
            except Exception:
                pass

        # Clean up test file unless caller requested saving it
        if not save_test_file and test_filepath.exists():
            try:
                test_filepath.unlink()
            except Exception:
                pass

    return results


def _parse_pytest_output(output: str, results: TestExecutionResults) -> None:
    """Parse pass/fail/error counts and failure lines from stdout/stderr."""
    # Match patterns like "3 passed, 1 failed, 2 errors in 0.12s"
    summary_match = re.search(r"(=+\s+)?(?P<counts>[0-9a-zA-Z,\s]+)\s+in\s+[0-9\.]+s", output)
    if summary_match:
        counts_str = summary_match.group("counts")
        passed_m = re.search(r"(\d+)\s+passed", counts_str)
        failed_m = re.search(r"(\d+)\s+failed", counts_str)
        errors_m = re.search(r"(\d+)\s+error", counts_str)

        if passed_m:
            results["passed"] = int(passed_m.group(1))
        if failed_m:
            results["failed"] = int(failed_m.group(1))
        if errors_m:
            results["errors"] = int(errors_m.group(1))

    # Collect failure messages
    lines = output.splitlines()
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if line_clean.startswith("FAILED ") or line_clean.startswith("ERROR "):
            results["failure_msgs"].append(line_clean)
        elif "AssertionError" in line_clean or "TypeError:" in line_clean or "ValueError:" in line_clean:
            # Include relevant traceback snippet
            results["failure_msgs"].append(line_clean)

    # Deduplicate and cap failure messages for prompt conciseness
    seen = set()
    deduped = []
    for msg in results["failure_msgs"]:
        if msg not in seen:
            seen.add(msg)
            deduped.append(msg)
    results["failure_msgs"] = deduped[:10]
