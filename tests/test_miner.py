import subprocess

import pytest

from buglens.mining.metric_extractor import extract_file_metrics

SIMPLE_CODE = '''
def risky_function(score, threshold, mode):
    if score < 0:
        raise ValueError("score must be non-negative")
    if mode == "strict":
        if score > threshold:
            return "FAIL"
        elif score == threshold:
            return "BORDERLINE"
        return "PASS"
    return "PASS" if score <= threshold else "FAIL"

def trivial(x):
    return x * 2
'''


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(SIMPLE_CODE)
    return str(path)


def test_returns_list(sample_file):
    assert isinstance(extract_file_metrics(sample_file), list)


def test_finds_functions(sample_file):
    assert "risky_function" in [item["function_name"] for item in extract_file_metrics(sample_file)]


def test_metrics_and_complexity(sample_file):
    functions = {item["function_name"]: item for item in extract_file_metrics(sample_file)}
    required = {"filepath", "function_name", "line_start", "line_end", "cyclomatic_complexity", "loc", "maintainability_index"}
    assert required <= functions["risky_function"].keys()
    assert functions["risky_function"]["cyclomatic_complexity"] > functions["trivial"]["cyclomatic_complexity"]


def test_bad_or_empty_files_return_no_metrics(tmp_path):
    bad, empty = tmp_path / "bad.py", tmp_path / "empty.py"
    bad.write_text("def broken(:\n    pass")
    empty.write_text("")
    assert extract_file_metrics(str(bad)) == []
    assert extract_file_metrics(str(empty)) == []


@pytest.fixture
def mini_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (["git", "init"], ["git", "config", "user.email", "test@test.com"], ["git", "config", "user.name", "Tester"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'hi'\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'hi'\n\ndef bye():\n    return 'bye'\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fix: add bye"], cwd=repo, check=True, capture_output=True)
    return str(repo)


def test_mine_git_features(mini_git_repo):
    from buglens.mining.git_miner import mine_repository
    df = mine_repository(mini_git_repo)
    assert not df.empty
    assert {"churn", "unique_developers", "bug_fix_commits"} <= set(df.columns)
    assert df["bug_fix_commits"].sum() >= 1


def test_feature_schema_length():
    from buglens.config import FEATURES
    assert len(FEATURES) == 8
