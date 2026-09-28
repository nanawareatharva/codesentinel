"""Mine per-file Git-history features with PyDriller."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pydriller import Repository

from buglens.config import BUG_FIX_KEYWORDS

GIT_COLUMNS = ["filepath", "commit_count", "churn", "unique_developers", "bug_fix_commits", "file_age_days", "recent_churn"]


def mine_repository(repo_path: str, since=None, to=None) -> pd.DataFrame:
    """Return per-Python-file churn, authorship, fix and age features.

    A mining error is reported and yields an empty, schema-correct frame so
    inference can zero-fill features for untracked or inaccessible repositories.
    """
    root = str(Path(repo_path).resolve())
    stats = defaultdict(lambda: {"commit_count": 0, "churn": 0, "developers": set(),
                                 "bug_fix_commits": 0, "first": None, "last": None,
                                 "recent_churn": 0})
    ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
    commits = 0
    try:
        for commit in Repository(root, since=since, to=to).traverse_commits():
            commits += 1
            for modified in commit.modified_files:
                path = modified.new_path or modified.old_path
                if not path or not path.endswith(".py"):
                    continue
                stat = stats[path]
                churn = (modified.added_lines or 0) + (modified.deleted_lines or 0)
                stat["commit_count"] += 1
                stat["churn"] += churn
                if commit.author.email:
                    stat["developers"].add(commit.author.email.lower().strip())
                if any(keyword in (commit.msg or "").lower() for keyword in BUG_FIX_KEYWORDS):
                    stat["bug_fix_commits"] += 1
                date = commit.author_date
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                stat["first"] = date if stat["first"] is None else min(stat["first"], date)
                stat["last"] = date if stat["last"] is None else max(stat["last"], date)
                if date >= ninety_days_ago:
                    stat["recent_churn"] += churn
    except Exception as exc:
        print(f"  Git mining error: {exc}")

    rows = []
    for filepath, stat in stats.items():
        age = max((stat["last"] - stat["first"]).days, 1) if stat["first"] and stat["last"] else 0
        rows.append({"filepath": filepath, "commit_count": stat["commit_count"], "churn": stat["churn"],
                     "unique_developers": len(stat["developers"]), "bug_fix_commits": stat["bug_fix_commits"],
                     "file_age_days": age, "recent_churn": stat["recent_churn"]})
    print(f"  Git mining complete: {commits} commits, {len(rows)} Python files")
    return pd.DataFrame(rows, columns=GIT_COLUMNS)
