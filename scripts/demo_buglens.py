"""
CodeSentinel — BugLens Demo & Inference Runner
==============================================
Runs full repository mining, LightGBM defect prediction,
and SHAP root-cause explanation on any Python codebase.

Usage:
    python scripts/demo_buglens.py
    python scripts/demo_buglens.py path/to/your/repo
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buglens.model.predict import predict_repo
from buglens.model.shap_explainer import explain_function


def run_demo(target_repo: str = "data/demo_repos/flask"):
    repo_path = Path(target_repo).resolve()
    if not repo_path.exists():
        print(f"Error: Target repository path not found: {repo_path}")
        return

    print("=" * 70)
    print(f"  CODESENTINEL / BUGLENS — LIVE DEFECT RISK INFERENCE")
    print("=" * 70)
    print(f"Target Repository : {repo_path}")

    # 1. Run inference
    results = predict_repo(str(repo_path))

    if not results:
        print("\nNo Python functions found or analyzed.")
        return

    # 2. Risk breakdown
    high = [r for r in results if r["risk_category"] == "HIGH"]
    med = [r for r in results if r["risk_category"] == "MEDIUM"]
    low = [r for r in results if r["risk_category"] == "LOW"]

    print("\n" + "=" * 70)
    print(f"  ANALYSIS SUMMARY ({len(results)} functions analyzed)")
    print("=" * 70)
    print(f"  • HIGH Risk   (>= 66%): {len(high):4d} functions")
    print(f"  • MEDIUM Risk (33-66%): {len(med):4d} functions")
    print(f"  • LOW Risk    (< 33%) : {len(low):4d} functions")

    # 3. Top 10 highest risk functions
    top_n = min(10, len(results))
    print("\n" + "=" * 70)
    print(f"  TOP {top_n} HIGHEST RISK FUNCTIONS (Priority Queue for AutoTestArmy)")
    print("=" * 70)
    print(f" {'#':<3} {'Risk':<8} {'Prob':<7} {'Function':<24} {'Location'}")
    print("-" * 70)

    for i, fn in enumerate(results[:top_n], 1):
        cat = fn["risk_category"]
        prob = f"{fn['bug_probability']:.1f}%"
        name = fn["function_name"]
        file_part = Path(fn["filepath"]).name
        location = f"{file_part}:L{fn['line_start']}-L{fn['line_end']}"
        print(f" {i:<3} [{cat:<6}] {prob:<7} {name:<24} {location}")

    # 4. SHAP Explanation for the Top Highest Risk Function
    top = results[0]
    print("\n" + "=" * 70)
    print(f"  SHAP ROOT-CAUSE EXPLANATION")
    print(f"  Function : {top['function_name']}")
    print(f"  File     : {Path(top['filepath']).name} (lines {top['line_start']}-{top['line_end']})")
    print(f"  Risk     : {top['bug_probability']:.1f}% [{top['risk_category']}]")
    print("=" * 70)

    expl = explain_function(top)
    print(f"\n  Key Summary : {expl['summary']}\n")
    print("  Feature Contribution Breakdown:")
    print("  " + "-" * 66)
    for driver in expl["top_risk_drivers"]:
        label = driver["label"]
        raw_val = driver["raw_value"]
        direction = driver["direction"]
        shap_val = driver["shap_value"]
        sign = "+" if shap_val > 0 else ""
        print(f"  • {label:<32} (value: {raw_val:<6}) -> {direction:<15} [SHAP: {sign}{shap_val:.4f}]")

    print("\n" + "=" * 70)
    print("  Demo complete. Phase 0, 1, 2 working as specified!")
    print("=" * 70)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/demo_repos/flask"
    run_demo(target)
