"""
CodeSentinel — End-to-End Live Demo on Any Python Repository
============================================================
Runs the complete CodeSentinel pipeline on any local Git repository:
1. BugLens: Dual mining (Git + AST) & LightGBM defect prediction.
2. SHAP: Root-cause explanation for the top-risk functions.
3. Priority Queue: Filters high-risk functions needing testing.
4. AutoTestArmy: LangGraph multi-agent loop (Analyzer -> Generator -> Critic).
5. Feedback: Records dynamic test outcomes into the feedback store.

Usage:
    python scripts/demo_end_to_end.py <path_to_repo>
    python scripts/demo_end_to_end.py <path_to_repo> --max-commits 50 --max-iterations 2
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from buglens.model.predict import predict_repo
from buglens.model.shap_explainer import explain_function
from autotestarmy.graph.pipeline import run_pipeline
from feedback.aggregator import record_outcomes


def run_end_to_end(
    repo_path_str: str,
    max_commits: int = 100,
    max_iterations: int = 2,
    num_targets: int = 2,
):
    repo_path = Path(repo_path_str).resolve()
    if not repo_path.exists():
        print(f"\n[ERROR] Target repository path does not exist: {repo_path}")
        return

    print("\n" + "=" * 70)
    print(f"  CODESENTINEL — END-TO-END DEMO ON TARGET REPOSITORY")
    print("=" * 70)
    print(f"Target Repository : {repo_path}")
    print(f"Mining Scope      : Last {max_commits} commits")
    print(f"Agent Iterations  : Max {max_iterations} self-correction loops")
    print("=" * 70)

    # ---------------------------------------------------------
    # STAGE 1: BUGLENS DEFECT PREDICTION
    # ---------------------------------------------------------
    print("\n[STEP 1/4] Mining Git history & AST complexity with BugLens...")
    t0 = time.time()
    predictions = predict_repo(str(repo_path), max_commits=max_commits)
    elapsed = time.time() - t0

    if not predictions:
        print("[!] No Python functions found in repository to analyze.")
        return

    high_risk = [p for p in predictions if p["risk_category"] == "HIGH"]
    med_risk = [p for p in predictions if p["risk_category"] == "MEDIUM"]
    low_risk = [p for p in predictions if p["risk_category"] == "LOW"]

    print(f"[*] Analyzed {len(predictions)} functions in {elapsed:.1f}s:")
    print(f"    - HIGH Risk   (>= 66%): {len(high_risk)} functions")
    print(f"    - MEDIUM Risk (33-66%): {len(med_risk)} functions")
    print(f"    - LOW Risk    (< 33%) : {len(low_risk)} functions")

    # ---------------------------------------------------------
    # STAGE 2: SHAP ROOT-CAUSE EXPLANATION
    # ---------------------------------------------------------
    top_function = predictions[0]
    prob_val = top_function['bug_probability']
    prob_str = f"{prob_val:.1f}%" if prob_val > 1.0 else f"{prob_val * 100:.1f}%"
    print(f"\n[STEP 2/4] Top-Risk Function: '{top_function['function_name']}' in {top_function['filepath']}")
    print(f"    Predicted Bug Probability: {prob_str} [{top_function['risk_category']} RISK]")
    
    explanation = explain_function(top_function)
    print("    Top Risk Drivers (TreeSHAP Decomposition):")
    for d in explanation.get("top_drivers", []):
        sign = "+" if d["shap_value"] > 0 else ""
        print(f"      • {d['label']:<28} (val={d['raw_value']:<5.1f}) -> {d['direction']:<14} [SHAP: {sign}{d['shap_value']:.4f}]")

    # ---------------------------------------------------------
    # STAGE 3: AUTOTESTARMY MULTI-AGENT TEST GENERATION
    # ---------------------------------------------------------
    # Select targets from priority queue (prefer HIGH/MED)
    candidates = high_risk + med_risk
    if not candidates:
        candidates = predictions[:num_targets]
    else:
        candidates = candidates[:num_targets]

    target_names = [f"'{c['function_name']}'" for c in candidates]
    print(f"\n[STEP 3/4] Triggering AutoTestArmy multi-agent loop for {len(candidates)} prioritized functions:")
    print(f"    Targets: {', '.join(target_names)}")
    print("    Executing: Analyzer Agent -> Generator Agent (Ollama phi3:mini) -> Critic Agent...")

    out_dir = ROOT_DIR / "data" / "generated_tests" / "demo_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_gen_start = time.time()
    final_state = run_pipeline(
        repo_path=str(repo_path),
        target_functions=candidates,
        max_iterations=max_iterations,
        save_output_dir=str(out_dir),
    )
    gen_elapsed = time.time() - t_gen_start

    print(f"[*] Multi-agent synthesis completed in {gen_elapsed:.1f}s!")
    print(f"    Loop Status   : {final_state.get('status')}")
    print(f"    Iterations    : {final_state.get('iteration')}")
    print(f"    Stopping Cause: {final_state.get('stop_reason')}")

    results = final_state.get("test_results") or {}
    print(f"    Pytest Results: {results.get('passed', 0)} passed, {results.get('failed', 0)} failed, {results.get('errors', 0)} errors")
    print(f"    Branch Coverage: {results.get('coverage_percent', 0.0):.1f}%")

    saved_test_file = out_dir / "generated_tests.py"
    if saved_test_file.exists():
        print(f"    Saved Test Suite: {saved_test_file}")

    # ---------------------------------------------------------
    # STAGE 4: CLOSED-LOOP FEEDBACK RECORDING
    # ---------------------------------------------------------
    print("\n[STEP 4/4] Ingesting test execution results into Closed-Loop Feedback...")
    rec_count = record_outcomes(candidates, results)
    print(f"[*] Recorded {rec_count} dynamic ground-truth labels to data/processed/feedback_labels.csv")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE: REPOSITORY TESTED & TEST CASES GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeSentinel End-to-End Demo on Any Repository")
    parser.add_argument("repo", help="Path to any local Python Git repository")
    parser.add_argument("--max-commits", type=int, default=50, help="Max commits to mine (default: 50)")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max agent self-correction iterations (default: 2)")
    parser.add_argument("--num-targets", type=int, default=2, help="Number of functions to generate tests for (default: 2)")
    args = parser.parse_args()

    run_end_to_end(
        repo_path_str=args.repo,
        max_commits=args.max_commits,
        max_iterations=args.max_iterations,
        num_targets=args.num_targets,
    )
