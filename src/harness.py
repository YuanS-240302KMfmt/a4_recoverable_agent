import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from .agent import MinimalRecoverableAgent, TaskRunResult
from .policies import BoundedRepairPolicy, RecoveryPolicy, SimpleRetryPolicy, TypedFeedbackPolicy
from .types import FailureCategory, PolicyName, TaskSpec


def build_policies() -> list[RecoveryPolicy]:
    return [
        SimpleRetryPolicy(),
        TypedFeedbackPolicy(),
        BoundedRepairPolicy(),
    ]


def load_tasks(path: str) -> list[TaskSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TaskSpec(**item) for item in data]


def print_summary(results: list[TaskRunResult], log_path: Path) -> None:
    by_policy: dict[PolicyName, list[TaskRunResult]] = defaultdict(list)
    for r in results:
        by_policy[r.policy].append(r)

    print(f"\n=== Replay Log Saved: {log_path} ===")
    print(
        f"{'Policy':<18} | {'RecoveryRate':>12} | {'AvgSuccessCost($)':>17} | "
        f"{'SideEffects':>11} | {'FailureBreakdown'}"
    )
    print("-" * 105)

    for policy_name, items in by_policy.items():
        total = len(items)
        successes = [x for x in items if x.success]
        recovery_rate = (len(successes) / total * 100) if total else 0.0
        avg_success_cost = (
            sum(x.cost_usd for x in successes) / len(successes) if successes else 0.0
        )
        side_effects = sum(x.side_effects for x in items)

        fail_counter = Counter(
            x.failure_category.value
            for x in items
            if x.failure_category != FailureCategory.NONE
        )
        fail_str = dict(fail_counter) if fail_counter else "{}"

        print(
            f"{policy_name.value:<18} | {recovery_rate:>11.1f}% | {avg_success_cost:>17.6f} | "
            f"{side_effects:>11d} | {fail_str}"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="A4 Recoverable Agent Evaluation Harness")
    parser.add_argument("--tasks", default="tasks/sample_tasks.json", help="Path to tasks JSON")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100], help="Random seeds")
    parser.add_argument("--out-dir", default="runs", help="Directory for replay JSONL logs")
    args = parser.parse_args()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.jsonl"

    tasks = load_tasks(args.tasks)
    policies = build_policies()
    all_results: list[TaskRunResult] = []

    with log_path.open("w", encoding="utf-8") as f:
        for policy in policies:
            agent = MinimalRecoverableAgent(policy=policy)
            for seed in args.seeds:
                for task in tasks:
                    res = agent.run(run_id=run_id, task=task, seed=seed)
                    all_results.append(res)
                    for ev in res.events:
                        f.write(ev.model_dump_json() + "\n")

    print_summary(all_results, log_path)


if __name__ == "__main__":
    main()