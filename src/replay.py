import argparse
import json
from pathlib import Path


def replay(log_path: str, task_id: str, policy: str, seed: int) -> None:
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    matched = [
        json.loads(line)
        for line in lines
        if json.loads(line)["task_id"] == task_id
        and json.loads(line)["policy"] == policy
        and json.loads(line)["seed"] == seed
    ]

    if not matched:
        print("No matching events found.")
        return

    print(f"=== Replay: {log_path} | task={task_id} | policy={policy} | seed={seed} ===")
    for ev in matched:
        step = ev["step"]
        etype = ev["event_type"]
        tok = f"[in={ev['prompt_tokens']}, out={ev['completion_tokens']}, ${ev['cost_usd']:.6f}]" if (ev["prompt_tokens"] or ev["completion_tokens"]) else ""
        print(f"Step {step} | {etype:<15} {tok}")
        print(f"  payload: {json.dumps(ev['payload'], ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a single task trajectory from JSONL log")
    parser.add_argument("--log", required=True, help="Path to runs/*.jsonl")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    replay(args.log, args.task_id, args.policy, args.seed)


if __name__ == "__main__":
    main()