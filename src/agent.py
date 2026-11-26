import json
import os
from dataclasses import dataclass
from typing import Any
from openai import OpenAI

from .faults import FaultInjector, ToolExecutionError
from .policies import BoundedRepairPolicy, RecoveryPolicy
from .tools import CompanyAssistantEnv
from .types import FailureCategory, PolicyName, ReplayEvent, TaskSpec


# DeepSeek 常见参考单价（USD / 1M tokens），按需在配置中微调以统计成本
INPUT_PRICE_PER_1M = 0.28
OUTPUT_PRICE_PER_1M = 0.42


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * INPUT_PRICE_PER_1M
        + completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M
    )


@dataclass
class TaskRunResult:
    run_id: str
    task_id: str
    policy: PolicyName
    seed: int
    success: bool
    failure_category: FailureCategory
    steps_used: int
    tool_calls_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    side_effects: int
    final_answer: str
    events: list[ReplayEvent]


class MinimalRecoverableAgent:
    def __init__(
        self,
        policy: RecoveryPolicy,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.policy = policy
        self.model = model or os.getenv("MODEL_NAME", "deepseek-chat")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )

    def run(self, run_id: str, task: TaskSpec, seed: int = 42) -> TaskRunResult:
        injector = FaultInjector(seed=seed, fault_schedule=task.injected_faults)
        env = CompanyAssistantEnv(injector=injector)
        events: list[ReplayEvent] = []

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a reliable company assistant. Current reference date is 2026-11-26 (Thursday). "
                    "Use tools accurately to look up data and create tickets. When a relative date like "
                    "'next Monday' or 'next Friday' is given, convert it to YYYY-MM-DD relative to 2026-11-26. "
                    "If a tool returns an error, inspect the feedback, adjust parameters if needed, and avoid "
                    "unwanted duplicate writes."
                ),
            },
            {"role": "user", "content": task.user_prompt},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        tool_calls_used = 0
        repair_counts: dict[str, int] = {}

        success = False
        failure_category = FailureCategory.NONE
        final_answer = ""

        allow_idem_guard = isinstance(self.policy, BoundedRepairPolicy)

        for step in range(1, task.max_steps + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=env.schemas(),
                tool_choice="auto",
                temperature=0.0,
            )

            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0
            step_cost = estimate_cost(p_tok, c_tok)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_cost += step_cost

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            events.append(
                ReplayEvent(
                    run_id=run_id,
                    task_id=task.task_id,
                    policy=self.policy.name,
                    seed=seed,
                    step=step,
                    event_type="llm_call",
                    payload={
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in tool_calls
                        ],
                    },
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    cost_usd=step_cost,
                )
            )

            # 无工具调用：模型认为已完成任务或给出最终回复
            if not tool_calls:
                final_answer = msg.content or ""
                # 若任务要求创建工单，检查是否至少创建过有效工单
                success = len(env.tickets) > 0 or "T1" in task.task_id
                if not success:
                    failure_category = FailureCategory.MODEL_NO_TOOL_PROGRESS
                break

            # 将 assistant 的 tool_calls 历史追加进上下文
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            aborted_by_policy = False
            for tc in tool_calls:
                if tool_calls_used >= task.max_tool_calls:
                    failure_category = FailureCategory.BUDGET_EXCEEDED
                    aborted_by_policy = True
                    break

                tool_calls_used += 1
                fn_name = tc.function.name
                raw_args = tc.function.arguments or "{}"

                events.append(
                    ReplayEvent(
                        run_id=run_id,
                        task_id=task.task_id,
                        policy=self.policy.name,
                        seed=seed,
                        step=step,
                        event_type="tool_call",
                        payload={"tool_call_id": tc.id, "name": fn_name, "arguments": raw_args},
                    )
                )

                try:
                    ok, output = env.execute(
                        name=fn_name,
                        raw_args=raw_args,
                        allow_idempotency_guard=allow_idem_guard,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output if ok else f"Error: {output}",
                        }
                    )
                except ToolExecutionError as err:
                    events.append(
                        ReplayEvent(
                            run_id=run_id,
                            task_id=task.task_id,
                            policy=self.policy.name,
                            seed=seed,
                            step=step,
                            event_type="fault_injected",
                            payload={
                                "tool": fn_name,
                                "fault_type": err.fault_type.value,
                                "retriable": err.retriable,
                                "message": err.message,
                            },
                        )
                    )

                    repair_key = f"{fn_name}:{err.fault_type.value}"
                    attempts = repair_counts.get(repair_key, 0)
                    feedback_msg, should_continue = self.policy.handle_tool_error(
                        err=err,
                        repair_attempts=attempts,
                        max_repair_attempts=2,
                    )
                    repair_counts[repair_key] = attempts + 1

                    events.append(
                        ReplayEvent(
                            run_id=run_id,
                            task_id=task.task_id,
                            policy=self.policy.name,
                            seed=seed,
                            step=step,
                            event_type="policy_action",
                            payload={
                                "policy": self.policy.name.value,
                                "repair_attempt": attempts + 1,
                                "should_continue": should_continue,
                                "feedback": feedback_msg,
                            },
                        )
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": feedback_msg,
                        }
                    )

                    if not should_continue:
                        failure_category = (
                            FailureCategory.PARAM_VALIDATION_FAILED
                            if not err.retriable
                            else FailureCategory.TRANSIENT_EXHAUSTED
                        )
                        aborted_by_policy = True
                        break

            if aborted_by_policy:
                break
        else:
            if not success and failure_category == FailureCategory.NONE:
                failure_category = FailureCategory.BUDGET_EXCEEDED

        # 副作用判定：存在非预期写入或重复工单
        if env.side_effect_count > task.expected_side_effects:
            if failure_category == FailureCategory.NONE:
                failure_category = FailureCategory.SIDE_EFFECT_VIOLATION

        events.append(
            ReplayEvent(
                run_id=run_id,
                task_id=task.task_id,
                policy=self.policy.name,
                seed=seed,
                step=step,
                event_type="final",
                payload={
                    "success": success,
                    "failure_category": failure_category.value,
                    "side_effects": env.side_effect_count,
                    "tickets_created": len(env.tickets),
                    "final_answer": final_answer,
                },
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cost_usd=total_cost,
            )
        )

        return TaskRunResult(
            run_id=run_id,
            task_id=task.task_id,
            policy=self.policy.name,
            seed=seed,
            success=success,
            failure_category=failure_category,
            steps_used=step,
            tool_calls_used=tool_calls_used,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost_usd=total_cost,
            side_effects=env.side_effect_count,
            final_answer=final_answer,
            events=events,
        )