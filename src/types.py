from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class PolicyName(str, Enum):
    SIMPLE_RETRY = "simple_retry"
    TYPED_FEEDBACK = "typed_feedback"
    BOUNDED_REPAIR = "bounded_repair"


class FaultType(str, Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    PHANTOM_TIMEOUT = "phantom_timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_ARGUMENT = "invalid_argument"
    PERSISTENT_INVALID_ARGUMENT = "persistent_invalid_argument"


class FailureCategory(str, Enum):
    NONE = "none"
    TRANSIENT_EXHAUSTED = "transient_exhausted"
    PARAM_VALIDATION_FAILED = "param_validation_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    SIDE_EFFECT_VIOLATION = "side_effect_violation"
    MODEL_NO_TOOL_PROGRESS = "model_no_tool_progress"


class TaskSpec(BaseModel):
    task_id: str
    user_prompt: str
    expected_side_effects: int = 0
    max_steps: int = 8
    max_tool_calls: int = 12
    injected_faults: dict[str, FaultType] = Field(default_factory=dict)


class ReplayEvent(BaseModel):
    run_id: str
    task_id: str
    policy: PolicyName
    seed: int
    step: int
    event_type: Literal["llm_call", "tool_call", "fault_injected", "policy_action", "final"]
    payload: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0