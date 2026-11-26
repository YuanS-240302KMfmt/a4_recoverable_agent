import time
from .faults import ToolExecutionError
from .types import FaultType, PolicyName


class RecoveryPolicy:
    name: PolicyName

    def handle_tool_error(
        self,
        err: ToolExecutionError,
        repair_attempts: int,
        max_repair_attempts: int = 2,
    ) -> tuple[str, bool]:
        """
        返回: (喂回给模型的 tool message 内容, 是否允许继续/重试)
        """
        raise NotImplementedError


class SimpleRetryPolicy(RecoveryPolicy):
    name = PolicyName.SIMPLE_RETRY

    def handle_tool_error(self, err: ToolExecutionError, repair_attempts: int, max_repair_attempts: int = 2):
        # 模糊报错 + 缺乏区分，容易在参数错误上反复撞墙或触发重复副作用
        return f"Tool failed: {err.message}. Please retry.", True


class TypedFeedbackPolicy(RecoveryPolicy):
    name = PolicyName.TYPED_FEEDBACK

    def handle_tool_error(self, err: ToolExecutionError, repair_attempts: int, max_repair_attempts: int = 2):
        msg = (
            f"[ErrorType={err.fault_type.value} | Retriable={err.retriable}] "
            f"{err.message}"
            + (f" Hint: {err.field_hint}" if err.field_hint else "")
        )
        return msg, True


class BoundedRepairPolicy(RecoveryPolicy):
    name = PolicyName.BOUNDED_REPAIR

    def handle_tool_error(self, err: ToolExecutionError, repair_attempts: int, max_repair_attempts: int = 2):
        # 1. 瞬时故障：做短暂退避，不浪费过多推理轮次
        if err.fault_type in (FaultType.TIMEOUT, FaultType.RATE_LIMIT):
            backoff = min(0.5 * (2 ** min(repair_attempts, 3)), 2.0)
            time.sleep(backoff)
            return (
                f"[TransientFault={err.fault_type.value}] Temporary outage mitigated with backoff; "
                f"you may retry this call once.",
                repair_attempts < max_repair_attempts,
            )

        # 2. 参数/语义错误：有界修复（超过上限直接中止该分支，防止死循环烧 Token）
        if repair_attempts >= max_repair_attempts:
            return (
                f"[BoundedRepairExhausted] Max repair attempts ({max_repair_attempts}) reached for "
                f"{err.fault_type.value}: {err.message}. Do not repeat identical invalid call; report blocker.",
                False,
            )

        return (
            f"[RepairableError={err.fault_type.value} | Attempt {repair_attempts + 1}/{max_repair_attempts}] "
            f"{err.message} Constraint: {err.field_hint or 'fix invalid fields before calling again.'}",
            True,
        )