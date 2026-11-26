import re
import random
from .types import FaultType


class ToolExecutionError(Exception):
    def __init__(self, fault_type: FaultType, message: str, retriable: bool, field_hint: str | None = None):
        super().__init__(message)
        self.fault_type = fault_type
        self.message = message
        self.retriable = retriable
        self.field_hint = field_hint


class FaultInjector:
    def __init__(self, seed: int = 42, fault_schedule: dict[str, str] | None = None):
        self.rng = random.Random(seed)
        # 支持 "timeout" / "rate_limit" / "invalid_argument" / "phantom_timeout" / "persistent_invalid_argument"
        self.fault_schedule = fault_schedule or {}
        self._hit_counts: dict[str, int] = {}

    def maybe_trigger(self, tool_name: str, args: dict) -> None:
        raw_fault = self.fault_schedule.get(tool_name, "none")
        if not raw_fault or raw_fault == "none":
            return

        key = f"{tool_name}:{raw_fault}"
        hits = self._hit_counts.get(key, 0)

        if raw_fault == "timeout":
            if hits == 0:
                self._hit_counts[key] = hits + 1
                raise ToolExecutionError(
                    fault_type=FaultType.TIMEOUT,
                    message=f"Simulated timeout (504) on tool '{tool_name}'.",
                    retriable=True,
                )

        elif raw_fault == "phantom_timeout":
            # 第一次调用时先让工具执行写入，再抛超时（由 tools.py 配合先写入状态）
            if hits == 0:
                self._hit_counts[key] = hits + 1
                raise ToolExecutionError(
                    fault_type=FaultType.TIMEOUT,
                    message=f"Simulated phantom timeout (504) after commit on tool '{tool_name}'. State may already exist.",
                    retriable=True,
                )

        elif raw_fault == "rate_limit":
            if hits == 0:
                self._hit_counts[key] = hits + 1
                raise ToolExecutionError(
                    fault_type=FaultType.RATE_LIMIT,
                    message=f"Simulated rate_limit (429) on tool '{tool_name}'.",
                    retriable=True,
                )

        elif raw_fault == "invalid_argument":
            due_date = str(args.get("due_date", ""))
            is_valid_date = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date))
            if not is_valid_date and hits == 0:
                self._hit_counts[key] = hits + 1
                raise ToolExecutionError(
                    fault_type=FaultType.INVALID_ARGUMENT,
                    message=f"Invalid arguments for '{tool_name}' with payload {args}.",
                    retriable=False,
                    field_hint="due_date must match YYYY-MM-DD (e.g. 2026-11-30), not relative phrases.",
                )

        elif raw_fault == "persistent_invalid_argument":
            # 连续最多触发 3 次，用来压测有界修复上限与止损成本
            if hits < 3:
                self._hit_counts[key] = hits + 1
                raise ToolExecutionError(
                    fault_type=FaultType.INVALID_ARGUMENT,
                    message=f"Persistent schema rejection ({hits + 1}/3) on '{tool_name}' with payload {args}.",
                    retriable=False,
                    field_hint="Check category enum [hardware, access, expense] and YYYY-MM-DD due_date.",
                )