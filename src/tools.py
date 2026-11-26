import json
from typing import Any
from .faults import FaultInjector, ToolExecutionError


class CompanyAssistantEnv:
    def __init__(self, injector: FaultInjector):
        self.injector = injector
        self.tickets: list[dict[str, Any]] = []
        self.side_effect_count: int = 0
        self._seen_keys: set[str] = set()

        self.employees = {
            "E101": {"name": "Alice", "dept": "Platform", "cost_center": "CC-01"},
            "E102": {"name": "Bob", "dept": "Data", "cost_center": "CC-02"},
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_employee",
                    "description": "Look up employee profile and cost center by employee_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "string", "description": "e.g. E101"}
                        },
                        "required": ["employee_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_support_ticket",
                    "description": "Create an IT or expense ticket for an employee.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "string"},
                            "category": {"type": "string", "enum": ["hardware", "access", "expense"]},
                            "due_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                            "summary": {"type": "string"},
                        },
                        "required": ["employee_id", "category", "due_date", "summary"],
                    },
                },
            },
        ]

    def execute(self, name: str, raw_args: str, allow_idempotency_guard: bool = False) -> tuple[bool, str]:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return False, f"JSONParseError: {e}"

        scheduled_fault = self.injector.fault_schedule.get(name, "none")

        # 模拟“服务端已实际写入，但返回前超时”的幽灵超时场景
        if name == "create_support_ticket" and scheduled_fault == "phantom_timeout":
            fault_key = f"{name}:phantom_timeout"
            if self.injector._hit_counts.get(fault_key, 0) == 0:
                emp_id = args.get("employee_id", "")
                category = args.get("category", "")
                due_date = args.get("due_date", "")
                summary = args.get("summary", "")
                if emp_id in self.employees and category in {"hardware", "access", "expense"} and len(due_date) == 10:
                    idem_key = f"{emp_id}:{category}:{summary}"
                    if idem_key not in self._seen_keys:
                        self._seen_keys.add(idem_key)
                        self.tickets.append({"id": f"T-{len(self.tickets)+1}", **args, "committed_before_timeout": True})
            self.injector.maybe_trigger(name, args)

        self.injector.maybe_trigger(name, args)

        if name == "get_employee":
            emp_id = args.get("employee_id", "")
            if emp_id not in self.employees:
                raise ToolExecutionError(
                    fault_type=__import__("src.types", fromlist=["FaultType"]).FaultType.INVALID_ARGUMENT,
                    message=f"Unknown employee_id '{emp_id}'. Valid IDs: {list(self.employees.keys())}.",
                    retriable=False,
                    field_hint="employee_id must be one of E101, E102.",
                )
            return True, json.dumps(self.employees[emp_id])

        if name == "create_support_ticket":
            emp_id = args.get("employee_id", "")
            category = args.get("category", "")
            due_date = args.get("due_date", "")
            summary = args.get("summary", "")

            if emp_id not in self.employees or category not in {"hardware", "access", "expense"} or len(due_date) != 10:
                raise ToolExecutionError(
                    fault_type=__import__("src.types", fromlist=["FaultType"]).FaultType.INVALID_ARGUMENT,
                    message="Validation failed: check employee_id, category enum, and YYYY-MM-DD due_date.",
                    retriable=False,
                    field_hint="due_date must be YYYY-MM-DD; category in [hardware, access, expense].",
                )

            idem_key = f"{emp_id}:{category}:{summary}"
            if idem_key in self._seen_keys:
                if not allow_idempotency_guard:
                    self.side_effect_count += 1
                    ticket = {"id": f"T-{len(self.tickets)+1}", **args, "duplicate_side_effect": True}
                    self.tickets.append(ticket)
                    return True, json.dumps({"status": "created_duplicate", "ticket": ticket})
                return False, f"IdempotencyGuardBlocked: duplicate ticket request for '{idem_key}' skipped."

            self._seen_keys.add(idem_key)
            ticket = {"id": f"T-{len(self.tickets)+1}", **args}
            self.tickets.append(ticket)
            return True, json.dumps({"status": "created", "ticket": ticket})

        return False, f"UnknownToolError: no tool named '{name}'."