from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ReplanFailedStep:
    action: str
    message: str
    arguments: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    step_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplanContext:
    original_task: str
    attempt: int
    previous_plan_summary: str
    failed_steps: list[ReplanFailedStep] = field(default_factory=list)
    completion_failures: list[str] = field(default_factory=list)
    successful_actions: list[str] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_steps"] = [step.to_dict() for step in self.failed_steps]
        return payload
