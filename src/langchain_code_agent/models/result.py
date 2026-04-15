from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_code_agent.models.plan import Plan


@dataclass(slots=True)
class ErrorContext:
    error_type: str
    message: str
    action: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    step_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FileChange:
    path: str
    change_type: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StepExecutionResult:
    action: str
    status: str
    ok: bool
    arguments: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_context: ErrorContext | None = None
    file_changes: list[FileChange] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.error_context is not None:
            payload["error_context"] = self.error_context.to_dict()
        payload["file_changes"] = [change.to_dict() for change in self.file_changes]
        return payload


@dataclass(slots=True)
class RunEvent:
    event_type: str
    level: str
    message: str
    action: str | None = None
    step_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error_context: ErrorContext | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.error_context is not None:
            payload["error_context"] = self.error_context.to_dict()
        return payload


@dataclass(slots=True)
class FinalReport:
    success: bool
    task_input: dict[str, Any]
    plan_summary: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    planned_steps: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    shell_outputs: list[dict[str, Any]] = field(default_factory=list)
    file_changes: list[FileChange] = field(default_factory=list)
    errors: list[ErrorContext] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "task_input": self.task_input,
            "plan_summary": self.plan_summary,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "planned_steps": self.planned_steps,
            "tool_calls": self.tool_calls,
            "shell_outputs": self.shell_outputs,
            "file_changes": [change.to_dict() for change in self.file_changes],
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(slots=True)
class RunResult:
    task: str
    workspace_root: str
    execution_mode: str
    planner: str
    plan: Plan
    events: list[RunEvent]
    step_results: list[StepExecutionResult]
    final_report: FinalReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "workspace_root": self.workspace_root,
            "execution_mode": self.execution_mode,
            "planner": self.planner,
            "plan": self.plan.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "step_results": [result.to_dict() for result in self.step_results],
            "final_report": self.final_report.to_dict(),
        }
