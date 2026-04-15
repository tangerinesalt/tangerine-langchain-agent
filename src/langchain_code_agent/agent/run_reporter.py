from __future__ import annotations

import json
import logging
from typing import Any

from langchain_code_agent.actions import action_produces_shell_output
from langchain_code_agent.models.result import ErrorContext, FileChange, FinalReport, RunEvent, RunResult


class RunReporter:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def record_event(
        self,
        events: list[RunEvent],
        *,
        event_type: str,
        level: str,
        message: str,
        action: str | None = None,
        step_index: int | None = None,
        details: dict[str, Any] | None = None,
        error_context: ErrorContext | None = None,
    ) -> None:
        event = RunEvent(
            event_type=event_type,
            level=level.upper(),
            message=message,
            action=action,
            step_index=step_index,
            details=details or {},
            error_context=error_context,
        )
        self.logger.log(
            getattr(logging, level.upper(), logging.INFO),
            json.dumps(event.to_dict(), ensure_ascii=False),
        )
        events.append(event)


def summarize_tool_data(data: dict[str, Any]) -> dict[str, Any]:
    summary = dict(data)
    if "stdout" in summary and isinstance(summary["stdout"], str):
        summary["stdout_preview"] = summary["stdout"][:500]
        del summary["stdout"]
    if "stderr" in summary and isinstance(summary["stderr"], str):
        summary["stderr_preview"] = summary["stderr"][:500]
        del summary["stderr"]
    if "content" in summary and isinstance(summary["content"], str):
        summary["content_preview"] = summary["content"][:500]
        del summary["content"]
    return summary


def extract_shell_output(
    action: str,
    data: dict[str, Any],
    step_index: int,
) -> dict[str, Any] | None:
    if not action_produces_shell_output(action):
        return None
    return {
        "step_index": step_index,
        "command": data.get("command"),
        "returncode": data.get("returncode"),
        "stdout": data.get("stdout", ""),
        "stderr": data.get("stderr", ""),
        "working_directory": data.get("working_directory"),
    }


def build_final_report(
    run_result: RunResult,
    *,
    task_input: dict[str, Any] | None = None,
    extra_errors: list[ErrorContext] | None = None,
) -> FinalReport:
    tool_calls: list[dict[str, Any]] = []
    shell_outputs: list[dict[str, Any]] = []
    file_changes: list[FileChange] = []
    errors: list[ErrorContext] = list(extra_errors or [])

    for index, step in enumerate(run_result.step_results, start=1):
        tool_calls.append(
            {
                "step_index": index,
                "action": step.action,
                "status": step.status,
                "ok": step.ok,
                "arguments": step.arguments,
            }
        )
        shell_output = extract_shell_output(step.action, step.data, index)
        if shell_output is not None:
            shell_outputs.append(shell_output)
        file_changes.extend(step.file_changes)
        if step.error_context is not None:
            errors.append(step.error_context)

    successful_steps = sum(
        1 for step in run_result.step_results if step.ok and step.status != "planned"
    )
    failed_steps = sum(1 for step in run_result.step_results if not step.ok)
    planned_steps = sum(1 for step in run_result.step_results if step.status == "planned")

    return FinalReport(
        success=failed_steps == 0 and not errors,
        task_input=task_input
        or {
            "task": run_result.task,
            "workspace_root": run_result.workspace_root,
            "execution_mode": run_result.execution_mode,
            "planner_backend": run_result.planner,
        },
        plan_summary=run_result.plan.summary,
        total_steps=len(run_result.step_results),
        successful_steps=successful_steps,
        failed_steps=failed_steps,
        planned_steps=planned_steps,
        tool_calls=tool_calls,
        shell_outputs=shell_outputs,
        file_changes=file_changes,
        errors=errors,
    )
