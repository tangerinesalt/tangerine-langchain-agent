from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_code_agent.actions import ActionRuntime
from langchain_code_agent.agent.completion_validator import validate_completion
from langchain_code_agent.agent.planner import build_planner
from langchain_code_agent.agent.run_reporter import (
    RunReporter,
    build_final_report,
    extract_shell_output,
    summarize_tool_data,
)
from langchain_code_agent.agent.step_executor import StepExecutor
from langchain_code_agent.config import AgentConfig
from langchain_code_agent.models.plan import Plan
from langchain_code_agent.models.result import (
    ErrorContext,
    FileChange,
    FinalReport,
    RunEvent,
    RunResult,
    StepExecutionResult,
)
from langchain_code_agent.models.task import Task
from langchain_code_agent.workspace.repository import Repository

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.repository = Repository(config.workspace_root, config.ignore_patterns)
        self.action_runtime = ActionRuntime(
            repository=self.repository,
            workspace_root=config.workspace_root,
            shell_timeout_seconds=config.shell_timeout_seconds,
            allowed_shell_commands=list(config.allowed_shell_commands),
            test_command=config.test_command,
        )
        self.step_executor = StepExecutor(self.action_runtime)
        self.reporter = RunReporter(logger)
        self.planner = build_planner(config)

    def run(self, task_text: str, *, execution_mode: str) -> RunResult:
        task = Task(
            goal=task_text,
            workspace_root=self.config.workspace_root,
            execution_mode=execution_mode,
        )
        events: list[RunEvent] = []
        self.reporter.record_event(
            events,
            event_type="task_received",
            level="INFO",
            message=f"Starting run in {execution_mode} mode.",
            details={
                "task": task.goal,
                "workspace_root": str(task.workspace_root),
                "execution_mode": execution_mode,
            },
        )
        try:
            plan = self.planner.create_plan(task)
        except Exception as exc:
            planning_error_context = ErrorContext(
                error_type=type(exc).__name__,
                message=str(exc),
            )
            self.reporter.record_event(
                events,
                event_type="planning_failed",
                level="ERROR",
                message="Planner failed to create a plan.",
                error_context=planning_error_context,
            )
            failed_result = RunResult(
                task=task.goal,
                workspace_root=str(task.workspace_root),
                execution_mode=task.execution_mode,
                planner=self.config.planner_backend,
                plan=_fallback_plan(),
                events=events,
                step_results=[],
                final_report=FinalReport(
                    success=False,
                    task_input=_task_input(task, self.config.planner_backend),
                    plan_summary="Planning failed before any step was created.",
                    total_steps=0,
                    successful_steps=0,
                    failed_steps=1,
                    planned_steps=0,
                    errors=[planning_error_context],
                ),
            )
            self.reporter.record_event(
                events,
                event_type="run_completed",
                level="ERROR",
                message="Run completed with planning failure.",
                details={"success": False, "total_steps": 0, "failed_steps": 1},
                error_context=planning_error_context,
            )
            failed_result.final_report = build_final_report(
                failed_result,
                task_input=_task_input(task, self.config.planner_backend),
                extra_errors=[planning_error_context],
            )
            return failed_result
        self.reporter.record_event(
            events,
            event_type="plan_created",
            level="INFO",
            message=f"Plan created with {len(plan.steps)} steps.",
            details=plan.to_dict(),
        )
        results: list[StepExecutionResult] = []

        for index, step in enumerate(plan.steps, start=1):
            self.reporter.record_event(
                events,
                event_type="step_started",
                level="INFO",
                message=step.description,
                action=step.action,
                step_index=index,
                details={"arguments": dict(step.arguments)},
            )
            if execution_mode == "dry-run":
                results.append(
                    StepExecutionResult(
                        action=step.action,
                        status="planned",
                        ok=True,
                        arguments=dict(step.arguments),
                        data={"arguments": step.arguments, "description": step.description},
                        completed_at=datetime.now(UTC).isoformat(),
                    )
                )
                self.reporter.record_event(
                    events,
                    event_type="step_skipped",
                    level="INFO",
                    message="Step recorded without execution.",
                    action=step.action,
                    step_index=index,
                    details={"mode": "dry-run"},
                )
                continue

            step_started_at = datetime.now(UTC).isoformat()
            before_state = self.repository.snapshot_file_state()
            tool_result, error_context = self.step_executor.execute_step(
                step.action,
                step.arguments,
                index,
            )
            after_state = self.repository.snapshot_file_state()
            file_changes = _diff_file_states(before_state, after_state)
            if file_changes:
                self.reporter.record_event(
                    events,
                    event_type="file_changes_detected",
                    level="INFO",
                    message=f"Detected {len(file_changes)} file changes.",
                    action=step.action,
                    step_index=index,
                    details={"file_changes": [change.to_dict() for change in file_changes]},
                )
            results.append(
                StepExecutionResult(
                    action=step.action,
                    status="completed" if tool_result.ok else "failed",
                    ok=tool_result.ok,
                    arguments=dict(step.arguments),
                    data=tool_result.data,
                    error=tool_result.error,
                    error_context=error_context,
                    file_changes=file_changes,
                    started_at=step_started_at,
                    completed_at=datetime.now(UTC).isoformat(),
                )
            )
            self.reporter.record_event(
                events,
                event_type="tool_called",
                level="INFO",
                message="Tool invocation finished.",
                action=step.action,
                step_index=index,
                details={
                    "arguments": dict(step.arguments),
                    "ok": tool_result.ok,
                    "data": summarize_tool_data(tool_result.data),
                },
                error_context=error_context,
            )
            shell_output = extract_shell_output(step.action, tool_result.data, index)
            if shell_output is not None:
                self.reporter.record_event(
                    events,
                    event_type="shell_output_captured",
                    level="INFO",
                    message="Captured shell output.",
                    action=step.action,
                    step_index=index,
                    details=shell_output,
                )
            self.reporter.record_event(
                events,
                event_type="step_completed" if tool_result.ok else "step_failed",
                level="INFO" if tool_result.ok else "ERROR",
                message=(
                    "Step completed."
                    if tool_result.ok
                    else f"Step failed: {tool_result.error}"
                ),
                action=step.action,
                step_index=index,
                details={"ok": tool_result.ok},
                error_context=error_context,
            )
        run_result = RunResult(
            task=task.goal,
            workspace_root=str(task.workspace_root),
            execution_mode=task.execution_mode,
            planner=self.config.planner_backend,
            plan=plan,
            events=events,
            step_results=results,
            final_report=FinalReport(
                success=False,
                task_input=_task_input(task, self.config.planner_backend),
                plan_summary=plan.summary,
                total_steps=0,
                successful_steps=0,
                failed_steps=0,
                planned_steps=0,
            ),
        )
        run_result.final_report = build_final_report(run_result)
        completion_errors = validate_completion(run_result)
        if completion_errors:
            run_result.final_report.errors.extend(completion_errors)
            run_result.final_report.success = False
            self.reporter.record_event(
                events,
                event_type="completion_validation_failed",
                level="ERROR",
                message="Run finished without the expected material output.",
                details={"errors": [error.to_dict() for error in completion_errors]},
                error_context=completion_errors[0],
            )
        self.reporter.record_event(
            events,
            event_type="run_completed",
            level="INFO" if run_result.final_report.success else "ERROR",
            message="Run completed.",
            details={
                "success": run_result.final_report.success,
                "total_steps": run_result.final_report.total_steps,
                "failed_steps": run_result.final_report.failed_steps,
            },
        )
        return run_result


def _diff_file_states(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> list[FileChange]:
    changes: list[FileChange] = []
    for path in sorted(set(before) | set(after)):
        before_state = before.get(path)
        after_state = after.get(path)
        if before_state is None and after_state is not None:
            changes.append(FileChange(path=path, change_type="added", after=after_state))
        elif before_state is not None and after_state is None:
            changes.append(FileChange(path=path, change_type="deleted", before=before_state))
        elif before_state != after_state and before_state is not None and after_state is not None:
            changes.append(
                FileChange(
                    path=path,
                    change_type="modified",
                    before=before_state,
                    after=after_state,
                )
            )
    return changes
def _task_input(task: Task, planner_backend: str) -> dict[str, Any]:
    return {
        "task": task.goal,
        "workspace_root": str(task.workspace_root),
        "execution_mode": task.execution_mode,
        "planner_backend": planner_backend,
    }


def _fallback_plan() -> Plan:
    return Plan(summary="Planning failed.", steps=[])
