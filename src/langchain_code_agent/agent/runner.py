from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_code_agent.actions import ActionRuntime
from langchain_code_agent.agent.completion_validator import validate_completion
from langchain_code_agent.agent.plan_validator import validate_plan, validate_task_specific_plan
from langchain_code_agent.agent.planner import build_planner
from langchain_code_agent.agent.replan_context import build_replan_context
from langchain_code_agent.agent.run_reporter import (
    RunReporter,
    build_final_report,
    extract_shell_output,
    summarize_tool_data,
)
from langchain_code_agent.agent.step_executor import StepExecutor
from langchain_code_agent.agent_config import AgentConfig
from langchain_code_agent.models.plan import Plan
from langchain_code_agent.models.replan import ReplanContext
from langchain_code_agent.models.result import (
    AttemptResult,
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
        run_id = uuid4().hex
        run_started_at = datetime.now(UTC).isoformat()
        run_started = time.perf_counter()
        self.reporter.current_run_id = run_id
        events: list[RunEvent] = []
        final_result: RunResult | None = None
        attempts: list[AttemptResult] = []
        replan_context = None
        self.reporter.record_event(
            events,
            event_type="task_received",
            level="INFO",
            message=f"Starting run in {execution_mode} mode.",
            details={
                "task": task_text,
                "workspace_root": str(self.config.workspace_root),
                "execution_mode": execution_mode,
            },
        )
        max_attempts = 1 if execution_mode == "dry-run" else self.config.max_replans + 1
        for attempt in range(1, max_attempts + 1):
            self.reporter.record_event(
                events,
                event_type="attempt_started",
                level="INFO",
                message=f"Starting attempt {attempt}.",
                details={"attempt": attempt, "task": task_text},
            )
            attempt_started = time.perf_counter()
            final_result = self._run_single_attempt(
                task_text,
                execution_mode=execution_mode,
                events=events,
                attempt=attempt,
                replan_context=replan_context,
                run_id=run_id,
                run_started_at=run_started_at,
            )
            attempt_duration_ms = _elapsed_ms(attempt_started)
            attempts.append(
                _attempt_result_from_run_result(
                    final_result,
                    attempt,
                    duration_ms=attempt_duration_ms,
                )
            )
            self.reporter.record_event(
                events,
                event_type="attempt_completed",
                level="INFO" if final_result.final_report.success else "ERROR",
                message=f"Attempt {attempt} completed.",
                details={
                    "attempt": attempt,
                    "success": final_result.final_report.success,
                    "duration_ms": attempt_duration_ms,
                },
            )
            if final_result.final_report.success or attempt == max_attempts:
                break

            replan_context = build_replan_context(task_text, attempts[-1])
            self.reporter.record_event(
                events,
                event_type="replan_requested",
                level="ERROR",
                message="Attempt failed; requesting one more plan.",
                details={
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "replan_context": replan_context.to_dict(),
                    "errors": [error.to_dict() for error in final_result.final_report.errors],
                },
                error_context=final_result.final_report.errors[0]
                if final_result.final_report.errors
                else None,
            )

        assert final_result is not None
        final_result.task = task_text
        final_result.run_id = run_id
        final_result.started_at = run_started_at
        final_result.attempts = attempts
        final_result.selected_attempt = attempts[-1].attempt if attempts else None
        final_result.final_report.task_input = _task_input(
            Task(
                goal=task_text,
                workspace_root=self.config.workspace_root,
                execution_mode=execution_mode,
            ),
            self.config.planner_backend,
        )
        run_duration_ms = _elapsed_ms(run_started)
        completed_at = datetime.now(UTC).isoformat()
        artifact_path = _run_artifact_path(self.config.workspace_root, run_id)
        final_result.completed_at = completed_at
        final_result.duration_ms = run_duration_ms
        final_result.artifact_path = str(artifact_path)
        final_result.final_report.run_id = run_id
        final_result.final_report.duration_ms = run_duration_ms
        final_result.final_report.artifact_path = str(artifact_path)
        self.reporter.record_event(
            events,
            event_type="run_completed",
            level="INFO" if final_result.final_report.success else "ERROR",
            message="Run completed.",
            details={
                "success": final_result.final_report.success,
                "total_steps": final_result.final_report.total_steps,
                "failed_steps": final_result.final_report.failed_steps,
                "attempts": final_result.final_report.attempts,
                "duration_ms": run_duration_ms,
                "artifact_path": str(artifact_path),
            },
        )
        _write_run_artifact(final_result, artifact_path)
        return final_result

    def _run_single_attempt(
        self,
        task_text: str,
        *,
        execution_mode: str,
        events: list[RunEvent],
        attempt: int,
        replan_context: ReplanContext | None,
        run_id: str,
        run_started_at: str,
    ) -> RunResult:
        task = Task(
            goal=task_text,
            workspace_root=self.config.workspace_root,
            execution_mode=execution_mode,
            replan_context=replan_context,
        )
        planning_started = time.perf_counter()
        planning_stage = "planner_call"
        planner_output: dict[str, Any] | None = None
        try:
            plan = self.planner.create_plan(task)
            planner_output = plan.to_dict()
            planning_stage = "validate_plan"
            plan = validate_plan(plan, existing_paths=set(self.repository.snapshot_file_state()))
            planning_stage = "validate_task_specific_plan"
            plan = validate_task_specific_plan(plan, task_text=task.goal)
        except Exception as exc:
            planning_duration_ms = _elapsed_ms(planning_started)
            planning_error_context = ErrorContext(
                error_type=type(exc).__name__,
                message=str(exc),
                stage=planning_stage,
                traceback=traceback.format_exc(),
            )
            self.reporter.record_event(
                events,
                event_type="planning_failed",
                level="ERROR",
                message="Planner failed to create a plan.",
                details={
                    "attempt": attempt,
                    "stage": planning_stage,
                    "duration_ms": planning_duration_ms,
                    "planner_output": planner_output,
                    "error_type": type(exc).__name__,
                },
                error_context=planning_error_context,
            )
            failed_result = RunResult(
                run_id=run_id,
                task=task.goal,
                workspace_root=str(task.workspace_root),
                execution_mode=task.execution_mode,
                planner=self.config.planner_backend,
                plan=_fallback_plan(),
                events=events,
                step_results=[],
                final_report=FinalReport(
                    success=False,
                    run_id=run_id,
                    task_input=_task_input(task, self.config.planner_backend),
                    plan_summary="Planning failed before any step was created.",
                    total_steps=0,
                    successful_steps=0,
                    failed_steps=1,
                    planned_steps=0,
                    attempts=attempt,
                    errors=[planning_error_context],
                ),
                started_at=run_started_at,
            )
            failed_result.final_report = build_final_report(
                failed_result,
                task_input=_task_input(task, self.config.planner_backend),
                extra_errors=[planning_error_context],
            )
            failed_result.final_report.attempts = attempt
            return failed_result

        planning_duration_ms = _elapsed_ms(planning_started)
        self.reporter.record_event(
            events,
            event_type="plan_created",
            level="INFO",
            message=f"Plan created with {len(plan.steps)} steps.",
            details={
                "attempt": attempt,
                "duration_ms": planning_duration_ms,
                "planner_output": planner_output,
                "validated_plan": plan.to_dict(),
            },
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
                details={"attempt": attempt, "arguments": dict(step.arguments)},
            )
            if execution_mode == "dry-run":
                step_completed_at = datetime.now(UTC).isoformat()
                results.append(
                    StepExecutionResult(
                        attempt=attempt,
                        action=step.action,
                        status="planned",
                        ok=True,
                        arguments=dict(step.arguments),
                        data={"arguments": step.arguments, "description": step.description},
                        completed_at=step_completed_at,
                        duration_ms=0,
                    )
                )
                self.reporter.record_event(
                    events,
                    event_type="step_skipped",
                    level="INFO",
                    message="Step recorded without execution.",
                    action=step.action,
                    step_index=index,
                    details={"attempt": attempt, "mode": "dry-run"},
                )
                continue

            step_started_at = datetime.now(UTC).isoformat()
            step_started = time.perf_counter()
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
                    details={
                        "attempt": attempt,
                        "file_changes": [change.to_dict() for change in file_changes],
                    },
                )
            step_duration_ms = _elapsed_ms(step_started)
            results.append(
                StepExecutionResult(
                    attempt=attempt,
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
                    duration_ms=step_duration_ms,
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
                    "attempt": attempt,
                    "arguments": dict(step.arguments),
                    "ok": tool_result.ok,
                    "duration_ms": step_duration_ms,
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
                    details={"attempt": attempt, **shell_output},
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
                details={"attempt": attempt, "ok": tool_result.ok},
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
                run_id=run_id,
                task_input=_task_input(task, self.config.planner_backend),
                plan_summary=plan.summary,
                total_steps=0,
                successful_steps=0,
                failed_steps=0,
                planned_steps=0,
                attempts=attempt,
            ),
            run_id=run_id,
            started_at=run_started_at,
        )
        run_result.final_report = build_final_report(run_result)
        run_result.final_report.attempts = attempt
        completion_errors = validate_completion(run_result)
        if completion_errors:
            run_result.final_report.errors.extend(completion_errors)
            run_result.final_report.success = False
            self.reporter.record_event(
                events,
                event_type="completion_validation_failed",
                level="ERROR",
                message="Run finished without the expected material output.",
                details={
                    "attempt": attempt,
                    "errors": [error.to_dict() for error in completion_errors],
                },
                error_context=completion_errors[0],
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


def _attempt_result_from_run_result(
    run_result: RunResult,
    attempt: int,
    *,
    duration_ms: int | None = None,
) -> AttemptResult:
    completion_errors = [
        error
        for error in run_result.final_report.errors
        if error.error_type == "IncompleteTaskResult"
    ]
    return AttemptResult(
        attempt=attempt,
        task=run_result.task,
        plan=run_result.plan,
        step_results=list(run_result.step_results),
        success=run_result.final_report.success,
        errors=list(run_result.final_report.errors),
        completion_errors=completion_errors,
        duration_ms=duration_ms,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _run_artifact_path(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / ".lca" / "runs" / run_id / "result.json"


def _write_run_artifact(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
