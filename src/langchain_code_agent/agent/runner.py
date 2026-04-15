from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from langchain_code_agent.agent.planner import build_planner
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
from langchain_code_agent.tools.base import ToolResult
from langchain_code_agent.tools.delete_file import delete_file_tool
from langchain_code_agent.tools.find_files_by_name import find_files_by_name_tool
from langchain_code_agent.tools.glob_files import glob_files_tool
from langchain_code_agent.tools.insert_text import insert_text_tool
from langchain_code_agent.tools.list_files import list_files_tool
from langchain_code_agent.tools.move_file import move_file_tool
from langchain_code_agent.tools.read_file import read_file_tool
from langchain_code_agent.tools.read_file_head import read_file_head_tool
from langchain_code_agent.tools.replace_in_file import replace_in_file_tool
from langchain_code_agent.tools.run_command import run_command_tool
from langchain_code_agent.tools.run_python_script import run_python_script_tool
from langchain_code_agent.tools.run_shell import run_shell_tool
from langchain_code_agent.tools.run_tests import run_tests_tool
from langchain_code_agent.tools.search_text import search_text_tool
from langchain_code_agent.tools.tree_view import tree_view_tool
from langchain_code_agent.tools.write_file import write_file_tool
from langchain_code_agent.workspace.repository import Repository

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.repository = Repository(config.workspace_root, config.ignore_patterns)
        self.planner = build_planner(config)

    def run(self, task_text: str, *, execution_mode: str) -> RunResult:
        task = Task(
            goal=task_text,
            workspace_root=self.config.workspace_root,
            execution_mode=execution_mode,
        )
        events: list[RunEvent] = []
        self._record_event(
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
            self._record_event(
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
            self._record_event(
                events,
                event_type="run_completed",
                level="ERROR",
                message="Run completed with planning failure.",
                details={"success": False, "total_steps": 0, "failed_steps": 1},
                error_context=planning_error_context,
            )
            failed_result.final_report = _build_final_report(
                failed_result,
                task_input=_task_input(task, self.config.planner_backend),
                extra_errors=[planning_error_context],
            )
            return failed_result
        self._record_event(
            events,
            event_type="plan_created",
            level="INFO",
            message=f"Plan created with {len(plan.steps)} steps.",
            details=plan.to_dict(),
        )
        results: list[StepExecutionResult] = []

        for index, step in enumerate(plan.steps, start=1):
            self._record_event(
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
                self._record_event(
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
            tool_result, error_context = self._execute_step(step.action, step.arguments, index)
            after_state = self.repository.snapshot_file_state()
            file_changes = _diff_file_states(before_state, after_state)
            if file_changes:
                self._record_event(
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
            self._record_event(
                events,
                event_type="tool_called",
                level="INFO",
                message="Tool invocation finished.",
                action=step.action,
                step_index=index,
                details={
                    "arguments": dict(step.arguments),
                    "ok": tool_result.ok,
                    "data": _summarize_tool_data(tool_result.data),
                },
                error_context=error_context,
            )
            shell_output = _extract_shell_output(step.action, tool_result.data, index)
            if shell_output is not None:
                self._record_event(
                    events,
                    event_type="shell_output_captured",
                    level="INFO",
                    message="Captured shell output.",
                    action=step.action,
                    step_index=index,
                    details=shell_output,
                )
            self._record_event(
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
        run_result.final_report = _build_final_report(run_result)
        self._record_event(
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

    def _execute_step(
        self,
        action: str,
        arguments: dict[str, object],
        step_index: int,
    ) -> tuple[ToolResult, ErrorContext | None]:
        try:
            validation_error = _validate_step_arguments(action, arguments)
            if validation_error is not None:
                error_context = ErrorContext(
                    error_type="InvalidStepArguments",
                    message=validation_error,
                    action=action,
                    arguments=dict(arguments),
                    step_index=step_index,
                )
                return ToolResult(ok=False, error=validation_error), error_context
            if action == "list_files":
                tool_result = list_files_tool(
                    self.repository,
                    limit=_coerce_int(arguments.get("limit"), 200),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "glob_files":
                tool_result = glob_files_tool(
                    self.repository,
                    pattern=str(arguments["pattern"]),
                    limit=_coerce_int(arguments.get("limit"), 200),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "find_files_by_name":
                tool_result = find_files_by_name_tool(
                    self.repository,
                    name=str(arguments["name"]),
                    limit=_coerce_int(arguments.get("limit"), 200),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "tree_view":
                tool_result = tree_view_tool(
                    self.repository,
                    path=str(arguments.get("path", ".")),
                    depth=_coerce_int(arguments.get("depth"), 2),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "read_file":
                tool_result = read_file_tool(self.repository, path=str(arguments["path"]))
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "read_file_head":
                tool_result = read_file_head_tool(
                    self.repository,
                    path=str(arguments["path"]),
                    start_line=_coerce_int(arguments.get("start_line"), 1),
                    max_lines=_coerce_int(arguments.get("max_lines"), 200),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "search_text":
                tool_result = search_text_tool(
                    self.repository,
                    query=str(arguments["query"]),
                    max_results=_coerce_int(arguments.get("max_results"), 20),
                    case_sensitive=bool(arguments.get("case_sensitive", False)),
                    use_regex=bool(arguments.get("use_regex", False)),
                    path_glob=_coerce_optional_str(arguments.get("path_glob")),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "replace_in_file":
                tool_result = replace_in_file_tool(
                    self.repository,
                    path=str(arguments["path"]),
                    old_text=str(arguments["old_text"]),
                    new_text=str(arguments["new_text"]),
                    count=_coerce_int(arguments.get("count"), 1),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "insert_text":
                tool_result = insert_text_tool(
                    self.repository,
                    path=str(arguments["path"]),
                    anchor=str(arguments["anchor"]),
                    text=str(arguments["text"]),
                    position=str(arguments.get("position", "after")),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "delete_file":
                tool_result = delete_file_tool(self.repository, path=str(arguments["path"]))
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "move_file":
                tool_result = move_file_tool(
                    self.repository,
                    source_path=str(arguments["source_path"]),
                    destination_path=str(arguments["destination_path"]),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "run_command":
                argv = arguments.get("argv")
                if not isinstance(argv, list):
                    raise TypeError("run_command expects argv to be a list of strings")
                tool_result = run_command_tool(
                    argv=[str(item) for item in argv],
                    workspace_root=self.config.workspace_root,
                    timeout_seconds=self.config.shell_timeout_seconds,
                    allowed_commands=self.config.allowed_shell_commands,
                    working_directory=_coerce_optional_str(arguments.get("working_directory")),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "run_python_script":
                tool_result = run_python_script_tool(
                    script=str(arguments["script"]),
                    workspace_root=self.config.workspace_root,
                    timeout_seconds=self.config.shell_timeout_seconds,
                    allowed_commands=self.config.allowed_shell_commands,
                    working_directory=_coerce_optional_str(arguments.get("working_directory")),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "run_shell":
                tool_result = run_shell_tool(
                    command=str(arguments["command"]),
                    workspace_root=self.config.workspace_root,
                    timeout_seconds=self.config.shell_timeout_seconds,
                    allowed_commands=self.config.allowed_shell_commands,
                    working_directory=_coerce_optional_str(arguments.get("working_directory")),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "run_tests":
                tool_result = run_tests_tool(
                    test_command=self.config.test_command,
                    workspace_root=self.config.workspace_root,
                    timeout_seconds=self.config.shell_timeout_seconds,
                    allowed_commands=self.config.allowed_shell_commands,
                    working_directory=_coerce_optional_str(arguments.get("working_directory")),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            if action == "write_file":
                tool_result = write_file_tool(
                    self.repository,
                    path=str(arguments["path"]),
                    content=str(arguments["content"]),
                    overwrite=bool(arguments.get("overwrite", False)),
                )
                return (
                    tool_result,
                    _tool_error_context(tool_result, action, arguments, step_index),
                )
            error_context = ErrorContext(
                error_type="UnsupportedActionError",
                message=f"Unsupported action: {action}",
                action=action,
                arguments=dict(arguments),
                step_index=step_index,
            )
            return ToolResult(ok=False, error=error_context.message), error_context
        except Exception as exc:
            error_context = ErrorContext(
                error_type=type(exc).__name__,
                message=str(exc),
                action=action,
                arguments=dict(arguments),
                step_index=step_index,
            )
            return ToolResult(ok=False, error=str(exc)), error_context

    def _record_event(
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
        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            json.dumps(event.to_dict(), ensure_ascii=False),
        )
        events.append(event)


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got: {type(value)!r}")


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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


def _extract_shell_output(
    action: str,
    data: dict[str, Any],
    step_index: int,
) -> dict[str, Any] | None:
    if action not in {"run_shell", "run_tests", "run_command", "run_python_script"}:
        return None
    return {
        "step_index": step_index,
        "command": data.get("command"),
        "returncode": data.get("returncode"),
        "stdout": data.get("stdout", ""),
        "stderr": data.get("stderr", ""),
        "working_directory": data.get("working_directory"),
    }


def _task_input(task: Task, planner_backend: str) -> dict[str, Any]:
    return {
        "task": task.goal,
        "workspace_root": str(task.workspace_root),
        "execution_mode": task.execution_mode,
        "planner_backend": planner_backend,
    }


def _fallback_plan() -> Plan:
    return Plan(summary="Planning failed.", steps=[])


def _summarize_tool_data(data: dict[str, Any]) -> dict[str, Any]:
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


def _build_final_report(
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
        shell_output = _extract_shell_output(step.action, step.data, index)
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


def _tool_error_context(
    tool_result: ToolResult,
    action: str,
    arguments: dict[str, object],
    step_index: int,
) -> ErrorContext | None:
    if tool_result.ok:
        return None
    return ErrorContext(
        error_type="ToolExecutionError",
        message=tool_result.error or "Tool execution failed.",
        action=action,
        arguments=dict(arguments),
        step_index=step_index,
    )


def _validate_step_arguments(action: str, arguments: dict[str, object]) -> str | None:
    allowed_arguments = {
        "glob_files": {"pattern", "limit"},
        "find_files_by_name": {"name", "limit"},
        "tree_view": {"path", "depth"},
        "list_files": {"limit"},
        "read_file": {"path"},
        "read_file_head": {"path", "start_line", "max_lines"},
        "search_text": {"query", "max_results", "case_sensitive", "use_regex", "path_glob"},
        "replace_in_file": {"path", "old_text", "new_text", "count"},
        "insert_text": {"path", "anchor", "text", "position"},
        "delete_file": {"path"},
        "move_file": {"source_path", "destination_path"},
        "run_command": {"argv", "working_directory"},
        "run_python_script": {"script", "working_directory"},
        "run_shell": {"command", "working_directory"},
        "run_tests": {"working_directory"},
        "write_file": {"path", "content", "overwrite"},
    }
    required_arguments = {
        "glob_files": {"pattern"},
        "find_files_by_name": {"name"},
        "read_file": {"path"},
        "read_file_head": {"path"},
        "search_text": {"query"},
        "replace_in_file": {"path", "old_text", "new_text"},
        "insert_text": {"path", "anchor", "text"},
        "delete_file": {"path"},
        "move_file": {"source_path", "destination_path"},
        "run_command": {"argv"},
        "run_python_script": {"script"},
        "run_shell": {"command"},
        "write_file": {"path", "content"},
    }

    if action not in allowed_arguments:
        return None

    unknown_arguments = sorted(set(arguments) - allowed_arguments[action])
    if unknown_arguments:
        return (
            f"Action '{action}' does not accept arguments: {', '.join(unknown_arguments)}"
        )

    missing_arguments = sorted(
        key for key in required_arguments.get(action, set()) if key not in arguments
    )
    if missing_arguments:
        return (
            f"Action '{action}' is missing required arguments: {', '.join(missing_arguments)}"
        )
    return None
