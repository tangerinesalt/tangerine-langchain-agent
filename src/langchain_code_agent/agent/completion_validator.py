from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_code_agent.models.plan import CompletionCheck, Plan, PlanStep
from langchain_code_agent.models.result import ErrorContext, FileChange, RunResult


def validate_completion(run_result: RunResult) -> list[ErrorContext]:
    if run_result.execution_mode != "execute":
        return []
    if any(not step.ok for step in run_result.step_results):
        return []
    checks = run_result.plan.completion_checks or derive_completion_checks(run_result.plan)
    errors: list[ErrorContext] = []
    for check in checks:
        if _check_satisfied(check, run_result):
            continue
        errors.append(
            ErrorContext(
                error_type="IncompleteTaskResult",
                message=_failure_message(check),
                arguments=dict(check.arguments),
            )
        )
    return errors


def derive_completion_checks(plan: Plan) -> list[CompletionCheck]:
    checks: list[CompletionCheck] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for index, step in enumerate(plan.steps):
        future_removed_paths = _paths_removed_by_steps(plan.steps[index + 1 :])
        derived = _derive_checks_for_step(
            step.action,
            step.arguments,
            future_removed_paths=future_removed_paths,
        )
        for check in derived:
            key = (
                check.check_type,
                tuple(sorted((name, repr(value)) for name, value in check.arguments.items())),
            )
            if key in seen:
                continue
            seen.add(key)
            checks.append(check)
    return checks


def _derive_checks_for_step(
    action: str,
    arguments: dict[str, object],
    *,
    future_removed_paths: set[str] | None = None,
) -> list[CompletionCheck]:
    future_removed_paths = future_removed_paths or set()
    if action == "write_file" and "path" in arguments:
        path = _normalize_path(str(arguments["path"]))
        if path in future_removed_paths:
            return []
        checks = [
            CompletionCheck(
                check_type="file_exists",
                arguments={"path": path},
            )
        ]
        content = arguments.get("content")
        if isinstance(content, str) and content:
            checks.append(
                CompletionCheck(
                    check_type="file_contains",
                    arguments={"path": path, "text": content},
                )
            )
        return checks
    if action == "move_file" and "destination_path" in arguments:
        path = _normalize_path(str(arguments["destination_path"]))
        if path in future_removed_paths:
            return []
        return [
            CompletionCheck(
                check_type="file_exists",
                arguments={"path": path},
            )
        ]
    if action == "delete_file" and "path" in arguments:
        return [
            CompletionCheck(
                check_type="file_absent",
                arguments={"path": str(arguments["path"])},
            )
        ]
    if action in {"insert_text", "replace_in_file"} and "path" in arguments:
        return [
            CompletionCheck(
                check_type="file_changed",
                arguments={"path": _normalize_path(str(arguments["path"]))},
            )
        ]
    if action == "run_tests":
        return [CompletionCheck(check_type="tests_passed", arguments={})]
    return []


def _paths_removed_by_steps(steps: list[PlanStep]) -> set[str]:
    removed: set[str] = set()
    for step in steps:
        if step.action == "delete_file" and "path" in step.arguments:
            removed.add(_normalize_path(str(step.arguments["path"])))
        elif step.action == "move_file" and "source_path" in step.arguments:
            removed.add(_normalize_path(str(step.arguments["source_path"])))
    return removed


def _check_satisfied(check: CompletionCheck, run_result: RunResult) -> bool:
    if check.check_type == "file_exists":
        return _workspace_path_exists(run_result, str(check.arguments["path"]))
    if check.check_type == "file_absent":
        return not _workspace_path_exists(run_result, str(check.arguments["path"]))
    if check.check_type == "file_changed":
        return _path_in_file_changes(
            run_result,
            str(check.arguments["path"]),
            {"added", "modified", "deleted"},
        )
    if check.check_type == "file_contains":
        return _workspace_file_contains(
            run_result,
            str(check.arguments["path"]),
            str(check.arguments["text"]),
        )
    if check.check_type == "action_succeeded":
        action = str(check.arguments["action"])
        return any(step.action == action and step.ok for step in run_result.step_results)
    if check.check_type == "command_exit_code":
        action = str(check.arguments["action"])
        code = int(check.arguments["code"])
        return any(
            step.action == action
            and step.data.get("returncode") == code
            for step in run_result.step_results
        )
    if check.check_type == "shell_output_contains":
        action = str(check.arguments["action"])
        text = str(check.arguments["text"])
        return any(
            step.action == action
            and text in str(step.data.get("stdout", ""))
            and step.ok
            for step in run_result.step_results
        )
    if check.check_type == "tests_passed":
        return any(
            step.action == "run_tests"
            and step.ok
            and step.data.get("returncode", 0) == 0
            for step in run_result.step_results
        )
    if check.check_type == "no_unexpected_file_changes":
        expected_paths = {
            _normalize_path(str(path)) for path in check.arguments.get("paths", [])
        }
        return all(change.path in expected_paths for change in _all_file_changes(run_result))
    return False


def _path_in_file_changes(
    run_result: RunResult,
    path: str,
    change_types: set[str] | None,
) -> bool:
    normalized = path.replace("\\", "/")
    for step in run_result.step_results:
        for change in step.file_changes:
            if change.path != normalized:
                continue
            if change_types is None or change.change_type in change_types:
                return True
    return False


def _workspace_path_exists(run_result: RunResult, path: str) -> bool:
    target = _resolve_workspace_path(run_result, path)
    return False if target is None else target.exists()


def _workspace_file_contains(run_result: RunResult, path: str, text: str) -> bool:
    target = _resolve_workspace_path(run_result, path)
    if target is None or not target.is_file():
        return False
    try:
        return text in target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def _resolve_workspace_path(run_result: RunResult, path: str) -> Path | None:
    root = Path(run_result.workspace_root).resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        return None
    return target


def _all_file_changes(run_result: RunResult) -> Iterator[FileChange]:
    for step in run_result.step_results:
        yield from step.file_changes


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _failure_message(check: CompletionCheck) -> str:
    if check.check_type == "file_exists":
        return f"Expected file to exist after run: {check.arguments['path']}"
    if check.check_type == "file_absent":
        return f"Expected file to be removed after run: {check.arguments['path']}"
    if check.check_type == "file_changed":
        return f"Expected file to change during run: {check.arguments['path']}"
    if check.check_type == "file_contains":
        return (
            f"Expected file {check.arguments['path']} to contain: "
            f"{check.arguments['text']}"
        )
    if check.check_type == "action_succeeded":
        return f"Expected successful action: {check.arguments['action']}"
    if check.check_type == "command_exit_code":
        return (
            f"Expected {check.arguments['action']} to exit with code: "
            f"{check.arguments['code']}"
        )
    if check.check_type == "shell_output_contains":
        return (
            "Expected shell output from "
            f"{check.arguments['action']} to contain: {check.arguments['text']}"
        )
    if check.check_type == "tests_passed":
        return "Expected run_tests to pass."
    if check.check_type == "no_unexpected_file_changes":
        return (
            "Expected no file changes outside: "
            f"{', '.join(str(path) for path in check.arguments['paths'])}"
        )
    return "Completion check failed."
