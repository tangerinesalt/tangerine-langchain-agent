from __future__ import annotations

from langchain_code_agent.models.plan import CompletionCheck, Plan
from langchain_code_agent.models.result import ErrorContext, RunResult


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
    for step in plan.steps:
        derived = _derive_checks_for_step(step.action, step.arguments)
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


def _derive_checks_for_step(action: str, arguments: dict[str, object]) -> list[CompletionCheck]:
    if action == "write_file" and "path" in arguments:
        return [
            CompletionCheck(
                check_type="file_exists",
                arguments={"path": str(arguments["path"])},
            )
        ]
    if action == "move_file" and "destination_path" in arguments:
        return [
            CompletionCheck(
                check_type="file_exists",
                arguments={"path": str(arguments["destination_path"])},
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
                arguments={"path": str(arguments["path"])},
            )
        ]
    if action == "run_tests":
        return [CompletionCheck(check_type="action_succeeded", arguments={"action": "run_tests"})]
    return []


def _check_satisfied(check: CompletionCheck, run_result: RunResult) -> bool:
    if check.check_type == "file_exists":
        return _path_in_file_changes(run_result, str(check.arguments["path"]), change_types=None)
    if check.check_type == "file_absent":
        return _path_in_file_changes(run_result, str(check.arguments["path"]), {"deleted"})
    if check.check_type == "file_changed":
        return _path_in_file_changes(
            run_result,
            str(check.arguments["path"]),
            {"added", "modified", "deleted"},
        )
    if check.check_type == "action_succeeded":
        action = str(check.arguments["action"])
        return any(step.action == action and step.ok for step in run_result.step_results)
    if check.check_type == "shell_output_contains":
        action = str(check.arguments["action"])
        text = str(check.arguments["text"])
        return any(
            step.action == action
            and text in str(step.data.get("stdout", ""))
            and step.ok
            for step in run_result.step_results
        )
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


def _failure_message(check: CompletionCheck) -> str:
    if check.check_type == "file_exists":
        return f"Expected file to exist after run: {check.arguments['path']}"
    if check.check_type == "file_absent":
        return f"Expected file to be removed after run: {check.arguments['path']}"
    if check.check_type == "file_changed":
        return f"Expected file to change during run: {check.arguments['path']}"
    if check.check_type == "action_succeeded":
        return f"Expected successful action: {check.arguments['action']}"
    if check.check_type == "shell_output_contains":
        return (
            "Expected shell output from "
            f"{check.arguments['action']} to contain: {check.arguments['text']}"
        )
    return "Completion check failed."
