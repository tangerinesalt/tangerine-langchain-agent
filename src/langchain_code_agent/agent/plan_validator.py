from __future__ import annotations

from pathlib import PurePosixPath

from langchain_code_agent.actions import validate_action_arguments
from langchain_code_agent.models.plan import CompletionCheck, Plan


READ_ACTIONS = {"read_file", "read_file_head"}
EDIT_ACTIONS = {"replace_in_file", "insert_text"}
SHELL_ACTIONS = {"run_command", "run_python_script", "run_shell", "run_tests"}


def validate_plan(plan: Plan, *, existing_paths: set[str] | None = None) -> Plan:
    for step in plan.steps:
        validation_error = validate_action_arguments(step.action, step.arguments)
        if validation_error is not None:
            raise ValueError(validation_error)
    for check in plan.completion_checks:
        validation_error = validate_completion_check(check)
        if validation_error is not None:
            raise ValueError(validation_error)
    if existing_paths is not None:
        _validate_plan_semantics(plan, existing_paths=existing_paths)
    return plan


def validate_completion_check(check: CompletionCheck) -> str | None:
    required_arguments = COMPLETION_CHECK_ARGUMENTS.get(check.check_type, frozenset())
    unknown_arguments = sorted(set(check.arguments) - required_arguments)
    if unknown_arguments:
        return (
            f"Completion check '{check.check_type}' does not accept arguments: "
            f"{', '.join(unknown_arguments)}"
        )

    missing_arguments = sorted(name for name in required_arguments if name not in check.arguments)
    if missing_arguments:
        return (
            f"Completion check '{check.check_type}' is missing required arguments: "
            f"{', '.join(missing_arguments)}"
        )
    return None


COMPLETION_CHECK_ARGUMENTS: dict[str, frozenset[str]] = {
    "file_exists": frozenset({"path"}),
    "file_absent": frozenset({"path"}),
    "file_changed": frozenset({"path"}),
    "action_succeeded": frozenset({"action"}),
    "shell_output_contains": frozenset({"action", "text"}),
}


def _validate_plan_semantics(plan: Plan, *, existing_paths: set[str]) -> None:
    known_files = {_normalize_path(path) for path in existing_paths}
    touched_files: set[str] = set()
    planned_actions = [step.action for step in plan.steps]

    for step in plan.steps:
        action = step.action
        arguments = step.arguments
        if action in READ_ACTIONS:
            path = _normalize_path(str(arguments["path"]))
            if path not in known_files:
                raise ValueError(
                    f"Action '{action}' references a file that is not available in the "
                    f"workspace or produced by earlier steps: {path}"
                )
        elif action in EDIT_ACTIONS:
            path = _normalize_path(str(arguments["path"]))
            if path not in known_files:
                raise ValueError(
                    f"Action '{action}' cannot modify a file that is not available: {path}"
                )
            touched_files.add(path)
        elif action == "write_file":
            path = _normalize_path(str(arguments["path"]))
            known_files.add(path)
            touched_files.add(path)
        elif action == "delete_file":
            path = _normalize_path(str(arguments["path"]))
            if path not in known_files:
                raise ValueError(f"Action 'delete_file' cannot remove missing file: {path}")
            known_files.remove(path)
            touched_files.add(path)
        elif action == "move_file":
            source_path = _normalize_path(str(arguments["source_path"]))
            destination_path = _normalize_path(str(arguments["destination_path"]))
            if source_path not in known_files:
                raise ValueError(
                    "Action 'move_file' references a source file that is not available: "
                    f"{source_path}"
                )
            known_files.remove(source_path)
            known_files.add(destination_path)
            touched_files.add(source_path)
            touched_files.add(destination_path)
        elif action in SHELL_ACTIONS:
            continue

    for check in plan.completion_checks:
        if check.check_type == "file_exists":
            path = _normalize_path(str(check.arguments["path"]))
            if path not in known_files:
                raise ValueError(
                    "Completion check 'file_exists' references a file that is not present "
                    f"after the planned steps: {path}"
                )
        elif check.check_type == "file_absent":
            path = _normalize_path(str(check.arguments["path"]))
            if path in known_files:
                raise ValueError(
                    "Completion check 'file_absent' references a file that still exists "
                    f"after the planned steps: {path}"
                )
        elif check.check_type == "file_changed":
            path = _normalize_path(str(check.arguments["path"]))
            if path not in touched_files:
                raise ValueError(
                    "Completion check 'file_changed' references a file that is not changed "
                    f"by the current plan: {path}"
                )
        elif check.check_type in {"action_succeeded", "shell_output_contains"}:
            action_name = str(check.arguments["action"])
            if action_name not in planned_actions:
                raise ValueError(
                    f"Completion check '{check.check_type}' references missing action: "
                    f"{action_name}"
                )


def _normalize_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")
