from __future__ import annotations

from pathlib import PurePosixPath

from langchain_code_agent.actions import get_action_spec, validate_action_arguments
from langchain_code_agent.agent.planning_failures import (
    INVALID_ACTION,
    INVALID_ACTION_ARGUMENTS,
    MISSING_EDIT_STEP,
    MISSING_VALIDATION_STEP,
    MISSING_WORKSPACE_PATH,
    UNSATISFIABLE_COMPLETION_CHECK,
    VALIDATION_BEFORE_EDIT,
    PlanValidationError,
)
from langchain_code_agent.models.plan import CompletionCheck, Plan

READ_ACTIONS = {"read_file", "read_file_head"}
EDIT_ACTIONS = {"replace_in_file", "insert_text"}
SHELL_ACTIONS = {"run_command", "run_python_script", "run_shell", "run_tests"}
MUTATING_ACTIONS = EDIT_ACTIONS | {"write_file", "move_file", "delete_file"}


def validate_plan(plan: Plan, *, existing_paths: set[str] | None = None) -> Plan:
    for step in plan.steps:
        if get_action_spec(step.action) is None:
            raise PlanValidationError(
                f"Unsupported action: {step.action}",
                failure_code=INVALID_ACTION,
            )
        validation_error = validate_action_arguments(step.action, step.arguments)
        if validation_error is not None:
            raise PlanValidationError(
                validation_error,
                failure_code=INVALID_ACTION_ARGUMENTS,
            )
    for check in plan.completion_checks:
        validation_error = validate_completion_check(check)
        if validation_error is not None:
            raise PlanValidationError(
                validation_error,
                failure_code=INVALID_ACTION_ARGUMENTS,
            )
    if existing_paths is not None:
        _validate_plan_semantics(plan, existing_paths=existing_paths)
    return plan


def validate_task_specific_plan(plan: Plan, *, task_text: str) -> Plan:
    if not is_fix_failing_tests_task(task_text):
        return plan

    last_edit_index = max(
        (index for index, step in enumerate(plan.steps) if step.action in MUTATING_ACTIONS),
        default=-1,
    )
    if last_edit_index == -1:
        raise PlanValidationError(
            "Fix-failing-tests tasks must include at least one edit step such as "
            "replace_in_file, insert_text, write_file, move_file, or delete_file.",
            failure_code=MISSING_EDIT_STEP,
        )

    if not any(step.action == "run_tests" for step in plan.steps):
        raise PlanValidationError(
            "Fix-failing-tests tasks must include a final run_tests verification step.",
            failure_code=MISSING_VALIDATION_STEP,
            repairable=True,
        )

    if not any(
        step.action == "run_tests" and index > last_edit_index
        for index, step in enumerate(plan.steps)
    ):
        raise PlanValidationError(
            "Fix-failing-tests tasks must run run_tests after the planned edit steps.",
            failure_code=VALIDATION_BEFORE_EDIT,
            repairable=True,
        )

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
                raise PlanValidationError(
                    f"Action '{action}' references a file that is not available in the "
                    f"workspace or produced by earlier steps: {path}",
                    failure_code=MISSING_WORKSPACE_PATH,
                )
        elif action in EDIT_ACTIONS:
            path = _normalize_path(str(arguments["path"]))
            if path not in known_files:
                raise PlanValidationError(
                    f"Action '{action}' cannot modify a file that is not available: {path}",
                    failure_code=MISSING_WORKSPACE_PATH,
                )
            touched_files.add(path)
        elif action == "write_file":
            path = _normalize_path(str(arguments["path"]))
            known_files.add(path)
            touched_files.add(path)
        elif action == "delete_file":
            path = _normalize_path(str(arguments["path"]))
            if path not in known_files:
                raise PlanValidationError(
                    f"Action 'delete_file' cannot remove missing file: {path}",
                    failure_code=MISSING_WORKSPACE_PATH,
                )
            known_files.remove(path)
            touched_files.add(path)
        elif action == "move_file":
            source_path = _normalize_path(str(arguments["source_path"]))
            destination_path = _normalize_path(str(arguments["destination_path"]))
            if source_path not in known_files:
                raise PlanValidationError(
                    "Action 'move_file' references a source file that is not available: "
                    f"{source_path}",
                    failure_code=MISSING_WORKSPACE_PATH,
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
                raise PlanValidationError(
                    "Completion check 'file_exists' references a file that is not present "
                    f"after the planned steps: {path}",
                    failure_code=UNSATISFIABLE_COMPLETION_CHECK,
                )
        elif check.check_type == "file_absent":
            path = _normalize_path(str(check.arguments["path"]))
            if path in known_files:
                raise PlanValidationError(
                    "Completion check 'file_absent' references a file that still exists "
                    f"after the planned steps: {path}",
                    failure_code=UNSATISFIABLE_COMPLETION_CHECK,
                )
        elif check.check_type == "file_changed":
            path = _normalize_path(str(check.arguments["path"]))
            if path not in touched_files:
                raise PlanValidationError(
                    "Completion check 'file_changed' references a file that is not changed "
                    f"by the current plan: {path}",
                    failure_code=UNSATISFIABLE_COMPLETION_CHECK,
                )
        elif check.check_type in {"action_succeeded", "shell_output_contains"}:
            action_name = str(check.arguments["action"])
            if action_name not in planned_actions:
                raise PlanValidationError(
                    f"Completion check '{check.check_type}' references missing action: "
                    f"{action_name}",
                    failure_code=UNSATISFIABLE_COMPLETION_CHECK,
                )


def _normalize_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def is_fix_failing_tests_task(task_text: str) -> bool:
    lowered = task_text.lower()
    fix_markers = ("fix", "repair", "resolve", "make ")
    test_markers = ("failing test", "failing tests", "pytest", "tests", "test suite")
    return any(marker in lowered for marker in fix_markers) and any(
        marker in lowered for marker in test_markers
    )
