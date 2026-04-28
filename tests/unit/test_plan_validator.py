import pytest

from langchain_code_agent.agent.plan_validator import validate_plan, validate_task_specific_plan
from langchain_code_agent.agent.planning_failures import PlanValidationError
from langchain_code_agent.models.plan import CompletionCheck, Plan, PlanStep


def test_validate_plan_accepts_valid_steps() -> None:
    plan = Plan(
        summary="Write a file.",
        steps=[
            PlanStep(
                action="write_file",
                description="Write notes.",
                arguments={"path": "notes.txt", "content": "hello"},
            )
        ],
    )

    assert validate_plan(plan) == plan


def test_validate_plan_rejects_invalid_arguments() -> None:
    plan = Plan(
        summary="Invalid plan.",
        steps=[
            PlanStep(
                action="list_files",
                description="Bad args.",
                arguments={"path": "."},
            )
        ],
    )

    with pytest.raises(ValueError, match="does not accept arguments"):
        validate_plan(plan)


def test_validate_plan_rejects_invalid_completion_check_arguments() -> None:
    plan = Plan(
        summary="Invalid completion check.",
        steps=[],
        completion_checks=[
            CompletionCheck(check_type="file_exists", arguments={"action": "write_file"})
        ],
    )

    with pytest.raises(ValueError, match="does not accept arguments: action"):
        validate_plan(plan)


def test_validate_plan_rejects_reading_file_not_in_workspace_or_prior_steps() -> None:
    plan = Plan(
        summary="Read missing file.",
        steps=[
            PlanStep(
                action="read_file_head",
                description="Read pytest log.",
                arguments={"path": "pytest.log", "max_lines": 20},
            )
        ],
    )

    with pytest.raises(ValueError, match="not available in the workspace"):
        validate_plan(plan, existing_paths={"test_ok.py"})


def test_validate_plan_classifies_invalid_action() -> None:
    plan = Plan(
        summary="Invent a tool.",
        steps=[
            PlanStep(
                action="edit_everything",
                description="Use an unsupported tool.",
                arguments={},
            )
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_plan(plan, existing_paths=set())

    assert exc_info.value.failure_code == "invalid_action"


def test_validate_plan_rejects_completion_check_for_missing_action() -> None:
    plan = Plan(
        summary="Inspect only.",
        steps=[
            PlanStep(
                action="list_files",
                description="Inspect files.",
                arguments={"limit": 10},
            )
        ],
        completion_checks=[
            CompletionCheck(check_type="action_succeeded", arguments={"action": "run_tests"})
        ],
    )

    with pytest.raises(ValueError, match="references missing action"):
        validate_plan(plan, existing_paths={"main.py"})


def test_validate_task_specific_plan_accepts_fix_failing_tests_with_edit_and_verification() -> None:
    plan = Plan(
        summary="Fix the implementation and verify.",
        steps=[
            PlanStep(
                action="read_file",
                description="Read the implementation.",
                arguments={"path": "math_utils.py"},
            ),
            PlanStep(
                action="replace_in_file",
                description="Fix the implementation.",
                arguments={
                    "path": "math_utils.py",
                    "old_text": "return value + factor + offset",
                    "new_text": "return value * factor + offset",
                },
            ),
            PlanStep(
                action="run_tests",
                description="Verify the fix.",
                arguments={},
            ),
        ],
    )

    assert (
        validate_task_specific_plan(plan, task_text="Fix the failing tests in this workspace.")
        == plan
    )


def test_validate_task_specific_plan_rejects_fix_failing_tests_without_edit_step() -> None:
    plan = Plan(
        summary="Inspect only.",
        steps=[
            PlanStep(
                action="read_file",
                description="Read the implementation.",
                arguments={"path": "math_utils.py"},
            ),
            PlanStep(
                action="run_tests",
                description="Run tests.",
                arguments={},
            ),
        ],
    )

    with pytest.raises(ValueError, match="must include at least one edit step"):
        validate_task_specific_plan(plan, task_text="Fix the failing tests in this workspace.")


def test_validate_task_specific_plan_classifies_missing_verification() -> None:
    plan = Plan(
        summary="Edit without verification.",
        steps=[
            PlanStep(
                action="replace_in_file",
                description="Fix the implementation.",
                arguments={
                    "path": "math_utils.py",
                    "old_text": "return value + factor + offset",
                    "new_text": "return value * factor + offset",
                },
            )
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_task_specific_plan(plan, task_text="Fix the failing tests in this workspace.")

    assert exc_info.value.failure_code == "missing_validation_step"
    assert exc_info.value.repairable is True


def test_validate_task_specific_plan_rejects_verification_before_edit() -> None:
    plan = Plan(
        summary="Run tests too early.",
        steps=[
            PlanStep(
                action="run_tests",
                description="Run tests.",
                arguments={},
            ),
            PlanStep(
                action="replace_in_file",
                description="Fix the implementation.",
                arguments={
                    "path": "math_utils.py",
                    "old_text": "return value + factor + offset",
                    "new_text": "return value * factor + offset",
                },
            ),
        ],
    )

    with pytest.raises(ValueError, match="must run run_tests after the planned edit steps"):
        validate_task_specific_plan(plan, task_text="Fix the failing tests in this workspace.")
