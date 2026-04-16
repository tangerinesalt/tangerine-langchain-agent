import pytest

from langchain_code_agent.agent.plan_validator import validate_plan
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
