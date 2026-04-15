import pytest

from langchain_code_agent.agent.plan_validator import validate_plan
from langchain_code_agent.models.plan import Plan, PlanStep


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
