from __future__ import annotations

from dataclasses import dataclass

from langchain_code_agent.agent.plan_validator import (
    MUTATING_ACTIONS,
    is_fix_failing_tests_task,
)
from langchain_code_agent.agent.planning_failures import (
    MISSING_VALIDATION_STEP,
    VALIDATION_BEFORE_EDIT,
    PlanningFailureCode,
)
from langchain_code_agent.models.plan import Plan, PlanStep

APPEND_RUN_TESTS_REPAIR = "append_run_tests_verification"


@dataclass(frozen=True, slots=True)
class PlanRepairResult:
    plan: Plan
    repair_code: str
    repaired_failure_code: PlanningFailureCode
    reason: str


def repair_plan(
    plan: Plan,
    *,
    task_text: str,
    failure_code: PlanningFailureCode,
) -> PlanRepairResult | None:
    if failure_code not in {MISSING_VALIDATION_STEP, VALIDATION_BEFORE_EDIT}:
        return None
    if not is_fix_failing_tests_task(task_text):
        return None

    last_edit_index = max(
        (index for index, step in enumerate(plan.steps) if step.action in MUTATING_ACTIONS),
        default=-1,
    )
    if last_edit_index == -1:
        return None
    if any(
        step.action == "run_tests" and index > last_edit_index
        for index, step in enumerate(plan.steps)
    ):
        return None

    repaired_steps = [
        *plan.steps,
        PlanStep(
            action="run_tests",
            description="Verify the test fix with the configured test command.",
            arguments={},
        ),
    ]
    return PlanRepairResult(
        plan=plan.model_copy(update={"steps": repaired_steps}),
        repair_code=APPEND_RUN_TESTS_REPAIR,
        repaired_failure_code=failure_code,
        reason="Added a final run_tests verification step after existing edit steps.",
    )
