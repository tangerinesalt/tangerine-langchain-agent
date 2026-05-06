from __future__ import annotations

from typing import Any

from langchain_code_agent.agent.planning.failures import (
    PlannerRevisionError,
    classify_planning_exception,
)
from langchain_code_agent.agent.planning.normalization_rules import apply_plan_normalization_rules
from langchain_code_agent.agent.planning.repair import PlanRepairResult, repair_plan
from langchain_code_agent.agent.planning.validator import (
    validate_plan,
    validate_task_specific_plan,
)
from langchain_code_agent.models.plan import Plan
from langchain_code_agent.models.task import Task


def validate_or_repair_task_specific_plan(
    plan: Plan,
    *,
    planner: Any,
    task: Task,
    task_text: str,
    existing_paths: set[str],
) -> tuple[Plan, PlanRepairResult | None, dict[str, str] | None]:
    try:
        return (
            validate_task_specific_plan(
                plan,
                task_text=task_text,
                planning_context=task.planning_context,
            ),
            None,
            None,
        )
    except Exception as exc:
        planning_failure = classify_planning_exception(
            exc,
            stage="validate_task_specific_plan",
        )
        if planning_failure.code == "missing_edit_step":
            try:
                revised_plan = planner.revise_plan(
                    task,
                    invalid_plan=plan,
                    failure_code=planning_failure.code,
                    failure_message=str(exc),
                )
            except Exception as revision_exc:
                raise PlannerRevisionError(revision_exc) from revision_exc
            revised_plan = apply_plan_normalization_rules(
                revised_plan,
                task=task,
                workspace_root=task.workspace_root,
            )
            revised_plan = validate_plan(revised_plan, existing_paths=existing_paths)
            revised_plan = validate_task_specific_plan(
                revised_plan,
                task_text=task_text,
                planning_context=task.planning_context,
            )
            return (
                revised_plan,
                None,
                {
                    "failure_code": planning_failure.code,
                    "failure_message": str(exc),
                },
            )
        repair_result = repair_plan(
            plan,
            task_text=task_text,
            failure_code=planning_failure.code,
        )
        if repair_result is None:
            raise

        repaired_plan = validate_plan(repair_result.plan, existing_paths=existing_paths)
        repaired_plan = validate_task_specific_plan(
            repaired_plan,
            task_text=task_text,
            planning_context=task.planning_context,
        )
        return repaired_plan, repair_result, None
