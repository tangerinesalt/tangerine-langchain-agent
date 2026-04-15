from __future__ import annotations

from langchain_code_agent.actions import validate_action_arguments
from langchain_code_agent.models.plan import Plan


def validate_plan(plan: Plan) -> Plan:
    for step in plan.steps:
        validation_error = validate_action_arguments(step.action, step.arguments)
        if validation_error is not None:
            raise ValueError(validation_error)
    return plan
