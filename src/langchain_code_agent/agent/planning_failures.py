from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

PlanningFailureCode = Literal[
    "json_format_error",
    "invalid_action",
    "invalid_action_arguments",
    "missing_workspace_path",
    "missing_edit_step",
    "missing_validation_step",
    "validation_before_edit",
    "unsatisfiable_completion_check",
    "unsupported_planner_response",
    "planner_call_error",
    "unknown_planning_failure",
]

JSON_FORMAT_ERROR: PlanningFailureCode = "json_format_error"
INVALID_ACTION: PlanningFailureCode = "invalid_action"
INVALID_ACTION_ARGUMENTS: PlanningFailureCode = "invalid_action_arguments"
MISSING_WORKSPACE_PATH: PlanningFailureCode = "missing_workspace_path"
MISSING_EDIT_STEP: PlanningFailureCode = "missing_edit_step"
MISSING_VALIDATION_STEP: PlanningFailureCode = "missing_validation_step"
VALIDATION_BEFORE_EDIT: PlanningFailureCode = "validation_before_edit"
UNSATISFIABLE_COMPLETION_CHECK: PlanningFailureCode = "unsatisfiable_completion_check"
UNSUPPORTED_PLANNER_RESPONSE: PlanningFailureCode = "unsupported_planner_response"
PLANNER_CALL_ERROR: PlanningFailureCode = "planner_call_error"
UNKNOWN_PLANNING_FAILURE: PlanningFailureCode = "unknown_planning_failure"


@dataclass(frozen=True, slots=True)
class PlanningFailure:
    code: PlanningFailureCode
    repairable: bool = False


class PlanValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        failure_code: PlanningFailureCode,
        repairable: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.repairable = repairable


def classify_planning_exception(exc: Exception, *, stage: str) -> PlanningFailure:
    if isinstance(exc, PlanValidationError):
        return PlanningFailure(code=exc.failure_code, repairable=exc.repairable)

    message = str(exc)
    if isinstance(exc, ValidationError):
        if "Unsupported action" in message:
            return PlanningFailure(code=INVALID_ACTION)
        return PlanningFailure(code=JSON_FORMAT_ERROR)

    if "Planner returned invalid JSON" in message:
        return PlanningFailure(code=JSON_FORMAT_ERROR)
    if "unsupported structured response" in message:
        return PlanningFailure(code=UNSUPPORTED_PLANNER_RESPONSE)
    if stage == "planner_call":
        return PlanningFailure(code=PLANNER_CALL_ERROR)
    return PlanningFailure(code=UNKNOWN_PLANNING_FAILURE)
