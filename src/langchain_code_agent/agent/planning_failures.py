from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

PlanningFailureOrigin = Literal[
    "agent_capability",
    "model_service",
    "unknown_runtime",
]

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
    "provider_timeout",
    "provider_rate_limit",
    "provider_auth_error",
    "provider_config_error",
    "provider_unknown_error",
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
PROVIDER_TIMEOUT: PlanningFailureCode = "provider_timeout"
PROVIDER_RATE_LIMIT: PlanningFailureCode = "provider_rate_limit"
PROVIDER_AUTH_ERROR: PlanningFailureCode = "provider_auth_error"
PROVIDER_CONFIG_ERROR: PlanningFailureCode = "provider_config_error"
PROVIDER_UNKNOWN_ERROR: PlanningFailureCode = "provider_unknown_error"
PLANNER_CALL_ERROR: PlanningFailureCode = "planner_call_error"
UNKNOWN_PLANNING_FAILURE: PlanningFailureCode = "unknown_planning_failure"


@dataclass(frozen=True, slots=True)
class PlanningFailure:
    code: PlanningFailureCode
    origin: PlanningFailureOrigin = "agent_capability"
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
        return PlanningFailure(
            code=exc.failure_code,
            origin="agent_capability",
            repairable=exc.repairable,
        )

    message = str(exc)
    if isinstance(exc, ValidationError):
        if "Unsupported action" in message:
            return PlanningFailure(code=INVALID_ACTION, origin="agent_capability")
        return PlanningFailure(code=JSON_FORMAT_ERROR, origin="agent_capability")

    if "Planner returned invalid JSON" in message:
        return PlanningFailure(code=JSON_FORMAT_ERROR, origin="agent_capability")
    if "unsupported structured response" in message:
        return PlanningFailure(code=UNSUPPORTED_PLANNER_RESPONSE, origin="unknown_runtime")
    if stage == "planner_call":
        return _classify_planner_call_exception(exc)
    return PlanningFailure(code=UNKNOWN_PLANNING_FAILURE, origin="agent_capability")


def _classify_planner_call_exception(exc: Exception) -> PlanningFailure:
    message = f"{type(exc).__name__}: {exc}".lower()
    if _contains_any(
        message,
        (
            "gateway timeout",
            "timed out",
            "timeout",
            "deadline exceeded",
            "read timeout",
            "connect timeout",
            "code 524",
            "'code': 524",
            "\"code\": 524",
            "status code 524",
        ),
    ):
        return PlanningFailure(code=PROVIDER_TIMEOUT, origin="model_service")
    if _contains_any(
        message,
        (
            "rate limit",
            "too many requests",
            "status code 429",
            "code 429",
        ),
    ):
        return PlanningFailure(code=PROVIDER_RATE_LIMIT, origin="model_service")
    if _contains_any(
        message,
        (
            "missing api key",
            "api key is required",
            "no api key",
            "model config",
            "configuration error",
            "invalid base url",
            "model profile",
        ),
    ):
        return PlanningFailure(code=PROVIDER_CONFIG_ERROR, origin="model_service")
    if _contains_any(
        message,
        (
            "unauthorized",
            "forbidden",
            "authentication",
            "invalid api key",
            "incorrect api key",
            "status code 401",
            "status code 403",
            "code 401",
            "code 403",
        ),
    ):
        return PlanningFailure(code=PROVIDER_AUTH_ERROR, origin="model_service")
    if _contains_any(message, ("provider returned error", "openrouter", "openai")):
        return PlanningFailure(code=PROVIDER_UNKNOWN_ERROR, origin="model_service")
    return PlanningFailure(code=PLANNER_CALL_ERROR, origin="unknown_runtime")


def _contains_any(message: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in message for pattern in patterns)
