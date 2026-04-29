from langchain_code_agent.agent.planning_failures import (
    PlanValidationError,
    classify_planning_exception,
)


def test_classify_planning_exception_marks_validation_failures_as_agent_capability() -> None:
    failure = classify_planning_exception(
        PlanValidationError(
            "missing path",
            failure_code="missing_workspace_path",
        ),
        stage="validate_plan",
    )

    assert failure.code == "missing_workspace_path"
    assert failure.origin == "agent_capability"


def test_classify_planning_exception_detects_provider_timeout() -> None:
    failure = classify_planning_exception(
        RuntimeError("Provider returned error with status code 524 gateway timeout."),
        stage="planner_call",
    )

    assert failure.code == "provider_timeout"
    assert failure.origin == "model_service"


def test_classify_planning_exception_detects_provider_timeout_from_response_shape() -> None:
    failure = classify_planning_exception(
        RuntimeError(
            "Response validation failed: input_value={'error': {'message': "
            "'Provider returned error', 'code': 524}}"
        ),
        stage="planner_call",
    )

    assert failure.code == "provider_timeout"
    assert failure.origin == "model_service"


def test_classify_planning_exception_detects_provider_rate_limit() -> None:
    failure = classify_planning_exception(
        RuntimeError("Provider returned error with status code 429 too many requests."),
        stage="planner_call",
    )

    assert failure.code == "provider_rate_limit"
    assert failure.origin == "model_service"


def test_classify_planning_exception_detects_provider_auth_and_config_errors() -> None:
    auth_failure = classify_planning_exception(
        RuntimeError("Unauthorized: invalid API key."),
        stage="planner_call",
    )
    config_failure = classify_planning_exception(
        RuntimeError("Missing API key for configured model profile."),
        stage="planner_call",
    )

    assert auth_failure.code == "provider_auth_error"
    assert auth_failure.origin == "model_service"
    assert config_failure.code == "provider_config_error"
    assert config_failure.origin == "model_service"


def test_classify_planning_exception_falls_back_to_unknown_runtime() -> None:
    failure = classify_planning_exception(
        RuntimeError("planner call crashed before any provider response"),
        stage="planner_call",
    )

    assert failure.code == "planner_call_error"
    assert failure.origin == "unknown_runtime"
