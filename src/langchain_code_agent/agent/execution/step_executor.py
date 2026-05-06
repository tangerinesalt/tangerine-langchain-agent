from __future__ import annotations

from langchain_code_agent.actions import ActionRuntime, execute_action, validate_action_arguments
from langchain_code_agent.models.result import ErrorContext
from langchain_code_agent.tools.base import ToolResult


class StepExecutor:
    def __init__(self, action_runtime: ActionRuntime) -> None:
        self.action_runtime = action_runtime

    def execute_step(
        self,
        action: str,
        arguments: dict[str, object],
        step_index: int,
    ) -> tuple[ToolResult, ErrorContext | None]:
        try:
            validation_error = validate_action_arguments(action, arguments)
            if validation_error is not None:
                error_context = ErrorContext(
                    error_type="InvalidStepArguments",
                    message=validation_error,
                    action=action,
                    arguments=dict(arguments),
                    step_index=step_index,
                )
                return ToolResult(ok=False, error=validation_error), error_context

            tool_result = execute_action(action, self.action_runtime, arguments)
            return (
                tool_result,
                _tool_error_context(tool_result, action, arguments, step_index),
            )
        except Exception as exc:
            error_context = ErrorContext(
                error_type=type(exc).__name__,
                message=str(exc),
                action=action,
                arguments=dict(arguments),
                step_index=step_index,
            )
            return ToolResult(ok=False, error=str(exc)), error_context


def _tool_error_context(
    tool_result: ToolResult,
    action: str,
    arguments: dict[str, object],
    step_index: int,
) -> ErrorContext | None:
    if tool_result.ok:
        return None
    return ErrorContext(
        error_type="ToolExecutionError",
        message=tool_result.error or "Tool execution failed.",
        action=action,
        arguments=dict(arguments),
        step_index=step_index,
    )
