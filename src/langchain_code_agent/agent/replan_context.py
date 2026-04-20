from __future__ import annotations

from langchain_code_agent.models.replan import ReplanContext, ReplanFailedStep
from langchain_code_agent.models.result import AttemptResult

MAX_REPLAN_OUTPUT_CHARS = 1200


def build_replan_context(original_task: str, attempt_result: AttemptResult) -> ReplanContext:
    attempt_failures = sorted(
        {
            error.message
            for error in attempt_result.errors
            if error.message and error.error_type != "IncompleteTaskResult"
        }
    )
    completion_failures = [
        error.message for error in attempt_result.completion_errors if error.message
    ]
    successful_actions = [step.action for step in attempt_result.step_results if step.ok]
    file_changes = sorted(
        {
            change.path
            for step in attempt_result.step_results
            for change in step.file_changes
        }
    )
    failed_steps = [
        ReplanFailedStep(
            action=step.action,
            message=step.error or "unknown error",
            arguments=dict(step.arguments),
            error_type=step.error_context.error_type if step.error_context is not None else None,
            step_index=step.error_context.step_index if step.error_context is not None else None,
            stdout_excerpt=_excerpt_output(step.data.get("stdout")),
            stderr_excerpt=_excerpt_output(step.data.get("stderr")),
        )
        for step in attempt_result.step_results
        if not step.ok
    ]
    return ReplanContext(
        original_task=original_task,
        attempt=attempt_result.attempt,
        previous_plan_summary=attempt_result.plan.summary,
        failed_steps=failed_steps,
        attempt_failures=attempt_failures,
        completion_failures=completion_failures,
        successful_actions=successful_actions,
        file_changes=file_changes,
    )


def _excerpt_output(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) <= MAX_REPLAN_OUTPUT_CHARS:
        return stripped
    return stripped[:MAX_REPLAN_OUTPUT_CHARS].rstrip() + "\n...[truncated]"
