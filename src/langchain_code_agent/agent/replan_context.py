from __future__ import annotations

from langchain_code_agent.models.replan import ReplanContext, ReplanFailedStep
from langchain_code_agent.models.result import AttemptResult


def build_replan_context(original_task: str, attempt_result: AttemptResult) -> ReplanContext:
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
        )
        for step in attempt_result.step_results
        if not step.ok
    ]
    return ReplanContext(
        original_task=original_task,
        attempt=attempt_result.attempt,
        previous_plan_summary=attempt_result.plan.summary,
        failed_steps=failed_steps,
        completion_failures=completion_failures,
        successful_actions=successful_actions,
        file_changes=file_changes,
    )
